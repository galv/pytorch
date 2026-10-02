"""Lossless EP=8 MoE routing with bounded, CUDA-graphed receive buffers.

Run with:
  NCCL_GRAPH_MIXING_SUPPORT=0 torchrun --standalone --nproc-per-node=8 \
      poster_examples/ep8_moe_backpressure.py
"""

import os

import torch
import torch.distributed as dist
from torch._higher_order_ops.cudagraph_conditional_nodes import (
    ControlFlowOpWarmupDispatchMode,
    CUDAGraphCaptureControlFlowOpDispatchMode,
)


WORLD_SIZE = 8
TOKENS_PER_RANK = 4096
RECEIVE_CAPACITY = 4096
PEER_CAPACITY = RECEIVE_CAPACITY // WORLD_SIZE
LOCAL_EXPERTS = 8
TOTAL_EXPERTS = WORLD_SIZE * LOCAL_EXPERTS
HIDDEN = 64
EXPERT_WIDTH = 128


@torch.library.custom_op("poster::sync_all_to_all", mutates_args=())
def sync_all_to_all(tensor: torch.Tensor) -> torch.Tensor:
    # Functional collectives use NCCL's auxiliary stream, which cannot join a
    # WHILE child capture after capture of its parent graph has begun.
    output = torch.empty_like(tensor)
    dist.all_to_all_single(output, tensor, async_op=False)
    return output


@sync_all_to_all.register_fake
def _(tensor):
    return torch.empty_like(tensor)


@torch.library.custom_op("poster::sync_all_reduce_max", mutates_args=())
def sync_all_reduce_max(tensor: torch.Tensor) -> torch.Tensor:
    output = tensor.clone()
    dist.all_reduce(output, op=dist.ReduceOp.MAX, async_op=False)
    return output


@sync_all_reduce_max.register_fake
def _(tensor):
    return torch.empty_like(tensor)


def ep8_moe(tokens, routes, local_weights):
    peers = torch.arange(WORLD_SIZE, device=tokens.device)[:, None]
    local_expert_ids = torch.arange(LOCAL_EXPERTS, device=tokens.device)
    destinations = routes // LOCAL_EXPERTS

    tokens_per_destination = (destinations[None, :] == peers).sum(1)
    local_rounds = ((tokens_per_destination + PEER_CAPACITY - 1) // PEER_CAPACITY).max()
    total_rounds = sync_all_reduce_max(local_rounds)

    done = torch.zeros(TOKENS_PER_RANK, dtype=torch.bool, device=tokens.device)
    output = tokens.new_zeros(TOKENS_PER_RANK, EXPERT_WIDTH)
    iteration = torch.zeros((), dtype=torch.int64, device=tokens.device)

    def cond(iteration, done, output):
        return iteration < total_rounds

    def body(iteration, done, output):
        candidates = ((destinations[None, :] == peers) & ~done[None, :]).to(torch.int64)
        valid, token_indices = torch.topk(candidates, PEER_CAPACITY, dim=1)

        send_tokens = tokens[token_indices]
        send_metadata = torch.stack(
            (routes[token_indices] % LOCAL_EXPERTS, valid), dim=-1
        )
        received_tokens = sync_all_to_all(send_tokens).flatten(0, 1)
        received_metadata = sync_all_to_all(send_metadata).flatten(0, 1)

        received_experts = received_metadata[:, 0]
        received_valid = received_metadata[:, 1].bool()
        sort_keys = torch.where(received_valid, received_experts, LOCAL_EXPERTS)
        order = torch.argsort(sort_keys, stable=True)
        packed_tokens = received_tokens[order]
        packed_experts = received_experts[order]
        packed_valid = received_valid[order]

        counts = (
            (packed_experts[:, None] == local_expert_ids[None, :])
            & packed_valid[:, None]
        ).sum(0)
        offsets = torch.cumsum(counts, 0, dtype=torch.int32)

        packed_output = torch.nn.functional.grouped_mm(
            packed_tokens,
            local_weights.transpose(-2, -1),
            offs=offsets,
        )

        packed_output = torch.where(
            packed_valid[:, None], packed_output, torch.zeros_like(packed_output)
        )
        received_output = torch.zeros_like(packed_output).index_copy(
            0, order, packed_output
        )
        returned_output = sync_all_to_all(
            received_output.view(WORLD_SIZE, PEER_CAPACITY, EXPERT_WIDTH)
        )

        returned_output = returned_output * valid[..., None]
        output = output.index_add(
            0, token_indices.flatten(), returned_output.flatten(0, 1)
        )
        selected = torch.zeros_like(done, dtype=torch.int64).scatter_add(
            0, token_indices.flatten(), valid.flatten()
        )
        return iteration + 1, done | selected.bool(), output

    iteration, _, output = torch.while_loop(cond, body, (iteration, done, output))
    return output, iteration


def capture(fn, *args):
    fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
    dist.barrier(device_ids=[torch.cuda.current_device()])
    with torch.no_grad():
        fn(*args)
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with (
        torch.no_grad(),
        torch.cuda.stream(stream),
        ControlFlowOpWarmupDispatchMode(),
    ):
        fn(*args)
    stream.synchronize()
    dist.barrier(device_ids=[torch.cuda.current_device()])

    graph = torch.cuda.CUDAGraph()
    with (
        torch.no_grad(),
        torch.cuda.graph(graph, stream=stream),
        CUDAGraphCaptureControlFlowOpDispatchMode(),
    ):
        output = fn(*args)
    stream.synchronize()
    dist.barrier(device_ids=[torch.cuda.current_device()])
    return graph, output


def main():
    os.environ.setdefault("NCCL_GRAPH_MIXING_SUPPORT", "0")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()

    if dist.get_world_size() != WORLD_SIZE:
        raise RuntimeError("run with torchrun --nproc-per-node=8")
    if torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("this 2D grouped_mm example requires Hopper or newer")

    torch.manual_seed(0)
    global_weights = torch.randn(
        TOTAL_EXPERTS,
        EXPERT_WIDTH,
        HIDDEN,
        dtype=torch.bfloat16,
        device="cuda",
    )
    local_weights = global_weights[
        rank * LOCAL_EXPERTS : (rank + 1) * LOCAL_EXPERTS
    ].contiguous()

    torch.manual_seed(rank + 1)
    tokens = torch.randn(TOKENS_PER_RANK, HIDDEN, dtype=torch.bfloat16, device="cuda")
    routes = torch.arange(TOKENS_PER_RANK, device="cuda") % TOTAL_EXPERTS
    graph, (output, rounds) = capture(ep8_moe, tokens, routes, local_weights)

    balanced = (
        torch.arange(TOKENS_PER_RANK, device="cuda") + rank * TOKENS_PER_RANK
    ) % TOTAL_EXPERTS
    asymmetric = balanced.clone()
    if rank == 0:
        asymmetric = torch.arange(TOKENS_PER_RANK, device="cuda") % LOCAL_EXPERTS

    for name, routing, expected_rounds in (
        ("balanced", balanced, 1),
        ("rank-0 source hotspot", asymmetric, 8),
    ):
        routes.copy_(routing)
        dist.barrier(device_ids=[local_rank])
        graph.replay()
        torch.cuda.synchronize()

        expected = torch.bmm(
            tokens[:, None, :], global_weights[routes].transpose(-2, -1)
        ).squeeze(1)
        torch.testing.assert_close(output, expected)
        if rounds.item() != expected_rounds:
            raise AssertionError(
                f"{name}: expected {expected_rounds} rounds, got {rounds.item()}"
            )
        dist.barrier(device_ids=[local_rank])

    if rank == 0:
        print("EP=8 CUDA graph passed (2D grouped_mm).")

    graph.reset()
    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
