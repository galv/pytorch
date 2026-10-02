"""Shape-stable dynamic control flow captured in CUDA graphs."""

import torch
from torch._higher_order_ops.cudagraph_conditional_nodes import (
    ControlFlowOpWarmupDispatchMode,
    CUDAGraphCaptureControlFlowOpDispatchMode,
)


def capture(fn, *args):
    """Compile ``fn`` and capture its HOPs as CUDA conditional nodes."""
    fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
    with torch.no_grad():
        fn(*args)  # Compile before stream capture.
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with (
        torch.no_grad(),
        torch.cuda.stream(stream),
        ControlFlowOpWarmupDispatchMode(),
    ):
        fn(*args)  # Warm every branch and one loop iteration.

    graph = torch.cuda.CUDAGraph()
    with (
        torch.no_grad(),
        torch.cuda.graph(graph, stream=stream),
        CUDAGraphCaptureControlFlowOpDispatchMode(),
    ):
        output = fn(*args)
    torch.cuda.current_stream().wait_stream(stream)
    return graph, output


# 1. Skip an optimizer step when any gradient is non-finite.
def safe_sgd(param, grad, lr):
    finite = torch.isfinite(grad).all()
    return torch.cond(
        finite,
        lambda p, g, rate: p - rate * g,
        lambda p, g, rate: p.clone(),
        (param, grad, lr),
    )


# 2. Reduce a runtime-sized embedding bag in fixed-size chunks.
def embedding_bag(table, token_ids, length):
    chunk_size = 4
    lanes = torch.arange(chunk_size, device=table.device)
    length = length.clamp(0, token_ids.numel())
    offset = torch.zeros((), dtype=torch.int64, device=table.device)
    total = torch.zeros(table.shape[1], dtype=table.dtype, device=table.device)

    def cond(offset, total):
        return offset < length

    def body(offset, total):
        positions = offset + lanes
        valid = positions < length
        positions = positions.clamp_max(token_ids.numel() - 1)
        chunk = table[token_ids[positions]]
        return offset + chunk_size, total + (chunk * valid[:, None]).sum(0)

    _, total = torch.while_loop(cond, body, (offset, total))
    return total


# 3. Conjugate gradient with a runtime convergence test.
def conjugate_gradient(A, b, tolerance, max_iterations):
    iteration = torch.zeros((), dtype=torch.int64, device=A.device)
    x = torch.zeros_like(b)
    residual = b - A @ x
    direction = residual.clone()
    residual_sq = residual @ residual

    def cond(iteration, x, residual, direction, residual_sq):
        return (iteration < max_iterations) & (residual_sq > tolerance.square())

    def body(iteration, x, residual, direction, residual_sq):
        Ad = A @ direction
        alpha = residual_sq / (direction @ Ad)
        x = x + alpha * direction
        residual = residual - alpha * Ad
        next_residual_sq = residual @ residual
        direction = residual + (next_residual_sq / residual_sq) * direction
        return iteration + 1, x, residual, direction, next_residual_sq

    iteration, x, _, _, _ = torch.while_loop(
        cond, body, (iteration, x, residual, direction, residual_sq)
    )
    return x, iteration


# 4. Lossless MoE routing through a capacity-limited receive buffer.
def buffered_moe(tokens, routes, weights):
    capacity = 4096
    expert_ids = torch.arange(weights.shape[0], device=tokens.device)[:, None]
    done = torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)
    output = tokens.new_zeros(tokens.shape[0], weights.shape[1])
    rounds = torch.zeros((), dtype=torch.int64, device=tokens.device)

    def cond(done, output, rounds):
        return (~done).any()

    def body(done, output, rounds):
        candidates = ((routes[None, :] == expert_ids) & ~done[None, :]).to(torch.int64)
        valid, indices = torch.topk(candidates, capacity, dim=1)

        # [experts, capacity, hidden]: one fixed receive buffer per expert.
        receive_buffer = tokens[indices]
        sent = torch.nn.functional.grouped_mm(receive_buffer, weights.transpose(-2, -1))
        sent = sent * valid[..., None]

        selected = torch.zeros_like(done, dtype=torch.int64).scatter_add(
            0, indices.flatten(), valid.flatten()
        )
        done = done | selected.bool()
        output = output.index_add(0, indices.flatten(), sent.flatten(0, 1))
        return done, output, rounds + 1

    _, output, rounds = torch.while_loop(cond, body, (done, output, rounds))
    return output, rounds


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")

    # cond: the same graph takes and skips the update.
    param = torch.tensor([1.0, 2.0], device=device)
    grad = torch.tensor([0.5, -1.0], device=device)
    lr = torch.tensor(0.1, device=device)
    graph, updated = capture(safe_sgd, param, grad, lr)
    graph.replay()
    torch.testing.assert_close(updated, torch.tensor([0.95, 2.1], device=device))
    grad[0] = torch.inf
    graph.replay()
    torch.testing.assert_close(updated, param)

    # while_loop: bag length changes without changing the allocation shape.
    table = torch.arange(40, dtype=torch.float32, device=device).reshape(10, 4)
    token_ids = torch.tensor([1, 3, 2, 8, 0, 6, 4, 9], device=device)
    length = torch.tensor(3, device=device)
    graph, reduced = capture(embedding_bag, table, token_ids, length)
    for n in (0, 3, 8):
        length.fill_(n)
        graph.replay()
        torch.testing.assert_close(reduced, table[token_ids[:n]].sum(0))

    # while_loop: convergence determines the replay-time iteration count.
    A = torch.tensor([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]], device=device)
    b = torch.tensor([1.0, 2.0, 3.0], device=device)
    tolerance = torch.tensor(1e-5, device=device)
    max_iterations = torch.tensor(20, device=device)
    graph, (solution, iterations) = capture(
        conjugate_gradient, A, b, tolerance, max_iterations
    )
    for rhs in ([1.0, 2.0, 3.0], [4.0, 1.0, -2.0]):
        b.copy_(torch.tensor(rhs, device=device))
        graph.replay()
        torch.testing.assert_close(
            solution, torch.linalg.solve(A, b), rtol=1e-4, atol=1e-5
        )
        if not 0 < iterations.item() <= max_iterations.item():
            raise AssertionError("conjugate gradient did not converge")

    # while_loop + grouped_mm: all experts consume one fixed buffer per round.
    hidden = 64
    expert_width = 128
    tokens = torch.randn(4096, hidden, dtype=torch.bfloat16, device=device)
    routes = torch.zeros(4096, dtype=torch.int64, device=device)
    weights = torch.randn(3, expert_width, hidden, dtype=torch.bfloat16, device=device)
    graph, (moe_output, rounds) = capture(buffered_moe, tokens, routes, weights)
    for routing, expected_rounds in (
        ([0] * 4096, 1),
        ([i % 3 for i in range(4096)], 1),
    ):
        routes.copy_(torch.tensor(routing, device=device))
        graph.replay()
        expected = torch.bmm(
            tokens[:, None, :], weights[routes].transpose(-2, -1)
        ).squeeze(1)
        torch.testing.assert_close(moe_output, expected)
        if rounds.item() != expected_rounds:
            raise AssertionError("unexpected number of MoE receive-buffer rounds")

    print("All four CUDA graph conditional-node examples passed.")


if __name__ == "__main__":
    main()
