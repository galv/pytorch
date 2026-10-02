"""Build the 36 x 24 inch poster; dependencies: reportlab, pygments, pymupdf."""

import argparse
import ast
import re
import textwrap
from pathlib import Path

import pymupdf
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Comment, Keyword, Number, String
from reportlab.lib.colors import CMYKColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent
WIDTH, HEIGHT = 36 * 72, 24 * 72
MARGIN, GAP = 48, 42
COL = (WIDTH - 2 * MARGIN - 2 * GAP) / 3
INK = CMYKColor(0, 0, 0, 0.94)
MUTED = CMYKColor(0, 0, 0, 0.64)
RULE = CMYKColor(0, 0, 0, 0.19)
TEAL = CMYKColor(0.88, 0.16, 0.42, 0.34)
BLUE = CMYKColor(0.87, 0.50, 0, 0.24)
ORANGE = CMYKColor(0, 0.70, 0.96, 0.22)
HIGHLIGHT = CMYKColor(0, 0.045, 0.20, 0)
WHITE = CMYKColor(0, 0, 0, 0)
HOPS = re.compile(r"torch\.(?:cond|while_loop|switch)\b")
PREREQUISITES = (
    (
        "Fix #191681: NCCL with torch.while_loop() for MoE (#4).",
        "https://github.com/pytorch/pytorch/issues/191681",
    ),
    (
        "Merge all three stacked PRs at #186056 to enable these features.",
        "https://github.com/pytorch/pytorch/pull/186056",
    ),
    (
        "Merge #192098 to prevent memory leaks.",
        "https://github.com/pytorch/pytorch/pull/192098",
    ),
)


SGD = """
def safe_sgd(param, grad, lr):
    finite = torch.isfinite(grad).all()
    return torch.cond(
        finite,
        lambda p, g, rate: p - rate * g,
        lambda p, g, rate: p.clone(),
        (param, grad, lr),
    )
"""

BAG = """
def embedding_bag(table, ids, n):
    C = 4
    lane = torch.arange(C, device=table.device)
    n = n.clamp(0, ids.numel())
    i = torch.zeros((), dtype=torch.int64,
                    device=table.device)
    acc = torch.zeros(table.shape[1],
                      dtype=table.dtype,
                      device=table.device)

    def cond(i, acc):
        return i < n

    def body(i, acc):
        pos = i + lane
        valid = pos < n
        pos = pos.clamp_max(ids.numel() - 1)
        chunk = table[ids[pos]]
        return i + C, acc + (chunk * valid[:, None]).sum(0)

    _, acc = torch.while_loop(cond, body, (i, acc))
    return acc
"""

CG = """
def conjugate_gradient(A, b, tol, max_iter):
    i = torch.zeros((), dtype=torch.int64,
                    device=A.device)
    x = torch.zeros_like(b)
    r = b - A @ x
    p = r.clone()
    rr = r @ r

    def cond(i, x, r, p, rr):
        return (i < max_iter) & (rr > tol.square())

    def body(i, x, r, p, rr):
        Ap = A @ p
        alpha = rr / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        next_rr = r @ r
        p = r + (next_rr / rr) * p
        return i + 1, x, r, p, next_rr

    i, x, _, _, _ = torch.while_loop(
        cond, body, (i, x, r, p, rr))
    return x, i
"""

MOE_A = """
def ep8_moe(x, route, W):
    peers = torch.arange(P, device=x.device)[:, None]
    experts = torch.arange(E, device=x.device)
    dest = route // E
    counts = (dest[None, :] == peers).sum(1)
    local_n = ((counts + C - 1) // C).max()
    n = global_max(local_n)
    done = torch.zeros(T, dtype=torch.bool,
                       device=x.device)
    y = x.new_zeros(T, D)
    i = torch.zeros((), dtype=torch.int64,
                    device=x.device)

    def cond(i, done, y):
        return i < n

    def body(i, done, y):
        mask = ((dest[None, :] == peers)
                & ~done[None, :]).to(torch.int64)
        valid, idx = torch.topk(mask, C, dim=1)
        send_x = x[idx]
        send_meta = torch.stack(
            (route[idx] % E, valid), dim=-1)
        recv_x = a2a(send_x).flatten(0, 1)
        recv_meta = a2a(send_meta).flatten(0, 1)
"""

MOE_B = """
        expert = recv_meta[:, 0]
        live = recv_meta[:, 1].bool()
        key = torch.where(live, expert, E)
        order = torch.argsort(key, stable=True)
        packed_x = recv_x[order]
        packed_e = expert[order]
        packed_live = live[order]

        counts = (
            (packed_e[:, None] == experts[None, :])
            & packed_live[:, None]
        ).sum(0)
        offs = torch.cumsum(counts, 0,
                            dtype=torch.int32)
        packed_y = torch.nn.functional.grouped_mm(
            packed_x, W.transpose(-2, -1), offs=offs)
        packed_y = torch.where(
            packed_live[:, None], packed_y,
            torch.zeros_like(packed_y))
        recv_y = torch.zeros_like(packed_y).index_copy(
            0, order, packed_y)
"""

MOE_C = """
        back = a2a(recv_y.view(P, C, D))
        back = back * valid[..., None]
        y = y.index_add(
            0, idx.flatten(), back.flatten(0, 1))
        selected = torch.zeros_like(
            done, dtype=torch.int64).scatter_add(
                0, idx.flatten(), valid.flatten())
        return i + 1, done | selected.bool(), y

    i, _, y = torch.while_loop(
        cond, body, (i, done, y))
    return y, i
"""

A2A = """
def a2a(t):
    out = torch.empty_like(t)
    dist.all_to_all_single(out, t, async_op=False)
    return out
"""

MAX = """
def global_max(t):
    out = t.clone()
    dist.all_reduce(out, op=dist.ReduceOp.MAX,
                    async_op=False)
    return out
"""


class Rename(ast.NodeTransformer):
    def __init__(self, names):
        self.names = names

    def visit_Name(self, node):
        node.id = self.names.get(node.id, node.id)
        return node

    def visit_arg(self, node):
        node.arg = self.names.get(node.arg, node.arg)
        node.annotation = None
        return node

    def visit_FunctionDef(self, node):
        node.name = self.names.get(node.name, node.name)
        node.decorator_list = []
        node.returns = None
        return self.generic_visit(node)


def verify_code():
    """Require the same AST as tested workloads, apart from identifier names."""
    sources = {}
    for filename in ("shape_stable_control_flow.py", "ep8_moe_backpressure.py"):
        tree = ast.parse((EXAMPLES / filename).read_text())
        sources.update(
            (node.name, node) for node in tree.body if isinstance(node, ast.FunctionDef)
        )
    cases = [
        ("safe_sgd", SGD, {}),
        (
            "embedding_bag",
            BAG,
            {
                "token_ids": "ids",
                "length": "n",
                "chunk_size": "C",
                "lanes": "lane",
                "offset": "i",
                "total": "acc",
                "positions": "pos",
            },
        ),
        (
            "conjugate_gradient",
            CG,
            {
                "tolerance": "tol",
                "max_iterations": "max_iter",
                "iteration": "i",
                "residual": "r",
                "direction": "p",
                "residual_sq": "rr",
                "Ad": "Ap",
                "next_residual_sq": "next_rr",
            },
        ),
        (
            "ep8_moe",
            MOE_A + MOE_B + MOE_C,
            {
                "tokens": "x",
                "routes": "route",
                "local_weights": "W",
                "local_expert_ids": "experts",
                "destinations": "dest",
                "tokens_per_destination": "counts",
                "local_rounds": "local_n",
                "total_rounds": "n",
                "output": "y",
                "iteration": "i",
                "candidates": "mask",
                "token_indices": "idx",
                "send_tokens": "send_x",
                "send_metadata": "send_meta",
                "received_tokens": "recv_x",
                "received_metadata": "recv_meta",
                "received_experts": "expert",
                "received_valid": "live",
                "sort_keys": "key",
                "packed_tokens": "packed_x",
                "packed_experts": "packed_e",
                "packed_valid": "packed_live",
                "offsets": "offs",
                "packed_output": "packed_y",
                "received_output": "recv_y",
                "returned_output": "back",
                "WORLD_SIZE": "P",
                "LOCAL_EXPERTS": "E",
                "TOKENS_PER_RANK": "T",
                "PEER_CAPACITY": "C",
                "EXPERT_WIDTH": "D",
                "sync_all_to_all": "a2a",
                "sync_all_reduce_max": "global_max",
            },
        ),
        (
            "sync_all_to_all",
            A2A,
            {
                "sync_all_to_all": "a2a",
                "tensor": "t",
                "output": "out",
            },
        ),
        (
            "sync_all_reduce_max",
            MAX,
            {
                "sync_all_reduce_max": "global_max",
                "tensor": "t",
                "output": "out",
            },
        ),
    ]
    for name, snippet, names in cases:
        expected = Rename(names).visit(sources[name])
        shown = ast.parse(snippet).body[0]
        if ast.dump(expected) != ast.dump(shown):
            raise ValueError(f"Poster code differs from the tested function: {name}")
    print("All six displayed function bodies match the tested source ASTs.")


def register_fonts():
    root = Path("/usr/share/fonts/truetype")
    for name, path in {
        "Sans": "lato/Lato-Regular.ttf",
        "SansBold": "lato/Lato-Bold.ttf",
        "Mono": "liberation/LiberationMono-Regular.ttf",
        "MonoBold": "liberation/LiberationMono-Bold.ttf",
    }.items():
        pdfmetrics.registerFont(TTFont(name, str(root / path)))


def text(c, x, top, value, size=22, font="Sans", color=INK, width=None):
    if width is not None and pdfmetrics.stringWidth(value, font, size) > width:
        raise ValueError(f"Text exceeds its column: {value}")
    c.setFont(font, size)
    c.setFillColor(color)
    c.drawString(x, HEIGHT - top - size, value)


def rule(c, x, top, width, color=RULE, thickness=1):
    c.setStrokeColor(color)
    c.setLineWidth(thickness)
    c.line(x, HEIGHT - top, x + width, HEIGHT - top)


def code(c, source, x, top, width=COL, size=20, leading=24, unindent=False):
    source = source.strip("\n")
    if unindent:
        source = textwrap.dedent(source)
    lines = source.splitlines()
    for index, line in enumerate(lines):
        baseline = HEIGHT - top - size - index * leading
        line_width = pdfmetrics.stringWidth(line, "Mono", size)
        if line_width > width:
            raise ValueError(
                f"Code exceeds column ({line_width:.1f} > {width}): {line}"
            )
        if HOPS.search(line):
            c.setFillColor(HIGHLIGHT)
            c.rect(x - 8, baseline - 5, width + 16, leading + 1, fill=1, stroke=0)
            c.setFillColor(ORANGE)
            c.rect(x - 8, baseline - 5, 3, leading + 1, fill=1, stroke=0)
        cursor = x
        offset = 0
        ranges = [match.span() for match in HOPS.finditer(line)]
        for token, value in lex(line, PythonLexer(stripnl=False, ensurenl=False)):
            value = value.rstrip("\n")
            color = INK
            font = "Mono"
            if token in Comment:
                color = MUTED
            elif token in Keyword:
                color, font = BLUE, "MonoBold"
            elif token in Number or token in String:
                color = TEAL
            if any(start <= offset < end for start, end in ranges):
                color, font = ORANGE, "MonoBold"
            c.setFillColor(color)
            c.setFont(font, size)
            c.drawString(cursor, baseline, value)
            cursor += pdfmetrics.stringWidth(value, font, size)
            offset += len(value)
    return top + len(lines) * leading


def heading(c, x, top, number, title, subline):
    text(c, x, top, number, 30, "SansBold", TEAL)
    text(c, x + 52, top, title, 30, "SansBold")
    text(c, x, top + 44, subline, 21, color=MUTED, width=COL)


def build_pdf(path, author):
    c = canvas.Canvas(
        str(path), pagesize=(WIDTH, HEIGHT), pageCompression=1, enforceColorSpace="CMYK"
    )
    c.setTitle("Shape-Stable Dynamic Control Flow in PyTorch CUDA Graphs")
    c.setAuthor(author)
    c.setSubject("Four executable workload examples using CUDA graph conditional nodes")
    c.setCreator("ReportLab; editable source: build_poster.py")
    c.setFillColor(WHITE)
    c.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)

    text(
        c,
        MARGIN,
        38,
        "Shape-Stable Dynamic Control Flow in PyTorch CUDA Graphs",
        59,
        "SansBold",
        width=WIDTH - 2 * MARGIN,
    )
    text(c, MARGIN, 116, author, 26, color=TEAL)
    text(
        c,
        1635,
        121,
        "GPU decisions. Fixed buffer shapes.",
        25,
        "SansBold",
        width=WIDTH - MARGIN - 1635,
    )
    rule(c, MARGIN, 166, WIDTH - 2 * MARGIN, TEAL, 2)
    text(
        c,
        MARGIN,
        182,
        "Conditional nodes evaluate GPU predicates during replay; branch outputs and loop-carried tensor metadata stay fixed.",
        25,
        width=WIDTH - 2 * MARGIN,
    )

    xs = [MARGIN + i * (COL + GAP) for i in range(3)]
    heading(
        c,
        xs[0],
        240,
        "01",
        "Skip non-finite optimizer steps",
        "torch.cond()  /  IF-ELSE conditional node",
    )
    heading(
        c,
        xs[1],
        240,
        "02",
        "Runtime-sized embedding bags",
        "torch.while_loop()  /  WHILE conditional node",
    )
    heading(
        c,
        xs[2],
        240,
        "03",
        "Conjugate gradient to convergence",
        "torch.while_loop()  /  WHILE conditional node",
    )

    code(c, SGD, xs[0], 336, size=23, leading=29)
    rule(c, xs[0], 604, COL)
    text(c, xs[0], 619, "Work Before General Use", 27, "SansBold", ORANGE)
    text(c, xs[0], 662, "Prototype examples; upstream work is still required.", 22)
    for index, (description, url) in enumerate(PREREQUISITES):
        top = 708 + index * 72
        text(c, xs[0], top, description, 22, width=COL)
        label = url.removeprefix("https://")
        text(c, xs[0], top + 30, label, 20, color=BLUE, width=COL)
        link_width = pdfmetrics.stringWidth(label, "Sans", 20)
        c.linkURL(
            url,
            (xs[0], HEIGHT - top - 55, xs[0] + link_width, HEIGHT - top - 30),
            relative=0,
            thickness=0,
        )
    code(c, BAG, xs[1], 336, size=21, leading=24)
    text(c, xs[1], 897, "GPU length n changes; the output vector does not.", 21)
    code(c, CG, xs[2], 336, size=21, leading=24)
    text(
        c,
        xs[2],
        914,
        "A is symmetric positive definite; b and tol are GPU tensors.",
        20,
    )

    for x in xs[1:]:
        c.setStrokeColor(RULE)
        c.setLineWidth(1)
        c.line(x - GAP / 2, HEIGHT - 240, x - GAP / 2, HEIGHT - 937)
    rule(c, MARGIN, 963, WIDTH - 2 * MARGIN, TEAL, 2)

    text(c, MARGIN, 984, "04", 30, "SansBold", TEAL)
    text(c, MARGIN + 52, 984, "Lossless MoE routing with backpressure", 30, "SansBold")
    text(
        c,
        1530,
        990,
        "torch.while_loop() + grouped_mm(..., offs=offs)",
        24,
        "SansBold",
        width=WIDTH - MARGIN - 1530,
    )
    text(
        c,
        MARGIN,
        1031,
        "EP=8  |  4,096 tokens/rank  |  8 experts/rank  |  4,096 receive slots/rank  |  512 slots/source  |  Top-1 routing; zero dropped tokens",
        22,
        width=WIDTH - 2 * MARGIN,
    )
    text(
        c,
        2090,
        1031,
        "Code continues left to right.",
        22,
        "SansBold",
        color=MUTED,
        width=WIDTH - MARGIN - 2090,
    )
    text(c, MARGIN, 1068, "P, E, T, C, D = 8, 8, 4096, 512, 128", 19, "Mono", TEAL)
    text(
        c,
        xs[1],
        1068,
        "GPU offsets partition a fixed 2D receive buffer.",
        21,
        color=TEAL,
    )
    text(
        c,
        xs[2],
        1068,
        "Global MAX keeps collective loop counts identical.",
        21,
        color=TEAL,
    )

    code(c, MOE_A, xs[0], 1104, size=19, leading=20.8)
    code(c, MOE_B, xs[1], 1104, size=19, leading=20.8)
    code(c, MOE_C, xs[2], 1104, size=19, leading=20.8)
    text(
        c,
        xs[1],
        1573,
        "Prototype + NCCL workaround: 1 or 8 rounds on 8 H100s.",
        20,
        "SansBold",
        width=COL,
    )
    text(
        c, xs[1], 1604, "All ranks' results match direct expert matmuls.", 20, width=COL
    )

    text(
        c,
        xs[2],
        1373,
        "Collective bodies; custom-op registration omitted",
        19,
        "SansBold",
    )
    code(c, A2A + MAX, xs[2], 1407, size=18, leading=20.8)

    rule(c, MARGIN, 1650, WIDTH - 2 * MARGIN)
    text(
        c,
        MARGIN,
        1665,
        "Runnable sources + capture/registration helpers: shape_stable_control_flow.py and ep8_moe_backpressure.py  |  NCCL_GRAPH_MIXING_SUPPORT=0",
        20,
        width=WIDTH - 2 * MARGIN,
    )
    text(
        c,
        MARGIN,
        1694,
        "Uses this PyTorch conditional-node implementation. torch.switch() is not needed here: grouped_mm evaluates all local experts together. Imports: torch; torch.distributed as dist.",
        18,
        color=MUTED,
        width=WIDTH - 2 * MARGIN,
    )
    c.showPage()
    c.save()


def inspect_and_render(path):
    doc = pymupdf.open(path)
    if len(doc) != 1 or tuple(doc[0].rect) != (0, 0, WIDTH, HEIGHT):
        raise ValueError("The poster must be one 36 x 24 inch page")
    page = doc[0]
    strings = page.get_text()
    for required in ("torch.cond", "torch.while_loop", "grouped_mm", "ep8_moe"):
        if required not in strings:
            raise ValueError(f"Missing poster content: {required}")
    links = {link.get("uri") for link in page.get_links()}
    for description, url in PREREQUISITES:
        if description not in strings or url not in links:
            raise ValueError(f"Missing prerequisite text or PDF link: {url}")
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            if not page.rect.contains(pymupdf.Rect(line["bbox"])):
                raise ValueError(f"Text outside page: {line}")
    for font in page.get_fonts():
        if font[1] != "n/a" and not doc.extract_font(font[0])[3]:
            raise ValueError(f"Font not embedded: {font}")
    streams = b"".join(doc.xref_stream(xref) for xref in page.get_contents())
    if re.search(rb"\s(?:rg|RG)\s", streams) or b" k" not in streams:
        raise ValueError("Expected CMYK-only drawing colors")
    page.get_pixmap(dpi=150, alpha=False).save(HERE / "poster_150dpi.png")
    page.get_pixmap(matrix=pymupdf.Matrix(0.75, 0.75), alpha=False).save(
        HERE / "poster_preview.png"
    )
    print("Verified: 1 page, 36 x 24 inches, CMYK vectors, embedded used fonts.")
    print(f"PDF: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author", default="Daniel Galvez | NVIDIA")
    args = parser.parse_args()
    verify_code()
    register_fonts()
    path = HERE / "shape_stable_cuda_graphs_poster.pdf"
    build_pdf(path, args.author)
    inspect_and_render(path)


if __name__ == "__main__":
    main()
