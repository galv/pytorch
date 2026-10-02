# CUDA Graphs Poster

`shape_stable_cuda_graphs_poster.pdf` is a one-page, **36 x 24 inch landscape**
poster for printing and Sessionize upload. Print at **100% / actual size**,
one-sided, without "fit to page". All artwork is vector text and rules with
CMYK colors; the used fonts are embedded. The PDF does not require bitmap
resolution scaling. No bleed or printer-specific ICC profile is included.

`poster_preview.png` is a small screen preview. `poster_150dpi.png` is an RGB
preview at 5400 x 3600 pixels; use the PDF, not the PNG, for printing.

The four workload bodies come from the adjacent runnable examples. Names are
shortened for layout, and the builder verifies their Python ASTs against those
files. The poster omits the shared capture harness, imports, custom-op
registration, and test setup. Full code is in:

- `../shape_stable_control_flow.py`
- `../ep8_moe_backpressure.py`

The MoE example uses top-1 routing, eight ranks, eight local experts per rank,
4096 tokens per rank, and a 4096-row receive buffer per rank. Its fixed
512-token allowance per source can underutilize the buffer under skew;
this is a correctness demonstration, not an optimized dispatcher.

To edit the poster, change `build_poster.py`, then run from the worktree root:

```sh
PYTHONPATH="$PWD/agent_space/poster_pdf_deps" \
  .venv/bin/python poster_examples/poster/build_poster.py
```

Dependencies are `reportlab`, `pygments`, and `pymupdf`. Fonts are the system
Lato and Liberation Mono families. Use `--author "Your author line"` to change
the author. The build checks code equivalence, page size, page count, color
operators, font embedding, and text bounds, and regenerates both previews.
