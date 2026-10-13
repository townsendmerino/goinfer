"""Chapter 2 animation: decode and prefill run the same matrix multiply, in two shapes.

Decode multiplies ONE row of activations by the weights, so every weight fetched from memory
feeds one multiply-add. Prefill multiplies MANY rows by the same weights, so every weight
fetched feeds one multiply-add per row. Same reads, more arithmetic: memory-bound versus
compute-bound. The counters are arithmetic about the drawing (8x8 weights, 1 or 6 rows), not
measurements, so nothing here is held by the site's claims check.

Render (ManimGX, Python 3.13+; https://manimgx.academa.ai):

    manimgx inspect docs/book/02-anim-matmul-shape.py     # storyboard + layout check
    manimgx render  docs/book/02-anim-matmul-shape.py -r 1280x720 --fps 30

The MP4 lands beside this file. Colours are the book's SVG palette.
"""

import manimgx as m

# The book's SVG palette (GitHub light).
BG = "#f6f8fa"
INK = "#1f2328"
MUTED = "#57606a"
GRID = "#8c959f"
CELL = "#eaeef2"
BLUE = "#0969da"
BLUE_SOFT = "#ddf4ff"
GREEN = "#1a7f37"
GREEN_SOFT = "#dafbe1"
GREEN_MID = "#aceebb"

m.config.background_color = BG

S = 0.34  # cell size
K = 8  # hidden size (rows of W, columns of X)
N = 8  # output size (columns of W and Y)
M_PREFILL = 6  # positions in the prefill pass
STEP = 0.42  # seconds per weight column



def grid(rows: int, cols: int, fill: str) -> m.VGroup:
    cells = [
        m.Square(side_length=S, color=GRID, stroke_width=1.5, fill_color=fill, fill_opacity=1)
        for _ in range(rows * cols)
    ]
    return m.VGroup(*cells).arrange_in_grid(rows=rows, cols=cols, buff=0)


def cell(g: m.VGroup, cols: int, r: int, c: int) -> m.Square:
    return g[r * cols + c]


def column(g: m.VGroup, rows: int, cols: int, c: int) -> list:
    return [cell(g, cols, r, c) for r in range(rows)]


class SansText(m.Text):
    """Text in the bundled Noto Sans, so the counters match the labels."""

    def __init__(self, text: str, **kwargs) -> None:
        kwargs.setdefault("font", "Noto Sans")
        super().__init__(text, **kwargs)


def counter(color: str) -> m.Integer:
    return m.Integer(0, mob_class=SansText, font_size=30, color=color)


def label(text: str, size: int = 22, color: str = MUTED) -> m.Text:
    return m.Text(text, font="Noto Sans", font_size=size, color=color)


class MatmulShape(m.Scene):
    def construct(self) -> None:
        title = label("one weight read — how much work does it buy?", 34, INK).to_edge(m.UP, buff=0.35)
        self.add(title)

        # W (K x N) top right; Y (rows x N) under it; X (rows x K) to Y's left.
        w = grid(K, N, CELL).move_to([2.7, 1.25, 0])
        w_lbl = label("weights\n(in memory)", 20).next_to(w, m.RIGHT, buff=0.25)


        def place_rows(rows: int):
            y = grid(rows, N, CELL)
            y.next_to(w, m.DOWN, buff=0.35)
            x = grid(rows, K, GREEN_SOFT)
            x.next_to(y, m.LEFT, buff=0.35).align_to(y, m.UP)
            return x, y

        x, y = place_rows(1)
        x_lbl = label("activations", 20).next_to(x, m.UP, buff=0.15)
        y_lbl = label("output", 20).next_to(y, m.RIGHT, buff=0.25)

        # Left panel: the regime and the two counters.
        mode = label("decode · 1 position", 30, INK).move_to([-4.6, 1.9, 0])
        reads_lbl = label("weights read", 22).move_to([-5.2, 0.75, 0])
        madds_lbl = label("multiply-adds", 22).move_to([-5.2, 0.15, 0])
        reads = counter(BLUE).next_to(reads_lbl, m.RIGHT, buff=0.4)
        madds = counter(GREEN).next_to(madds_lbl, m.RIGHT, buff=0.4)
        reads.align_to(madds, m.LEFT)

        self.play(
            m.FadeIn(w), m.FadeIn(w_lbl), m.FadeIn(x), m.FadeIn(x_lbl), m.FadeIn(y), m.FadeIn(y_lbl),
            run_time=0.8,
        )
        self.play(m.FadeIn(mode), m.FadeIn(reads_lbl), m.FadeIn(madds_lbl), m.FadeIn(reads), m.FadeIn(madds),
                  run_time=0.6)
        self.wait(0.4)

        def sweep(x: m.VGroup, y: m.VGroup, rows: int) -> None:
            for c in range(N):
                wcol = column(w, K, N, c)
                ycol = column(y, rows, N, c)
                # Counters jump rather than animate: tweening a number morphs its glyphs.
                reads.set_value((c + 1) * K)
                madds.set_value((c + 1) * K * rows)
                self.play(
                    *[sq.animate.set_fill(BLUE, 1) for sq in wcol],
                    *[sq.animate.set_fill(GREEN_MID, 1) for sq in x],
                    *[sq.animate.set_fill(GREEN, 1) for sq in ycol],
                    run_time=STEP,
                )
                self.play(
                    *[sq.animate.set_fill(BLUE_SOFT, 1) for sq in wcol],
                    *[sq.animate.set_fill(GREEN_SOFT, 1) for sq in x],
                    *[sq.animate.set_fill(GREEN_SOFT, 1) for sq in ycol],
                    run_time=STEP * 0.4,
                )

        sweep(x, y, 1)
        verdict1 = label("each weight read feeds 1 multiply-add  →  waiting on memory: memory-bound", 24, INK)
        verdict1.move_to([0, -3.05, 0])
        self.play(m.FadeIn(verdict1, shift=m.UP * 0.2), run_time=0.6)
        self.wait(2.0)

        # Prefill: same W, six positions at once.
        x6, y6 = place_rows(M_PREFILL)
        mode6 = label(f"prefill · {M_PREFILL} positions", 30, INK).move_to(mode)
        note = label("(a real prompt has thousands)", 20).next_to(mode6, m.DOWN, buff=0.12)
        # The decode row stays put as row 0; the five new positions fade in under it.
        self.remove(x, y)
        x6_new = m.VGroup(*list(x6)[K:])
        y6_new = m.VGroup(*list(y6)[N:])
        self.add(m.VGroup(*list(x6)[:K]), m.VGroup(*list(y6)[:N]))
        for sq in list(y6)[:N]:
            sq.set_fill(GREEN_SOFT, 1)
        self.play(
            m.FadeOut(verdict1),
            m.FadeOut(mode),
            *[sq.animate.set_fill(CELL, 1) for sq in w],
            *[sq.animate.set_fill(CELL, 1) for sq in list(y6)[:N]],
            run_time=0.6,
        )
        reads.set_value(0)
        madds.set_value(0)
        self.play(
            m.FadeIn(x6_new, shift=m.DOWN * 0.2),
            m.FadeIn(y6_new, shift=m.DOWN * 0.2),
            y_lbl.animate.next_to(y6, m.RIGHT, buff=0.25),
            m.FadeIn(mode6),
            m.FadeIn(note),
            run_time=0.9,
        )
        self.wait(0.5)

        sweep(x6, y6, M_PREFILL)
        verdict6 = label(f"each weight read feeds {M_PREFILL} multiply-adds  →  waiting on arithmetic: compute-bound", 24, INK)
        verdict6.move_to([0, -3.05, 0])
        self.play(m.FadeIn(verdict6, shift=m.UP * 0.2), run_time=0.6)
        self.wait(1.0)

        closing = label(f"same {K * N} weight reads.  {M_PREFILL}× the work.", 30, BLUE).move_to([0, -3.65, 0])
        self.play(m.Write(closing), run_time=1.0)
        self.wait(3.0)
