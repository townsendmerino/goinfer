"""Chapter 10 animation: the same int4 bytes in two packings, and what unpacking each one costs.

Obvious packing: byte k holds weights 2k and 2k+1. A mask yields the even weights, a shift the
odd ones, and two shuffles weave them back into order. Split-half: byte i holds weights i and
i+4 (i+16 in the real 32-weight group). The mask and the shift each yield a run already in order,
so both shuffles go. Drawn at 8 weights per group so the nibbles are readable; the counts are
about the drawing, not measurements.

Render (ManimGX, Python 3.13+; https://manimgx.academa.ai):

    manimgx inspect docs/book/10-anim-nibble-unpack.py
    manimgx render  docs/book/10-anim-nibble-unpack.py -r 1280x720 --fps 30

Colours are the book's SVG palette.
"""

import manimgx as m

BG = "#f6f8fa"
INK = "#1f2328"
MUTED = "#57606a"
GRID = "#8c959f"
CELL = "#eaeef2"
BLUE = "#0969da"
BLUE_SOFT = "#ddf4ff"
GREEN = "#1a7f37"
GREEN_SOFT = "#dafbe1"
RED = "#cf222e"

m.config.background_color = BG

FONT = "Noto Sans"
NW, NH = 0.72, 0.6  # nibble cell
X0 = -1.0  # centre of the working area
Y_BYTES, Y_MASK, Y_SHIFT, Y_OUT = 1.45, 0.25, -0.85, -2.25
LANE = 0.86  # spacing of the 4 lanes of a mask/shift result
SLOT = 0.8  # spacing of the 8 output slots


def label(text: str, size: int = 22, color: str = MUTED) -> m.Text:
    return m.Text(text, font=FONT, font_size=size, color=color)


def nib(w: int) -> m.VGroup:
    fill = BLUE_SOFT if w < 4 else GREEN_SOFT
    box = m.Rectangle(width=NW, height=NH, color=GRID, stroke_width=1.5, fill_color=fill, fill_opacity=1)
    return m.VGroup(box, label(f"w{w}", 22, INK).move_to(box))


def byte_row(pairs: list) -> list:
    """pairs[k] = (high, low). Returns [(high_nib, low_nib, outline)] laid out across the row."""
    out = []
    gap = 0.3
    width = 4 * 2 * NW + 3 * gap
    left = X0 - width / 2
    for k, (hi, lo) in enumerate(pairs):
        x = left + k * (2 * NW + gap)
        h = nib(hi).move_to([x + NW / 2, Y_BYTES, 0])
        l = nib(lo).move_to([x + 1.5 * NW, Y_BYTES, 0])
        frame = m.Rectangle(width=2 * NW, height=NH, color=INK, stroke_width=2.5).move_to([x + NW, Y_BYTES, 0])
        out.append((h, l, frame))
    return out


def lane_x(i: int) -> float:
    return X0 + (i - 1.5) * LANE


def slot_x(j: int) -> float:
    return X0 + (j - 3.5) * SLOT


class NibbleUnpack(m.Scene):
    def construct(self) -> None:
        title = label("same bytes, two packings — what does unpacking cost?", 32, INK).to_edge(m.UP, buff=0.35)
        self.add(title)

        row_lbls = m.VGroup(
            label("4 bytes").move_to([-5.6, Y_BYTES, 0]),
            label("mask").move_to([-5.6, Y_MASK, 0]),
            label("shift").move_to([-5.6, Y_SHIFT, 0]),
            label("in order").move_to([-5.6, Y_OUT, 0]),
        )
        guide = m.VGroup(*[
            m.Rectangle(width=NW, height=NH, color=GRID, stroke_width=1, fill_color=BG, fill_opacity=0)
            .set_stroke(opacity=0.45).move_to([slot_x(j), Y_OUT, 0])
            for j in range(8)
        ])

        # Right panel: the instruction count.
        count_lbl = label("instructions per group", 22).move_to([5.0, 1.6, 0])
        count = m.Integer(0, mob_class=_Sans, font_size=56, color=BLUE).move_to([5.0, 0.85, 0])
        ops = m.VGroup()  # the op names, listed under the count

        def bump(n: int, name: str, color: str = INK) -> None:
            count.set_value(n)
            op = label(f"{n}  {name}", 22, color)
            op.move_to([5.0, 0.15 - 0.45 * (n - 1), 0])
            ops.add(op)
            self.play(m.FadeIn(op, shift=m.LEFT * 0.15), run_time=0.3)

        def run(heading: str, pairs: list, interleave: bool) -> m.VGroup:
            head = label(heading, 26, INK).move_to([X0, 2.45, 0])
            nibs = byte_row(pairs)
            self.play(m.FadeIn(head), *[m.FadeIn(m.VGroup(h, l, f)) for h, l, f in nibs], run_time=0.8)
            self.wait(0.6)

            # mask: the low nibbles drop into the mask lanes
            lows = [l.copy() for _, l, _ in nibs]
            self.play(*[c.animate.move_to([lane_x(i), Y_MASK, 0]) for i, c in enumerate(lows)], run_time=0.7)
            bump(1, "mask")
            # shift: the high nibbles drop into the shift lanes
            highs = [h.copy() for h, _, _ in nibs]
            self.play(*[c.animate.move_to([lane_x(i), Y_SHIFT, 0]) for i, c in enumerate(highs)], run_time=0.7)
            bump(2, "shift")
            self.wait(0.4)

            if interleave:
                # Evens in one register, odds in the other: weave them back together.
                order = [lows[0], highs[0], lows[1], highs[1], lows[2], highs[2], lows[3], highs[3]]
                self.play(*[order[j].animate.move_to([slot_x(j), Y_OUT, 0]) for j in range(4)], run_time=0.8)
                bump(3, "shuffle", RED)
                self.play(*[order[j].animate.move_to([slot_x(j), Y_OUT, 0]) for j in range(4, 8)], run_time=0.8)
                bump(4, "shuffle", RED)
            else:
                # Each lane is already a contiguous run: weights 0-3 and 4-7.
                note = label("already in order — nothing to shuffle", 22, GREEN).move_to([X0, Y_OUT - 0.75, 0])
                self.play(
                    *[c.animate.move_to([slot_x(i), Y_OUT, 0]) for i, c in enumerate(lows)],
                    *[c.animate.move_to([slot_x(4 + i), Y_OUT, 0]) for i, c in enumerate(highs)],
                    m.FadeIn(note),
                    run_time=0.8,
                )
                lows.append(note)
            self.wait(1.4)
            return m.VGroup(head, *[m.VGroup(h, l, f) for h, l, f in nibs], *lows, *highs)

        self.play(m.FadeIn(row_lbls), m.FadeIn(guide), m.FadeIn(count_lbl), m.FadeIn(count), run_time=0.6)

        # Obvious packing: byte k = (w2k+1 high, w2k low).
        first = run("obvious packing: byte k holds w2k and w2k+1",
                    [(2 * k + 1, 2 * k) for k in range(4)], interleave=True)
        self.wait(0.6)
        count.set_value(0)
        self.play(m.FadeOut(first), m.FadeOut(ops), run_time=0.6)
        ops.remove(*list(ops))
        self.wait(0.3)

        # Split-half: byte i = (wi+4 high, wi low).
        run("split-half packing: byte i holds wi and wi+4", [(i + 4, i) for i in range(4)], interleave=False)

        closing = label("same data. two fewer instructions, every group, forever.", 28, BLUE).move_to([0, -3.6, 0])
        self.play(m.Write(closing), run_time=1.0)
        self.wait(3.0)


class _Sans(m.Text):
    def __init__(self, text: str, **kwargs) -> None:
        kwargs.setdefault("font", FONT)
        super().__init__(text, **kwargs)
