"""Chapter 4 animation: eight rounds of the loop, without and with the KV cache.

Without the cache every round recomputes K and V for every position so far (1+2+...+8 = 36).
With it, each round computes K and V for the one new position and appends them (8 in all), but
attention still reads every stored key, which is the term Chapter 8 is about. The counts are
arithmetic about eight rounds, not measurements.

Render (ManimGX, Python 3.13+; https://manimgx.academa.ai):

    manimgx inspect docs/book/04-anim-kv-cache.py
    manimgx render  docs/book/04-anim-kv-cache.py -r 1280x720 --fps 30

Colours are the book's SVG palette.
"""

import manimgx as m

BG = "#f6f8fa"
INK = "#1f2328"
MUTED = "#57606a"
GRID = "#8c959f"
BLUE = "#0969da"
BLUE_SOFT = "#ddf4ff"
GREEN = "#1a7f37"

m.config.background_color = BG

FONT = "Noto Sans"
ROUNDS = 8
C = 0.62  # cell size
X_LEFT = -4.4  # left edge of the position cells
Y_WITHOUT, Y_WITH = 1.25, -1.15


class _Sans(m.Text):
    def __init__(self, text: str, **kwargs) -> None:
        kwargs.setdefault("font", FONT)
        super().__init__(text, **kwargs)


def label(text: str, size: int = 22, color: str = MUTED) -> m.Text:
    return m.Text(text, font=FONT, font_size=size, color=color)


def pos_cell(i: int, y: float) -> m.VGroup:
    box = m.Square(side_length=C, color=GRID, stroke_width=1.5, fill_color=BLUE_SOFT, fill_opacity=1)
    box.move_to([X_LEFT + C / 2 + i * (C + 0.08), y, 0])
    return m.VGroup(box, label(str(i + 1), 20, INK).move_to(box))


class KVCache(m.Scene):
    def construct(self) -> None:
        title = label("the loop, eight rounds — what gets recomputed?", 32, INK).to_edge(m.UP, buff=0.35)
        self.add(title)

        lane1 = label("without the cache", 26, INK).move_to([X_LEFT, Y_WITHOUT + 0.75, 0], aligned_edge=m.LEFT)
        lane2 = label("with the cache", 26, INK).move_to([X_LEFT, Y_WITH + 0.75, 0], aligned_edge=m.LEFT)
        sub1 = label("K,V recomputed for every position, every round", 20).next_to(lane1, m.RIGHT, buff=0.35)
        sub2 = label("K,V computed once, for the new position, and kept", 20).next_to(lane2, m.RIGHT, buff=0.35)

        cnt_lbl1 = label("K,V computed", 20).move_to([4.9, Y_WITHOUT + 0.05, 0])
        cnt_lbl2 = label("K,V computed", 20).move_to([4.9, Y_WITH + 0.05, 0])
        cnt1 = m.Integer(0, mob_class=_Sans, font_size=44, color=BLUE).move_to([4.9, Y_WITHOUT - 0.6, 0])
        cnt2 = m.Integer(0, mob_class=_Sans, font_size=44, color=BLUE).move_to([4.9, Y_WITH - 0.6, 0])
        rnd = label("round 0", 24, INK).move_to([0, -3.0, 0])

        legend = m.VGroup(
            m.Square(side_length=0.28, color=GRID, stroke_width=1.5, fill_color=BLUE, fill_opacity=1),
            label("computed this round", 18),
            m.Square(side_length=0.28, color=GREEN, stroke_width=3, fill_color=BLUE_SOFT, fill_opacity=1),
            label("read by attention", 18),
        ).arrange(m.RIGHT, buff=0.18).move_to([0, -3.55, 0])
        legend[2].shift(m.RIGHT * 0.3)
        legend[3].shift(m.RIGHT * 0.3)

        self.play(*[m.FadeIn(x) for x in (lane1, lane2, sub1, sub2, cnt_lbl1, cnt_lbl2, cnt1, cnt2, legend)],
                  run_time=0.7)

        top, bot = [], []
        total1 = 0
        for r in range(ROUNDS):
            step = 0.75 if r < 3 else 0.42
            a, b = pos_cell(r, Y_WITHOUT), pos_cell(r, Y_WITH)
            top.append(a)
            bot.append(b)
            new_rnd = label(f"round {r + 1}", 24, INK).move_to(rnd)
            if r == 0:
                self.play(m.FadeIn(a), m.FadeIn(b), m.FadeIn(new_rnd), run_time=step * 0.6)
            else:
                self.play(m.FadeIn(a), m.FadeIn(b), m.FadeOut(rnd), m.FadeIn(new_rnd), run_time=step * 0.6)
            rnd = new_rnd

            total1 += r + 1
            cnt1.set_value(total1)
            cnt2.set_value(r + 1)
            self.play(
                # without: every position so far is recomputed
                *[c[0].animate.set_fill(BLUE, 1) for c in top],
                *[c[1].animate.set_color(BG) for c in top],
                # with: only the new one; attention reads all the earlier, stored keys
                b[0].animate.set_fill(BLUE, 1),
                b[1].animate.set_color(BG),
                *[c[0].animate.set_stroke(GREEN, 3) for c in bot[:-1]],
                run_time=step,
            )
            self.play(
                *[c[0].animate.set_fill(BLUE_SOFT, 1) for c in top],
                *[c[1].animate.set_color(INK) for c in top],
                b[0].animate.set_fill(BLUE_SOFT, 1),
                b[1].animate.set_color(INK),
                *[c[0].animate.set_stroke(GRID, 1.5) for c in bot[:-1]],
                run_time=step * 0.5,
            )

        self.wait(0.6)
        v1 = label("work per round grows with N  →  total work ~ N²", 22, INK)
        v1.move_to([X_LEFT, Y_WITHOUT - 0.72, 0], aligned_edge=m.LEFT)
        v2 = label("work per round is constant  →  total work ~ N", 22, INK)
        v2.move_to([X_LEFT, Y_WITH - 0.72, 0], aligned_edge=m.LEFT)
        note = label("the comparison against every stored key still happens — Chapter 8", 22, BLUE)
        note.move_to(rnd)
        self.play(m.FadeIn(v1), m.FadeIn(v2), m.FadeOut(rnd), m.FadeIn(note), run_time=0.8)
        self.wait(3.5)
