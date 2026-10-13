"""Chapter 9 animation: one round of speculative decoding, on the chapter's own example.

The drafter proposes five tokens one at a time, cheaply. The model checks all five in one pass.
The leading run of agreements is accepted, the model's own token is taken at the first
disagreement, and everything after it is discarded: four tokens for one pass of the model.
The tokens are the chapter's example, not a recorded run.

Render (ManimGX, Python 3.13+; https://manimgx.academa.ai):

    manimgx inspect docs/book/09-anim-speculative.py
    manimgx render  docs/book/09-anim-speculative.py -r 1280x720 --fps 30

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
RED_SOFT = "#ffebe9"

m.config.background_color = BG

FONT = "Noto Sans"
DRAFT = ["the", "cat", "sat", "on", "the"]
MODEL = ["the", "cat", "sat", "in", "a"]
AGREE = 3  # leading run of agreements
TW, TH, GAP = 1.05, 0.62, 0.18  # token box
X0 = -0.75  # centre of the token columns
Y_DRAFT, Y_MODEL, Y_CMP, Y_OUT = 1.75, 0.55, -0.4, -1.75


class _Sans(m.Text):
    def __init__(self, text: str, **kwargs) -> None:
        kwargs.setdefault("font", FONT)
        super().__init__(text, **kwargs)


def label(text: str, size: int = 22, color: str = MUTED) -> m.Text:
    return m.Text(text, font=FONT, font_size=size, color=color)


def col_x(i: int) -> float:
    return X0 + (i - 2) * (TW + GAP)


def token(word: str, i: int, y: float, fill: str) -> m.VGroup:
    box = m.RoundedRectangle(corner_radius=0.08, width=TW, height=TH, color=GRID, stroke_width=1.5,
                             fill_color=fill, fill_opacity=1).move_to([col_x(i), y, 0])
    return m.VGroup(box, label(word, 24, INK).move_to(box))


class Speculative(m.Scene):
    def construct(self) -> None:
        title = label("guess five, check once", 34, INK).to_edge(m.UP, buff=0.35)
        self.add(title)

        lx = col_x(0) - TW / 2 - 0.35
        rows = m.VGroup(
            label("drafter proposes").move_to([lx, Y_DRAFT, 0], aligned_edge=m.RIGHT),
            label("model would say").move_to([lx, Y_MODEL, 0], aligned_edge=m.RIGHT),
            label("compare").move_to([lx, Y_CMP, 0], aligned_edge=m.RIGHT),
            label("output").move_to([lx, Y_OUT, 0], aligned_edge=m.RIGHT),
        )
        self.play(m.FadeIn(rows[0]), run_time=0.4)

        # Right panel.
        px = 5.15
        tok_lbl = label("tokens produced", 20).move_to([px, 1.3, 0])
        toks = m.Integer(0, mob_class=_Sans, font_size=44, color=GREEN).move_to([px, 0.75, 0])
        pass_lbl = label("passes of the model", 20).move_to([px, 0.05, 0])
        passes = m.Integer(0, mob_class=_Sans, font_size=44, color=BLUE).move_to([px, -0.5, 0])
        self.play(*[m.FadeIn(x) for x in (tok_lbl, toks, pass_lbl, passes)], run_time=0.4)

        # 1. The drafter proposes five, one at a time; each guess is cheap.
        draft = [token(w, i, Y_DRAFT, CELL) for i, w in enumerate(DRAFT)]
        cheap = label("cheap, one at a time", 20).next_to(m.VGroup(*draft), m.UP, buff=0.18)
        self.play(m.FadeIn(cheap), run_time=0.3)
        for t in draft:
            self.play(m.FadeIn(t, shift=m.RIGHT * 0.15), run_time=0.28)
        self.wait(0.4)

        # 2. The model runs once over all five.
        sweep = m.Rectangle(width=5 * TW + 4 * GAP + 0.3, height=Y_DRAFT - Y_MODEL + TH + 0.3,
                            color=BLUE, stroke_width=3).move_to([X0, (Y_DRAFT + Y_MODEL) / 2, 0])
        once = label("one pass of the model, all five positions", 20, BLUE).next_to(sweep, m.UP, buff=0.1)
        model = [token(w, i, Y_MODEL, BLUE_SOFT) for i, w in enumerate(MODEL)]
        passes.set_value(1)
        self.play(m.FadeOut(cheap), m.Create(sweep), m.FadeIn(rows[1]), m.FadeIn(once), run_time=0.6)
        self.play(*[m.FadeIn(t) for t in model], run_time=0.6)
        self.play(m.FadeOut(sweep), m.FadeOut(once), run_time=0.4)

        # 3. Compare, left to right, until the first disagreement.
        self.play(m.FadeIn(rows[2]), run_time=0.3)
        marks = []
        for i in range(5):
            if i < AGREE:
                mk = label("✓", 34, GREEN)
            elif i == AGREE:
                mk = label("✗", 34, RED)
            else:
                mk = m.Line([-0.18, 0, 0], [0.18, 0, 0], color=MUTED, stroke_width=4)
            mk.move_to([col_x(i), Y_CMP, 0])
            marks.append(mk)
            self.play(m.FadeIn(mk, scale=1.3), run_time=0.3)
        self.wait(0.4)

        # 4. Accept the agreeing run, take the model's token at the disagreement, discard the rest.
        self.play(m.FadeIn(rows[3]), run_time=0.3)
        out = []
        for i in range(AGREE):
            o = token(DRAFT[i], i, Y_OUT, GREEN_SOFT)
            out.append(o)
        toks.set_value(AGREE)
        self.play(*[m.TransformFromCopy(draft[i], out[i]) for i in range(AGREE)], run_time=0.7)
        fix = token(MODEL[AGREE], AGREE, Y_OUT, GREEN_SOFT)
        toks.set_value(AGREE + 1)
        self.play(m.TransformFromCopy(model[AGREE], fix), run_time=0.6)
        self.play(
            *[draft[i][0].animate.set_fill(RED_SOFT, 1) for i in range(AGREE, 5)],
            *[draft[i].animate.set_opacity(0.35) for i in range(AGREE, 5)],
            # the model's tokens after the disagreement were conditioned on a wrong guess
            *[model[i].animate.set_opacity(0.35) for i in range(AGREE + 1, 5)],
            run_time=0.5,
        )
        discard = label("discarded — the next round starts from “in”", 20, RED)
        discard.next_to(m.VGroup(*draft[AGREE:]), m.UP, buff=0.18)
        self.play(m.FadeIn(discard), run_time=0.4)
        self.wait(0.8)

        plain = label("plain decode:\n4 tokens = 4 passes", 22, INK).move_to([px, -1.5, 0])
        self.play(m.FadeIn(plain), run_time=0.5)
        closing = label("the drafter proposes; the model decides. the output is the same either way.", 26, BLUE)
        closing.move_to([0, -3.2, 0])
        self.play(m.Write(closing), run_time=1.0)
        self.wait(3.5)
