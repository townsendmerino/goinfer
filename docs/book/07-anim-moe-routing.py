"""Chapter 7 animation: a router picks 8 of 64 experts per token, and content decides the cost.

One layer's 64 experts start on disk. Each token's 8 picks are either already in memory (a hit)
or fetched (a miss). A run of Go tokens keeps choosing from the same experts, so after the first
token little is fetched; a French word routes elsewhere and the fetches come back. The routing is
illustrative (a fixed seed, two clusters), not a recording of a real model, and the cache is
drawn without a capacity limit, so nothing is evicted.

Render (ManimGX, Python 3.13+; https://manimgx.academa.ai):

    manimgx inspect docs/book/07-anim-moe-routing.py
    manimgx render  docs/book/07-anim-moe-routing.py -r 1280x720 --fps 30

Colours are the book's SVG palette.
"""

import random

import manimgx as m

BG = "#f6f8fa"
INK = "#1f2328"
MUTED = "#57606a"
GRID = "#8c959f"
CELL = "#eaeef2"
BLUE = "#0969da"
BLUE_SOFT = "#b6e3ff"  # a step darker than the other scenes: it must read against on-disk grey
GREEN = "#1a7f37"

m.config.background_color = BG

FONT = "Noto Sans"
N, TOPK = 64, 8
S = 0.44  # expert cell
GRID_CENTER = [2.9, 0.15, 0]
TOKENS = ["func", "main", "(", ")", "{", "bonjour", "le", "chat"]
FRENCH_FROM = 5  # index of the first French token


def routing() -> list:
    """Each token's 8 experts: Go tokens draw from one cluster, French ones from another."""
    rng = random.Random(7)
    cells = list(range(N))
    rng.shuffle(cells)
    code, french = cells[:12], cells[12:24]
    picks = []
    for t in range(len(TOKENS)):
        pool = code if t < FRENCH_FROM else french
        chosen = rng.sample(pool, TOPK if t != FRENCH_FROM else TOPK - 1)
        if t == FRENCH_FROM:
            chosen.append(rng.choice(code))  # a word shares a little with what came before
        picks.append(chosen)
    return picks


def label(text: str, size: int = 22, color: str = MUTED) -> m.Text:
    return m.Text(text, font=FONT, font_size=size, color=color)


class MoERouting(m.Scene):
    def construct(self) -> None:
        title = label("8 of 64 experts per token — which 8 depends on the words", 32, INK)
        title.to_edge(m.UP, buff=0.35)
        self.add(title)

        experts = m.VGroup(*[
            m.Square(side_length=S, color=GRID, stroke_width=1.2, fill_color=CELL, fill_opacity=1)
            for _ in range(N)
        ]).arrange_in_grid(rows=8, cols=8, buff=0.05).move_to(GRID_CENTER)
        grid_lbl = label("64 experts, one layer", 22, INK).next_to(experts, m.UP, buff=0.18)

        def swatch(fill: str, stroke: str = GRID) -> m.Square:
            return m.Square(side_length=0.26, color=stroke, stroke_width=1.2, fill_color=fill, fill_opacity=1)

        legend = m.VGroup(
            m.VGroup(swatch(CELL), label("on disk", 18)).arrange(m.RIGHT, buff=0.12),
            m.VGroup(swatch(BLUE_SOFT), label("in memory", 18)).arrange(m.RIGHT, buff=0.12),
            m.VGroup(swatch(BLUE), label("fetched now", 18)).arrange(m.RIGHT, buff=0.12),
            m.VGroup(swatch(GREEN), label("used now", 18)).arrange(m.RIGHT, buff=0.12),
        ).arrange(m.RIGHT, buff=0.3).next_to(experts, m.DOWN, buff=0.25)

        # Left: the token stream and what each token cost.
        lx, rx, top = -6.3, -2.6, 1.85
        head = m.VGroup(
            label("token", 20).move_to([lx, top + 0.5, 0], aligned_edge=m.LEFT),
            label("experts fetched", 20).move_to([rx, top + 0.5, 0], aligned_edge=m.RIGHT),
        )
        router = m.RoundedRectangle(corner_radius=0.12, width=1.6, height=0.9, color=INK, stroke_width=2,
                                    fill_color=BG, fill_opacity=1).move_to([-0.95, 0.15, 0])
        router_lbl = label("router", 22, INK).move_to(router)
        arrow = m.Arrow(router.get_right(), experts.get_left(), buff=0.12, color=MUTED, stroke_width=3)
        per_tok = label("each token: 8 of 64 = 1/8 of the expert weights", 20)
        per_tok.move_to([lx, -2.6, 0], aligned_edge=m.LEFT)

        self.play(*[m.FadeIn(x) for x in (experts, grid_lbl, legend, head, router, router_lbl, arrow)],
                  run_time=0.8)
        self.play(m.FadeIn(per_tok), run_time=0.4)

        resident = set()
        picks = routing()
        for t, word in enumerate(TOKENS):
            fast = 1.0 if t in (0, 1, FRENCH_FROM) else 0.7
            y = top - t * 0.46
            tok = label(word, 24, INK).move_to([lx, y, 0], aligned_edge=m.LEFT)
            self.play(m.FadeIn(tok, shift=m.RIGHT * 0.15), router.animate.set_stroke(BLUE, 3),
                      run_time=0.3 * fast)

            hits = [i for i in picks[t] if i in resident]
            misses = [i for i in picks[t] if i not in resident]
            self.play(
                *[experts[i].animate.set_fill(BLUE, 1) for i in misses],
                *[experts[i].animate.set_fill(GREEN, 1) for i in hits],
                run_time=0.5 * fast,
            )
            cost = label(str(len(misses)), 24, BLUE if misses else MUTED)
            cost.move_to([rx, y, 0], aligned_edge=m.RIGHT)
            self.play(*[experts[i].animate.set_fill(GREEN, 1) for i in misses], m.FadeIn(cost), run_time=0.3 * fast)
            resident.update(picks[t])
            self.play(*[experts[i].animate.set_fill(BLUE_SOFT, 1) for i in picks[t]],
                      router.animate.set_stroke(INK, 2), run_time=0.3 * fast)
            if t == FRENCH_FROM - 1:
                self.wait(0.6)

        self.wait(0.5)
        closing = label("same number of tokens, different words: content is routing, and routing is cost", 24, BLUE)
        closing.move_to([0, -3.6, 0])
        self.play(m.Write(closing), run_time=1.0)
        self.wait(3.5)
