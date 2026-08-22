# Plan: demo refresh — new models in the existing demos, one tape, no new modules

> **STATUS 2026-08-22 — Tier 1 SHIPPED; Tier 2 gates 1–4 run, gate 4 KILLS the straight swap.**
> Full numbers: `docs/measurements/demo-chat-tier2-gates-2026-08-22.md` and
> `docs/measurements/demo-chat-incumbent-2026-08-22.md`.
>
> - **Tier 1 done** — and it was a correctness fix, not a currency one: every Gemma 3 checkpoint
>   these READMEs told users to download is `gated=manual`, so the first copy-pasteable command on
>   the page did not run for a new user. Gemma 4 E2B is ungated.
> - **Gate 1 PASS** (apache-2.0, per-repo). **Gate 2 PASS** — loads and generates through the
>   existing dense `qwen3_5` adapter with NO loader work; the multimodal wrapper, `text_config`
>   nesting, `model.language_model.*` prefix and 3:1 `DDDS…` hybrid are all already handled.
> - **Gate 3**: the template is thinking-capable (11 `think` markers) — needs the non-thinking render.
> - **Tier 3 BLOCKED on this box, two independent reasons** — neither is a code problem:
>   1. **`ttyd` is not installed**, and `vhs` cannot render without it. Installing a system package
>      is an owner decision, not something to do under a demo task.
>   2. **No runnable v1 DFlash pair on this card.** The only paired target present is
>      `gemma-4-26b-a4b-it` + `gemma4-26b-dflash` (the gpt-oss-20B target is absent, and
>      `dflash2-qwen38` is a DFlash **2** checkpoint the v1 loader correctly refuses, per P15).
>      That pair OOMs on 8 GB — and instructively: the target DOES go resident via C′, then
>      `C′ cache: 64 slots/layer would need 6.5 GB VRAM but only 3.8 GB free — capping to 32
>      (3.3 GB)` consumes the headroom, and the drafter fails afterwards on an 11.5 MB allocation
>      (`cuda/drafter.go:134`). **The C′ slot auto-cap sizes itself against free VRAM at that
>      moment and does not reserve for a drafter that loads later.** Whether the fix is a reserve,
>      a lower cap under `--drafter`, or nothing is a design call; recorded, not patched.
>
> - **Gate 4 KILL**: 13.5 tok/s against the incumbent 0.5B's 28.1 — **2.08× slower**, and barely
>   ahead of the 1.5B tier. The doc's own fallback applies: keep the Qwen2.5 tier rather than
>   replacing it.
>
> **One correction to this doc's own reasoning.** The pre-registered risk blamed "the hybrid's
> DeltaNet layers". The arithmetic says otherwise: Qwen3.5's vocabulary is 248 320 against
> qwen2.5's 151 936, making the LM head alone **1.87× larger** against a measured 2.08× slowdown.
> Since that vocabulary is shared across the whole Qwen3.5 line, **no member of the family escapes
> it** — so "try the 2B instead" is not a way out. If Tier 2 is still wanted, Gemma 4 E2B is the
> better candidate: dense, no vocab penalty, apache-2.0 and ungated (which also retires this doc's
> "Gemma stays flag-loaded, never embedded" constraint for Gemma 4 specifically — that constraint is
> a Gemma 3 fact).


> **Audience:** internal planning (2026-08-20). The demos' example and embedded models are
> 2024–early-2025 releases, while `docs/capability-matrix.md` now carries Gemma 4, Qwen3.5/3.6/3.8,
> Qwen3-Next, GLM-4.5/4.6, Llama 4, Kimi K2.x, gpt-oss, Nemotron-H, Granite-4.0-H, Laguna and
> Mellum2. The visible mismatch is the front page: the README GIF, the quickstart and the release
> assets all say **Qwen2.5-Coder**, which understates a runtime that ships DeltaNet hybrids and
> block drafting. This doc consolidates the whole refresh: what changes, what it costs, the gates
> on the one expensive piece, and what is deliberately NOT added.
>
> **Provenance rule (inherited from `benchmarks.md`):** any tok/s or size figure that lands in a
> demo README ships with a traceable run — commit, model+quant, box, config. The current "~57
> tok/s" claims predate the P12 quant fix and must be re-measured, not carried forward.

## Inventory — every model reference the demos make today

| where | reference | age class |
|---|---|---|
| `demo/chat` (embedded; release assets; README; root README GIF + quickstart) | Qwen2.5-Coder-Instruct 0.5B / 1.5B | 2024 |
| `demo/agent/README.md` | qwen2.5-coder-0.5b, gemma-3-4b-it, `~/.ken/model` | 2024–25 |
| `demo/gemma/README.md` | gemma-3-270m, gemma-3-1b, tinyllama-1.1b | 2023–25 |
| `demo/gemma-web/README.md` | gemma-3-270m(-it) | 2025 |
| root `README.md` | qwen2.5-coder-0.5b quickstart; **gemma-4-E2B QAT (already current)** | mixed |

Three of the four demos take `--model` — refreshing them is a docs pass. Only `demo/chat` bakes
weights into the binary, and that is where all the real cost and all the gates live.

## Tier 1 — README example sweep (an afternoon, mostly mechanical)

Point the flag-loaded examples at models the matrix supports today. Suggested mapping, not
binding — pick per demo at edit time:

- `demo/agent`: qwen2.5-coder-0.5b → a Qwen3.5-small (4B) or Gemma 4 E4B example;
  keep the `~/.ken/model` line (ken integration is a feature, not staleness).
- `demo/gemma` / `demo/gemma-web`: gemma-3-270m/1b → Gemma 4 E2B/E4B (`google/gemma-4-E2B[-it]`),
  keeping one Gemma 3 line since that family stays supported; drop the tinyllama example
  (it demonstrates nothing gemma-specific and dates the page).
- root README quickstart: align with whatever Tier 2 decides; the gemma-4-E2B QAT paragraph is
  already right and stays.

Constraint carried from the capability matrix: **Gemma models stay flag-loaded, never embedded** —
the Gemma license attaches redistribution terms, and `demo/chat`'s whole mechanism is
redistributing weights inside a release binary. Apache-2.0/MIT models only for embedding.

## Tier 2 — the embedded model in `demo/chat` (the one real task; gated)

**Candidate: the Qwen3.5 small series (0.8B / 2B / 4B / 9B, announced 2026-08)** — same Qwen3.5
foundation as `qwen3_5`/`qwen3_5_moe`. This is the interesting swap: the headline demo would then
run the DeltaNet hybrid family this repo built the adapter, the P12 quant fix and the GGUF loader
for — demonstrating goinfer's own differentiating work instead of an architecture every runtime
runs. The 0.8B replaces the 0.5B tier, the 2B replaces the 1.5B tier.

Gates, in kill order — each is cheap and each can end the tier:

1. **License, per-repo, on the repo page.** Embedding is redistribution. Expect Apache-2.0
   (Qwen precedent), verify anyway — the HF list endpoint has already lied to this program once
   (P10 increment 1). No weights into `build-embed.sh` before this clears.
2. **Loader + model_type check.** The small series is stated as natively multimodal; goinfer is
   text-only. Confirm the checkpoints load through the dense `qwen3_5` adapter (text_config
   nesting has the `qwen3_5_moe_text` precedent) and that a GGUF or the prequant `.giw` path
   covers them — the matrix lists dense `qwen3_5` as safetensors; the Qwen3.8 GGUF work (P13)
   may already close this. A tensor dump accounted against the config settles it.
3. **Template + thinking mode.** The P10 lesson, standing: render the model's own template,
   non-thinking, and let the harness's hard-errors guard it. A chat demo that silently runs a
   thinking-mode model gives a visibly worse first impression at double the latency.
4. **Measured tok/s on a laptop CPU, before committing.** The demo's pitch is "tiny + fast".
   The current claims (~57 tok/s 0.5B, ~26 tok/s 1.5B) are stale either way (pre-P12) — re-measure
   the incumbent AND the candidate on the same box, same quant, same commit. Pre-registered risk:
   the hybrid's DeltaNet layers may decode slower per-token than the old dense 0.5B at equal
   size. **If the 0.8B loses the feel test, keep the Qwen2.5 tier alongside rather than
   replacing it** — the README then says plainly which tier is fast and which is current.
5. **Asset size + canned demos.** Rebuild both tiers via the prequant path; confirm the ~617 MB
   class holds (0.8B int8 ≈ +60% params vs 0.5B — the README size table updates from the built
   artifacts, not projections); re-run every `/demo` prompt on both tiers and swap out any that
   regress (the `json`/`extract` constrained-decoding pair must stay — it is the party trick).

Lands with: new release assets + `RELEASE_TEMPLATE.md` names, the README size/speed tables from
measured runs, and a re-recorded `docs/assets/demo.gif` — the GIF is the single most-seen stale
artifact in the repo.

## Tier 3 — one drafter tape (show the differentiator; no code)

Nothing demos `serve --drafter`, a shipped production feature with measured end-to-end numbers
(P10: code 1.44×, math 1.58× on the CUDA box). The cheap, zero-maintenance form already has a
precedent in-repo: `demo/mellum2-gpu.tape`. Record `demo/drafter.tape` — same prompt side by side,
`--drafter` off then on, on the linux box (the feature is CUDA-only today; Metal stays blocked on
§A2 divergence — the tape must not imply otherwise). Caption the tape with the measured figures
and their provenance per the rule above, and link it from the serve section of the README next to
the existing `--drafter` mention in `docs/api-tiers.md`.

## Non-goals (deliberate, with reasons)

- **No new `demo/` modules.** `demo/agent` is part of the tri-module tag; every demo module is
  permanent release surface. The drafter and GPU-resident stories are told in tapes and README
  sections at near-zero carrying cost.
- **No Gemma embed** (license, above) and **no embedded-model change to `demo/agent`** — it is
  flag-loaded and its value is the RAG/agent loop, not the model.
- **No benchmark claims beyond the demo's own box.** Cross-engine comparisons live in
  `docs/benchmarks.md` with their provenance; demo READMEs quote only their own measured runs.

## Order of work

Tier 1 is unblocked now and independent. Tier 2 gates run 1→5 and stop at the first kill; a kill
at gate 1 or 2 leaves Tier 1's sweep still worth shipping and the incumbent embed in place. Tier 3
is independent of both and can ride any release. Suggested queue filing: one entry in
`docs/queue-engineering.md` citing this doc; Tier 2's gate-4 measurement lands in
`docs/measurements/` like any other number.
