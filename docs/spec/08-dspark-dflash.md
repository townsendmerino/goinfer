# 08 — DSpark / DFlash block drafters

> Status: **proposal — not built** (cut 2026-08-15, queued as **P10** in
> [queue-performance](../queue-performance.md)). This is the α lever
> [05](./05-eagle3-head.md)/[06](./06-acceptance-analysis.md) named, arriving pretrained:
> block drafters whose draft cost is per-*block* instead of per-token, with published
> acceptance far above EAGLE's measured ~1.6 tok/verify. Depends on
> [00-core](./00-core.md); composes with [03](./03-router-tree.md); its verify shape is
> exactly the short linear draft [07](./07-stageb-gemm-verify.md)'s large-dim re-measure
> asked for. Numbers quoted below from the upstream projects are **their claims on their
> hardware**, not ours — reproducing them on our harness is increment 0 of nothing and
> gate for everything.

## Provenance

Surfaced via `ARahim3/mlx-dspark` (an MLX port measured on an M4 Pro; r/LocalLLaMA post
`1vokrcy`, 2026-08). The methods are DeepSeek's **DSpark** and z-lab's **DFlash**; both
publish trained drafter checkpoints on HF for targets goinfer already runs (Gemma-4,
Qwen3, Nemotron — confirmed pairs listed under increments; the post's headline model,
Qwen3.8-27B, implies newer pairs — re-check HF at spike time). We would import **weights
and method**, not code: the MLX/PyTorch implementations serve as the parity oracle, same
as AngelSlim did for 05.

## Idea

Both are lossless drafters under the standard verify — everything 00-core already
enforces (greedy bit-exact, sampled in-distribution) holds regardless of drafter quality.
What is new is the draft-side economics:

- **DSpark** (semi-autoregressive block drafter). A small parallel backbone (~5
  transformer layers) reads the target's hidden states — our `ForwardCapture` seam — and
  proposes a **7-token block** in essentially one pass; a **rank-256 Markov head** then
  adds cheap sequential token-to-token dependency across the block (correcting the
  suffix decay that pure parallel prediction suffers), and a **confidence head** scores
  each position so block length adapts per round (our [04](./04-adaptive-depth.md), but
  learned). Draft cost per round ≈ one pass of a 5-layer backbone + a rank-256 chain —
  versus EAGLE's head forward *per drafted token* plus a per-round correction forward,
  the exact overhead that 05's scorecard says ~1.6 accepted tokens couldn't pay back.
- **DFlash** (block-diffusion drafter). Denoises a **16-token block in one parallel
  pass**, and reuses the **target's own embedding and LM head** — the drafter ships only
  the denoiser trunk. Upstream reports acceptance length ~**6.0 on code/math** (up to
  ~2.1× end-to-end on the M4 Pro) and ~**0.98× on open chat** — a drafter that is
  strongest precisely on goinfer's marquee traffic (structured output, code, tool
  calls), where our own calibrated floor is α̂_grammar ≈ 0.20 (06).

Per the [README thesis](./README.md#thesis) — `speedup ≈ accepted/verify ÷ draft
cost/verify` — these attack **both terms at once**, which no source in the current
router does: n-gram (02) has a free draft but only fires on copy traffic; EAGLE (05) has
model-quality α on novel text but a draft cost that sank it; grammar (01) is the floor.

## Why goinfer is well-placed

- **The substrate is built and gated.** `Drafter` interface, batched verify
  (`forwardN` / `PrefillLastN`), `cache.TruncateTo` rollback, the 03 router with
  calibrated α̂ + online correction, and the hidden-state seam (`Model.ForwardCapture`,
  gated logits-byte-identical) that 05 already paid for. A new drafter is a new leaf,
  not new plumbing.
- **The verify amortization already exists where it matters.** On the resident CUDA
  path the batched M=k verify is measured **2.5–3.6× cheaper than k decodes at k=4–8**
  (`TestSpecVerifyCeiling`, D1) — and a 7- or 16-token **linear** block is exactly the
  short-linear-draft shape 07's large-dim re-measure said amortizes (M=4 → 1.58× at 70B
  dims), where trees were the wrong shape. What D1 lacks is a drafter with model-quality
  α on novel text; that is precisely what these are.
- **Same regime as the source numbers.** mlx-dspark's setting is single-user batch-1
  local decode on unified memory — goinfer's declared axis — so the published speedups
  are at least the right *kind* of number, unlike datacentre-batching results.
- **DFlash × `constrain` is a differentiated story.** The verify already applies the
  grammar mask, so constrained speculative decode stays lossless for free; and because
  DFlash reuses the target's LM head, the *same* mask can later be applied to the
  drafter's logits — the tokenization-aware forced drafter 01/06 said fundamentally
  needs the model, without training one.

## What it requires (the honest cost)

- **Two new drafter forwards.** DSpark: a 5-layer backbone (ops we have: attention +
  MLP + norms) + a rank-256 Markov head + a confidence head — closest to existing code.
  DFlash: **bidirectional (non-causal) attention over a masked block** — a genuinely new
  forward type for `decoder`, though bounded (one trunk, one pass, no KV politics).
- **Per-target trained checkpoints, instruct-matched.** These help specific presets,
  not every family; upstream reports base-vs-instruct mismatch craters acceptance
  (~47% vs ~82%). Availability is the same gating problem 05 named.
- **Protocol extraction, again.** As with EAGLE, the repos will not document every
  wiring choice (which layers are tapped, norm placement, mask schedule). The 05 lesson
  is priced in: a from-scratch forward that is "structurally correct" can still sit at
  α≈0.3 — so **reference-parity against the upstream implementation comes before any
  acceptance measurement**, not after a disappointing one.
- **Loader + fixture work.** Safetensors loaders for both drafter formats, converted
  fixtures, and a dumped-tensor CONFIRMED section in this doc once shapes are in hand
  (the 05 pattern).

## Licensing / IP note

Not yet checked — a **gate, not a footnote** (the 05 discipline). Verify the license on
the *specific* HF checkpoint repos (`deepseek-ai/dspark_*`, `z-lab/*DFlash*`) before
importing weights; reimplementing the method in Go is the clean path for the code side,
and we import no Python either way. Record the decision in the PR as 05 did.

## Where it can win — and where it will lose (pre-registered)

- **CPU: expect the 05 kill-gate to fire again, but measure once.** The verify node
  costs ~0.5 of a target step; even a *perfect* 7-token block tops out near ~2.2× (a
  ceiling mlx-dspark states for M-series and our 00-core economics reproduce). DSpark's
  per-block draft cost is the one term that changed since 05's CPU loss (~0.4×) — so one
  measured run documents whether it changes the verdict. Prediction: **no** on small
  targets; the entry is the GPU paths.
- **Resident CUDA (`linux`, 2070S 8 GB): Qwen3-4B int4/int8 + DSpark block7.** The D1
  verify amortization is already measured on this box. This is the cheapest end-to-end
  venue and the first gate.
- **Metal resident (`mac`, M1 Pro 16 GB): Gemma-4 12B int4 + either drafter.** The
  bigger target makes the draft relatively cheaper (06's "translates only when the
  target step is expensive enough"); 12B at 8-bit does not fit 16 GB with a drafter
  alongside, so int4 target + 4-bit drafter (~1.8 GB) is the configuration.
- **DFlash routes to constrained/tool traffic; DSpark to open text.** Both enter as
  router sources with trace-fit α̂ (06's runbook), not as replacements for n-gram —
  which stays the copy-traffic winner and costs nothing.

## Kill-gates (pre-registered, in order)

1. **Parity gate.** The Go drafter forward matches the upstream reference on dumped
   fixtures (logit cosine + argmax at drafted positions). Not cleared → do not measure
   acceptance; debug or stop. (This is the gate 05 cleared too late.)
2. **Acceptance gate.** Measured tok/verify on the 00-core suites (`chat` / `code` /
   constrained), trace-fit α̂ per 06. Bar: **≥3.0 tok/verify** on the paired target on
   at least one suite — anything near EAGLE's 1.6 means the protocol is wrong (back to
   gate 1) or the claims don't transfer (stop, record).
3. **Wall-clock gate.** End-to-end ≥**1.3×** vs plain resident decode (the 07 funding
   bar) on ≥1 real workload on ≥1 GPU backend, lossless gates green
   (`TestEagleSpecParity` shape). Miss broadly → park with numbers, like Stage B.
4. **Router gate.** With the drafter routed by α̂, the *mixed*-workload number must not
   regress vs n-gram-only — a drafter that wins its suite but mis-routes is a 03
   calibration item, not a ship.

## Build increments (planned)

1. **License + checkpoint audit — DONE 2026-08-15, not a kill, but a real flag.** Enumerate the HF
   pairs (confirmed at cut time: `deepseek-ai/dspark_gemma4_12b_block7`,
   `deepseek-ai/dspark_qwen3_4b_block7`, `z-lab/gemma4-12B-it-DFlash`,
   `z-lab/Qwen3-4B-DFlash-b16`), licenses, sizes. Kill here was free; the result doesn't kill it,
   but it isn't clean either:

   | repo | exists | license | size | notes |
   |---|---|---|---|---|
   | `deepseek-ai/dspark_gemma4_12b_block7` | yes | **NONE — no model card at all** | ~3.43B params, BF16, ~6.86GB | drafts for Gemma4; `Gemma4DSparkModel` |
   | `deepseek-ai/dspark_qwen3_4b_block7` | yes | **NONE — no model card at all** | ~1.39B params, BF16, ~2.79GB | drafts for Qwen3; `Qwen3DSparkModel` |
   | `z-lab/gemma4-12B-it-DFlash` | yes | **NONE — no model card at all** | ~0.73B params, BF16, ~5.82GB | drafts for Gemma4-12B-it; tagged `model_type: qwen3` (an internal arch label, not a Qwen target) |
   | `z-lab/Qwen3-4B-DFlash-b16` | yes | **MIT**, documented | ~0.54B params, BF16, ~1.08GB | explicitly "must be used in conjunction with the target model Qwen/Qwen3-4B"; needs `trust_remote_code=True`; cites arXiv:2602.06036 |

   **3 of the 4 confirmed pairs ship no license whatsoever** — not restrictive, not permissive,
   just absent (HF's bare "No model card" state). That is not automatically a blocker (DeepSeek and
   z-lab's OWN base-model releases are typically permissive, and the omission may just be a
   checkpoint-repo oversight), but it is not something to import and ship on the assumption it's
   fine either — **resolve before increment 2 touches these specific four weights**: either an
   explicit license appears upstream, or ask the authors, or treat it as blocking for those four
   and lean on the ones below that ARE licensed.

   **Re-checked HF for newer pairs, per the doc's own note.** The source post's headline model
   (Qwen3.8-27B) has **no official deepseek-ai or z-lab pair** — the only DSpark checkpoint
   targeting it is a **third-party community one**, `RadixArk/Qwen3.8-27B-DSpark` (license `other`,
   trained via SpecForge, not deepseek-ai/z-lab — do not treat as a confirmed pair). But deepseek-ai
   and z-lab both ship **more official pairs than the four originally listed**: deepseek-ai adds
   `dspark_qwen3_8b_block7`, `dspark_qwen3_14b_block7`, and full-size `DeepSeek-V4-{Flash,Pro}-DSpark`
   (**MIT**, documented); z-lab adds ~20 more DFlash targets (`Qwen3.6-27B-DFlash`,
   `Qwen3-8B-DFlash-b16`, `gpt-oss-120b-DFlash`, `Kimi-K2.6-DFlash`, `gemma-4-31B-it-DFlash`, among
   others — inconsistent naming, "gemma-4" vs "gemma4"). **No Nemotron pair exists from either
   deepseek-ai or z-lab** (confirms the doc's own honest gap) — but **NVIDIA ships its own
   first-party pair**, `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-{DSpark,DFlash}` (license
   `other`), a genuinely new option this audit surfaced that the original scoping didn't know about.
2. **Protocol extraction + fixtures.** Read the reference implementations
   (mlx-dspark being MLX, closest in spirit to our decode loop), dump per-position
   tensors for a short trace, convert one DSpark checkpoint to f32 safetensors, write
   the CONFIRMED-shapes section here.
3. **DSpark forward + loader** (gate 1), then **Drafter wiring** into the existing
   verify on CPU — correctness first where debugging is cheap; take the one CPU
   wall-clock measurement while there.
4. **Resident CUDA end-to-end** on Qwen3-4B (gates 2–3). This is the go/no-go for the
   rest of the program.
5. **DFlash forward** (the new bidirectional block pass) + constrained-traffic
   measurement vs the 01/02 router baseline (gates 2–4).
6. **Metal 12B run** (`mac`), only if 4 clears.

## Validation plan

- Correctness: lossless by construction (verifier owns `p`); output ≡ non-speculative,
  bit-exact greedy + sampled in-distribution, per backend — the standing 00-core bar.
- Acceptance: reproduce upstream acceptance on a matching model/workload before
  claiming transfer; publish α̂ tables per 06 with the same held-out discipline.
- Speed: 00-core harness suites on both boxes; every number lands in
  [experiments.md](./experiments.md) dated, with the losing runs kept.
