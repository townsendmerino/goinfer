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
`1vokrcy`, 2026-08). **The upstream is `deepseek-ai/DeepSpec` (github, MIT)** — DeepSeek's own
training + inference implementation for DSpark, DFlash *and* Eagle3, which mlx-dspark's `NOTICE`
names as its source and whose README lists the HF checkpoint repos as its released outputs
(paper: arXiv:2607.05147). Found during increment 2; it is the authoritative reference for the
parity gate, and it documents far more of the wiring than this page assumed. The methods are
DeepSeek's **DSpark** and z-lab's **DFlash**; both
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
   **CLEARED for the DFlash trunk, 2026-08-15 (`linux-62gb`)** — `TestDFlash_referenceParity`
   over `decoder/dflash.go`, fed the reference's own `fused_context`/`block_in`:

   | trace | ctx | layer_out.0–4 cosine | trunk_out cosine | maxAbs |
   |---|---|---|---|---|
   | `raw` | 5 | 1.00000000 ×5 | **1.00000000** | 5.2e-05 |
   | `chat` | 23 | 1.00000000 ×5 | **1.00000000** | 3.1e-05 |

   **And the gate was shown to FAIL**, per this repo's "a gate must be able to run, and able
   to fail" rule — a clean first-run cosine of 1.0 is exactly when that rule earns its keep.
   Four mutations, each one of the wiring details identified as a silent-divergence risk,
   all correctly rejected: **causal block** instead of bidirectional; **norming the fused
   context** with `input_layernorm` the way the block is (the natural-looking wrong port);
   **q roped at the block-local position** instead of the absolute one; **dropping the
   per-head `k_norm`**. The unmutated forward passes. So the 1.0 is a measurement, not a
   tautology.

   **Second half also CLEARED the same day — `TestDFlash_targetEndToEnd`.** The trunk gate
   feeds the reference's own inputs, which proves the trunk but says nothing about the two
   ends DFlash borrows from the target. This runs the whole path on goinfer's own Qwen3-4B
   (f32): `ForwardCapture` → `FuseContext` → trunk → target `lm_head`, and compares the
   drafted token ids. **15/15 exact on both traces**, with the anchor token agreeing before
   the drafter even runs. So our capture seam, the 5-tap convention *and its +1 layer-output
   indexing*, the fusion, the trunk, and the borrowed embed/head all agree with the
   reference at the token level — the seam question flagged above is answered: no off-by-one.

   **Gate 1 is therefore CLEARED for DFlash, both halves.** Acceptance measurement is
   unblocked; per the pre-registration, that is gate 2 and it happens on the GPU paths.
2. **Acceptance gate.** Measured tok/verify on the 00-core suites (`chat` / `code` /
   constrained), trace-fit α̂ per 06. Bar: **≥3.0 tok/verify** on the paired target on
   at least one suite — anything near EAGLE's 1.6 means the protocol is wrong (back to
   gate 1) or the claims don't transfer (stop, record).
   **CLEARED 2026-08-15 (`linux-62gb`, `TestDFlashAcceptance`) — at 6.14/7.32 tok/verify,
   but only after THREE harness errors were found and fixed, each of which had produced a
   confident wrong number first.** The full sequence is kept below because the errors are
   the transferable part; the final result is in "RE-MEASURED IN NON-THINKING MODE".

   The first reading to survive scrutiny was a MISS. At 32 new tokens the gate read 3.53
   (over the bar) and that was an artifact of generation length — the sustained number was
   below it:

   | suite | 32 new tokens | **160 new tokens** |
   |---|---|---|
   | **code** | 3.53 ✅ | **2.90** ❌ |
   | math | 2.44 | 2.79 |
   | chat | 2.79 | 2.13 |

   (160-token run: code 168 rounds / 487 tokens, math 115 / 321, chat 151 / 322.)

   **Why the short run lied, and why it was checked anyway.** A 32-token generation is
   mostly the boilerplate opening — *"Sure! Here's a Python function that…"* — which is
   exactly the text a block drafter nails. Extending to 160 tokens moves the measurement
   into the body, where code drops **3.53 → 2.90** and chat **2.79 → 2.13**. (Math rises,
   2.44 → 2.79: at 32 tokens it had barely started its working-out. Small samples both
   ways.) Nothing forced this re-run — the gate had already gone green. It was run because
   a bar cleared by 0.53 on the easiest possible sample is not a result, and the cost of
   being wrong here is an entire GPU increment built on a number that was never real.

   Qwen3-4B **int8** target (the precision the resident GPU paths run), greedy, ChatML,
   32 new tokens/prompt. Acceptance is a numerics property, so it transfers to the GPU
   paths at equal precision — **wall-clock does not, and this harness deliberately does
   not measure it** (it verifies with 16 sequential forwards, not one batched M=16 pass).
   Losslessness is structural here: every emitted token is the *target's* own argmax.

   **The two optimistic readings this program produced, and what they cost.** Increment 2
   reported **11.0 tok/verify** — one round, one prompt, labelled a smoke signal. The
   32-token sweep reported **3.53** — a real sweep, but on the easiest part of the output.
   The sustained number is **2.90**. Each reading was ~2–4× the next, and each was honest
   about its own scope at the time; what would have been dishonest is planning on any of
   them. **Plan against 2.90.**

   **What the miss does NOT mean.** Gate 1 is cleared and mutation-verified, so this is
   not "the protocol is wrong (back to gate 1)" — our forward reproduces the reference to
   cosine 1.0 and 15/15 drafted ids. That leaves the pre-registration's other branch, "the
   claims don't transfer" — but three differences from upstream's setup remain
   **unseparated**, and declaring transfer failure before separating them would be its own
   unfounded claim:
   - **int8 target vs their bf16** — quantizing the target changes `p`, and the drafter
     was trained against an unquantized one. The prime suspect.
   - **ChatML vs Qwen3's own template** (ours omits the thinking block).
   - **greedy-only** vs whatever sampling upstream measured under.

   **THE ATTRIBUTION RAN, AND IT IS THE SECOND BRANCH: the claims DO transfer, and the
   gap is ours.** z-lab's own `spec_generate` (verbatim from the MIT `dflash.py`, counter
   added) at their bf16 precision and their template, same prompts, same 160 tokens:

   | suite | ours (int8, ChatML) | **reference (bf16, own template)** | gap |
   |---|---|---|---|
   | code | 2.90 | **6.37** | 2.2× |
   | math | 2.79 | **3.86** | 1.4× |
   | chat | 2.13 | 2.31 | 1.08× |

   **6.37 on code reproduces upstream's published ~6.0 on our box.** So the drafter is
   as good as advertised and the pre-registered "stop, record" branch does NOT apply —
   what is wrong is our harness's configuration, which is a fixable thing rather than a
   verdict on the method.

   **The precision hypothesis was stated with a mechanism, and it is WRONG.** The gap's
   shape — largest where acceptance is highest (code 2.2×), absent where it is low (chat
   1.08×) — is the signature of a per-token α drop compounding over a long accepted run
   (geometric fit: reference α≈0.85, us α≈0.65), and an **int8 target** was the obvious
   culprit: every position where int8's argmax differs from bf16's is a forced rejection,
   a class this repo has already measured at ~7.5% of positions. **The f32-target control
   refutes it: code 2.76 at f32 vs 2.90 at int8 — no improvement, slightly worse.**
   Recorded because a mechanism-plus-precedent story that survives one plausibility check
   is exactly the kind that gets believed without a control.

   **The actual cause is THINKING MODE, and upstream documented the hazard.** Our harness
   rendered with `chat.ChatML()`, which stops at `<|im_start|>assistant\n`. Qwen3's own
   template with `enable_thinking=False` appends **`<think>\n\n</think>\n\n`** (ids
   `151667, 271, 151668, 271` — verified: ChatML's ids plus those four are *exactly*
   `apply_chat_template`'s). Without it, Qwen3-4B runs in **thinking mode** — and the
   DFlash drafter was trained on **non-thinking** output. DeepSpec's own README says so
   twice: the checkpoints are trained on data "generated by its corresponding target model
   in non-thinking mode", and it warns to re-finetune "especially if the target model is
   expected to run in thinking mode".

   So the 2.90 measured the drafter against a distribution it was never trained on. The
   gap's suite ordering follows: thinking-mode divergence is worst on code/math, where the
   chain-of-thought is longest, and mildest on open chat — which is precisely the observed
   2.2× / 1.4× / 1.08×.

   **This is the THIRD instance of one error class in this program**, and that is the part
   worth keeping: increment 2's raw-vs-chat prompt (0/15 vs 10/15 accepted), the 32-vs-160
   token length artifact (3.53 vs 2.90), and now templated-but-thinking (2.90 vs the
   re-measure). Every one of them made the drafter look different than it is, and every one
   was an **input-distribution** mistake rather than a code defect — with gate 1 green and
   mutation-verified throughout. **For a speculative drafter, the harness's prompt
   construction is part of the measurement apparatus, not setup.** Re-measured in
   non-thinking mode below.

   ### Gate 2 — RE-MEASURED IN NON-THINKING MODE: **CLEARED**, decisively

   | suite | thinking (the wrong harness) | **non-thinking (correct)** | reference bf16 |
   |---|---|---|---|
   | code | 2.90 | **6.14** ✅ | 6.37 |
   | math | 2.79 | **7.32** ✅ | 3.86 |
   | chat | 2.13 | 2.18 | 2.31 |

   Qwen3-4B **int8** target, greedy, 160 new tokens. Best suite **7.32 tok/verify against
   a ≥3.0 bar** — not marginal, and **code lands within 4% of the bf16 reference (6.14 vs
   6.37)**, which is the cross-check that says our implementation is right and not merely
   lucky. **Gate 2 is cleared.**

   **The precision worry is retired, and that is the load-bearing result for increment 4.**
   The earlier note here warned that if int8 cost half the acceptance, the resident GPU
   increment would inherit the problem, since a quantized resident target is the entire
   point of it. It does not: int8 costs ~4% against bf16 on code. **A quantized resident
   target does not crater this drafter**, so increment 4's economics survive — the f32
   control that refuted the precision hypothesis was measuring the same thing from the
   other side.

   **Two things NOT to over-read.** (1) Our math 7.32 *exceeds* the bf16 reference's 3.86,
   which is not a win — it is 2 prompts / 44 rounds, and int8's own generated text simply
   happened to be more draftable than bf16's. Unexplained, small sample, recorded as such;
   do not quote it as evidence goinfer beats the reference. (2) **`chat` barely moved
   (2.13 → 2.18) and sits near the reference's 2.31** — open chat is genuinely hard to
   draft, exactly as upstream reports (~0.98× end-to-end there). The router (03) should
   expect DFlash to earn its keep on code/math and to be roughly a wash on open chat,
   which is the same conclusion 06 reached from the other direction.
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
   (~~**MIT**, documented~~ — **CORRECTED 2026-08-15: only the two `DeepSeek-V4-*` repos are MIT.
   `dspark_qwen3_{8b,14b}_block7` carry NO license and no model card, exactly like the 4b — this
   sentence's trailing "(MIT, documented)" read as covering the whole list and sent increment 2 at
   the 8b as a licensed checkpoint. The HF *list* endpoint reports `license=None` for every repo
   including the genuinely-MIT ones, so only per-repo detail queries separate them**); z-lab adds
   ~20 more DFlash targets (`Qwen3.6-27B-DFlash`,
   `Qwen3-8B-DFlash-b16`, `gpt-oss-120b-DFlash`, `Kimi-K2.6-DFlash`, `gemma-4-31B-it-DFlash`, among
   others — inconsistent naming, "gemma-4" vs "gemma4"). **No Nemotron pair exists from either
   deepseek-ai or z-lab** (confirms the doc's own honest gap) — but **NVIDIA ships its own
   first-party pair**, `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-{DSpark,DFlash}` (license
   `other`), a genuinely new option this audit surfaced that the original scoping didn't know about.
2. **Protocol extraction + fixtures — PROTOCOL HALF DONE 2026-08-15 (`linux-62gb`); the
   checkpoint half is BLOCKED, see below.** Read the reference implementations
   (mlx-dspark being MLX, closest in spirit to our decode loop), dump per-position
   tensors for a short trace, convert one DSpark checkpoint to f32 safetensors, write
   the CONFIRMED-shapes section here.

   **The license gate increment 1 raised did not clear — it got worse on inspection, and it
   blocks exactly the half that needs weights.** Every `deepseek-ai/dspark_*_block7` drafter
   (4b, 8b, 14b, gemma4_12b) carries **no license and no model card** — three files each
   (`.gitattributes`, `config.json`, `model.safetensors`). The only MIT DSpark repos with a
   `LICENSE` are `DeepSeek-V4-{Flash,Pro}-DSpark` at **166.9 GB / 892.8 GB**, which are full-size
   V4 models rather than standalone drafters. Increment 1's own summary said the 8b/14b were
   MIT; that was an artifact of the HF *list* endpoint, which reports `license=None` for every
   repo including the two that really are MIT — corrected in `queue-performance.md`. **No weights
   were downloaded.**

   **What replaced the weights, and why it is stronger than a skim.** mlx-dspark's `NOTICE`
   names the upstream the design page did not know about: **`deepseek-ai/DeepSpec`
   (github, MIT)** — DeepSeek's own training + inference implementation, whose README lists
   these exact HF repos as its released checkpoints and whose `config/dspark/*.py` are the
   configs that produced them. So the protocol came from **first-party MIT source**, not from
   the paper and not only from a third-party port. See the CONFIRMED section below.
3. **DSpark forward + loader** (gate 1), then **Drafter wiring** into the existing
   verify on CPU — correctness first where debugging is cheap; take the one CPU
   wall-clock measurement while there. **RE-TARGET TO DFLASH (2026-08-15)** while the DSpark
   license is unresolved: the licensed checkpoint, the f32 conversion and the parity fixtures
   are DFlash's, so gate 1 runs there. The forward is *smaller* than DSpark's (no embed, no
   lm_head, no Markov chain, no confidence head — 58 tensors, one pass) but carries the two
   details a port loses silently: the split RoPE application (q at block positions, k over
   context+block) and `logits_start = 1`. Seam prerequisite either way: `ForwardCapture`
   handles 5 taps already but **rejects `gemma4`**, so the `mac` leg needs it extended.
4. **Resident CUDA end-to-end** on Qwen3-4B (gates 2–3). This is the go/no-go for the
   rest of the program. **Gate 2 is already cleared on CPU** (above — acceptance is
   numerics and transfers), so what remains here is gate 3, wall-clock.

   **ARCHITECTURE DECIDED 2026-08-15 BY MEASUREMENT: the drafter must be GPU-RESIDENT
   TOO. A resident target with a CPU drafter is not a viable configuration.**
   `BenchmarkDFlashTrunk`, one block draft, f32 CPU:

   | context | uncached | with the context cache | 
   |---|---|---|
   | 64 | 1,640 ms | **1,406 ms** |
   | 512 | 3,875 ms | **1,926 ms** (−50%) |
   | 2048 | 12,871 ms | **6,090 ms** (−53%) |

   A resident 4B decode step is ~10–16 ms, and a block buys ~6 accepted tokens ≈ 100 ms of
   decode. At 1.4 s per draft that is a **>10× net loss** — worse than Lever 2's measured
   0.11×, which is the same wall from the same cause ("the DRAFT was the wall, not the
   verify"). Even granting a generous 5–10× from the two known inefficiencies below, the
   CPU trunk lands at 150–300 ms and still loses. mlx-dspark reaches its numbers with the
   drafter on the GPU as well; so must we.

   **The context cache landed as part of finding this** (`DFlashContext`, matching
   mlx-dspark's `CtxCache` and dflash.py's `DynamicCache`): the first implementation
   re-projected the entire context in every layer on every round, which is O(ctx × layers)
   of repeat work per round. Gate 1 re-passes bit-for-bit through the cached path (cosine
   1.0, identical maxAbs), so it is an optimization, not a numerics change. **Any GPU port
   should carry the cached design** — projecting committed positions once, at commit.

   **Two known CPU inefficiencies, deliberately NOT fixed** (this path is not production —
   fixing them would be optimizing a configuration just measured as unviable), but recorded
   because the GPU port must not inherit them: the block's projections run as 16 separate
   `M=1` matmuls where one `M=16` batch would reuse the weights, and the attention inner
   loop is scalar f64 Go rather than a `linalg` kernel.

   **The other prerequisite — resident hidden capture — is BUILT and gated (2026-08-15).**

   ~~The resident runner has no capture seam.~~ **That claim, made twice, was wrong**, and
   how it was wrong is the part worth keeping: the search was for the CPU seam's identifiers
   (`captureLayers`, `ForwardCapture`), and their absence in `cuda/` was read as absence of
   the *mechanism*. `cuda/resident.go` already had **`layerCap`** — a divergence-localization
   probe that snapshots the residual `r.x` after **every** layer. **Absence of a name is not
   absence of a capability.**

   What `layerCap` is not, is a production seam: every layer, a `stream.Sync()` and a
   download each, into an unbounded buffer — 36 syncs per token on a 36-layer model, fine to
   bisect a bug with, far too expensive to decode against. So the work was smaller than
   "build a seam" and realer than "it already exists":

   **`cudaResident.SetHiddenCapture(taps []int)` / `HiddenCapture()`** — only the TAPPED
   layer outputs, into fixed slots (5 syncs, not 36), ascending taps validated, `nil`
   disarms, off by default so no other family pays for it. Reuses the existing `capVec`
   helper, at the same point in `launchToken` as `layerCap` (after segC, where `r.x` is
   unambiguously the layer output).

   **Gated by `TestResidentHiddenCapture`** — resident int4 captures vs
   `decoder.Model.ForwardCapture`, 5 taps × 5 positions: cosine **0.99945 / 0.99824 /
   0.99801 / 0.99828 / 0.99793**. **And it fails when it should:** an off-by-one tap (`l+1`)
   drops layer 1 to **0.264**. A wrong layer yields plausible-looking vectors, so only the
   cross-check against the HF-gated CPU seam catches it.

   The `mac` side still has the sibling gap increment 2 found (`ForwardCapture` rejects
   `gemma4`): the CPU seam remains single-arch even though the resident one is now general.

   ### Gate 3 PROJECTED from measured inputs, before building the trunk

   The trunk is the expensive build here (non-causal cross-attention kernels at M=16 over
   `[ctx‖block]`). Both terms of the speedup are measurable without it, so they were.

   **Measured** (`TestSpecVerifyCeiling`, extended to k=16 and pointed at the real pairing
   through a new `GOINFER_CUDA_MODEL` override — Qwen3-4B int4 resident, depth 1024, 2070S):

   | k | batched M=k | k× sequential | cheaper by |
   |---|---|---|---|
   | 8 | 26.4 ms | 89.4 ms | 3.39× |
   | **16** | **46.4 ms** | **178.9 ms** | **3.86×** |

   M=1 decode is **11.12 ms**; fitting `T(M) = W + M·C` gives an **8.77 ms** weight-read term
   plus **2.35 ms/row**, so a 16-wide verify costs ~4.2 plain decodes.

   **Modelled draft:** 0.537 B drafter params against 4.02 B target = ratio **0.134**, which
   independently matches its 5-of-36 layer share (0.139). Bracketing the draft cost:

   | draft cost | code (α 6.14) | math (α 7.32) | chat (α 2.18) |
   |---|---|---|---|
   | 6.2 ms (param-scaled fit) | **1.30×** | **1.55×** | 0.46× |
   | 2.7 ms (bandwidth + FLOP floor) | **1.39×** | **1.66×** | 0.49× |

   **Gate 3 clears on math either way, code sits ON the 1.3× bar, and chat is a ~2× LOSS.**
   The bar is ≥1.3× on ≥1 real workload, so the trunk is worth funding — but the honest
   headline will be *"1.3–1.7× on structured traffic"*, not a flat speedup, and **gate 4's
   router becomes mandatory rather than a nicety**: at 0.46× on open chat a DFlash that
   fires indiscriminately is worse than no drafter. That agrees with 06's α̂-routing
   conclusion and with upstream's own ~0.98× on open chat.

   ### The draft term, MEASURED — and it kills the optimistic half of the bracket

   The bracket above spanned 1.30–1.39× on code because the draft cost was modelled. It is
   measurable without the trunk, on a structural fact worth stating plainly: **the DFlash
   drafter is exactly FIVE LAYERS OF THE TARGET'S OWN LAYER SHAPE plus `fc`** — same hidden
   2560, same 32/8 GQA at head_dim 128, same 9728 SwiGLU. 5 × 100.9 M + 32.8 M = **537.4 M**,
   matching the checkpoint to the digit. The resident runner already executes that layer 36
   times per token, so the drafter's cost can be read off the target instead of modelled.

   `TestDFlashDraftCostProbe` isolates the LM head via `launchToken`'s `head` flag — the head
   is 389 M the drafter does **not** have (it borrows the target's, already paid inside the
   verify), so charging it to the drafter would overstate the draft:

   | quantity (Qwen3-4B int4, depth 1024, 2070S) | measured |
   |---|---|
   | M=1 decode, with head | 11.105 ms |
   | M=1, layers only | 10.144 ms |
   | LM head | **0.961 ms** (9%) |
   | **per-layer, M=1** | **0.2818 ms** |

   Scaling the 36-layer M=16 verify the same way gives per-layer(M=16) ≈ 1.23 ms, so the
   drafter's 5 layers + `fc` ≈ **6.6 ms** — and that is insensitive to the one soft input
   (varying the head's M=16 cost from 1 to 5 ms moves code by 0.02×):

   | | code (α 6.14) | math (α 7.32) | chat (α 2.18) |
   |---|---|---|---|
   | earlier modelled bracket | 1.30–1.39× | 1.55–1.66× | ~0.47× |
   | **measured draft (6.6 ms)** | **1.29×** | **1.54×** | ~0.46× |

   **The 2.7 ms "bandwidth+FLOP floor" was wrong — the real draft is ~6.6 ms, the pessimistic
   end of the bracket.** So **code lands at 1.29×, just UNDER the 1.3× bar**, and it can only
   get worse: this still excludes the drafter's non-causal attention over `[ctx‖block]`,
   which has no counterpart in the target's per-token path.

   **Gate 3 therefore projects to PASS ON MATH ONLY (~1.5×), with code missing.** **[SUPERSEDED — see "All three suites swept" at the end: those figures all verify 16 positions. At each suite's own optimum width DFlash measures 1.74× code / 2.25× math / 0.92× chat, so code clears the bar and DSpark's margin closes.]** The bar is
   ≥1.3× on ≥1 real workload, so the trunk remains fundable — but the claim it can support
   has narrowed twice: from "1.3–1.7× on structured traffic" to **"~1.5× on math-like
   traffic, break-even on code, 2× loss on open chat"**. Fund the kernel work against that
   sentence, not the earlier one.
5. **DFlash forward** (the new bidirectional block pass) + constrained-traffic
   measurement vs the 01/02 router baseline (gates 2–4).
6. **Metal run** (`mac`), only if 4 clears. **RE-TARGET (2026-08-15): not the 12B.** Every
   Gemma-4-12B drafter is unlicensed — `z-lab/gemma4-12B-it-DFlash`,
   `deepseek-ai/dspark_gemma4_12b_block7` and `deepseek-ai/dflash_gemma4_12b_block7` are all bare
   repos (no license, no README, no LICENSE), verified per-repo. But z-lab ships **apache-2.0,
   documented** drafters for other Gemma-4 targets that the increment-1 sweep did not reach:
   **`z-lab/gemma-4-26B-A4B-it-DFlash`** and `z-lab/gemma-4-31B-it-DFlash`. Note the naming split
   increment 1 flagged is exactly the licensing split: the unlicensed one is `gemma4-…`, the
   licensed ones are `gemma-4-…`. 26B-A4B is a model this repo already runs resident, so it is the
   natural mac target. (Same org also ships an apache-2.0 **`PARO`** family — 5 repos, a method
   neither this page nor the audit knows about. Unexamined thread, recorded so it is not lost.)

## CONFIRMED architecture (`deepseek-ai/dspark_qwen3_8b_block7`, derived 2026-08-15)

> **Read the provenance line before the numbers.** 05's CONFIRMED section was *dumped from a
> checkpoint on this box*. This one is **derived from first-party MIT source and validated against
> the published checkpoint's exact byte count** — because the weights are unlicensed and were not
> downloaded (see increment 2). Every number below traces to one of:
>
> - **`deepseek-ai/DeepSpec`** (github, **MIT**) — `config/dspark/dspark_qwen3_8b.py` (the config
>   that produced this checkpoint), `deepspec/modeling/dspark/{common,markov_head}.py`,
>   `deepspec/modeling/dspark/qwen3/{config,modeling}.py`, `deepspec/eval/dspark/{draft_ops,
>   evaluator}.py`.
> - **`ARahim3/mlx-dspark`** (**MIT**) — the MLX inference port; read as the second source, and it
>   agrees with DeepSpec everywhere the two overlap.
> - **`Qwen/Qwen3-8B`** `config.json` (**Apache-2.0**) — the target dims the drafter inherits.
> - HF **blob metadata** for the drafter repo (file size only — no tensor data fetched).
>
> **What is NOT confirmed: any VALUE.** No per-position tensor dump, no f32 conversion, no logit
> cosine. That is the rest of increment 2 and it needs the license decision. Treat the shapes as
> load-bearing and the *forward* as reference-read; the parity gate (kill-gate 1) is still ahead.

**The byte-exact cross-check — why the derivation is trustworthy without the weights.** The
architecture below implies **2,371,081,729 parameters**, i.e. **4,742,163,458 bytes** at bf16.
The published `model.safetensors` is **4,742,170,330 bytes**. The **6,872-byte** difference is the
safetensors JSON header + its 8-byte length prefix, which is the right magnitude for the 64 tensors
listed below (~107 B/entry). A missing or mis-shaped tensor would move this by megabytes. So the
tensor inventory is complete and correctly shaped, established from published metadata alone.

**Config (DeepSpec `dspark_qwen3_8b.py`, verbatim):** `block_size=7`, `num_draft_layers=5`,
`target_layer_ids=[1, 9, 17, 25, 33]`, `mask_token_id=151669`, `num_anchors=512`,
`markov_rank=256`, `markov_head_type='vanilla'`, `confidence_head_alpha=1.0` (⇒
`enable_confidence_head=True`), `confidence_head_with_markov=True`.

`build_draft_config` **deep-copies the target config** and overrides only
`num_hidden_layers=5`, `tie_word_embeddings=False`, `architectures=["Qwen3DSparkModel"]`, plus the
DSpark fields — so every other dim is Qwen3-8B's: hidden 4096, heads 32/8 GQA, head_dim 128,
intermediate 12288, vocab 151936, `rms_norm_eps` 1e-6, `rope_theta` 1e6, no attention bias.

Tensors (`[out, in]`, 64 total, bf16):

| tensor | shape | note |
|---|---|---|
| `embed_tokens.weight` | [151936, 4096] | drafter's OWN copy, initialized from the target and **frozen** during training (`initialize_embeddings_and_head(freeze=True)`) |
| `fc.weight` | [4096, **20480**] | fuses the **5** captured target hidden states (5·4096) → 4096. 05's EAGLE-3 `fc` is the same idea at 3 layers |
| `hidden_norm.weight` | [4096] | RMSNorm on the fused target feature |
| `layers.{0..4}.self_attn.q_proj.weight` | [4096, 4096] | ×5 |
| `layers.{0..4}.self_attn.{k,v}_proj.weight` | [1024, 4096] | ×5 each, GQA 32/8 |
| `layers.{0..4}.self_attn.o_proj.weight` | [4096, 4096] | ×5 |
| `layers.{0..4}.self_attn.{q,k}_norm.weight` | [128] | ×5 each — Qwen3 per-head RMSNorm over head_dim |
| `layers.{0..4}.mlp.{gate,up}_proj.weight` | [12288, 4096] | ×5 each (SwiGLU) |
| `layers.{0..4}.mlp.down_proj.weight` | [4096, 12288] | ×5 |
| `layers.{0..4}.{input_layernorm,post_attention_layernorm}.weight` | [4096] | ×5 each — 2-norm (qwen/llama) layout, not gemma's 4-norm |
| `norm.weight` | [4096] | final, pre-head |
| `lm_head.weight` | [151936, 4096] | own copy, frozen; **full target vocab — no reduced draft vocab, so no `d2t`/`t2d`** (unlike 05's EAGLE-3 head) |
| `markov_w1.weight` | [151936, **256**] | rank-256 Markov: embedding indexed by the PREVIOUS token id |
| `markov_w2.weight` | [151936, 256] | projects the rank-256 latent back to full-vocab logit bias |
| `confidence_head.proj.{weight,bias}` | [1, **4352**], [1] | `AcceptRatePredictor` = one `Linear(→1)`; in_dim = hidden 4096 + markov_rank 256 because `confidence_head_with_markov=True` |

**Where the weight actually is:** embed + lm_head are **1.24 B of the 2.37 B params (52%)** and are
frozen copies of the target's. The 5 backbone layers are 0.96 B (193 M each); the Markov head is
77.8 M; the confidence head is 4,353 params. A loader that reuses the target's embed/head instead of
storing copies would halve the drafter's footprint — worth checking whether the two copies are
bit-identical to Qwen3-8B's own, since `freeze=True` says they must be.

### CONFIRMED forward (DeepSpec `eval/dspark/{evaluator,draft_ops}.py` + `mlx-dspark/model.py`)

Per drafting round, with the target's committed context already in the drafter's per-layer context
cache:

1. **Context update (once per accepted chunk).** `fused = hidden_norm(fc(concat(h[1], h[9], h[17],
   h[25], h[33])))` over the newly committed positions; each drafter layer projects it to K/V and
   appends to its own **`CtxCache`** — K is RoPE'd at its absolute position, **V is not**. This cache
   is append-only and is never rolled back (unlike the target KV cache).
2. **Block input.** `draft_input_ids = [last_committed_token, MASK ×6]` where MASK = `151669`, so
   the block is `block_size=7` wide.
3. **Backbone.** `noise_embedding = embed_tokens(draft_input_ids)` → 5 layers, each doing
   cross-attention with **Q from the block** and **K/V from `concat(ctx_cache, block)`**, then the
   standard Qwen3 2-norm layer → `norm`.
4. **The block attention is BIDIRECTIONAL — this is the load-bearing detail.** `draft_ops.py` calls
   the backbone with `is_causal=False`, and the training mask
   (`common.create_dspark_attention_mask`) allows `q_block_id == kv_block_id` with **no `q≥k`
   constraint** — every block position attends every other. Context keys are masked to
   `kv_idx < anchor_pos`. mlx-dspark independently confirms it (`mask=None`, "full bidirectional
   block"). **Consequence for us: the block width is not truncatable** — the trained distribution
   depends on all 7 positions, so "draft only 3" is not free (mlx-dspark's `draft_width` makes
   exactly this distinction and allows truncation only for causal DFlash-lineage heads).
5. **Logits.** `base_logits = lm_head(block_hidden[:, :7, :])` — `logits_start = 0`, so slot 0 both
   embeds the known token *and* predicts the first draft token. **All 7 positions are drafts**;
   a block7 head proposes 7 tokens/round and verifies 8.
6. **Markov correction — sequential, and not free.** For `i` in 0..6:
   `step_logits[i] = base_logits[i] + markov_w2(markov_w1[prev])`, `next = argmax/sample`,
   `prev = next`. So the block is parallel in the backbone and **serial in a 7-step scalar chain**.
   Each step is a [151936, 256] matvec ≈ **38.9 M MACs**, ×7 ≈ 272 M per round — small against a
   backbone pass but *not* the "cheap" the Idea section implies; it is ~0.3 of one 4096-dim layer's
   projection work, and it is latency-serial. Budget it explicitly on the CPU path.
7. **Confidence → adaptive block length.** `confidence_head(concat(block_hidden,
   markov_w1[prev_ids]))` → per-position logit → `sigmoid` → the proposal is truncated at the
   **first position below `confidence_threshold`** (`_confident_prefix_length`). This is 04's
   adaptive depth, learned and per-round, exactly as the Idea section claims.

### What this changes in the plan

- **The seam requirement is 5 layers, not 3, and it is config-driven per target.**
  `Model.ForwardCapture(id, cache, layers []int)` already takes an arbitrary layer list, so Qwen3
  needs no seam change. **But it hard-rejects architectures with their own `runLayers`** —
  including `gemma4` — so `dspark_gemma4_12b_block7`, the `mac` half of this program, needs the
  seam extended before it can run at all. That is a real increment-3 prerequisite the design page
  did not list.
- **Capture-point convention must be checked, not assumed.** DeepSpec indexes
  `hidden_states[layer_id + 1]` with `-1` meaning the embedding output, i.e. `target_layer_ids=[1,
  9, ...]` means *the output of layers 1, 9, …*. Confirm our seam's off-by-one against this before
  the parity gate — 05's lesson was that a structurally-correct forward can still sit at α≈0.3.
- **"The repos will not document every wiring choice" (the What-it-requires section) is now too
  pessimistic for DSpark.** DeepSpec documents the tapped layers, norm placement, mask schedule and
  block protocol outright. The remaining unknowns are values, not wiring — which is precisely what
  the blocked half would settle.

## CONFIRMED tensor structure (`z-lab/Qwen3-4B-DFlash-b16`, dumped 2026-08-15)

**Why DFlash and not DSpark.** Decided 2026-08-15 by Francis after the license finding above:
`z-lab/Qwen3-4B-DFlash-b16` is **MIT with a real model card**, 1.07 GB, and explicitly paired with
`Qwen/Qwen3-4B` (Apache-2.0). It is the only licensed, feasible drafter of either family, so
increment 2 completed against it. **This reorders the program** — DFlash's forward was increment 5;
its fixtures now exist first, and increments 3–4 (Go forward → resident CUDA) should follow DFlash
unless the DSpark license resolves. What increments 3+ inherit unchanged: the 5-tap seam, the
`fc`+`hidden_norm` fusion, and the bidirectional block — DSpark and DFlash agree on all three, and
even share `target_layer_ids=[1,9,17,25,33]` and `mask_token_id=151669`.

**Environment (checked, not assumed).** `~/.venv-vl` — 05's env — exists and carries
**torch 2.12.0+cpu, safetensors 0.8.0, transformers 5.12.0**, which is everything the conversion and
the reference forward need. The one gap: the checkpoint's `utils.py` imports `datasets` at module
scope for an eval helper the trace never calls, and that is not installed. Stubbed in
`scripts/pin_dflash_trace.py` rather than installed, so the shared parity venv keeps its pinned
dependency set — **the reference code itself runs unmodified**, which is the point of using it as
the oracle.

**Byte-exact accounting.** `model.safetensors` is **1,074,860,568 bytes** = 537,427,200 params ×2
(bf16) + a 6,160-byte header + its 8-byte prefix, **exactly**. 58 tensors, all BF16:

| tensor | shape | note |
|---|---|---|
| `fc.weight` | [2560, **12800**] | fuses the 5 captured target hidden states (5·2560) → 2560 |
| `hidden_norm.weight` | [2560] | RMSNorm on the fused context |
| `layers.{0..4}.self_attn.q_proj.weight` | [4096, 2560] | 32 heads × 128 |
| `layers.{0..4}.self_attn.{k,v}_proj.weight` | [1024, 2560] | GQA 32/8 |
| `layers.{0..4}.self_attn.o_proj.weight` | [2560, 4096] | |
| `layers.{0..4}.self_attn.{q,k}_norm.weight` | [128] | Qwen3 per-head RMSNorm |
| `layers.{0..4}.mlp.{gate,up}_proj.weight` | [9728, 2560] | SwiGLU |
| `layers.{0..4}.mlp.down_proj.weight` | [2560, 9728] | |
| `layers.{0..4}.{input,post_attention}_layernorm.weight` | [2560] | 2-norm qwen layout |
| `norm.weight` | [2560] | final |

**What is ABSENT is the headline.** No `embed_tokens`, no `lm_head`, no Markov head, no confidence
head. The Idea section's claim — "reuses the **target's own embedding and LM head**; the drafter
ships only the denoiser trunk" — is **confirmed by the tensor inventory**, not just quoted: the 58
tensors account for the file to the byte, so there is nowhere for a vocab-sized tensor to hide. At
537 M params this drafter is **4.4× smaller than DSpark-8b's 2.37 B**, and the entire difference is
that DSpark ships frozen copies of the target's embed + head (1.24 B, 52% of it) plus the
Markov/confidence heads.

Config: `block_size=16`, `num_hidden_layers=5`, `hidden 2560 / heads 32:8 / head_dim 128 /
intermediate 9728 / vocab 151936`, `rope_theta 1e6`, `rms_norm_eps 1e-6`,
`target_layer_ids=[1,9,17,25,33]` over `num_target_layers=36`, `mask_token_id=151669`.

**f32 fixture:** `scripts/convert_dflash_f32.py` → `~/models/qwen3-4b-dflash-f32` (2,149,714,968 B,
58 tensors), the 05 precedent (`.bin`/bf16 → f32 safetensors so the Go loader reads one format).

### CONFIRMED forward (`dflash.py` from the checkpoint, MIT, run unmodified)

1. **Context fusion, once per round:** `target_hidden = hidden_norm(fc(concat(h[1], h[9], h[17],
   h[25], h[33])))`. `extract_context_feature` indexes `hidden_states[layer_id + 1]`, so
   `hidden_states[0]` is the embedding output and the ids name **layer outputs** — the same +1
   convention DeepSpec uses, and the off-by-one a Go port loses silently.
2. **Block:** slot 0 = the last committed token, slots 1..15 = `MASK` (151669).
   `noise_embedding = target.model.embed_tokens(block_ids)` — the **target's** embedding.
3. **Trunk:** 5 layers. Each does Q from the block only; `k = cat([k_proj(fused_ctx),
   k_proj(block)])`, same for v — so the fused context is re-projected per layer (no separate ctx
   cache in this reference; a `DynamicCache` holds the concatenation and is `crop`ped back to
   `start` after each round).
4. **`is_causal = False`, hard-coded in the attention class** — the block is **bidirectional**, same
   as DSpark. Not truncatable for the same reason.
5. **RoPE quirk, load-bearing for parity:** `position_embeddings` are built over the whole
   `[ctx + block]` span; then **q takes `cos[..., -q_len:, :]`** (the block's own positions) while
   **k takes the full `cos`** (context + block). A port that ropes q at the block's local offset
   will be subtly wrong rather than obviously wrong.
6. **Head:** `target.lm_head(trunk_out[:, -15:, :])` → **15 drafted tokens** from a 16-wide block
   (slot 0 is a pure anchor — `logits_start = 1`, unlike DSpark's 0). Verify is 16 wide.
7. **One pass. No iterative denoising at inference** — despite "block diffusion", `spec_generate`
   calls the trunk exactly once per round. The Idea section's "one parallel pass" is literal.

### The fixture, and what it already says

`testdata/dflash_qwen3_4b_golden.json` (committed; weights are not) — `scripts/pin_dflash_trace.py`,
f32, greedy, two fixed prompts recorded verbatim. Per trace: prompt ids, the anchor, the 16 block
ids, per-tensor stats (`shape/mean/std/min/max/first8`) for `fused_context`, `block_in`,
`layer_out.{0..4}` and `trunk_out`, plus the 15 drafted ids, top-8 logits per drafted position, and
the first 64 raw logits at position 0. **The harness was cross-checked against the reference's own
`spec_generate` call path (cache + `is_causal=False`): drafted ids byte-identical**, so the dump is
the reference's behaviour and not an approximation of it.

**Two prompts, because one would have lied.** Accepted length of the dumped block, measured against
the target's own next-token predictions:

| trace | ctx | accepted | tok/verify |
|---|---|---|---|
| `raw` — `"The capital of France is"`, bare completion | 5 | **0 / 15** | 1.0 |
| `chat` — Qwen chat template, "write a Fibonacci function" | 23 | **10 / 15** | **11.0** |

A spike run only on the bare prompt would have concluded DFlash does not transfer. It is the
[quant-eval lesson](../../docs/parity-coverage-policy.md) in the acceptance domain: a raw completion
prompt on an instruction-tuned target is off-distribution, and here it costs *everything*, not a
little. Both traces are kept for exactly that reason.

**Treat the 11.0 as a smoke signal, not kill-gate 2.** One prompt, one round, their implementation,
f32 on CPU. But it is 3.7× the ≥3.0 bar and above DFlash's own published ~6.0 on code/math, on the
traffic class 06 says we care about — which is the first evidence in this program that the published
numbers might transfer. End-to-end smoke through `spec_generate` (64 tokens, chat prompt) produced
correct, coherent Python.

## Validation plan

- Correctness: lossless by construction (verifier owns `p`); output ≡ non-speculative,
  bit-exact greedy + sampled in-distribution, per backend — the standing 00-core bar.
- Acceptance: reproduce upstream acceptance on a matching model/workload before
  claiming transfer; publish α̂ tables per 06 with the same held-out discipline.
- Speed: 00-core harness suites on both boxes; every number lands in
  [experiments.md](./experiments.md) dated, with the losing runs kept.

## Performance levers and the DSpark pivot (2026-08-15, post gate 2/3)

DFlash cleared gates 1–2 (parity 1.0; acceptance 6.14 code / 7.32 math / 2.18 chat
tok-per-verify); gate 3 projects 1.29× code / 1.54× math / 0.46× chat **[at verify width 16;
measured optima are 1.74× / 2.25× / 0.92× — see the end of this document]**; gate 4 (router) is
now mandatory. **Decision (2026-08-15, Francis): explore the DSpark path — `license=None`
accepted for exploration.** The work originally pivoted to DFlash only for licensing; with
that constraint lifted for a spike, DSpark's structural advantages come back into play, and
they line up with the performance problem gate 3 exposed.

**Why DSpark is the better bet to explore:**

1. **DSpark ships the router mechanism** — its confidence head is a built-in acceptance
   predictor. The single worst number (chat 0.46×) is fixed by not firing when acceptance is
   low. DFlash needs an external router built for that; DSpark's per-draft confidence head
   IS that signal, for free — fire/don't-fire and adaptive block length both read off it.
   The chat-loss fix is in the checkpoint, not a separate build. **This is the strongest
   reason.**
2. **Smaller blocks waste less verify on low-acceptance traffic.** DSpark drafts 7-token
   blocks vs DFlash's 16; on chat (accept ~1–2), a 7-wide verify wastes far less than a
   16-wide one, so DSpark's chat floor is structurally higher before the confidence head
   even gates.
3. ~~**DSpark likely sidesteps DFlash's largest build.**~~ **CONFIRMED FALSE, 2026-08-15 —
   the build-cost gate this reason asked for has been run, from first-party source.**
   DSpark's block attention is **also bidirectional**: in the upstream DeepSpec tree (not
   vendored here), `eval/dspark/draft_ops.py` calls the backbone with `is_causal=False`, and
   the training mask
   (`common.create_dspark_attention_mask`) allows `q_block_id == kv_block_id` with **no
   `q ≥ k` constraint** — every block position attends every other. This is the same fact
   already recorded in this page's DSpark CONFIRMED forward, §4. **DSpark needs the identical
   non-causal cross-attention kernel, at M=7 instead of M=16.** It does not sidestep the
   trunk build; it re-shapes it.

   And on weights it runs the other way too. `dspark_qwen3_4b_block7` is **1.39 B** against
   DFlash's 537 M, decomposing exactly as:

   | component | bytes | note |
   |---|---|---|
   | trunk (5 layers + `fc`) | 537 M | **identical to DFlash's whole drafter** |
   | `embed_tokens` + `lm_head` | 778 M | frozen COPIES of the target's — duplicated, not new |
   | Markov head (`w1`+`w2`) | 77.8 M | plus a **7-step serial chain** of [151936, 256] matvecs per round |
   | confidence head | 4.4 K | |

   A loader that reuses the resident target's embed/head skips the 778 M, leaving **~615 M
   resident — only ~15% above DFlash.** So the weight cost is close to a wash, but the
   *build* is strictly larger: same novel kernel, plus two extra heads and a serial chain
   whose 7 dependent launches land inside the draft term gate 3 is most sensitive to.

   **Reasons 1 and 2 stand and are the strong ones.** The pivot may well still be right — it
   rests on the confidence head, not on a cheaper build.

**Levers that apply to either drafter:**

- **Routed economics is the honest baseline — free.** The 0.46× chat figure is
  indiscriminate firing, which gate 4 forbids. With a router — or DSpark's confidence head —
  chat floors at 1.0×, so the real economics are **1.29–1.54× on structured, 1.0× on chat**,
  a win on every class. Evaluate against this, never the un-routed loss.
- **Bigger target (mac leg) — measure, don't assume.** The draft/verify layer ratio barely
  moves with depth (5/36 → ~5/40); the real mechanism is verify amortization (currently
  3.86× at k=16). Whether it rises at scale is empirical — goinfer decode runs at 6–10% of
  peak bandwidth, so the direction is not guaranteed. Two cheap probes settle it: (a) verify
  amortization on a 12B/26B target (`cuda/spec_verify_ceiling_test.go` takes the
  `GOINFER_CUDA_MODEL` override), and (b) acceptance at scale. Do both before any mac trunk.
- **The 7.32 math anomaly is a caution, not a win.** Acceptance is measured against the int8
  target's *own* output distribution; int8 biases greedy toward higher-probability tokens
  (flattens the tail), making text more predictable — hence more draftable — on
  normally-high-entropy content. Math on 2 prompts / 44 rounds is where that shows.
  Testable: int8 vs bf16 output entropy on those prompts. The trustworthy cross-check
  remains code 6.14 vs 6.37 bf16 (within 4%). **Do not quote 7.32 as beating upstream.**

**License, with the user's decision.** Francis accepts `license=None` for exploration
(2026-08-15) — the `dspark_*_block7` drafters are in scope for the spike. Filing one upstream
issue is still worth it (DeepSpec is MIT and lists these as releases — likely an oversight),
but that is for eventual distribution/ship, not a blocker on exploring. Draft issue kept at
`docs/prompts/dspark-license-issue.md`. The apache-2.0 PARO family in the same org remains a
cleanly-licensed fallback.

**Sequencing — explore DSpark first.** (1) Reframe against routed economics (free);
(2) ~~confirm DSpark avoids the non-causal M=16 trunk~~ — **done, it does not** (above), so
re-price the build as *same kernel at M=7 + Markov chain + confidence head*; (3) run DSpark's
gates 1–2 (parity + acceptance) reusing the `HiddenCapture` seam and the DFlash harness, with
the confidence head gating chat; (4) measure verify-amortization + acceptance on a 12B/26B
target for the mac premise; (5) file the DSpark license issue + keep PARO as a fallback, in
parallel. Compare DSpark's confidence-gated blended economics against DFlash's 1.29–1.54×
projection before committing either trunk.

**Autoresearch tie-in:** the confidence-head threshold + block length are a bounded search
with a clean metric (blended tok/s) and the airtight lossless gate 1 — an E9 target
([task-autoresearch-loop](../task-autoresearch-loop.md)) once the DSpark path runs.

### DSpark RE-PRICED against the corrected build cost (2026-08-15)

Reason 3 of the pivot is dead (the kernel is needed either way), so the case had to be re-made
on economics rather than on build savings. **It re-prices substantially BETTER than DFlash**,
and for a reason neither the pivot nor this page had identified.

**The mechanism: block-7 verifies EIGHT positions, not sixteen — and that is already measured.**

| term | DFlash | DSpark | source |
|---|---|---|---|
| verify | k=16 → **46.385 ms** | k=8 → **26.383 ms** | <span>measured, both</span> |
| draft | 6.6 ms (M=16 trunk) | 4.3 ms (M=7 trunk 3.5 + Markov chain 0.8) | modelled |
| **round** | **53.0 ms** | **30.7 ms** | |

The 16-wide block pays **20 ms more verify** than the 8-wide one. Acceptance does not repay
that, because accepted length is *concave* in block width — the tail positions contribute
little. Fitting a per-token accept probability to DFlash's measured acceptance and carrying it
to a 7-token block:

| suite | DFlash α (measured) | implied p/token | DSpark α (projected) | DFlash × | **DSpark ×** |
|---|---|---|---|---|---|
| code | 6.14 | 0.849 | 4.83 | 1.29× | **1.75×** |
| math | 7.32 | 0.882 | 5.36 | 1.54× | **1.94×** |
| chat | 2.18 | 0.541 | 2.16 | 0.46× | **0.78×** |

**Code clears the 1.3× bar with room, where DFlash missed it.** **[SUPERSEDED — this compares
DSpark's 7-wide block against DFlash at 16. Measured DFlash at a 7-wide VERIFY is 1.74× vs
DSpark's projected 1.75×: the advantage was block width, not DSpark. See the end of this
document.]** And chat's un-routed floor rises
0.46× → 0.78× before the confidence head gates anything — reason 2 of the pivot, quantified: a
7-wide block wastes less verify when acceptance is low. Under gate 4's routing both floor at 1.0×,
so the routed picture is **1.75× / 1.94× / 1.0×**.

**Break-even, which is the honest way to read a projection with a transferred input:**

| | needs | of a ceiling of | fraction |
|---|---|---|---|
| DSpark → 1.3× | α ≥ **3.59** | 8.00 | 45% |
| DFlash → 1.3× | α ≥ 6.19 | 16.00 | 39% |

DSpark needs a *higher fraction* of its ceiling but a much *lower absolute* acceptance, and the
transferred estimate (4.83 on code) clears it by 35%.

**THE LOAD-BEARING ASSUMPTION, stated so it is not mistaken for a measurement: DSpark's
per-token acceptance is TRANSFERRED from DFlash, not measured.** Both are 5-layer trunks on the
same taps against the same target, which is why the transfer is defensible at all — but DSpark's
Markov head exists specifically to correct suffix decay, so its p could be *higher* than
DFlash's, and it is a differently-trained model, so it could be lower. **Measuring DSpark's
actual acceptance is the single highest-value next step** — it is gate 2 for DSpark, it reuses
the `HiddenCapture` seam and the DFlash harness, and it needs no kernel work.

**What this does to the funding question.** The DFlash trunk was worth ~1.5× on math alone. The
DSpark trunk — same kernel at M=7, plus two heads — projects to **1.75–1.94× on structured
traffic with chat routed to 1.0×**. The larger build buys a materially better return, so
"DSpark is more build" is true and no longer the deciding fact.

### Probe A — does a bigger target help? MEASURED, and the premise is wrong as stated

The mac leg rests on "the bigger target makes the draft relatively cheaper". The page already
hedged it as empirical. It is now measured, on three real targets (`TestSpecVerifyCeiling` with
`GOINFER_CUDA_MODEL`, int4 resident, depth 1024, 2070S):

| target | layers | M=1 | k=8 | k=16 | amort@8 | amort@16 |
|---|---|---|---|---|---|---|
| qwen2.5-1.5B | 28 | 5.39 ms | 11.07 | 17.95 | 3.90× | **4.81×** |
| Qwen3-4B | 36 | 11.12 ms | 26.38 | 46.39 | 3.37× | **3.84×** |
| qwen2.5-7B | 28 | 15.17 ms | 34.62 | 58.93 | 3.50× | **4.12×** |

**Verify amortization does NOT rise with target size — within one family it FALLS** (qwen2.5:
1.5B 4.81× → 7B 4.12×). Qwen3-4B is lowest of all at 3.84× while being the *deepest* of the
three, so architecture matters as much as scale here. Either way the hoped-for direction is
absent.

And the projected DSpark speedup is **flat across a 4.7× range of model size**:

| target | draft | round | **projected ×** |
|---|---|---|---|
| qwen2.5-1.5B | 2.76 ms | 13.83 ms | **1.88×** |
| Qwen3-4B | 4.43 ms | 30.81 ms | **1.74×** |
| qwen2.5-7B | 6.92 ms | 41.55 ms | **1.76×** |

**The mechanism, which is the transferable part: the draft's relative cost is set by `5/L` — the
target's DEPTH — not by its parameter count.** The drafter is five layers *of the target's own
shape*, so a WIDER target makes drafter and target more expensive in the same proportion and the
ratio does not move. 1.5B and 7B both have 28 layers, hence identical 5/28 shares despite 4.7×
the parameters.

**So the mac premise survives only in its depth form, and weakly.** Gemma-4-12B has **48 layers**
→ 5/48 = 0.104, the best ratio available here and a real improvement on Qwen3-4B's 0.139 — but
depth grows far more slowly than size, so the lever is worth a fraction of what "bigger target"
implied. **Do not fund the mac leg on the size argument; fund it on depth, or not at all.**

**Probe A's second finding, unplanned: Gemma-4-12B DECLINES residency on CUDA** (`BuildResident
declined`, with and without `GOINFER_GEMMA4_RESIDENT`). So the 48-layer point — the one that
would actually test the depth lever — cannot be measured on this box today. That is a second
`gemma4` gap alongside `ForwardCapture`'s rejection, and it blocks the mac premise from being
settled here at all.

**Probe B (acceptance at scale) NOT RUN — it needs a checkpoint pair we do not have.** DFlash is
paired to Qwen3-4B; measuring acceptance on a bigger target means `z-lab/Qwen3-8B-DFlash-b16`
(exists, per increment 1) plus its target — a download and a fresh harness run, not a re-run.
Recorded as not-done rather than folded into Probe A's result.

### DSpark gate 2 — MEASURED, and it beats its own projection on every suite

The re-pricing above rested on one transferred input: a per-token accept probability fitted to
DFlash and carried to a 7-token block. **Measured** — `scripts/ref_dspark_accept.py`, which
drives **DeepSpec's own** `generate_decoding_sample` with the DSpark evaluator's actual
`_init_context`/`_propose`/`_update` methods (a shim supplies only the attributes they read), on
`deepseek-ai/dspark_qwen3_4b_block7` + Qwen3-4B, bf16, greedy, non-thinking template, 160 new
tokens, the same suites as the DFlash harness:

| suite | projected α | **measured α** | ungated × | **gated ×** | DFlash × |
|---|---|---|---|---|---|
| code | 4.83 | **5.76** | 2.09× | **2.12×** | 1.29× |
| math | 5.36 | **5.73** | 2.09× | **2.15×** | 1.54× |
| chat | 2.16 | **3.04** | 1.11× | **1.29×** | 0.46× |

**The transfer was CONSERVATIVE on all three, and most on chat (2.16 → 3.04, +41%).** The likely
cause is the component DFlash does not have: the **rank-256 Markov head** exists precisely to
correct the suffix decay a parallel block suffers, so DSpark's per-token acceptance is higher
than DFlash's rather than equal to it. Carrying DFlash's `p` across was the conservative choice
and it under-sold DSpark.

**DSpark wins on EVERY traffic class, including chat.** That is the headline: no routing is
needed to *avoid a loss* — chat is 1.11× before any gating, against DFlash's 0.46×. Two
independent causes, both structural: DSpark's raw chat acceptance is 40% higher, and its 8-wide
verify costs 20 ms less than DFlash's 16-wide one.

**Reason 1 of the pivot is CORRECT but for the wrong reason, and the correction matters.** The
claim was that the confidence head fixes chat by *not firing when acceptance is low*. Gating at
0.5 does not raise acceptance — it slightly *lowers* it (3.04 → 2.96). What it does is cut the
**proposal width**, and therefore the **verify**:

    chat: proposal 6.96 -> 4.87, so verify k 7.96 -> 5.87 = 26.3 ms -> 21.2 ms
          acceptance barely moves (3.04 -> 2.96)
          net 1.11x -> 1.29x

So the confidence head is not an accept-rate router; it is an **adaptive block-length control that
buys back verify cost**. Same conclusion — it is the thing that makes chat work — but the
mechanism is 04's adaptive depth, learned, exactly as the Idea section originally said, and *not*
a fire/don't-fire gate. On code and math it barely engages (proposal 6.82 and 6.56 of 7), which is
the right behaviour and further evidence it is measuring confidence rather than thresholding
noise.

**Where this leaves the program.** Gate 2 is cleared for DSpark at **5.76 / 5.73 / 3.04**, and
gate 3 projects **2.12× / 2.15× / 1.29×** — over the 1.3× bar on all three, against DFlash's
1.29× / 1.54× / 0.46×. The remaining projected term is the same modelled 4.3 ms draft; everything
else is measured. **DSpark is the better drafter on this pairing by every measured axis, and the
larger build now buys a win on traffic DFlash loses outright.**

### DSpark gate 1 — CLEARED, and the trunk is genuinely SHARED

`decoder/dspark.go` + `TestDSpark_referenceParity`, against DeepSpec's own modeling code
(`scripts/pin_dspark_trace.py`, f32, greedy, both traces):

| | layer_out.0–4 | trunk_out | markov logits | drafted ids | confidence |
|---|---|---|---|---|---|
| raw (ctx 5) | 1.00000000 ×5 | 1.00000000 | 1.00000000 | **7/7 exact** | ±1e-6 |
| chat (ctx 23) | 1.00000000 ×5 | 1.00000000 | 1.00000000 | **7/7 exact** | ±1e-6 |

**The shared-trunk claim is now measured, not just read from source.** `dspark.go` reuses
`blockTrunk` — the same Go code that passes the DFlash fixture — and it reproduces DSpark's
per-layer outputs at cosine 1.0. Two independently-developed drafters, one trunk implementation.
That is a real reduction in what the resident port has to build: **one non-causal block trunk
serves both**, and only the heads differ.

**Gate falsifiable, checked on DSpark's own failure modes** (all three rejected): the Markov
chain not advancing (feeding the anchor at every step instead of the previously drafted token),
the Markov bias dropped entirely, and the confidence feature missing its Markov half.

**What the Go side now has**: loader (64 tensors incl. own embed/lm_head, rank-256 Markov,
confidence head), `EmbedBlock`, `BaseLogits`, sequential `SampleBlock`, and `Confidence`. Two
loader details worth noting — DSpark's config carries RoPE only as nested `rope_parameters`
(the mirror of granite's flat-only released config, so the loader accepts both spellings), and
it refuses `markov_head_type != "vanilla"` rather than silently running the vanilla chain
against gated weights.

**Remaining for DSpark**: the end-to-end logit half (drive it from goinfer's own target through
`ForwardCapture`, as `TestDFlash_targetEndToEnd` does) and then the resident trunk — which is now
one kernel serving both drafters rather than two.

### Which models can actually benefit — the binding constraint is OUR seam, not the checkpoints

Gate 1's second half is cleared for DSpark too (`TestDSpark_targetEndToEnd`: **7/7 drafted ids
exact on both traces**, driven from goinfer's own target through `ForwardCapture`). So the
question becomes which pairings this can be pointed at. The published catalogue is **38 drafter
repos** across deepseek-ai and z-lab. Crossed against the 23 families goinfer runs, and against
`Model.ForwardCapture`'s architecture rejection in `decoder/model.go` — it refuses any arch
with its own `runLayers`, and the identical guard appears twice (ForwardCapture and its
sub-capture sibling), which is why this cites the function rather than a line:

| model goinfer runs | drafter published | CPU seam | status |
|---|---|---|---|
| **Qwen3-4B** | DSpark + DFlash + z-lab | ✅ | **measured: 2.12× / 2.15× / 1.29×** |
| **Qwen3-8B / 14B** | DSpark + DFlash + z-lab (8B) | ✅ | same arch, same path — checkpoint swap |
| **Llama-3.1-8B-Instruct** | `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat` | ✅ | unexercised, should work |
| Qwen3.6-35B-A3B | `z-lab/Qwen3.6-35B-A3B-DFlash` | ❌ `qwen35` | **blocked by our seam** — model is on this box |
| gpt-oss-20b | `z-lab/gpt-oss-20b-DFlash` | ❌ `gptoss` | **blocked by our seam** — model is on this box |
| Gemma-4-26B-A4B | `z-lab/gemma-4-26B-A4B-it-DFlash` (apache-2.0) | ❌ `gemma4` | **blocked by our seam** — model is on this box |
| Gemma-4-31B / 12B | z-lab (31B apache-2.0; 12B unlicensed) | ❌ `gemma4` | blocked; 12B also declines CUDA residency |
| Kimi-K2.5 / K2.6 | z-lab | ❌ `mla` | blocked; also far past this box |
| GLM-5.1 | z-lab | ✅ `glm4_moe` | goinfer runs GLM-4.5/4.6, not 5.1 — no matching pair |

**The finding: availability is not the constraint any more — our own seam is.** Three models
already sitting on this box (Qwen3.6-35B-A3B, gpt-oss-20b, Gemma-4-26B-A4B) have published
drafters and are refused by `ForwardCapture` purely because their families carry their own
`runLayers`. That single arch check gates more of this program's addressable surface than the
licensing question ever did.

**And the resident seam does NOT share the limitation.** `cudaResident.SetHiddenCapture` captures
`r.x` after any tapped layer in the resident runner's own loop, with no architecture predicate —
so a resident target can feed a drafter for *any* family the resident runner already supports.
Extending `ForwardCapture` to the own-`runLayers` families is therefore worth doing on its own
merits, but the GPU path — which is where this program is going anyway — may not need it.

**Size is not the axis** (Probe A): the benefit is roughly flat from 1.5B to 7B, because the
drafter is five layers of the target's own shape. What moves it is DEPTH (`5/L`) and acceptance,
not parameter count. So the interesting targets are the deep ones and the ones whose traffic is
structured, not simply the big ones.

### Can Qwen3.6-35B-A3B be supported? YES — one blocker, one caveat, one correction

Asked directly, and answered by loading it rather than by reasoning about it.

**The drafter is the easy part, and it already works.** `z-lab/Qwen3.6-35B-A3B-DFlash` is
**apache-2.0, documented, 0.77 GB**. It now loads and its trunk runs through the SAME shared
`blockTrunk` (`TestDFlash_secondPairingLoads`): block 16, **8 taps** `[1,6,11,16,22,27,32,37]`
of 40 target layers, **6** trunk layers, hidden 2048, heads 32/8. The 4B gate re-passes
bit-identically, so nothing regressed.

Getting there needed two loader fixes, and they are the third instance of a class this program
keeps meeting: **z-lab ships two config dialects across its own checkpoints** — `block_size`
nested inside `dflash_config` (top-level on the 4B) and RoPE as `rope_parameters` (flat
`rope_theta` on the 4B). The 4B-only loader reported this supported pairing as
*"block_size must be >= 2, got 0"*. After granite's flat-only rope and nemotron's
`hybrid_override_pattern`, that is three publishers whose *second* checkpoint used a different
spelling than their first.

**The blocker is ours: `ForwardCapture` refuses `qwen35`.** But `runLayersQwen35` has a clean
per-layer loop with `h` as the residual, so the capture hook is the same few lines the generic
path already carries. Cost is not the code — it is that `decoder/model.go` (core) and
`decoder/forward_qwen35.go` (`qwen3_5_moe`'s own) are both in the parity manifest, so it needs a
goldens-gated refresh. That is the sanctioned path, and the change is additive and inert for
every other family.

**The caveat is the box, not the method.** 35B-A3B at int4 is a 26.5 GB `.giw` against 8 GB of
VRAM, so a resident target here means the MoE expert-streaming path (~5 tok/s measured on the
26B), where gate 3's economics are entirely different and probably unfavourable. **Gates 1 and 2
are runnable here** — they are numerics — **gate 3 is not.** This is a correctness-and-acceptance
target on this box and a wall-clock target only on a bigger one.

**And the correction, which matters for every projection above.** This page said the drafter is
*"exactly five layers of the target's own layer shape"*. **That is true of the Qwen3-4B pairing
and does NOT generalize.** The 35B drafter is 6 layers at hidden 2048 with 32×128 heads, while
its target has 16×256 heads and an MoE FFN — same hidden width, different geometry, different
depth. So the draft-cost model that gave gate 3's 4.3–6.6 ms **is per-pairing and must be
re-derived** for any target other than Qwen3-4B. The identity was a fact about one pair that read
like a fact about the method.

### The other two blocked seams, checked — and gemma4 already HAS one

Asked whether `gptoss` and `gemma4` should be checked too. Checked, and it changed the answer for
one of them.

**gemma4 already has the seam, under another name.** `decoder/forward_gemma4.go` carries
**`g4traceHidden(l, h)`** — called after *every* layer, plus `-1` for post-embedding — which is
exactly the per-layer residual a block drafter taps. It is a debug package-global (`var
g4traceHidden func(layer int, h []float32)`), not the per-stream `cache.captureLayers`, so the
work is **wiring an existing call site to the production mechanism, not building capture**.

That is the *second* time in this program that a capability I recorded as missing turned out to
exist under a different name — after `layerCap` in the CUDA resident runner. Both times the
search was for the CPU seam's identifiers. **The lesson has now repeated often enough to state as
a rule: before recording a capability as absent, search for the BEHAVIOUR (a per-layer residual
callback), not the NAME.**

**gptoss has no hook but a clean loop.** `runLayersGptOss` is a single `for l` with
`h[i] += attnOut[i]` / `h[i] += moeOut[i]`, so the hook is the same few lines the generic path
carries — no structural obstacle.

**Both drafters are cleanly licensed and DIMENSIONALLY MATCH our local targets** (checked rather
than assumed, after the 35B taught that the pairing geometry does not generalize):

| drafter | licence | size | trunk | taps | block | target dims | our local target |
|---|---|---|---|---|---|---|---|
| `z-lab/gpt-oss-20b-DFlash` | **MIT** | 1.57 GB | 8 layers, h 2880 | 5 of 24 | 8 | vocab 201088 | **exact match** (h 2880, 24 layers, vocab 201088) |
| `z-lab/gemma-4-26B-A4B-it-DFlash` | **apache-2.0** | 0.86 GB | 5 layers, h 2816 | 6 of 30 | 16 | vocab 262144 | **exact match** (h 2816, 30 layers, vocab 262144) |

Both spell `block_size` at top level, so the 35B's nested spelling was the outlier and the
both-spellings loader covers all four pairings.

**So the addressable set on this box is three targets, all present, all cleanly licensed, all
blocked by one arch check**: Qwen3.6-35B-A3B, gpt-oss-20b, Gemma-4-26B-A4B. Two need a few lines
in their forward loop; gemma4 needs a rewire of a hook it already has. None needs a new drafter,
a licence resolution, or a download.

## Increment: the capture seam, wired for the three addressable families

`ForwardCapture`'s arch guard was the single thing standing between the measured 4B result and
every other target on this box. It is now shrunk, and the three families whose targets we hold
locally with a licensed drafter — `qwen3_5_moe`, `gemma4`, `gpt-oss` — capture through one
shared helper (`decoder/capture.go`) rather than four copies of the same seven lines.

**The gate is about placement, not shapes.** The failure mode here is quiet: a capture taken
after ATTENTION rather than after the whole layer has the right shape, the right dtype, and
plausible magnitudes. Every shape assertion passes, and the only symptom is that the drafter
accepts less — which reads as *"block drafting is weak for this family"* rather than as a bug,
and would have been written into this document as a finding about the family.

So `TestCaptureSeam_ownRunLayers` asserts the one identity that pins placement without needing a
reference dump: **`runLayersX` returns the residual after the last layer, so capturing
`NumLayers-1` must reproduce it bitwise.** Plus two properties that are cheap and are real bugs
otherwise — the seam must not perturb the forward (it is documented read-only), and distinct
layers must yield distinct tensors (the natural bug is a shared backing array).

**Mutation-tested, because a gate that has never failed is not a gate:**

| mutant | result |
|---|---|
| capture moved after the attention add | **FAILS** — "the copy is NOT taken after the full layer" |
| rows alias `h` instead of copying | **FAILS** — "the rows alias" |

`granite` and `nemotron_h` stay rejected **deliberately**, not by oversight: they interleave
recurrent mixers, so "the residual after layer l" is a decision, not an assumption. `mla` and
`llama4_text` are simply not done. `ForwardSubCapture`'s guard is unchanged — sub-layer capture
really is still unwired there.

Manifest cost, paid once rather than three times: this is the sanctioned goldens-gated exception
(a guarded diagnostic seam in a `core` file), refreshed at **42 goldens passed / 0 skipped / 0
failed**, `validated_at` preserved.

### The hardcoded think-ids — a latent bug the second pairing exposed

The acceptance harness carried `qwen3NoThinkSuffix = []int{151667, 271, 151668, 271}`, pinned
from Qwen3-4B. **Qwen3.6-35B-A3B has a 248320-token vocab in which `<think>` is 248068 and
151667 is an unrelated token.** The literal would have fed the 35B four wrong tokens.

That failure does not error. It depresses acceptance, and it looks exactly like *"the drafter
does not transfer to this target"* — and given the thinking-vs-non-thinking gap already measured
on the 4B (0/15 vs 10/15 accepted), it is precisely the kind of number this document would have
attributed to the drafter.

Now resolved through the target's own tokenizer and **verified rather than trusted**: the encode
must agree with the ids the tokenizer itself reports, and be 4 long. Checked on both —

| target | resolved suffix |
|---|---|
| Qwen3-4B | `[151667 271 151668 271]` — **byte-identical to the old literal**, so no recorded number moves |
| Qwen3.6-35B-A3B | `[248068 271 248069 271]` |

The harness also now checks the pairing at load (target hidden == drafter hidden; every tap in
range of the target's layer count), so a mismatched pair fails immediately instead of producing
a number.

## Probe B ANSWERED: acceptance transfers to a second pairing — measured, not projected

Probe B ("does acceptance hold at scale, or is 6.14 a Qwen3-4B artifact?") was open because it
needed a second target we could actually run. Wiring the capture seam supplied one:
**Qwen3.6-35B-A3B**, drafter and target both local, both cleanly licensed.

**At the length gate 2 is recorded at, the second pairing is slightly BETTER:**

| suite `code`, int8 target, greedy | Qwen3-4B | Qwen3.6-35B-A3B |
|---|---|---|
| **tok/verify @ maxNew=160** | **6.14** | **6.78** |
| steady state (`1 + mean accepted`) | 6.10 | 6.71 |
| rounds | 80 (26.7/prompt) | 49 (16.3/prompt) |
| tokens generated | 491 | 332 |

Both clear the ≥3.0 bar decisively. **Acceptance is not a 4B artifact.** The 35B has a 6-layer
trunk against 40 layers (ratio 0.15) versus the 4B's 5-against-36 (0.14) — near-identical depth
ratios, and probe A already established depth ratio as the mechanism, so this is the outcome
that theory predicts and it is now measured rather than projected.

**The 4B re-measured to 6.14 exactly** — the recorded value, to the digit, under a harness that
had since gained a tokenizer-resolved template suffix, a maxNew guard and a second metric. That
is the control saying those changes are numerically inert, and it is the reason the 35B number
next to it can be trusted.

### The length series, and why one matched length proves nothing

| maxNew | Qwen3-4B | Qwen3.6-35B-A3B |
|---|---|---|
| 16 | 7.11 | 4.77 |
| 48 | 6.75 | **8.15** |
| 160 | 6.14 | 6.78 |

**The 4B declines monotonically. The 35B is non-monotonic and peaks in the middle.** So the two
pairings rank *35B worse* at 16, *35B much better* at 48, and *35B slightly better* at 160. Any
single length would have supported a confident and differently-wrong sentence, and two of the
three would have been misleading about the magnitude.

The cause is content position, not noise: the 35B's predictable region is the code block, which
its answers reach after a prose preamble. A 16-token run stops before it; a 48-token run sits
inside it; a 160-token run runs past it into the prose that follows. This retires the "easy
prefix" mechanism recorded earlier in this document — the easy part is not reliably at the
front, and **the sign of the truncation bias is model-dependent**.

**A residual confound, stated rather than buried:** even at maxNew=160 the two runs do not cover
the same amount of text. The 35B hit EOS earlier (332 tokens over 3 prompts, 111/prompt) than
the 4B (491, 164/prompt), so a matched *cap* is still not a matched *workload*. The 35B's
shorter answers are plausibly weighted toward its code-block region. The 1.10× edge should be
read as "the second pairing is at least as good", not as a measured superiority.

### The error record for this probe

Two conclusions were drawn and retracted before this one, both worth keeping:

1. **Wrong baseline.** The 35B's first number (4.77 at maxNew=16) was compared against **2.90** —
   a value this document had already retired two sections earlier as the thinking-mode
   measurement. The correct baseline was 6.14. The inflation figure derived from it (2.45×) was
   wrong by ~4×.
2. **A matched control that was too narrow.** Re-measuring the 4B at maxNew=16 gave 7.11 and
   licensed "at matched settings the 35B accepts less". That was true at 16 and false at 48.
   A matched setting is necessary for a comparison and demonstrably not sufficient.

Both errors were in the *comparison*, not the measurement — gate 1 green, harness
mutation-tested, every individual number reproducible. That is now the fourth and fifth
instance in P10 of the same class: **for a speculative drafter, the apparatus around the number
is where the mistakes live.**

### gpt-oss is blocked on a missing chat template, not on the seam

The capture seam works for `gpt-oss` and its drafter loads and runs the shared trunk. It still
cannot be acceptance-measured, for an unrelated reason worth recording so it is not rediscovered:
**goinfer has no harmony template.**

- The HF checkpoint ships **no** `chat_template` in `tokenizer_config.json` (length 0).
- The MXFP4 GGUF ships a **16.7 KB** harmony Jinja template.
- The harmony markers `<|start|>`, `<|channel|>`, `<|message|>`, `<|end|>`, `<|return|>` are all
  present in the vocab, and none of them is a marker `chat.Detect` tests.

So `Detect` returns `ErrUnknownTemplate` by both routes — template-string fingerprint and
bare-checkpoint token heuristic — and the harness's hard error fires. **This is the correct
outcome**: the measurement stops instead of producing a number.

**Checked rather than assumed, because the near-miss is real:** `Detect`'s first branch matches
`<|channel>`, and harmony's marker is `<|channel|>`. Those differ by one character in a position
that makes `strings.Contains` return false — `<|channel>` needs `l` followed by `>`, harmony has
`l` followed by `|`. Verified against the actual 16.7 KB template: **no branch matches**, so
gpt-oss is not silently rendered as Gemma-4. Had it matched, the harness would have measured a
gpt-oss target through Gemma-4's turn markers and printed an acceptance figure for it.

Unblocking gpt-oss means writing a `Harmony()` template in `chat/templates.go` — contained, but
a genuinely new chat format with channel semantics (analysis vs final), and a wrong one corrupts
the measurement in exactly the silent way the hard error exists to prevent. **Recorded as a
decision, not done here:** it is chat-template work, not block-drafting work, and P10 has no
claim that depends on it.

### An instrument that hid the thing it was watching

Recorded because it cost a run and because the class is already known here.

The Gemma-4 acceptance run showed **zero output for ten minutes** at 670% CPU with RSS flat at
4.2 GB — far under what a 26B target should need. I checked `/proc/PID/maps`, saw the drafter's
safetensors mapped and **no GGUF**, and concluded the target load was stuck. I killed it with
SIGQUIT to get a stack dump.

Both inferences were wrong, and the run was healthy:

- **The GGUF was absent from `maps` because loading had FINISHED.** `loadGGUFWeights` closes the
  mapping once the weights are built — its own comment says so ("the mapping is unneeded once
  the build returns"). I read the absence as "not yet mapped" when it meant "already done".
- **The output was not missing; my pipeline was withholding it.** The run was piped through
  `grep -E` without `--line-buffered`, so ~4 KB of output sat in grep's buffer. The pairing line
  had in fact been emitted, and appeared the moment the process died. The program was writing
  the whole time.

So the diagnostic apparatus suppressed the signal, and I read the silence as a fault in the
subject. **This is the same class as the concurrency-instrument lesson already in this
repository** — the instrument you reach for to verify a property destroys or hides the property
— and it is worth noting that the tooling documentation warns about this exact flag. Knowing the
class was not enough; the check that would have caught it is *"if this process were healthy,
would my pipeline show me anything?"*, which is the same question the monitor guidance asks
about failure signatures, pointed the other way.

Practical rule for this harness: **any pipe used to watch a long run must be line-buffered**, and
a silent stream is evidence about the pipe before it is evidence about the program.

### The Gemma-4 pairing: q4_0 destroyed it, and the drafter was never the problem

The Gemma-4 pairing first measured **0.00 mean accepted** over 477 rounds (1.01 tok/verify).
That was recorded as *our* bug rather than as a result, per the asymmetry pre-registered for a
pairing with no gate-1 reference dump. It was the right call: the cause was **the target's
quantization**, and it is now isolated by a controlled swap.

Same drafter, same prompt, same code, only the target's weights differ:

| | target = q4_0 GGUF | target = bf16 safetensors |
|---|---|---|
| target's own anchor token | `"0"` — nonsense for the prompt | `"There"` — sensible |
| distinct drafted ids (of 15) | **1** | **8** |
| drafted text | `","` repeated | `" are" " several" " ways" " to" " approach"` |
| acceptance | **0.00** | measured below |

The **target itself** was broken at q4_0 — its own greedy continuation of *"Write a Python
function that returns the nth Fibonacci number"* was `"0"`. The drafter was being handed hidden
states from a degraded forward and had no chance. Nothing was wrong with the capture seam, the
chat template, the embedding convention or the drafter.

**Why this was not predicted, and what it corrects.** This document already records that "int8
costs only ~4% of acceptance, so a quantized resident target does NOT crater this drafter". That
finding stands *for int8 derived from bf16*, which is what it measured. It does **not** extend to
q4_0 — the crudest 4-bit format, no k-quant grouping — and the failure there is not a 4%
degradation but a total collapse. **Block drafters read the target's hidden manifold, not just
its argmax, so they are more sensitive to target quantization than the target's own output
quality suggests**: a target that still produces text can already be too degraded to draft
against.

Both Qwen3 pairings used q8_0/f32-derived targets, which is why neither hit this.

**Method note.** The controlled swap is what made this cheap. Four suspects had been cleared by
inspection (prompt id-exactness, embedding scale, LM-head handling, seam placement) and the
remaining evidence — degenerate output with *sane-magnitude* inputs — did not point anywhere on
its own. Changing exactly one variable and re-running a 30-second diagnostic settled it, where
more inspection would not have.

### Gemma-4 on a sound target: **4.80 tok/verify** — three families, all over the bar

With the q4_0 confound removed (target = the bf16 safetensors, quantized to int8 the same way
both Qwen3 pairings were), the Gemma-4 pairing measures **4.80 tok/verify** — mean accepted
3.77/15 over 104 rounds, 499 tokens. It clears the ≥3.0 bar.

That is the *informative* direction for this pairing. It has no gate-1 reference dump, so a low
number would have been ambiguous between "the drafter does not transfer" and "our path is
wrong"; a number this far above the bar can only be produced by a working path.

| pairing | trunk / target layers | depth ratio | **tok/verify @160** |
|---|---|---|---|
| Qwen3.6-35B-A3B | 6 / 40 | 0.150 | **6.78** |
| Qwen3-4B | 5 / 36 | 0.139 | **6.14** |
| Gemma-4-26B-A4B | 5 / 30 | 0.167 | **4.80** |

**Three model families, three architectures, one shared trunk, every one over the bar.** Probe B
asked whether acceptance was a Qwen3-4B artifact. It is not, and the answer no longer rests on a
single second pairing that happened to share the 4B's publisher and tokenizer family.

Gemma is the weakest of the three and also has the **highest** depth ratio, i.e. the most
expensive drafter relative to its target — so it is the worst pairing on both axes at once.
Whether those two facts are connected is **not** established here: probe A tied depth ratio to
drafter *cost*, and nothing yet ties it to *acceptance*. Three points across three architectures
cannot separate that from ordinary per-family variation, and the gpt-oss pairing — the extreme
at 8/24 = 0.333 — is the one that would test it, which is a further reason its harmony template
is worth having.

### Does 4-bit crater block drafting? **goinfer's int4 does not** — increment 4's economics hold

The q4_0 collapse raised a question with consequences past P10: the resident CUDA path is
**W4A8**, and increment 4's whole economic case is a resident *quantized* target. One 4-bit data
point showing total collapse is not something to design around, nor to dismiss.

Run on the **Qwen3-4B pairing** specifically — the only one with a gate-1 reference dump behind
it, so the result is attributable to quantization rather than ambiguous the way Gemma's first
number was. One variable changed:

| target quantization | tok/verify @160 | mean accepted | rounds / tokens |
|---|---|---|---|
| **q4_0** (llama.cpp legacy 4-bit) | **1.01** — collapse | 0.00 | 477 / 480 |
| **goinfer int4** (W4A8) | **6.45** | 5.41 | 76 / 490 |
| **goinfer int8** (from bf16) | **6.14** | 5.10 | 80 / 491 |

**Two 4-bit formats, opposite outcomes.** goinfer's int4 preserves acceptance completely;
llama.cpp's q4_0 destroys it. That is the measured justification for having refused to
generalize from the q4_0 result by bit-width — these are different quantizers that happen to
share a width, and goinfer's already measured 92.5% agreement against f32 on nemotron where
q4_0 evidently would not.

**Do not read 6.45 > 6.14 as "int4 is better."** Different quantization yields a different token
stream (76 rounds / 490 tokens vs 80 / 491), so the two runs cover different content. The
supportable claim is **int4 is not worse**, not that it improves anything.

`int4mix` was not measurable on this pairing: it is **GGUF-only** and the gate-1 target is
safetensors. Recorded rather than worked around, since substituting a different target to
exercise the flag would have given up the one property that made this measurement
interpretable.

**Consequence:** the resident W4A8 target is cleared for block drafting, and increment 4
proceeds on the economics it was designed with. The scope correction stands unchanged —
*quantizer quality*, not bit-width, is what a block drafter is sensitive to.

## The fourth pairing (gpt-oss): 2.30 — and the comparison is INVALID, not merely confounded

With the harmony template added, gpt-oss became measurable. It reads **2.30 tok/verify** (mean
accepted 1.28/7, 210 rounds) — the only pairing to miss the ≥3.0 bar. **That number must not be
placed beside the other three**, and the reason is worth more than the number.

**The run labelled `code` did not measure code.** Harmony declares "Channel must be included for
every message" and has no non-thinking form — the diagnostic confirms the target's very first
token is `<|channel|>`, entering **analysis**. So the target spent the run emitting *reasoning
prose*, while the other three pairings were measured on the code they were asked for.

The drafter is **not** off-distribution: it drafts `":"`, `"We"`, `" need"` — reasoning-style
tokens, 4 of 7 distinct, healthy monotonic tap profile (rms 4.26 → 331.93). z-lab evidently
trained it on analysis-channel output. The path works; the *workload* differs.

And this document already records what content type alone is worth on a **fixed** pairing:
Qwen3-4B scores **6.14 on code and 2.18 on chat**. gpt-oss's 2.30 on reasoning prose sits
essentially on top of the 4B's 2.18 on prose. **Content type is a sufficient explanation for the
entire gap**, with no need to invoke anything about the pairing.

### The depth-ratio hypothesis remains UNTESTED

Normalizing to per-token acceptance α (which removes the block-width difference — gpt-oss uses
block 8, so its `tok/verify` is capped at 8 where the others cap at 16):

| pairing | block | mean accepted | tok/verify | **α** | depth ratio |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | 15 | 5.71 | 6.78 | **0.866** | 0.150 |
| Qwen3-4B | 15 | 5.10 | 6.14 | **0.848** | 0.139 |
| Gemma-4-26B-A4B | 15 | 3.77 | 4.80 | **0.796** | 0.167 |
| gpt-oss-20b | 7 | 1.28 | 2.30 | **0.566** | 0.333 |

α does decline as depth ratio rises, and gpt-oss is the extreme on both. **This is not evidence
for the hypothesis.** gpt-oss is simultaneously the highest depth ratio AND the only pairing
measured on prose, and the 4B's own code-vs-chat spread (0.848 vs roughly 0.57 implied by 2.18)
shows content type alone spans the entire observed α range. The two explanations are not merely
correlated here — they are indistinguishable from these four points, and the cheaper one already
suffices.

The 4B and 35B also rank *against* the hypothesis (0.139 → α 0.848, 0.150 → α 0.866), though
they differ by ~0.02 in both and should be read as a tie.

**To actually test it** would need gpt-oss measured on the same content type as the others,
which harmony's mandatory channel makes awkward, or a pairing with a high depth ratio whose
target does not reason by default. Neither is available today. Recorded as open rather than
resolved in the direction the numbers superficially suggest.

### Harness: print what the target PRODUCED, and a variance caveat it immediately exposed

The suite names (`code`/`math`/`chat`) describe the **prompt**. Acceptance is a property of the
**output**, and on a reasoning-by-default model the two diverge silently — gpt-oss's `code` run
measured the analysis channel and produced a number that looked comparable to three code numbers
and was not. That was caught only by inspecting an anchor token in a separate diagnostic, which
is not a process that scales to the next family.

Every run now prints a preview of what the target actually emitted, next to the per-prompt
progress. The 4B re-measures to **6.75 at maxNew=48, unchanged**, so the instrumentation is inert.

It exposed something the aggregate was hiding on the very first run — the **per-prompt spread
within a single suite**:

| prompt | target produced | mean accepted |
|---|---|---|
| Python / Fibonacci | `Sure! Here's a Python function … ```python` | **9.40** |
| Go / reverse a slice | `Here's a Go function … ```go func ReverseIntSlice` | **2.62** |
| SQL / top 5 customers | `To select the top 5 customers … ```sql SELECT` | **9.00** |

**3.6× between prompts in the same suite**, from three prompts total. Every recorded tok/verify
in this document is a mean over that spread, so cross-pairing differences smaller than roughly a
factor of two are not resolvable by this suite — a caveat that applies to the 6.78-vs-6.14
comparison (35B vs 4B), which should be read as *"the second pairing is at least as good"* and
not as a measured ordering. It does **not** touch the conclusions that rest on large gaps: the
q4_0 collapse (1.01 vs 6.45) and the int4 clearance are both far outside this range.

Worth noting the Go prompt is the low one, and its answer opens with a *named function*
(`ReverseIntSlice`) rather than boilerplate — consistent with acceptance tracking how
predictable the specific continuation is, rather than the language or the suite.

## MEASURED: verify width 7 projects **1.74×** on code, against 1.28× at 16

Gate 3 projected **1.29× on code** — under its own 1.3× bar — and the recorded conclusion was
"pass on math only". That projection verified all 16 of DFlash's drafted positions. It did not
have to.

**The drafter still drafts its full trained block; only the number of positions the TARGET
verifies is capped** (`GOINFER_DFLASH_VERIFY_WIDTH`). Drafting a narrower block instead would
change the non-causal attention pattern over `[ctx‖block]` and take the drafter off-distribution
— a different experiment. This one holds the draft fixed and varies only the verify.

Measured on the gate-1-backed Qwen3-4B pairing, code suite, int8, maxNew=160:

| verify width | mean accepted | tok/verify | geometric fit | excess | **projected speedup** |
|---|---|---|---|---|---|
| **16** (current) | 5.10 | 6.14 | 5.10 | — | **1.28×** ← reproduces the recorded 1.29× |
| 12 | 5.01 | 6.05 | 4.66 | +7.5% | 1.53× |
| 10 | 4.64 | 5.67 | 4.31 | +7.7% | 1.61× |
| 8 | 4.24 | 5.27 | 3.82 | +11.1% | 1.71× |
| **7** | **3.97** | **5.00** | 3.50 | +13.4% | **1.74×** ← optimum |
| 6 | 3.41 | 4.44 | 3.13 | +9.0% | 1.66× |
| 4 | 2.45 | 3.47 | 2.18 | +12.6% | 1.55× |

**k=16 reproduces 6.14 exactly** — 80 rounds, 491 tokens, identical to the uncapped run — so the
cap is a genuine no-op at full width and the narrower rows are interpretable.

### Why: the last four positions are nearly free acceptance at full price

Marginal accepted tokens per position added, against 2.35 ms of verify per position:

| positions added | accepted gained | per slot |
|---|---|---|
| 4→5 | +0.96 | 0.480 |
| 6 | +0.56 | 0.560 |
| 7 | +0.27 | 0.270 |
| 8–9 | +0.40 | 0.200 |
| 10–11 | +0.37 | 0.185 |
| **12–15** | **+0.09** | **0.022** |

**Positions 12–15 land 0.09 tokens between them and cost 9.4 ms of verify every round.** The
block is paying full batched-verify price for four slots that hit 2% of the time.

### The geometric model was wrong in the favourable direction

Fitting a single per-token α to the full block gives 0.8477, and that α **understates measured
acceptance at every narrower width, by 7.5–13.4%**. Acceptance is not position-independent: it is
**front-loaded**, which the marginal table shows directly. Mechanically that is what to expect —
the drafter predicts position 1 from the anchor's true hidden state and position 15 through
fifteen intervening masks.

So the earlier block-width projection built on that fitted α (optimum 1.58× at k=7) was a
**floor**, not a target. Measured is 1.74×.

### What this changes

- **Gate 3 goes from "pass on math only" to passing on code as well.** 1.74× against a ≥1.3× bar,
  where the recorded figure was 1.29× and missing.
- **DSpark's headline advantage evaporates.** This document projects DSpark at **1.75× on code**
  versus DFlash's 1.29×, and treats that as a reason the pivot may be right. Measured
  DFlash-at-7 is **1.74×** — the same number. **The advantage was BLOCK WIDTH, not DSpark**, and
  DFlash gets it from a one-line cap while being the licensed one. Reason 2 of the pivot
  ("a 7-wide block wastes less verify when acceptance is low") is confirmed as a real effect and
  refuted as a DSpark-specific one.

### Standing caveats

This is still a **projection**: acceptance is measured, wall-clock is not. It composes measured
acceptance with the measured verify curve (`T(M) = 8.77 + 2.35M` ms) and measured draft (6.6 ms)
against an 11.12 ms decode — all pinned on the 4B on an RTX 2070 SUPER. A different target or GPU
moves the curve, and gate 3 remains unmeasured. **Gate 3 should now be run at k=7, not k=16.**

Unlike the cross-pairing comparisons, this sweep is **paired** — same prompts, same target, same
drafter, only the width varies — so the 3.6×-per-prompt suite variance that makes 6.78-vs-6.14
unresolvable does not undermine it. Seven widths, monotonic on both sides of a clean peak.

### All three suites swept: the optimum width TRACKS acceptance

| suite | k=16 (current) | k=12 | k=10 | k=8 | k=7 | k=6 | k=4 | **optimum** |
|---|---|---|---|---|---|---|---|---|
| **math** | 1.53× | — | — | **2.25×** | — | 1.80× | 1.70× | **k=8 → 2.25×** |
| **code** | 1.28× | 1.53× | 1.61× | 1.71× | **1.74×** | 1.66× | 1.55× | **k=7 → 1.74×** |
| chat | 0.45× | — | — | 0.70× | — | 0.80× | **0.92×** | **k=4 → 0.92×** |

Each suite's k=16 row reproduces its recorded figure (1.54 / 1.29 / 0.46), so the sweep is
anchored to the numbers already in this document.

**The optimum moves with acceptance: math (α highest) peaks widest at 8, code at 7, chat (α
lowest) narrowest at 4.** That is the concavity argument working in both directions — tail
positions are worth keeping exactly when they land often enough to pay their 2.35 ms.

**Gate 3, re-projected at each suite's own optimum: code 1.74× and math 2.25×, both clearing the
≥1.3× bar decisively** where the recorded figures were 1.29× (missing) and 1.54×. The routed
headline moves from *"~1.5× on math-like traffic"* to **1.74× code / 2.25× math / 1.0× chat**.

### Chat: the biggest relative gain, and the router still stays

Chat improves **2.02×** — more than either other suite — and still lands at **0.92×**, under
break-even. The reason is visible in its acceptance column: **1.16 → 1.15 → 1.11 → 1.04 across a
16→4 cut.** The drafter lands about one token per round regardless of width, so the wide block
was buying nothing measurable while costing 28 ms of verify per round. Narrowing stops the waste;
it cannot manufacture acceptance that was never there.

Nor can narrowing further fix it. At k=4 the round already costs 6.6 ms draft + 18.2 ms verify
for ~2.0 tokens; the **draft term alone** puts a ceiling near 1.1× even at perfect acceptance,
and chat's first-position acceptance is roughly 0.5. **So gate 4's router remains necessary** —
but the penalty for firing indiscriminately falls from a 2.2× loss to a 1.09× loss, which
demotes it from "mandatory or the feature is harmful" to "worth having".

### A width-selecting router beats an on/off one

The three optima are 8 / 7 / 4 against α of roughly 0.87 / 0.85 / 0.54. Since the optimum is a
function of predicted acceptance, and gate 4's router already has to predict acceptance to decide
*whether* to fire, **it can pick the width for free** — the α̂ predictors from 05/§9 are the same
signal. That is strictly better than gating on/off: on chat, width 4 recovers 0.92× of the 1.0×
that skipping would give, so a mispredicted fire costs 8% rather than 55%.

**Caveat unchanged:** acceptance is measured, wall-clock is not. These compose measured
acceptance with the measured verify curve and draft cost on one target and one GPU. Gate 3 should
be run per-suite at these widths.

### The second pairing CORRECTS the front-loading claim — and the practical answer improves

The width sweep above was run on the 4B only, and it concluded that "acceptance is FRONT-LOADED,
not position-independent." Repeating it on Qwen3.6-35B-A3B shows **that is a property of the
pairing, not of block drafting.**

| | Qwen3-4B | Qwen3.6-35B-A3B |
|---|---|---|
| α fitted at k=16 | 0.848 | 0.866 |
| measured vs geometric, narrow k | **+7.5% … +13.4%** (above) | **−3.4% … −5.8%** (below) |
| shape | clear peak, **k=7 → 1.74×** | **flat plateau, ~1.58×** across k=6–10 |

The 35B's k=16 control reproduces 6.78 exactly, so the sweep is anchored. Its measured acceptance
(5.71 / 4.53 / 3.86 / 3.19 at k=16/10/8/6) sits slightly *below* its own geometric fit at every
narrow width, where the 4B sat consistently above. **The 4B's early positions out-accept its
fitted α; the 35B's do not.**

**So the corrected claim is:** narrowing the verify width is a large win on both pairings, but
*how much* the tail costs you is pairing-specific, and a single fitted α is not reliably
conservative — it under-predicted the 4B by up to 13% and over-predicted the 35B by up to 6%.

**The practical consequence is better than a per-pairing rule, not worse.** The 35B's curve is
*flat* across k=6–10, so width choice barely matters for it, while the 4B peaks at 7 and is
within 2% of peak at 8. **A single default of k≈8 serves both** — 4B 1.71× against its 1.74×
optimum, 35B 1.58× on its plateau. No per-pairing calibration step is needed for increment 4;
width can default to 8 and be refined per traffic class, which is the axis that actually moved
the number (math 8 / code 7 / chat 4).

**Caveat specific to this table:** the 35B speedups compose its *measured* acceptance with the
*4B's* verify curve and draft cost, because the 35B's own constants have never been measured.
The acceptance column is real; the speedup column is illustrative, and the flatness in particular
is partly an artifact of borrowed constants. What is robustly measured here is the **acceptance
profile** — near-geometric for the 35B, front-loaded for the 4B.

## Metal — verify curve measured, and the leading finding isn't the numbers

Per `docs/prompts/metal-verify-curve.md` (dispatched to the M1 Pro leg 2026-08-16). Result up
front: **Metal doesn't currently amortize batched verify at all** — ceiling ~1.13× *even with
free drafting*, against CUDA's measured 1.74× at k=7. And that ceiling number is arguably the
second finding, not the first — see below.

**Corrected before measuring: the prompt's harness pointer was wrong.** It said "mirror the
method on the `gpu/` (Metal) side" and named `gpu/`'s benchmark files as the template. Checked,
not assumed: `gpu/` is the WebGPU backend (wgpu-native, cgo, `-tags gpu`) — it has **no
`PrefillLast`** at all, only `ForwardN`. The native cgo-free Metal backend (`metal/`, purego,
dlopen `Metal.framework` — the one this repo's whole Metal story is actually about) **does** have
`PrefillLast`, matching CUDA's structure directly. Built the test in `metal/`
(`metal/spec_verify_curve_test.go`), not `gpu/`.

**The load-bearing finding: `PrefillLast` declines by default on Metal, and forcing it on for
measurement surfaces why nobody ships it.** `metal/backend.go` refuses batched prefill unless
`GOINFER_METAL_BATCHED_PREFILL=1` — its own f16-MMA activation kernels are **not bit-identical**
to decode's int8 path (54% stream divergence measured previously, §A2-Metal). P10's verify step
needs exact reproduction of sequential greedy (00-core's lossless contract), so **this kernel
cannot serve as P10's verify oracle on Metal as it stands, independent of its speed** — that gap
would need closing first, before any of the numbers below mean "shippable."

Measured anyway, because the task asked for the timing shape regardless: qwen2.5-coder-1.5b
(int4, W4A8 — CUDA's own default fixture in `spec_verify_ceiling_test.go`, not the project's
headline Qwen3-4B, which isn't on this Mac), depth 1024 (matches the CUDA test), best-of-5 per
point, `GOINFER_METAL_BATCHED_PREFILL=1` forced on for the batched calls only.

| M | T(M) ms |
|---|---|
| 1 | 91.4 |
| 2 | 91.6 |
| 4 | 103.6 |
| 6 | 105.6 |
| 8 | 119.1 |
| 10 | 153.4 |
| 12 | 156.8 |
| 16 | 169.7 |

Fit: **W = 80.42 ms, C = 5.90 ms** (least-squares over the 8 points).

**Real `decode_ms` = 27.40 ms** (~36.5 tok/s at this depth) — measured *separately* via `Forward`
(the shipped int8 path), mirroring what `spec_verify_ceiling_test.go` actually does (it computes
`one` independently, not by extrapolating the batched fit to M=1) rather than what this task's
prompt said to do ("T(1) is decode_ms"). The two diverge by **3.15×** on Metal specifically
because `PrefillLast` and `Forward` are different kernel families here — on CUDA the prompt's
simplification is fair because they're close; on Metal it silently substitutes an
never-tuned-for-small-M kernel's cost for real decode cost, understating the true ratio the
speedup formula needs. Use the separately-measured `Forward` number, not the fit's T(1), for
`decode_ms` on any backend where verify and decode aren't the same kernel family.

**Composed ceiling** (`decode_ms·(1+accepted)/verify_ms(k)`, `draft_ms = 0` — an upper bound, not
a real number, since actual drafting can only cost more):

| k | accepted (from the acceptance table above) | verify_ms(k) = W+Ck | ceiling speedup |
|---|---|---|---|
| 16 | 5.10 | 174.75 | 0.956× |
| 12 | 5.01 | 151.17 | 1.089× |
| 10 | 4.64 | 139.38 | 1.109× |
| **8** | 4.24 | 127.59 | **1.125×** |
| 7 | 3.97 | 121.69 | 1.119× |
| 6 | 3.41 | 115.79 | 1.043× |
| 4 | 2.45 | 104.00 | 0.909× |

Peaks at k=8, **1.125×, essentially flat**, and this is the *best possible* case. `draft_ms` was
**not measured** — no DFlash checkpoint or torch conversion environment (`~/.venv-vl`, per the
05-eagle3 precedent) exists on this Mac, and building that pipeline from scratch to get a real
(necessarily nonzero) number would not change the qualitative answer: any real `draft_ms > 0`
only lowers every row in the table above. The ceiling already answers the practical question.

**Bottom line for P10's Metal leg:** two independent reasons this path isn't ready, not one —
(1) the verify kernel is non-bit-identical to decode and would need a real fix before it's a
legal verify oracle at all, and (2) even ignoring that and assuming it were free to fix, the
*timing* ceiling tops out at ~1.13× versus CUDA's 1.74×, because `W` (Metal's batched-dispatch
fixed cost, 80 ms) dwarfs `C` (the per-row marginal cost, 5.9 ms) by a much larger margin than on
CUDA — the fixed cost is the whole story on this backend, and it is not amortizing at any of the
tested widths. Not a build target until (1) is resolved, and even then the ceiling says this is
a marginal win at best on the current kernel, not the ~1.5-1.7× the CUDA leg found.

### The draft term's omitted half: context work, and a hard requirement for increment 4

The 6.6 ms draft is derived by scaling the target's per-layer GPU cost across the drafter's 5
layers, and `cuda/dflash_draftcost_test.go` states its own limit: it excludes "the drafter's
non-causal attention over `[ctx‖block]`… a floor with a named omission, not a prediction." This
measures the omission's **shape** (`TestDFlashDraftScaling`).

There are two draft paths, asymptotically different **per round**:

| | what it does | per round |
|---|---|---|
| `DraftBlock(fused, block)` | `NewContext` + `ExtendContext` over the WHOLE fused context, then draft | **O(ctx)** |
| `DraftBlockCtx(ctx, block)` | caller keeps the context; only newly accepted rows are projected in | **O(new)** |

Measured on CPU (Qwen3-4B DFlash, 5 layers, hidden 2560, block 16, 5 new rows per round, warm-up
discarded):

| ctx | rebuild ms | incremental ms | ratio |
|---|---|---|---|
| 64 | 1708 | 1492 | 1.1× |
| 128 | 2073 | 1576 | 1.3× |
| 256 | 2682 | 1710 | 1.6× |
| 512 | 3974 | 1980 | 2.0× |
| 1024 | 6635 | 2709 | **2.4×** |

**Two findings, with different confidence.**

**1. Architectural, and it transfers: increment 4 MUST use `DraftBlockCtx`.** The rebuild path
costs 2.4× the incremental one at a 1024-token context and the gap widens with length — it
re-runs a full drafter-prefill of the entire context on *every block*. The acceptance harness
uses `DraftBlock`, which is correct there (identical numerics, and acceptance is all it measures)
but would be a serious defect in a serving path. This is now measured rather than assumed.

**2. Quantitative, and it does NOT transfer: the draft has a context-dependent term the floor
omits.** Even incremental, cost rises 1492 → 2709 ms from ctx 64 → 1024 — roughly half the draft
at a long context is attention over the context, which the 6.6 ms figure does not include. **The
CPU ratio must not be carried to the GPU**: attention over 1024 keys is bandwidth-cheap on a GPU
and expensive on this CPU path (f32, allocate-per-call), so the GPU magnitude could be far
smaller. What is established is that the term **exists and grows with context**, not how big it
is on CUDA.

**So the 1.74× projection still rests on an unmeasured draft**, and it is the term most likely to
move. For scale: if the GPU draft doubled to 13.2 ms, code at k=7 falls 1.74× → 1.44× — still
over the bar, but the headline changes. Measuring the resident draft remains the gate on
increment 4; this narrows *where* to look rather than settling it.

#### CORRECTION to the section above: 6.6 ms does NOT omit the context attention

The section above claims the 6.6 ms draft "omits" the context-attention term and calls it "the
draft term's omitted half". **That is wrong, and the error is in reading rather than in any
measurement.**

`cuda/dflash_draftcost_test.go` computes an **M=1** floor — `5.33 × per-layer(M=1)` — and its
"excludes the drafter's non-causal attention" note attaches to **that** figure. Re-run today to
confirm: per-layer(M=1) = **0.2830 ms**, M=1 decode 11.122 ms, floor **1.51 ms**. Those reproduce
the recorded 0.2818 / 11.105 / 1.50.

But **6.6 ms is a different number**: `5.33 × per-layer(M=16) ≈ 5.33 × 1.23`, where
per-layer(M=16) comes from the 36-layer M=16 batched verify **measured at depth 1024**. That
verify attends over 1024 keys with 16 queries per layer. The drafter's per-layer work is 16
queries over ctx+16 keys, non-causal — structurally the same shape at the same depth. **So the
context attention is already inside the 6.6 ms**, and I attached a caveat written about the
1.51 ms floor to a figure it does not describe.

What the CPU scaling test did show remains true and unchanged: cost grows with context length.
But so does the target's verify, for the same reason, and the `per-layer(M=16)` scaling already
carries it. Growth with context is a **shared property already captured**, not an omission.

**Finding 1 of that section stands unaffected** — `DraftBlockCtx` is required, the rebuild path
costs 2.4× at ctx=1024 and worsens. That was measured, not inferred, and it is the useful result.

**The real residual risk is different, and narrower than I made it sound:** whether a **5-layer**
drafter achieves the **36-layer** target's per-layer efficiency. The 6.6 ms assumes it does. A
36-layer model amortizes launch latency across 36 dispatches; five layers have far less to hide
it behind, which is precisely the mechanism that made the CUDA-graphs projection (1.4–1.7×) land
at 1.01×, and the mechanism behind Lever 2's "the draft was the wall". That is a **dispatch
efficiency** question, not a missing-attention question — and it is still the gate on
increment 4.
