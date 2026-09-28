# The five CPU-int4-golden drifts — not defects, one threshold fix and one test-design fix (2026-09-28)

**2026-09-28 · `linux` (nobara-pc) · goinfer `cfd3ce25`+ (pre-push), aikit `v1.50.1`.** Follow-up to today's DeltaNet
prefill work (`docs/tasks/task-cuda-deltanet-prefill-2026-09.md`), which found five CUDA-package tests red after
aikit v1.50.0's binary16 int4 group scales: `TestGemma3ResidentReal_gate`, `TestGemma3ImgPrefillResidentReal_gate`,
`TestQwen25VLResidentReal_gate`, `TestQwen25VLMRoPEPrefillResidentReal_gate`, `TestMLAResidentParityCUDA`. The task
doc's own status called this "a separate follow-up, not this task's." This is that follow-up.

**Verdict: no defect in goinfer's forward code, in any of the five.** Two different, unrelated pre-existing
miscalibrations, both now fixed and evidenced below.

## Method

Both mismatches needed a direct A/B against pre-f16-scale aikit, not just "the number is lower than a round
threshold." A throwaway git worktree (`git worktree add --detach 3cd62e6d`) with `go.work` pointed at a second
worktree of aikit checked out at its `v1.49.0` tag (not the live local aikit checkout, which floats — that trap cost
one wasted run: `go.work` unioning the local aikit tree silently ran CURRENT aikit regardless of which goinfer
commit was checked out, so the first attempt at this comparison showed no change at all).

## 1 & 2. Gemma3 and Qwen2.5-VL: `cpuInt4VsF32GoldenFloor` was an int8-era bar applied to an int4 comparison

Both `TestGemma3Resident(Img)Real_gate` and `TestQwen25VL(MRoPE)ResidentReal_gate` start with a PRECONDITION check
— "does the reference itself hold, before testing anything gap-0-specific" — comparing a CPU **int4** forward
against an **f32** HF golden, at a 0.99 floor inherited unchanged from the introducing commits (`84850d62`,
`a630a2cc`), which never distinguished it from an int8-vs-f32 or f32-vs-f32 comparison. `decoder/real_oracle_test.go`
already has the right precedent for this exact situation (`oracleCosFloor`): int8-vs-bf16/f32 gets 0.99, int4 gets
0.98, pre-registered before any qwen3_next number existed. These four CUDA-package tests never inherited that split.

| | pre-f16-scale (aikit v1.49.0) | today (aikit v1.50.1) |
|---|---:|---:|
| gemma3 (introduction, `84850d62`) | 0.998167 | — |
| gemma3 (re-measured, this worktree) | 0.997912 | 0.989747 |
| qwen2.5-vl (this worktree; no number recorded at introduction) | 0.989088 | 0.988259 |

- **Gemma3: the f16-scale change is the entire cause.** 0.998 → 0.990, an ~0.008 drop, matching the order of
  magnitude of L1's own accepted quality cost (`task-cpu-decode-peer-gap-2026-09.md`'s pre-registered gate measured
  a worst-case drop of 0.0168 on isolated kernels; the owner overrode a FAIL verdict there — "for consistency, let's
  override" — and shipped it). Nothing else changed between the two aikit tags on this code path (`git log
  8fa0e3da..84ee8f49 -- cuda/ decoder/` — see §3 below for the same diff, reused).
- **Qwen2.5-VL: already below the 0.99 bar BEFORE the f16-scale work**, at 0.989088 — this was a pre-existing
  miscalibration the heavy tier never surfaced (`cmd/gate/gpu.go`'s own comment: the heavy tier "existed and was
  never executed by anything" until this week). The f16-scale change adds only ~0.0008 more. Confirmed as
  quantization noise, not a defect: `TestQwen25VLReal_gate` (`decoder` package) loads the SAME checkpoint at **f32**
  and passes at cosine 0.999459, exact argmax — goinfer's Qwen2.5-VL forward matches HF; the int4 CPU kernel path is
  simply noisier on this checkpoint than Gemma3's.

**Fix:** `cpuInt4VsF32GoldenFloor = 0.98` (`cuda/realforward_test.go`), used by all four precondition checks in
place of the literal `0.99`. Mutation-checked (raising it to 0.999 turns the check red again).

**With the fix, all four tests reach their REAL assertion for the first time** — the `t.Fatalf` precondition had
never let them get there — and it passes clean:

| test | CPU argmax | hybrid/resident argmax | cosine |
|---|---:|---:|---:|
| `TestGemma3ResidentReal_gate` | 15004 | 15004 | 0.998271 |
| `TestGemma3ImgPrefillResidentReal_gate` | 496 | 496 | 0.996878 |
| `TestQwen25VLResidentReal_gate` | 1697 | 1697 | 1.000000 |
| `TestQwen25VLMRoPEPrefillResidentReal_gate` | 264 | 264 | 0.998644 |

## 3. MLA: a genuine near-tie flip, not a routing defect — confirmed by inspecting the actual logit margins

`TestMLAResidentParityCUDA` (a synthetic/random-weight `deepseek-tiny` fixture, not a real checkpoint) diverges at
greedy generation step 5: CPU picks token 51, CUDA picks token 87. Bisected earlier to `2669bf11` (the CPU f16-scale
commit): passes at `8fa0e3da`, fails at `84ee8f49`. That commit's own diff touches only scale-accessor plumbing
(`wm.Int4()` → `wm.Int4F16()` in `decoder/moepool.go` and `decoder/int4f16scales.go`'s removal) — no routing/gating
math — and its own pre-registered gate 3 claims CUDA's upload numerics are byte-identical old-vs-new. So only CPU's
rounding changed; CUDA didn't move at all.

The test's own comment explains why this can't be waved off on sight: a prior real bug (an `nGroup`/`topkGroup`
transposition in the CUDA router launch) left the per-position cosine check untouched and only showed up several
tokens into free generation — exactly this failure's surface shape. So this needed direct verification, not
pattern-matching.

**Debug replay** (temporary instrumentation, `cuda/mla_resident_test.go`, removed before the final fix): raw top-2
logit margins at every generation step —

| step | cpu top1/top2 gap | cuda top1/top2 gap |
|---:|---:|---:|
| 0-4 | 0.030–0.046 | 0.032–0.047 |
| **5 (diverges)** | **0.000746** | **0.002231** |
| 6-7 | 0.008–0.046 | — |

At step 5 — and only step 5 — both backends' margins collapse by more than an order of magnitude, and each side's
runner-up **is** the other side's winner (token 51 is CUDA's #2 pick; token 87 is CPU's #2 pick). That is the MoE
router-flip noise floor this repo already has a memory for (a bit-identical router flips top-k under ~0.5% input
noise) — an already-near-tied pick on a random-weight fixture, decided by backend-implementation rounding, not a
wrong computation. A real routing defect produces a confidently wrong pick, not a swap between each side's own
top-2, which the fix's mutual-containment check below still catches.

**Fix, `cuda/mla_resident_test.go`:**
- The generation check now walks ONE teacher-forced trajectory — both arms fed the same, CPU-chosen token at every
  step (matched observations, not two independently-diverging free-running arms; the same principle this repo's
  measurement discipline names for a peer comparison, applied here to a backend comparison). This also means a
  divergence at one step no longer confounds every later step: cosine stays ≥0.99993 through generation, before AND
  after the tolerated flip.
- A token divergence still hard-fails UNLESS it is a **mutual near-tie**: the CPU's pick is in CUDA's own top-2,
  AND CUDA's pick is in CPU's own top-2. Mutation-checked both directions (forcing either side's containment check
  to `false` turns the test red again) — the guard is live, not a rubber stamp.
- The per-position prompt-cosine floor (0.99) is unchanged and now also covers the generation region.

`TestMLAResidentParityCUDA` passes clean: worst cosine 0.999846 over all 14 positions (6 prompt + 8 generated), one
tolerated near-tie flip logged by name at step 5.

## Artifacts

`five-tests-fixed.log` — the final `-v` run of all five, clean.

## Not in scope here

- Whether `cpuInt4VsF32GoldenFloor` should be split further (a per-checkpoint bar, rather than one shared 0.98) if
  a future family lands closer to that bar than Qwen2.5-VL's 0.988 already does.
- Re-running the full GPU gate to confirm these five clear in the gate's own harness, not just standalone — covered
  by tonight's queued `gate-gpu-cuda` run (`docs/tasks/task-cuda-deltanet-prefill-2026-09.md`).
