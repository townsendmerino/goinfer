# P10 block drafters on Metal: a projection, before any port — 2026-09-28

**Status: written for the owner's decision; NOT started (owner, 2026-09-28: "write it up but we won't start now").**
P10 is DFlash/DSpark block drafting ([`docs/spec/08-dspark-dflash.md`](../spec/08-dspark-dflash.md)). It shipped on
CUDA behind `--drafter`. Metal gained a verify it can use on 2026-09-27: the MC3 step kernels, bit-identical to decode
([`metal-spec-step-verify-2026-09-27.md`](metal-spec-step-verify-2026-09-27.md)). This page asks whether porting
the rest is worth it, from measured terms only. Nothing here was run for it.

## 1. What transfers from CUDA, and what does not

Units: one target decode token, t_d. CUDA's figures are the RTX 2070 SUPER on a Qwen3-4B int8 target at depth ~1024,
from spec 08: t_d = 11.12 ms, the verify curve 8.77 + 2.35·M ms, and the drafter 8.82 ms.

| term | CUDA (measured) | Metal | transfers? |
|---|---|---|---|
| **tokens per round, non-thinking** (acceptance) | code 4.71 (k 7), math 5.88 (k 8), chat 1.96 (k 4) | the same | **yes.** Greedy acceptance is a property of the model pair and the prompts, not the device. One caveat: CUDA's drafter ran int8, and its math acceptance fell 15% against an f32 drafter. |
| **tokens per round, thinking template** | about half (5.76 → 3.00) | the same | yes. It is below break-even on either backend. |
| **verify of k + 1 rows** | 2.48 t_d at 8 rows (0.21 per extra row) | **1.98–2.54 t_d at 8 rows** (0.14–0.22 per extra row) | **measured on Metal, bit-identical.** Rows are 0.11–0.19 of a token on the 1.5B and 0.16–0.24 on the 7B at M = 8; the 4B is interpolated. The step fragment is 8 rows wide, so k ≤ 7 per verify piece. |
| **drafter trunk** (5 layers at M = 16) | 0.79 t_d, with a 1.28× per-layer premium at M = 16 | **unknown: bounded at 0.79–1.2 t_d** | **no.** It is the one term not measured on Metal. The drafter need not be bit-identical, so it may use the f16-MMA prefill path. The upper bound is 1.5× CUDA's relative cost, for Metal's weaker small-M GEMM. |
| **seam and composition** (capture, host argmax, per-round residual) | ~0.4 t_d a round (+11% measured composition) | assumed the same relative cost | an assumption. |

## 2. The projection (non-thinking target, Qwen3-4B class)

Round cost = draft + verify(k + 1) + ~0.4 t_d. Speedup = tokens per round ÷ round cost.

| suite | k | tokens / round | Metal round (t_d) | **Metal projection** | CUDA, shipped |
|---|---|---|---|---|---|
| code | 7 | 4.71 | 3.17–4.14 | **1.14–1.49×** | 1.44× |
| math | 7 (the fragment cap; CUDA ran 8) | ~5.5 (between CUDA's 7- and 8-wide acceptance, estimated) | 3.17–4.14 | **1.33–1.73×** | 1.58× |
| chat | 4 | 1.96 | 2.75–3.48 | **0.56–0.71× unguarded**, ~0.9× guarded (CUDA's guard measured 0.92×) | 0.92× guarded |

**The composition is optimistic.** On CUDA, projections ran 5% above measurement (code) to 23% below (math). The
+11% composition residual appeared only once the loop was actually built. Read the low end as the likelier one.

## 3. What it would compete with on Metal

Metal already has lossless n-gram speculation on the step verify: **2.08× on copy-heavy traffic and 1.07× on chat**
(1.5B, served, 2026-09-27).
- The drafter's niche is **non-copy code and math generation**, where n-gram has nothing to copy.
- It loses on chat and on thinking-mode templates unless guarded. The guard needs a fast greedy fallback, which
  CUDA's loop lacked at first.

## 4. What the port is

Everything but the verify:
- the drafter's 5-layer trunk on the Metal resident, at M = 16;
- block attention (`attn_block_full`);
- resident hidden capture for the draft's context features;
- the GPU context cache (`DFlashContext`);
- the round loop over the step verify, capped at 8 rows per piece;
- the acceptance guard with a fast greedy fallback;
- the drafter's `.giw` / loader path.

Weeks, not days. The pairs the Mac can hold are **Qwen3-4B** (`dspark_qwen3_4b_block7`, `z-lab/Qwen3-4B-DFlash-b16`).
The Gemma-4-12B pairs (~7 GB int4 plus the drafter) are at the edge of this 16 GB Mac's resident budget.

## 5. Recommendation, and the cheap first step if it is ever started

- **Do not port on this projection.** Code spans 1.14–1.49× against spec 08's 1.3× bar, so the band straddles it.
  Math clears at 1.33× or better. Chat is a loss that needs the guard.
- **Step 0 when it is picked up:** measure the one unknown term, the drafter trunk's cost on Metal. Run a 5-layer
  truncated resident forward at M = 16 against a decode token, the Metal twin of CUDA's
  `TestDFlashDispatchAmortization`. It is an in-process, by-day measurement (TE5(b)). It collapses the band:
  - **at ≤ 0.8 t_d,** code projects at 1.26–1.48× across the verify band (1.36× at its middle), and the port is worth
    registering;
  - **at ≥ 1.1 t_d,** code projects at 1.17–1.35× (1.26× at the middle), and P10-on-Metal is parked on its own number;
  - **in between,** it is the owner's call.
- **Pre-registration still owed before any port:** the served bars, spec 08's gate 3 shape (≥ 1.3× on at least one
  real workload, lossless), and a do-nothing arm that includes n-gram spec, so the drafter has to beat what Metal
  already has.
