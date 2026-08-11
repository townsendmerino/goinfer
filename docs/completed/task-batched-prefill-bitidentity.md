# Batched prefill vs sequential decode bit-identity — FOUND, FIXED

> **ARCHIVED — a record, not instructions.** This file is closed work kept for its reasoning and
> its numbers. Checkboxes record the state at the moment it was archived: an unticked box means
> "not ticked when this closed", **not** "still to do", and nothing in `docs/completed/` is
> actionable. If you need a task, use the live docs; if something here reads as an instruction to
> a future reader, it was missed at archival — see the doc-closeout rule in
> `docs/parity-coverage-policy.md`, and move it to live policy or strike it.


> Found 2026-08-04 while building D1 (spec-decode); **FIXED same day** (explicit-FMA everywhere).
> Status: **batched prefill default-ON again, bit-identical** — `TestPrefillDivergenceRate` 0/50
> (was 42/50), enforced by `cuda.TestKernelFMALint`. The root cause was **compiler FMA contraction**,
> not a reduction reorder; the fix and the durable gate are at the bottom of this doc.

## The claim that was false

The batched-prefill campaign shipped "the K/V it writes is BIT-IDENTICAL to the sequential path
row-for-row … decode from the last prompt token stays byte-identical," gated by
`TestPrefillLast_e2e`. **On real models it is not true.**

## Measured (real qwen2.5-coder-1.5B)

- **Token-stream divergence: 42/50 (84%)** — `TestPrefillDivergenceRate`. Batched-prefill-then-decode
  vs sequential-prefill-then-decode, 50 distinct prompts × 128 greedy tokens. Mean first-divergent
  token 27; many at token 0. Not "rare ties": a ~1e-6 KV perturbation compounds under greedy.
- **KV differs:** batched-prefill KV vs sequential-decode KV over 28 layers × 40 rows = 528,420
  mismatches.
- **The unit:** the batched forward's projected **Q differs from the decode forward's by ~2e-6**
  (313/1536 elems), even with layer-0 **input x identical** and the **quantized activation identical**
  (int + scale, 0/384). `TestBatchedVsDecodeGap` is byte-identical at startPos=0 (single-key attention
  ignores Q → the gap was invisible) and diverges at startPos>0 — the regime production never used
  `PrefillLast` in (it is always called at startPos=0), which is why no gate caught it.

## Why the gate missed it — the fifth representativeness axis: DISTRIBUTION

`TestPrefillLast_e2e` uses `mistral-tiny-window` (a small fixture); the GEMV/RMS bit-identity gates
use **uniform random** weights. The divergence is a **data-dependent last-ULP difference** in the
per-word float scale-accumulate — `facc += (float)p * scale`. Float rounding of a reordered or
re-contracted sum matters most when summands vary widely in magnitude; uniform random int4 gives
similar group-scale magnitudes that round identically and cancel, while **real weights have
outlier-scale channels** that expose it. A green gate on uniform data is exercising a different
numerical regime than production.

**Record DISTRIBUTION alongside depth, width, discreteness, degeneracy** as the fifth axis of fixture
representativeness — all five are the same failure: the gate is green because it runs a different code
path or numerical regime than production. A GEMV bit-identity fixture qualifies only if it deliberately
includes **outlier-scale groups** (assert it at construction, the way the MoE fixture asserts a routing
flip).

## What is (and isn't) the cause — investigation record

Localized by elimination, all on the real 1.5B, in one process:

- layer-0 **input x**: identical.
- **RMS** (`rmsnorm_quant` vs `rmsnorm_quant_batched`): bit-identical in isolation at real widths
  (1536, 8960); the **quantized activation they produce in the real forward is identical**.
- fused **fQKV** == unfused rms+gemv (K1 exonerated).
- decode GEMV confirmed = `gemv_w4a8_fwd`; isolated `gemv_w4a8_fwd == gemv_w4a8_rn` on random data.
- both paths **deterministic**.
- **PTX shows different FMA contraction:** `gemv_w4a8_fwd` = 11 `fma.rn.f32` + 3 `mul.f32`;
  `gemv_w4a8_rn` = 192 `fma` + 0 `mul` — the scale-accumulate `facc += p*s` is **mul+add (2 roundings)
  in decode, fma (1 rounding) in batched**. fma vs mul+add differ ~1 ULP data-dependently.

**BUT:** forcing `gemv_w4a8_rn`'s loop accumulate to `__fmul_rn` (mul+add, matching fwd) dropped its
fma count 192→96 yet **did not move the Q gap** (still 313/1536). So the loop accumulate is not the
(sole) source. The divergence is a composition of contraction/rounding-regime differences across the
separately-compiled batched kernels that resists single-op localization in bounded time. This is why
the fallback (default-off) is correct now, and the fix is a scoped follow-up.

## The fix — LANDED (explicit `__fmaf_rn` everywhere)

The root cause is FMA CONTRACTION, confirmed at the PTX: `gemv_w4a8_fwd` compiled to 11 `fma` + 3
`mul`, `gemv_w4a8_rn` to 192 `fma` + 0 `mul` — the scale-accumulate `facc += p*s` is mul+add in one and
fma in the other. Not a reduction reorder (a targeted `__fmul_rn` on the rn loop did NOT move it — the
divergence is a *composition* of contraction differences across separately-compiled kernels).

**Fix: remove the compiler's freedom everywhere.** Every float multiply-accumulate under the bit-identity
contract is now an explicit intrinsic (`__fmaf_rn` fused — fewer instructions AND one rounding, so both
faster and more accurate than mul+add), across BOTH paths' kernels:
- aikit `gemv_quant.cu` (`gemv_w4a8_fwd`, `gemv_w8a8_fwd`) — the decode GEMV, in its own repo.
- goinfer `gemv_w4a8_rn` / `_batched` / `_staged`, `fused_qkv.cu` (fQKV/fGU), `glue.cu` +
  `prefill_batched.cu` (rmsnorm/rope/attention/glu/V-sum), `decode_splitkv.cu`, `gemv_fwd.cu` (rope_kv).

Verified against the exposing measurements: batched-vs-decode gap byte-identical (was 138275),
`TestPrefillDivergenceRate` **0/50** (was 42/50), decode speed unchanged (fused = fewer instructions).
Then re-enabled batched prefill by default and restored the bit-identity claim (§B2, §9).

**Cross-repo note:** `gemv_w4a8_fwd` lives in `aikit/gpu` (external module). The decode-side fix shipped
as **`aikit/gpu@v0.25.0`** (commit `be049df`), and goinfer's go.mod is bumped to it — no `replace`, CI
uses the published version. Verified 0/50 against the real dependency. aikit's `gemv_quant.cu` header
carries the same explicit-FMA rule so a future aikit kernel edit can't silently re-break the pair.

## The durable gate — TestKernelFMALint

`cuda.TestKernelFMALint` scans every contracted kernel and FAILS THE BUILD on any bare float MAC
(`a*b + c`), before any numerical test and independent of the NVRTC version. New kernels inherit the
rule automatically. The standing rule is recorded in `ollama-chase.md` §2 (next to Metal's
reduction-width contract — same family) and in aikit's `gemv_quant.cu` header. Owed follow-ups: the
same explicit-intrinsic audit on Metal/AIR (see task-6 finding), and a PTX instruction-histogram
cross-check as a second backstop.

## Gate work owed (the durable deliverable)

- `TestPrefillDivergenceRate` (token-stream), `TestBatchedVsDecodeGap` (unit) — promote to permanent
  gates.
- A `gemv_w4a8_rn == gemv_w4a8_fwd` gate on **outlier-scale (real-distribution) weights**, asserting the
  fixture actually contains outlier groups at construction.
- Audit every bit-identity gate over a float reduction for the same defect (uniform-data-deep).
- Q compared alongside KV in the parity harness (Q is never cached, so was never compared).
