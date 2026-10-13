# CUDA f16 KV cache, and f16 as the CUDA default (2026-10-12)

Owner request, 2026-10-12: "f16 KV by default on CUDA ... roughly doubles the context ... needs a fidelity gate, plus a paired speed check against
today's f32." **The request assumed `-kv f16` already worked on CUDA. It does not**: CUDA's resident build allocates f32 K/V whatever `--kv` says
(`decoder/residency.go:ResidentKVPrecision`, `decoder/residentneed.go:cudaKVBytes`, `docs/measurements/memory-accounting-cuda-2026-09-25.md`
finding 1), and every attention kernel reads `const float* kc`. So this is a build, then a gate, then a default. Owner decision the same day:
**a phased project, one push at the end.** The push waits for the gates below; nothing here is on `main` until they have been read.

## The design

- **Storage:** K and V as IEEE half (`__half`), arithmetic unchanged: every load converts to f32 (exact), every store rounds to nearest even.
  The softmax, the dot products and the V-sum run in f32 exactly as today, so the only change to a result is the rounding of the stored K and V.
- **Kernels:** each KV-touching `.cu` is compiled twice from one source: once as today (`KV_T` = `float`; its PTX must regenerate byte for byte
  unchanged, which is the proof the f32 path is untouched) and once with `-DKV_F16` as a `_kv16` twin module. The Go side loads the twin into
  the same pipeline fields when the build is f16, so no launch site changes.
- **Exact-input identity (the kernel test):** on K and V that are already f16-representable, a `_kv16` kernel must produce **the same bits** as
  its f32 original, because the loads are exact and the arithmetic order is the same. That is a bit-for-bit test per kernel, not a tolerance.
- **Selection:** `--kv auto` (the new default; today's `--kv` default `f32` becomes `auto`). On CUDA it means f16 when the family is cleared and
  nothing in the build declines it, else f32. `--kv f32` is exact and is what today does; `--kv f16` on CUDA asks for f16 on any build that
  supports it. Metal (f16 always) and WebGPU (f32 unless asked) are unchanged. A build that declines f16 says why in the banner.
- **Declines to f32, named (v1 scope):** MLA latents, Gated-DeltaNet hybrids, sliding-window or KV-shared layers (Gemma 3/4, windowed KV),
  gpt-oss attention sinks, the C′ paged-experts path, a drafter or speculative decode (its own f32 cache shares the attention pipelines), the
  VL image-prefill and m-RoPE kernels, Cohere's pairwise RoPE. Each is a later widening with its own gate; none is claimed here.
- **Capacity:** `cudaKVBytes` and `Plan` price f16 at half. The context a given card admits then follows from the existing fit logic.

## The gates (registered before any code runs; the criteria are the R6 flash-decode gate's, unchanged)

Instrument: `cuda.TestFlashDecodeGateVsReference`'s shape (`cuda/flash_decode_gate_test.go`), arms `f32` (exact) and `f16` in one binary and
session, teacher-forced on the reference tokens, against the **already-built** CPU f32/f64 reference rows of held-out prompt set B
(`~/goinfer-logs/prefill-ref-b`, D7-K8000 and S-K3900, ten prompts of 64 positions each; `testdata/prefill-gate-prose-b/`). Set B was scored once
for the flash-decode lane; this candidate has no tunable parameter and was never scored on B, so B is still not a set this decision was fitted to,
but it is no longer unseen for the lane, and this record says so. Both arms run the production decode lane (flash-decode default-on at 2,048 keys
and over), so the gate measures f16 KV as it would ship.

- **Cells:** D7 (qwen2.5-7b-instruct-q4_k_m, nH=28 nKV=4 hd=128) at K=8000, the **decision** cell; S (qwen2.5-coder-1.5b-instruct-q4_k_m) at K=3900,
  the **confirmation** cell, which cannot rescue a D7 failure.
- **Per cell, all three:** (a) hard flips: f16 <= f32 + 2*sqrt(f32). (b) agreement, paired: f16's mean teacher-forced agreement >= f32's mean - 1.0 pt AND
  f16 >= f32 on at least half the prompts. (c) KL: f16's mean continuation KL <= 1.10 x f32's.
- **Ambiguous is parked:** (b) within its last 0.2 pt, or (c) in 1.05-1.10x, is inconclusive. A re-run needs a named mechanism, never a re-roll.
  A (c) pass whose paired delta sits more than 2 standard errors above zero is reported with the delta and its s.e.
- **Preconditions (a failed one voids the cell; it is not a fail):**
  1. **Vacuity:** the f16 arm's KV buffers are half the f32 arm's bytes (read from the resident), the `_kv16` pipelines are the ones launched (launch
     count), and its decode rows differ from f32's on at least one position of every prompt.
  2. **A/A:** two f16 runs on the first prompt agree bit for bit over all 64 rows.
  3. **Seed row:** the first (prefill-seeded) row is compared and reported; f16 prefill stores the same K/V rounded, so it may differ and is not required equal.
  4. **Overflow survey, before any gate is read:** on the real K/V of the gate prompts at every layer, max|K| and max|V| must be under 65504/2. A family
     within 2x of f16's range is refused f16 by the default, whatever the gate says (R6's rule). The result is a table in the record.
  5. **Exact-input identity** passes for every `_kv16` kernel the gate's arms launch (the kernel test above).
- **Prediction (written now):** both cells PASS (0.8); the f16 arm's mean KL is within 3% of f32's (0.6) and its agreement within 0.5 pt (0.7); a real
  family fails the overflow survey (0.05).

## The paired speed check (registered separately; fidelity does not gate it and it does not gate fidelity)

Served, greedy, `scripts/bench_peer.py` protocol with the goinfer-only two-arm form (`BENCH_ENGINES=goinfer,goinfer_old`, no peer arm): the arms are `--kv f32`
and `--kv f16`, same binary, fresh serve per arm, alternating order, an idle box, one timed run on the box (the timing lock), at {128, 512, 2048, 3900}
on qwen2.5-coder-0.5b / 1.5b and phi3-mini, and 8000 on D7 (qwen2.5-7b). The do-nothing arm is `--kv f32`, the one the default would replace.

- **Ships (as the default):** the median f16/f32 decode rate is **>= 0.98 at every cell**. Decode attention is byte-bound at depth, so f16 is expected to
  win there; short cells have little KV to read, and the bar only forbids a regression.
- **Parked:** the worst cell in 0.95-0.98. **Killed:** any cell below 0.95.
- **Prediction (written now):** D7 at 8000 reads >= 1.10x (0.6); the 128-key cells read within +-2% of 1.00 (0.7); no cell reads under 0.98 (0.75).

## Decision

The default flips to `auto` (f16 on a cleared CUDA family) **only if** the fidelity gate reads PASS on both cells, the overflow survey is clean, and the
speed check ships. PARKED on either leaves `--kv f16` available and `auto` at f32. FAIL on fidelity leaves the f16 path built, opt-in, and records why.
Nothing moves a golden or a parity row unless the gates say so.

## Tier and cost (CLAUDE.md, "Gate tiers")

| check | tier | instrument | stopping rule | cost basis |
|---|---|---|---|---|
| per-kernel exact-input identity; exact PTX regen of the f32 modules | day | the tests | fixed | seconds |
| overflow survey | day | real K/V of the gate prompts, per layer | the ten set-B prompts at D7 and S | minutes (the R6 oracle test's K=8000 prefill) |
| the fidelity gate | night | `TestFlashDecodeGateVsReference`'s shape, two arms | all 10 prompts x 64, no early stop | the flash-decode gate's own wall: D7@8000 1,110 s, S@3900 well under; queue at 60 |
| the paired speed check | night | `bench_peer.py`, two goinfer arms | fixed rounds, pre-registered here | cells x runs x per-cell time plus idle-gate waits: estimated when queued |

## Phases (all local commits; one push at the end)

0. This record, committed before any gate. 1. The kernel twins and their identity tests, f32 PTX byte-identical. 2. The Go plumbing: options, buffers,
twin loading, `UploadKV` conversion, declines, pricing, banner. 3. End-to-end tests (f16 resident vs f32 resident logits), the gate and the night scripts.
4. The overflow survey and, at night, the two gates. 5. The default, only on a pass; docs and flags; one push.
