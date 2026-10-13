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

## Results (2026-10-13; the night of 2026-10-12, `76abc5bc`)

Log: `docs/measurements/cuda-f16-kv-2026-10/kv16-gate-2026-10-12.log.gz`. Both cells ran to the end; every precondition held (the f16 K buffer is half the f32 one, the lane launched at every decode row, 63 of 63 decode rows differ
on every prompt, A/A bit-identical over 64 rows on both cells, the seed row identical on 10 of 10 prompts in both).

- **Overflow survey: clean.** Over the ten set-B prompts at every layer, the largest |K| is 423.3 (D7, last layer) and 360.6 (S, layer 0), and the largest |V| is 34 or less, against the 32,752 limit.
- **D7, K=8000, the decision cell: AMBIGUOUS, PARKED.** Mean teacher-forced agreement f16 92.03% against f32 92.97% (paired difference -0.94 pt, s.e. 0.67; the bar is -1.0, and the parked band is its last 0.2); f16 >= f32 on 5 of 10 prompts
  (the bar is half); hard flips 9 against 11 (criterion a passes); mean KL 0.053865 against 0.053184, ratio 1.0128 (criterion c passes, below the 1.05 band), paired KL delta +0.00068, s.e. 0.0015. Four of the ten prompts sit at or near
  100% on both arms and say nothing; the other six move by one to three positions each, in both directions. One position is 1.56 points, so -0.94 pt over 640 positions is six positions.
- **S, K=3900, the confirmation cell: PASSES** (agreement 90.94% against 91.09%, f16 >= f32 on 6 of 10, hard flips 8 against 9, KL ratio 1.0057). It cannot rescue the decision cell.
- **Prediction scored:** both cells PASS (0.8) came out one of two; KL within 3% (0.6) yes on both; agreement within 0.5 pt (0.7) yes on S, no on D7; a family fails the survey (0.05) no.

**Decision, by the rule above: the default does not flip.** `--kv auto` stays f32; `-kv f16` stays available on the five dense families and is the opt-in path. Per the registration, a re-run needs a named mechanism, not a re-roll, and there is
none: the f16 arm's only change is the rounding of stored K and V, which the exact-input identity tests prove is the only change. Whether a PARKED decision cell is enough to turn the default on anyway is the owner's call
(R6's flash-decode lane and R8's vision tower were each turned on by owner decision), and it would be recorded as that, not as a pass.

The paired speed check did not run: its first night run was refused by `bench_peer.py` on a 1-minute load of 1.36, left by the gate before it (no cell was measured). The script now waits for the load to settle before each pass; it is queued again.

## Speed read and owner override (2026-10-13)

**First served read (old kernel), stopped by the owner:** pass 1 of 2 (f16 first) complete, plus the d=128 cells of pass 2; records `docs/measurements/cuda-f16-kv-2026-10/speed-partial-2026-10-13-*.json.gz`. f16 against f32:
0.5B 1.00 / 0.94 / 1.00 / 1.02 at 128 / 512 / 2048 / 3900; 1.5B 0.96 / 0.87 / 1.02 / 1.02; phi3-mini 0.98 / 0.95 / **0.88 / 0.84**. The 1.5B d=128 cell read 0.960 and 0.959 in the two arm orders, so the slow cells were real. By the bands
above that read was KILLED (any cell under 0.95).

**Cause and fix (`9a1a0f7e`):** Phi-3 decodes through `attn_batched`, whose V-sum is a latency-bound walk down the keys; half-width loads put half the bytes in flight per request, so the f16 reader ran 231 us a call at 2048 keys
against 161 for f32. The f16 build now keeps sixteen V loads in flight, FMAs in the same ascending order. The reader takes 133 us (f32 161), the whole phi3 token reads 1.02 / 1.06 / 1.10 times f32 at 512 / 2048 / 3900 in process, the f32
module's PTX is byte-identical, and the exact-input identity holds (hd 64 and 128 by the tests, hd 96 by an output hash that matches across arms for all four readers). Two other attempts (a two-keys-at-once score loop, a 4-load unroll) changed
nothing and were dropped.

**Second served read (new kernel), stopped by the owner at 12 of 24 cells of pass 1:** the shallow cells only (0.5B at 128 / 512 / 2048 / 3900, 1.5B at 128, phi3-mini at 128) read 1.013, 1.002, 0.991, 1.007, 1.011, 1.006. Not measured with the new kernel:
1.5B and phi3-mini at 512 and over, 7B at 8000, the reversed arm order. It is not a result.

**Owner override.** The owner approved the default on 2026-10-13, over two registered gates that did not pass: the fidelity decision cell read PARKED, and the speed check was stopped before its deep cells and its second order ran, because
the KV cache is about to be re-architected. `--kv auto` is now f16 on CUDA for llama, mistral, qwen2, qwen3 and phi3; `--kv f32` is the exact cache. This is recorded as an override, not as a pass. **Owed:** the paired speed check, run
as registered, on whatever the KV re-architecture leaves; until then the deep-cell speeds of the new kernel are the in-process figures above.
