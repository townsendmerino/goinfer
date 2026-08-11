# Task (goinfer): quantized CPU KV cache (int8 now, int4 later)

> **ARCHIVED — a record, not instructions.** This file is closed work kept for its reasoning and
> its numbers. Checkboxes record the state at the moment it was archived: an unticked box means
> "not ticked when this closed", **not** "still to do", and nothing in `docs/completed/` is
> actionable. If you need a task, use the live docs; if something here reads as an instruction to
> a future reader, it was missed at archival — see the doc-closeout rule in
> `docs/parity-coverage-policy.md`, and move it to live policy or strike it.


> **For:** `~/tmcode/goinfer` (+ one small aikit kernel). Follows the shape of
> `docs/completed/task-gpu-f16-kv.md` — the GPU cache already got its precision knob
> (`--kv f16`, landed 2026-06-10); this is the CPU-side counterpart, and it goes
> further (int8 = 4×, not 2×) because on CPU integer kernels beat software f16.
> Maps to the current literature: per-token KV quantization (KIVI-family / the
> "watch TurboQuant" roadmap item), applied with goinfer's existing
> argmax+cosine quant-gate discipline. **Lossy ⇒ opt-in knob; the f32 default
> stays bit-exact.**

## Problem

`decoder/kvcache.go` stores K and V as `[][]float32` per layer, appended
`[pos*kvDim]`. Everything downstream reads f32:

- decode: `attendQuery` (attention.go:133) — scalar f64-accum dots over the
  whole stored history, per head, per token
- batched prefill: `attendBatchedHeads` — copies cache slices into per-head
  scratch (`attnBatchBufs`) then `MatmulBT`
- sessions: `kvsnapshot.go` persists the cache raw f32 to `.giw-kv`

Cost per token (K+V, all layers, f32):

| model | geometry | KV bytes/token | 16k ctx | 32k ctx |
|---|---|---|---|---|
| Qwen2.5-0.5B | 24L × 2KV × 64 | 24 KB | 393 MB | 786 MB |
| Qwen2.5-1.5B | 28L × 2KV × 128 | 57 KB | 940 MB | 1.88 GB |
| Qwen2.5-7B | 28L × 4KV × 128 | 115 KB | 1.88 GB | 3.76 GB |

Three felt pains, in order:

1. **Heap.** Weights are mmap'd/aliased, so at long context the KV cache *is*
   the heap — the README's "<100 MB heap" story dies at ~4k context even on the
   0.5B demo (24 KB/token × 4k ≈ 100 MB by itself).
2. **Session disk + restore I/O.** A 4k-token warm session is 230 MB on disk
   for the 1.5B (470 MB for the 7B), × `--kv-sessions` (default 4). Agent loops
   with fat system prompts (the `/v1/messages` + Claude Code story) hit this
   directly.
3. **Long-context decode bandwidth.** Every decode token re-reads the whole
   cache: at 16k on the 7B that's 1.88 GB/token of KV traffic on top of the
   ~4 GB weight stream. Quartering KV bytes is a real tok/s lever where context
   is long relative to model size.

## Why int8, not f16, on CPU

The GPU chose f16 because pack2x16float is free in WGSL. On CPU Go has no
hardware f16 — every read would be a software unpack with no kernel support.
Meanwhile aikit/linalg already ships the entire int8 toolchain, SIMD'd on both
arches (NEON SDOT + AVX2):

- `QuantizeRowInt8` / `DequantizeRowInt8` (per-row symmetric maxabs/127)
- `DotI8` (quant_export.go:10) — the integer dot the score loop needs
- `MatmulBTW8A8` — f32 activation × int8 rows with per-row scales: exactly the
  shape of "query × quantized keys" if ever needed for prefill
- precedent: `ann.FlatI8` measured **Δ0 recall vs f32** on real embeddings, and
  the int8 full-model gate (Qwen3.6 Gate 2) holds argmax 92.5% / cosine 0.9986
  — KV-only int8 is strictly gentler than quantizing every weight.

int8 is smaller than f16 (4× vs 2×), *faster* to read (¼ bandwidth + SDOT
beats scalar f64 dots), and reuses shipped, fuzz-tested kernels. f16-on-CPU is
dominated on every axis; skip it.

## Design

### Quantization granularity

Per **(position, KV-head)** sub-row, symmetric int8: each appended K (and V)
of `kvDim` floats is quantized as `nKV` independent rows of `headDim`, one
scale each. Rationale:

- `headDim` is 64/128 — multiple of 16, so `DotI8`'s SDOT/AVX2 paths apply
  per head slice with no tail handling.
- Per-position quantization keeps positions independent ⇒ `TruncateTo`
  (speculative rollback + prefix-reuse rewind) stays a reslice, untouched.
- Per-head scales bound RoPE'd-key outlier channels far better than one scale
  per kvDim row; this is the cheap version of KIVI's observation. If int8
  per-head passes the gate (expected), full per-channel key quantization is
  noise we don't need until int4.
- Scale overhead: `2 (K,V) × layers × nKV × 4 B/token` — 0.9 KB/token on the
  7B, <1% of the int8 payload. (f16 scales later if anyone cares.)

### Storage

```go
type kvQuant uint8 // KVF32 (default) | KVI8

// inside KVCache, parallel to keys/vals — used iff quant == KVI8:
keysQ, valsQ     [][]int8    // per layer, [pos*kvDim]
keyScale, valScale [][]float32 // per layer, [pos*nKVperLayer]
```

f32 and int8 paths stay separate fields + a branch in Append/read — **no
interface indirection on the hot path**. Gemma 4's per-layer widths and
KV-shared zero-length layers already force per-layer stride derivation
(`TruncateTo` does `len/pos`); the int8 arrays follow the identical rule, with
scales strided the same way.

### Read paths

- **Decode (`attendQuery`)**: quantize the query per head once per step
  (`QuantizeRowInt8` on `hd` floats — trivial), then
  `score = qScale·kScale[s,h]·DotI8(qHead, kHead)`. The f64-accum scalar loop
  becomes an SDOT dot — the one place decode attention gets *faster*, not just
  smaller. The V-side weighted sum dequantizes inline:
  `acc += w·vScale[s,h]·float32(v[d])` — one new fused kernel in aikit
  (`AxpyI8`: f32 += scalar × int8-row) keeps it SIMD; scalar fallback is fine
  to ship first.
- **Batched prefill (`attendBatchedHeads`)**: it already materializes per-head
  K/V scratch copies (`attnBatchBufs`). Swap the copy for
  `DequantizeRowInt8`-into-scratch — **zero changes to the matmul kernels**,
  prefill numerics shift only by the quant rounding already covered by the gate.
- **MoE families**: excluded in v1. They route attention through the acc64
  batched kernel *specifically* for bit-stable expert routing across
  prefill/decode; quantized KV would reopen that. `arch.MoE != nil ⇒ force
  KVF32` (same shape as the hybrid-cache exclusions).

### Knob plumbing

`Options.KVQuant` → `Model.NewCache` → serve `--kv-quant i8|f32` (default
f32), mirroring the GPU `--kv f16|f32` knob. Default path byte-identical:
`TestDecodeParity` and every existing golden stay bit-exact.

### Sessions (`.giw-kv` v2)

Bump the snapshot version (magic+version guard already rejects unknown
versions loudly — the format policy explicitly allows this pre-1.0). Persist
the int8 payload + scales when the cache is int8; an f32 cache snapshots as
today. A restored int8 session is bit-identical to the int8 cache that wrote
it, so prefix-reuse exactness *within the chosen precision* is preserved.
4× smaller files, ~4× faster restore.

## Increments

### ✅ DONE — per-head int8 Increment 1 SHIPPED (commit d389c5a)

Per-(position,KV-head) symmetric int8, opt-in `Options.KVQuant=="i8"` (default
f32 bit-exact), decode via DotI8 + inline V dequant, for global + ring layers.
Excludes MoE/gemma4/qwen3_5_moe; int8 forces sequential prefill (batched int8
prefill is a follow-up). **Gate (corrected): teacher-forced per-step argmax 87–92%
/ avg logit cosine 0.993 on gemma-3-4b-it, coherent generation** — in line with
the shipped full-int8-WEIGHTS precedent (92.5% argmax). The opt-in lossy KV's
gate is argmax/cosine, NOT the 0.999 logit-cosine bar (unreachable over 34 layers
and the wrong metric).

> **Lesson (a real detour):** a first pass reported per-head int8 "insufficient"
> (cosine 0.73, argmax flips) and concluded per-CHANNEL K (KIVI) + chunked storage
> was needed. **That was a TEST BUG** — the gate fed *foreign token-ids* (a Qwen
> prompt) to gemma's tokenizer, i.e. garbage input where argmax is a coin-flip and
> cosine is noise. On a *coherently tokenized* prompt, per-head int8 is fine.
> Block size barely moves it (per-head 0.9938 vs per-block-16 0.9935 cosine over
> 153 tokens), so per-head (simplest, ~4×, best cosine) is the right granularity —
> **no per-channel KIVI / chunked storage needed.** Always tokenize gate prompts
> with the model's own tokenizer.

### Increment 1 — storage + decode path (the core)
- `KVCache` int8 fields, `Append` quantize-on-write, `TruncateTo`/`Pos`
  untouched semantics; `attendQuery` int8 branch (DotI8 + inline V dequant).
- aikit: nothing required yet (`DotI8` ships); `AxpyI8` optional follow-up.
- [ ] **Gate (the f16-KV shape, run long):** int8-KV vs f32-KV greedy decode
      over ≥8k-key contexts on a real checkpoint (1.5B + the Gemma E2B for
      sliding-window coverage): **argmax preserved per step** (near-tie rule:
      flips only within the established 3%-of-range guard), **logit cosine
      ≥ 0.999** (KV-only should comfortably beat the 0.99 full-model-int8 bar;
      tighten the bar to what's measured). Default f32: bit-exact, all
      existing goldens green.
- [ ] **Memory gate:** measured heap at 16k ctx, 1.5B: ~940 MB → **~245 MB**.

### ✅ Increment 2 — batched prefill + knob — DONE
- Batched int8 prefill: the int8 cache is dequantized into the caller's f32
  scratch (`dequantGlobalLayer` for global; `batchReadLocal` int8 branch for
  rings — ring history dequant + the K new rows round-tripped quantize→dequant so
  prefill attends the SAME values decode will) — `attendBatchedHeads`'s f32
  matmul is unchanged (f32 path bit-exact). Dropped the Inc-1 `canBatchN` int8
  forcing, so int8 prefills batched, not sequentially.
- `Options.KVQuant` plumbed to serve `--kv-quant f32|i8`.
- **Gates (green):** batched int8 prefill vs batched f32 on gemma-3-4b-it —
  argmax 92.5% / cosine 0.9934. **TTFT regression 2.3%** (int8/f32 = 1.023,
  meets ≤5%) — after pre-sizing the int8 arrays + quantizing in place to kill
  the per-token allocation churn (a naive append-with-make was 18%; the fix also
  speeds the decode Append path). Spec-decode rollback under int8:
  `TestKVI8_truncateReappend` — TruncateTo+re-append leaves the int8 cache
  (global + ring) byte-identical to a fresh one. f32 default bit-exact.

### ✅ Increment 3 — `.giw-kv` v2 — DONE (merged ring + int8)
- One version bump (v1→v2) covering BOTH KV programs: ring windowed persistence ×
  {f32 | int8}. Per layer, in cache-structure order (loader rebuilds rings + quant
  from the model, no per-layer tags): ring layers store `count` + only the **live
  window** (`min(count,W)` rows, unwrapped — restore re-wraps into slots, so short
  sessions aren't bloated to the full W); global layers append-forever. Header
  gained `headDim` + `quant`; geometry guard checks both. sessionLRU untouched
  (`Snapshot` no longer refuses ring/int8; still refuses qwen3_5_moe — its DeltaNet
  state isn't serialized).
- **Gate (green):** `TestKVI8_snapshotRoundtrip` on gemma-3-4b-it int8 (int8 rings
  + int8 global): snapshot→restore→next-token logits **bit-identical** to
  never-snapshotted. Compact serialization cut the blob ~9.6× (62 MB → 6.5 MB on a
  93-token session). **v1 policy:** the version guard rejects v1 blobs loudly
  (`SnapshotError`) → caller cold-prefills; snapshots are a regenerable cache, so a
  bump just re-prefills.

### Increment 4 (deferred, separately gated) — int4 KV, group=32
- `QuantizeGroupInt4Row` / W4A8 kernels exist; 6.6–7× vs f32 (scales weigh
  more at int4). **This is where per-channel key treatment (KIVI) likely
  becomes necessary** — keys are the outlier-prone side post-RoPE. Trigger:
  int8 ships clean AND a real >32k or multi-session RAM complaint exists.
  Gate bar: the looser full-int4 shape (argmax ≥ 90%, cosine ≥ 0.994).

### Non-goals (assessed, rejected)
- **CPU f16 KV** — dominated by int8 on CPU (see above).
- **XQuant-style K/V rematerialization** — trades memory for recompute; CPU
  decode is already compute/weight-stream-bound. Wrong direction here.
- **Attention-score token eviction (AhaKV-class)** — quality-risky, breaks
  exact prefix-reuse semantics (evicted positions ≠ rewindable), and
  sliding-window eviction already exists for the families that want it
  (Gemma local layers, Mellum2). Research track at best.
- **GPU cache** — already has f16; GPU int8 KV is a separate doc if VRAM
  pressure returns.

## Estimated impact

| metric | today (f32) | int8 (Inc 1–3) | int4 (Inc 4) |
|---|---|---|---|
| KV RAM, 1.5B @ 16k | 940 MB | **245 MB (3.8×)** | ~140 MB (6.6×) |
| KV RAM, 7B @ 32k | 3.76 GB | **0.96 GB** | ~0.57 GB |
| heap headline, 0.5B demo @ 32k ctx | ~790 MB | **~200 MB** | ~120 MB |
| `.giw-kv` 4k-token session, 1.5B | 230 MB | **58 MB** | ~33 MB |
| decode tok/s @ short ctx (≤1k) | — | **neutral** (weight-stream-bound; GPU f16 measured 0.99× here — same expectation) | neutral |
| decode tok/s @ 16k, 7B | — | **~1.2–1.4×** (KV traffic 1.88→0.47 GB/token vs ~4 GB weights; plus SDOT replacing scalar f64 scores) | ~1.3–1.5× |
| decode tok/s @ 16–32k, small models (0.5–1.5B) | — | **~1.5–2.5×** (KV read rivals or exceeds weight read here — the demo/agent regime) | ~2–3× |
| TTFT | — | ~neutral (≤5% gate) | ~neutral |
| quality | bit-exact | argmax-preserved + cosine ≥0.999 expected (KV-only ≪ full-model int8, which already holds 0.9986) | needs the int4 gate; KIVI-style keys if it misses |
| max context in fixed RAM | 1× | **~4×** | ~7× |

Speedup estimates are bandwidth arithmetic + the scalar→SDOT swap, stated as
ranges; Increment 1's bench makes them measured numbers before any README
claim (house rule: every cell with provenance).

## Effort

| increment | size | risk |
|---|---|---|
| 1 — storage + decode + gates | 3–4 days | low: kernels exist, gate shape exists |
| 2 — prefill + knob + spec-decode tests | 2–3 days | medium: TruncateTo×quant interplay |
| 3 — snapshot v2 | 1–2 days | low: versioning policy already allows it |
| 4 — int4 | ~1 wk | medium-high: likely needs per-channel keys |

**Recommendation: ship Increments 1–3 as one release feature** ("4× KV memory,
4× smaller warm sessions, faster long-context decode — opt-in, default
bit-exact"). It's the same story shape as the GPU f16 landing, lands almost
entirely on existing aikit kernels, and directly serves the two marquee
use-cases: the single-file demo's memory claim and Claude-Code-against-goinfer
agent sessions. Hold int4 for a real trigger, per house style.
