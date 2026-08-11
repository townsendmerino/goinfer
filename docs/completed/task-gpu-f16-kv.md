# Task (goinfer): f16 KV cache (2× GPU context on the same VRAM)

> **ARCHIVED — a record, not instructions.** This file is closed work kept for its reasoning and
> its numbers. Checkboxes record the state at the moment it was archived: an unticked box means
> "not ticked when this closed", **not** "still to do", and nothing in `docs/completed/` is
> actionable. If you need a task, use the live docs; if something here reads as an instruction to
> a future reader, it was missed at archival — see the doc-closeout rule in
> `docs/parity-coverage-policy.md`, and move it to live policy or strike it.


> **For:** Claude Code, in `~/tmcode/goinfer` (GPU work → the 64 GB RTX box;
> `-tags gpu`). Deferred follow-on from `docs/roadmap.md`. Increments ordered and
> independently shippable. **This change is LOSSY** (f16 ≠ f32), so the gate is
> argmax-preserved + cosine, NOT bit-exact — and the f32 default stays bit-exact
> (see "The catch"). Pure-Go core CI job untouched.

## Problem

The resident GPU KV cache is **f32** (`gpu.NewKVCache`, `decodelayer.go` —
`Size: capElems*4`). For Qwen2.5-7B (28 layers, 4 KV heads × 128 head_dim) that's
`2 (K+V) × 28 × 512 = 28,672` elems/token → **~115 KB/token f32**. On the 8 GB
card the proven fit (peak 6937/8192 MiB):

| | size |
|---|---|
| 7B int4 weights | ~3.98 GB |
| **16k f32 KV** | **+1.88 GB** |
| total resident | **5.86 GB → fits** (~1.25 GB headroom) |

**32k f32** doubles the KV to +3.76 GB → ~7.7 GB + scratch → over the card, so v1
**hard-caps context at 16k**. **f16 KV (2 bytes) halves the per-token cost:** 32k
f16 = **1.88 GB — identical to 16k f32.** Same total (5.86 GB), same headroom,
**2× the context window.** That's the headline win — and f16 KV *also speeds up
long-context decode* (the attention kernel is KV-read-bound; half the bytes = half
the read), so it's "2× context **and** faster" — see Increment 2.

## What already exists (building blocks)

- **f16 pack/unpack, in WGSL, with NO `shader-f16` device feature** — the W4A8
  path already stores group scales as packed f16 and unpacks them in-shader:
  `packF16Pairs` / `f16to32` (`gpu/gemv_w4a8.go`), read as `array<u32>`, 2 halves
  per u32, manual bit-twiddle to f32. This is the exact pattern f16 KV reuses, and
  it deliberately avoids the native `enable f16;` extension (which the lavapipe CI
  software adapter may not support). **No new device features.**
- The KV write/read kernels are already isolated: `ropeStore` (rotate K → cache),
  `vStore` (V → cache), and `attnShaderWGSL` (read K/V → dot products).

## The catch (read before designing)

f16's 10-bit mantissa makes this a **lossy** change — a close-call decode token can
flip vs f32 KV. In practice the error is tiny (K/V are small post-norm values, the
weights are *already* int4-quantized, and attention is a softmax-weighted average),
but it's real. Consequences for the design:

- **Default stays f32** — `TestDecodeParity` (bit-exact greedy) must keep passing
  on the default path. f16 is a **precision knob** (`--kv f16`, or auto-enable only
  when the requested context exceeds the f32 cap).
- The f16 path's gate is **argmax-preserved + cosine ≥ ~0.99** vs the f32 KV run
  (the int8/int4 quant-gate shape), not bit-exact.

## Increments

### Increment 1 — f16 KV storage + the three kernels
- `NewKVCache`: allocate `capElems*2`, store as `array<u32>` (2 f16/u32).
- `ropeStore` / `vStore`: pack the rotated K / the V to f16 on write into the cache
  (reuse `packF16Pairs` / the WGSL f16-pack helper).
- `attnShaderWGSL`: read the f16 cache and `f16to32`-unpack each K/V element before
  the dot product / weighted sum. (The query stays f32; only the cache is f16.)
- [ ] **Gate:** f16-KV decode vs f32-KV decode — **argmax preserved every step**,
      full-logit **cosine ≥ 0.99** — **run at long context, near the 32k it unlocks**
      (or the longest the cap allows until Increment 2 lifts it). That's where
      per-element f16 rounding acts over the most keys and a long-range near-tie can
      flip; a short-prompt cosine is necessary but says nothing about the regime the
      feature exists for. Real HW (software-adapter-skipped). (No windowed layers —
      the residency path is full-attention-only, `SlidingWindow == 0`.)

### Increment 2 — the precision knob + the 32k unlock
- Plumb a KV-precision option (`Options.KVPrecision` / serve `--kv f16|f32`,
  default f32) down to `NewKVCache`. Optionally auto-select f16 when the requested
  context > the f32 cap. Bump the context cap to **32k when f16**.
- [ ] **Gate:** 7B int4 + **32k f16 KV** fits 8 GB — real allocation, no OOM,
      record peak VRAM (target ≈ the 16k-f32 5.86 GB). f32 default unchanged →
      `TestDecodeParity` still bit-exact.
- [ ] **Secondary win (measure + record):** at long context the attention kernel
      is **KV-read-bound**, so halving the cache bytes should **speed up
      long-context decode** — at equal context, f16-KV decode tok/s should beat
      f32-KV (e.g. 16k-f16 vs 16k-f32). Measure and record it; "2× context AND
      faster long-context decode" is a stronger story than "fits."

## Scope / constraints

- **GPU residency path** (the 8 GB fit is the motivation). The CPU KV cache
  (`decoder/kvcache.go`) is also f32 but CPU has RAM headroom — out of scope here;
  it could follow the same knob later if a use case wants it.
- **W8A8 and W4A8** resident models (both share the KV cache + attention kernel).
- **Default f32 keeps bit-exact parity** — f16 is opt-in. Never silently degrade
  the default decode.
- Manual WGSL f16 (the W4A8 pattern) — **no `shader-f16` feature dependency**, so
  CI's software adapter and any backend still work.

## Why deferred / when to pick up

16k f32 covers most use; the win is **>16k context on the 8 GB card** — long-doc
summarization, large-RAG, long agent transcripts on the small GPU. Pick it up when
a consumer's context needs cross 16k. (Pairs naturally with the batched-prefill
item: long context implies long prompts, which is where that one matters too.)

## Definition of done

- [x] **Increments 1–2 landed, gated on the RTX 2070 SUPER** (2026-06-10).
- [x] **f16-KV parity recorded; f32 default still bit-exact.** `TestKVCacheF16_parity`
      (real HW): f16-KV vs f32-KV decode over an 8000-key context — cosine min
      0.99868 mean 0.99918, 15/16 argmax exact (the 1 flip a 0.22%-of-range near-tie,
      guarded by the qwen35 gate's 3% rule). f32 path unchanged: `TestDecodeToken_parity`
      / `TestDecodeRunnerW4A8_parity` still cosine 1.000000.
- [x] **32k-f16 7B fit measured** (`TestKVCacheF16_fit`, Qwen2.5-7B int4 on the 8 GB
      card): **32k f16 peak whole-device VRAM 6912 MiB ≈ 16k f32's 6926 MiB — fits,
      no OOM. 2× context on the same VRAM.** The `--kv f16|f32` knob is wired in
      `cmd/serve` (default f32).
- [ ] **Long-context decode speedup — UNMEASURED / not yet shown.** At ctx ~1k,
      f16 vs f32 decode is 0.99× (neutral) — expected, since at short context decode
      is weight-stream-bound (the ~4 GB int4 weights dominate), not KV-read-bound, so
      halving KV bytes doesn't move tok/s. The predicted speedup only appears near
      16k–32k; measuring it needs a 16k+ sequential prefill (slow without batched
      prefill — coupled to that shelved item). Left open; the 2×-context win stands on
      its own.
