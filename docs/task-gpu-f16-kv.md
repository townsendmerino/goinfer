# Task (goinfer): f16 KV cache (2× GPU context on the same VRAM)

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
**2× the context window.** That is the entire win.

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
- [ ] **Gate:** on a fixed prompt, f16-KV decode vs f32-KV decode — **argmax
      preserved every step**, full-logit **cosine ≥ 0.99**. Real HW (software-
      adapter-skipped). Confirm both local (windowed) and global layers.

### Increment 2 — the precision knob + the 32k unlock
- Plumb a KV-precision option (`Options.KVPrecision` / serve `--kv f16|f32`,
  default f32) down to `NewKVCache`. Optionally auto-select f16 when the requested
  context > the f32 cap. Bump the context cap to **32k when f16**.
- [ ] **Gate:** 7B int4 + **32k f16 KV** fits 8 GB — real allocation, no OOM,
      record peak VRAM (target ≈ the 16k-f32 5.86 GB). f32 default unchanged →
      `TestDecodeParity` still bit-exact.

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

- [ ] Increments 1–2 landed, each gated on real hardware.
- [ ] f16-KV cosine/argmax parity recorded; **f32 default still bit-exact**
      (`TestDecodeParity` untouched).
- [ ] 32k-f16 7B fit measured (peak VRAM) + noted in the GPU campaign doc /
      CHANGELOG; the `--kv` knob documented in the serve README.
