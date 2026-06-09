# Task (goinfer): f16 group scales in the `.giw` (smaller W4A8 files)

> **For:** Claude Code, in `~/tmcode/goinfer`. Deferred follow-on from
> `docs/roadmap.md`. **This is a FILE-SIZE win only — it does NOT speed decode**
> (see "Rationale, corrected"). It is **lossy for the CPU W4A8 path** (GPU is
> unaffected), so it needs a cosine re-gate, and it's a **versioned-format change**
> needing a bump + back-compat read. Lowest-value of the open follow-ons.

## Problem

The `.giw` serialized-weights format stores int4 (W4A8, weightMat kind 3) **group
scales as f32** — `serialize.go`: `case 3: w.f32(m.q4s); w.bytesField(m.q4)`. Per
row, per group of 32 weights that's **16 bytes of nibbles + a 4-byte f32 scale** =
20 B/group, so the scales are **~20% of the W4A8 byte stream**. Storing them
**f16** (2 B) → 18 B/group → **~10% smaller `.giw` files** (a 7B int4 ≈ 4 GB →
~400 MB saved). Pure distribution/download win.

## Rationale, corrected (verify-don't-trust)

The roadmap line says *"the kernel is ALU-bound so it wouldn't speed decode."*
**The conclusion is right; the stated reason is wrong** — `gpu/gemv_w4a8.go`'s own
roofline says the W4A8 GPU decode is **weight-bandwidth-bound**: *"the decode
roofline is weight bytes/token … this is the lever on the 4.3 ms gemv floor."*
The real reasons on-disk f16 won't move decode:
1. **The GPU buffer is already f16** (`bScales: array<u32>`, packed by
   `packF16Pairs`). The on-disk f32 is widened/re-packed to f16 at upload today, so
   the decode-time representation is *unchanged* by an on-disk format change.
2. **The CPU path widens to f32 on load** into the same `q4s []float32` the kernel
   reads — the on-disk bytes aren't the decode-time representation there either.

So the `.giw` change shrinks the *file*, not the *working set*. Decode-neutral.

## What already exists (building blocks)

- **The f16 codec** — `packF16Pairs` / `f16to32` (`gpu/gemv_w4a8.go`), already used
  to build the GPU's f16 scale buffer. Reuse for the giw writer/reader. (CPU-side
  f16↔f32 is a plain `math.Float*` round-trip; no device feature.)
- **The versioned format** — `serialize.go`: `giwMagic="GINFW"`, `giwVersion=1`;
  the reader rejects any non-matching version. weightMat kind 3 is the only one
  that carries `q4s`.
- **The scale source** — `weightmat.go`: `q4s []float32 [rows*nGroups]` from
  `QuantizeGroupInt4Row`; `Int4()` exposes it.

## The catch (read before designing)

- **GPU: bit-identical.** The GPU already rounds these scales to f16 at upload, so
  storing f16 on disk yields the *same* f16 buffer — **no GPU parity change** (as
  long as the on-disk f16 uses the same round-to-nearest as `packF16Pairs`).
- **CPU: lossy.** The CPU W4A8 decode uses full **f32** scales today; reading f16
  (widened to f32) means f16-rounded scales → a tiny numeric change. It needs the
  **cosine gate re-run** (the W4A8 quant-gate shape). Upside: it makes the CPU and
  GPU W4A8 paths use the *same* f16 scales — more consistent, not less.
- **Versioning.** Changing `q4s` to f16 is a format change. The reader currently
  **hard-rejects** other versions, so existing v1 `.giw` files would break. Need a
  **`giwVersion` bump + version-aware read** (v1 → read f32; v2 → read f16), or a
  documented re-quant.

## Increments

### Increment 1 — f16 scale codec in the giw writer/reader + version bump
- Add an `f16([]float32)` writer (pack via `packF16Pairs`-equivalent) and an
  `f16() []float32` reader (unpack via `f16to32`) to the giw codec.
- Kind 3 writes `q4s` as f16. Bump `giwVersion` to 2.
- Make the reader **version-aware**: v1 reads `q4s` as f32 (existing files keep
  working), v2 reads f16. (Don't hard-reject v1.)
- [ ] **Gate:** round-trip a real W4A8 model → v2 file is ~10% smaller; reading a
      banked **v1** file still loads (back-compat); v2 `q4s` round-trips to the
      f16-rounded values.

### Increment 2 — re-gate parity
- [ ] **GPU:** assert **bit-identical** decode vs the f32-on-disk path (same f16
      buffer either way) — a no-op if the rounding matches; confirm it does.
- [ ] **CPU:** W4A8 cosine gate with f16-on-disk scales — **argmax preserved,
      cosine ≥ ~0.99** vs the f32-scale reference. Record it.

## Scope / constraints

- **W4A8 (kind 3) only.** q8 (kind 2) and f32 (kind 1) weightMats are untouched.
- **Back-compat is mandatory** — v1 `.giw` files must keep loading (version-aware
  read), or every shipped int4 model breaks on upgrade.
- Same round-to-nearest f16 as `packF16Pairs`, so the GPU buffer is unchanged.

## Why deferred / when to pick up

**Lowest-value of the open follow-ons:** ~10% smaller files, **zero** decode or
fit improvement, and it adds a format version + a lossy CPU re-gate. Pick it up
only when **`.giw` distribution size is a real cost** — shipping a model zoo,
bandwidth-constrained downloads, or bundling int4 weights into a release artifact.
Until then the f32 scales are simpler and bit-exact on CPU. Strictly behind the
batched-prefill and f16-KV items, which themselves are behind release-gating work.

## Definition of done

- [ ] Increments 1–2 landed; v2 writer, version-aware reader, both gates green.
- [ ] File-size delta measured on a real W4A8 model + recorded; CPU cosine recorded;
      GPU bit-identical confirmed.
- [ ] Back-compat with banked v1 `.giw` files verified (a v1 fixture in the test).
