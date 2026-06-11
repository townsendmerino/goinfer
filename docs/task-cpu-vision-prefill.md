# Task (goinfer): speed up the CPU vision-tower prefill

> **For:** Claude Code, in `~/mycode/goinfer` (pure-Go CPU path; no `-tags gpu`).
> Follow-on from multimodal P4 (`docs/multimodal.md`). Increments ordered and
> independently shippable. The SigLIP attention vectorization (Increment 0) has
> **already landed**; the rest is open. Numerics are parity-gated (the encoder /
> projector goldens under `vision/`), so every increment's gate is
> argmax-preserved + cosine ≥ the existing thresholds — **never a silent
> precision regression**.

## Problem

A Gemma 3 vision turn runs the image through the SigLIP tower
(`vision.Encoder.Forward`) before the text decoder sees it. Gemma 3 fixes the
input at **896×896, patch 14 → a 64×64 grid = 4096 patches**, through a
**27-layer** ViT (hidden 1152, intermediate ~4304, 16 heads × 72 head-dim). On
this 16-core CPU box, measured against the real `google/gemma-3-4b-it` tower
(f32):

| stage | time |
|---|---|
| `LoadEncoder` | 1.5 s |
| **`encoder.Forward`** | **~190 s** (after Increment 0; was >400 s) |
| `projector.Forward` | 14 ms |

So a single image is **~3 minutes of prefill** before any text is generated.
That's correct (the encoder/projector match the HF goldens at cosine 1.0) but
makes `/v1/messages`+`/v1/chat/completions` image requests — and the planned
`demo/agent` image drop — slow enough to need a 400 s+ client timeout and a
"processing image…" spinner. This is the "ViT prefill cost on CPU, acceptable
v1" caveat in `docs/multimodal.md` made concrete.

### Where the time goes (per image, ~2.7 TMAC total over 27 layers)

- **FFN + q/k/v/o projections: ~1.7 TMAC** — already on aikit's multi-threaded
  `linalg.MatmulBT` (f32 SIMD, forks across output columns). This is the floor.
- **Attention QKᵀ + scores·V: ~1.0 TMAC** — QKᵀ on the f64-accumulating
  `MatmulBTAcc64` (for parity), scores·V on `MatmulBT`.

`MatmulBT` is already parallel across cores, so ~190 s is close to the honest
f32 floor on this hardware — the win has to come from **fewer/cheaper MACs**
(quantization) or **not redoing the work** (feature caching), not from more
threading.

## Increment 0 — vectorize the SigLIP attention ✅ DONE

Replaced the scalar per-(head,query,key) triple-loop in `vision/encoder.go`
`attention` with per-head SIMD: QKᵀ via `linalg.MatmulBTAcc64` (f64-accumulating,
so the parity golden is unchanged), row-softmax, then scores·V via
`MatmulBT(scores, Vᵀ)`. **>400 s → ~190 s**, `TestSiglipEncoder_parity` still
cosine 1.0 (max abs diff 1.19e-6). Mirrors the text path's `attendBatchedHeads`.

## Increment 1 — int8 (W8A8) vision tower — ⚠️ EVALUATED, NO CPU WIN (a wash on AVX2)

**Built and measured; it does NOT speed up the encoder on a non-VNNI CPU.** The
tower's projection + FFN weights quantize to int8 (`--vision-quant int8`, the
`qmat` type + `linalg.MatmulBTW8A8`); parity holds (encoder golden cosine
**0.99996**, f32 stays 1.0). But on the real gemma-3-4b-it tower:

| | encoder.Forward |
|---|---|
| f32 | **3m11s** |
| int8 W8A8 | **3m18s** (slightly *slower*) |

**Why the decoder's int8 win doesn't transfer.** The decoder's ~2–4× is on
**bandwidth-bound M=1 decode** — int8 weights are half the bytes to stream from
RAM, and decode is memory-bound. The vision prefill is the opposite regime:
**compute-bound, M=4096**. On this **AVX2-only** CPU there is no int8 SIMD
dot-product (AVX512-VNNI), so the integer matmul is no faster than f32 SIMD FMA —
and W8A8 *adds* work (it dynamically re-quantizes the 4096-row activation matrix
every matmul). Net: a wash, and lossier, so **f32 is the default**.

The int8 path stays as an opt-in (`--vision-quant int8`) — it should win on an
**AVX512-VNNI** host (Intel Ice Lake+/Sapphire Rapids, AMD Zen4+) where the int8
dot is hardware-accelerated. The `qmat` abstraction is also the right seam for a
GPU int8 path (where the win is real).

**Conclusion: there is no big CPU-only lever.** The ~190 s f32 prefill is close
to the compute floor on AVX2. The real fix is **GPU residency for the tower**
(below). The remaining CPU options are marginal (Increment 3) or about *not
redoing* the work (Increment 2, feature cache).

## Increment 2 — projected-feature cache (multi-turn / repeated image)

In a chat where the same image recurs across turns (the `demo/agent` case, or a
follow-up question about one image), re-encoding it every turn is pure waste.
Cache the projector output (`[MMTokens*textHidden]`) keyed by a content hash of
the decoded image bytes (+ the tower fingerprint), bounded LRU. A cache hit
turns a ~3-min turn into a normal text turn.

- Hash the raw image bytes (sha256) → key; small LRU on the `loadedModel` /
  agent `Session`.
- **Gate:** a second request with the same image skips `encoder.Forward`
  (assert via timing or a counter); output identical to the uncached path.

## Increment 3 — f32 QKᵀ (small, optional)

Drop the `MatmulBTAcc64` on QKᵀ to plain `MatmulBT` (f32) if the parity golden
still holds — f64-accumulate is ~2–4× slower per MAC and the attention block is
~0.5 TMAC of it. Modest (the f32 FFN floor dominates), low effort, only worth
doing if Increment 1 doesn't already subsume the attention path.

- **Gate:** `TestSiglipEncoder_parity` cosine still ≥ 0.9999; if it drops, keep
  Acc64.

## The real fix (now the recommended next step): GPU tower

With int8 a wash on AVX2 (Increment 1), there is **no big CPU-only win** — the
~190 s f32 prefill is near the compute floor. The latency fix is to run the SigLIP
tower on the **GPU** (`-tags gpu` WebGPU backend), where the ~5 TFLOP forward is
~20 ms instead of minutes. The WebGPU backend is currently matmul-substitution
for the decoder; hosting the ViT (patch-embed + 27 transformer blocks) on it is
its own project, tracked with the GPU work. This is the path that makes a vision
turn feel like the hosted apps.

## Non-goals / explicitly deferred
- **Reducing patch count** — Gemma 3 fixes 896×896 → 4096 patches; pan-and-scan
  is already out of scope. Can't trim without diverging from HF.
- **Async/background encode** — a UX patch (return a job id, poll), not a speedup;
  consider only if the latency stays user-facing after Increments 1–2.

## Verification

- `go test ./vision/` green (encoder + projector parity at the int8 tolerance).
- `go test ./decoder/ -run TestGemma3VL` (end-to-end image parity) argmax-exact.
- Re-time `encoder.Forward` on the real `~/models/gemma-3-4b-it` tower; land the
  before/after in `docs/benchmarks.md`.
