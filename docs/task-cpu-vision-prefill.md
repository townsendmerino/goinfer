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

## Increment 1 — int8 (W8A8) vision tower  ← the main CPU lever

Quantize the tower's projection + FFN weights to int8 and run them through
aikit's `linalg.MatmulBTW8A8` (the same W8A8 kernel the decoder's int8 path
already uses), instead of f32 `MatmulBT`. The encoder is matmul-bound and that
~1.7 TMAC of projections/FFN is the bulk of the 190 s, so int8 is the biggest
single CPU win (the decoder sees ~2–4× on equivalent matmul-bound work).

- Load the tower at a chosen quant (a `--vision-quant int8` knob, mirroring
  `--quant`); store int8 weights + per-row scales in `Encoder`.
- Swap the projection/FFN `MatmulBT` calls for `MatmulBTW8A8`; keep LayerNorm,
  softmax, GELU, and the patch-embed conv in f32.
- Decide whether QKᵀ/scores·V also go int8 (activations are dynamic) or stay f32.
- **Gate:** `TestSiglipEncoder_parity` / `TestProjector_parity` re-pinned at an
  int8 tolerance (cosine ≥ 0.999, like the W8A8 decode gate); end-to-end
  `gemma3_vl_tiny` image parity still argmax-exact. Record the new
  `encoder.Forward` time in `docs/benchmarks.md`.

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

## Non-goals / explicitly deferred

- **GPU residency for the tower** — the real answer for latency, but the WebGPU
  backend is matmul-substitution only; a resident ViT is its own project
  (tracked with the GPU work, not here). Out of scope for the pure-Go CPU path.
- **Reducing patch count** — Gemma 3 fixes 896×896 → 4096 patches; pan-and-scan
  is already out of scope. Can't trim without diverging from HF.
- **Async/background encode** — a UX patch (return a job id, poll), not a speedup;
  consider only if the latency stays user-facing after Increments 1–2.

## Verification

- `go test ./vision/` green (encoder + projector parity at the int8 tolerance).
- `go test ./decoder/ -run TestGemma3VL` (end-to-end image parity) argmax-exact.
- Re-time `encoder.Forward` on the real `~/models/gemma-3-4b-it` tower; land the
  before/after in `docs/benchmarks.md`.
