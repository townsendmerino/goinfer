# Image resend on a Qwen3.5 hybrid: what the tower cache buys, and where the time actually is (2026-10-06, exploratory)

**Exploratory, one run per cell, never to be quoted as a result.** Machine nobara-pc (Ryzen 7 3700X, `-backend cpu`, the GPU idle at 468 MiB, load 1.6 at the
start), `serve` built from the working tree on top of 37e7db48 plus the uncommitted feature cache, `~/models/qwen3.5-0.8b` (safetensors), `-ctx 4096`, temperature 0,
`max_tokens` 24, streamed. Two synthetic 1024x640 images (`shot1`, `shot2`, ~10.7 KB PNGs of green glyph-like blocks), the first image-in-prompt run of a
fresh process. Time to first content token:

| request | TTFT | server log |
|---|---|---|
| F1 new image (shot1) | 38.27 s | `encoded a 10687-byte image in 3.374s` |
| F2 the same bytes, a new question | 34.48 s | `reused the cached encode of this image` |
| F3 a different image (shot2) | 37.71 s | `encoded a 10669-byte image in 3.046s` |
| F4 the same bytes as F3, a new question | 34.57 s | `reused the cached encode of this image` |

For comparison, the cold-user run on v0.20.0 (`cold-user-2026-10-05-nobara-pc.md`, section F) read 41.95 / 37.88 / 37.78 s for the same three request shapes on the same
model, with `prefill_reused_tokens=0` on the resend. That run used different images and a v0.20.0 binary, so only the paired contrasts inside this run are comparable.

## What this says

- **The tower is about 3 s of the ~38 s.** The two encodes took 3.37 s and 3.05 s, and the cache took 3.8 s and 3.1 s off the resends. That is the whole effect of the
  cache on this model and hardware: a resend is about 9% faster to first token.
- **The other ~34 s is the prefill of the image turn on the CPU.** `prefillLogitsQwen35VL` (`decoder/generate_vl_qwen35.go`) runs the ~670 tokens through the model one at a
  time, because the Gated-DeltaNet recurrent state has to see each token in order and there is no batched hybrid prefill (`canBatchN` is false for the family). 670 tokens in
  ~34 s is ~20 tokens/s, decode speed. The GPU does not help because a recurrent family takes no resident branch for image turns (`GenerateQwenVL`: `!recurrent`), which is
  why `-backend cuda` read 37.5 s in the cold-user run.
- **Image-prefix reuse (P9a) does not apply either**, for the same reason: it is resident-only and excludes recurrent families, so `prefill_reused_tokens` stays 0.
- So the cold-user finding (resend costs the same as a new image) is explained, and a feature cache is the wrong lever for it on this family. It is the right one where the tower
  dominates and the prefill is batched (Gemma 3's SigLIP tower on the CPU, measured at 171 s in `docs/multimodal.md`, before the resident encoder), which this run did not measure.

## After P26a: the image prefill batched (same day, same machine, same harness, exploratory)

`prefillLogitsQwen35VL` now runs the batched hybrid forward (`runLayersQwen35N`, which already existed for the embedding route) over the whole prompt, with m-RoPE positions, instead of one token at a time. Same four requests, `serve` built from the
working tree, `-backend cpu`, load average 5.0 at the start (my own gate tests had just finished, so if anything this under-reads the change):

| request | TTFT before | TTFT after | server log |
|---|---|---|---|
| F1 new image (shot1) | 38.27 s | **13.81 s** | `encoded … in 3.243s` |
| F2 the same bytes, a new question | 34.48 s | **10.67 s** | `reused the cached encode` |
| F3 a different image (shot2) | 37.71 s | **13.72 s** | `encoded … in 3.083s` |
| F4 the same bytes as F3 | 34.57 s | **10.59 s** | `reused the cached encode` |

That is 2.8x on a new image and 3.2x on a resend (the cache removes the ~3 s tower; what is left, about 10.5 s, is the prefill of ~670 tokens at ~63 tokens/s, against ~20 before). One run per cell; the replies are not byte-identical to the per-token ones (a
batched matmul reduces in a different order) and read the same. Gates, all passing: the tiny fixture's HF goldens (cosine 0.9999, argmax, 8-token continuation past the image), a new test that holds the batched path to the per-token loop over four prompt layouts
(cosine >= 0.999999, same argmax, same 8-step continuation from each path's own cache; red when the m-RoPE positions are dropped), `TestQwen35VLReal_G2` and `TestServe_qwen35Image_G4` on the real 0.8B (exact reply text against HF for three images).

## What would move the number further

A batched prefill for the hybrid family, or a snapshot of the recurrent state and KV at the end of the image block that a resend restores. Both are decoder work, which re-stales the
parity manifest, and neither is built. Queued as a performance item (`docs/queue-performance.md`, P26).
