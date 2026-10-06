# P26b: the CUDA-resident image prefill and decode for a Gated-DeltaNet hybrid (2026-10-06)

Qwen3.5's image turn ran its prefill on the CPU on every backend, because the resident path was refused for recurrent families (`GenerateQwenVL`: no resident hybrid m-RoPE prefill, and the CPU-prefill-then-`UploadKV` bridge copies only layers that have a KV, so a DeltaNet layer's state would be left zeroed). The CUDA resident already had the pieces
(a DeltaNet-aware batched prefill, `prefillDeltaNetRows`, and an m-RoPE prefill entry point); what it lacked was the layout: Qwen3.5 uses Qwen3-VL's **interleaved** m-RoPE (`decoder.mropeComponentInterleaved`), and `rope_kv_mrope_batched` only knew
Qwen2.5-VL's contiguous sections.

## What changed

- `rope_kv_mrope_batched` takes a `mode` argument: 0 is the old contiguous rule, unchanged; 1 is the interleaved rule (`d%3==1 && d<3*section[1]` height, `d%3==2 && d<3*section[2]` width, else temporal). `rope_mrope_prefill.ptx` was regenerated with the pinned NVRTC 12.9.86, which reproduced the previous PTX **byte for byte**
  first, so the diff is only this change. The pairwise twin (GLM-OCR) is a different kernel with a different argument list and is untouched.
- The resident picks the mode from the model (`Model.MRopeInterleavedResident`), refuses pairwise-with-interleaved, and builds the argument list from the kernel actually bound (`mropeTakesMode`).
- `decoder.ResidentHybridMRoPEPrefill` is a new optional marker. The CUDA resident claims it for **dense** hybrids only (`HybridMRoPEPrefill`); an MoE hybrid has no gate here and stays on the CPU. `GenerateQwenVL` lets a recurrent family into the resident m-RoPE prefill branch only for such a resident, still excludes it from the
  reuse fast path and the `UploadKV` bridge, and forgets the resident's recorded ids before the attempt (a prefill that fails partway has already overwritten the recurrent state).

## Time to first token, Qwen3.5-0.8B, `-backend cuda`, the cold-user scenario (exploratory, one run per cell, idle machine)

| request | before (cold-user, v0.20.0) | this tree |
|---|---|---|
| new image | 37.75 s | **3.61 s** (3.28 s of it the CPU vision tower; 0.3 s the prefill) |
| the same bytes, a new question | 37.50 s | **0.30 s** (tower output cached; resident prefill) |
| a different image | 37.53 s | **3.29 s** |
| resend of that | — | **0.28 s** |

(The same four requests on `-backend cpu` after the CPU batched prefill: 13.8 / 10.7 / 13.7 / 10.6 s, `docs/measurements/image-resend-2026-10-06.md`.) The tower is now the whole cost of a new image, on both backends.

## Gates, and what each one actually proves

| gate | proves |
|---|---|
| `TestRopeKVMRoPEBatched_interleavedModeMatchesCPUReference` | the kernel, in both modes, against `decoder.ApplyMRoPEForTest` on rows whose components all differ; red when the interleaved run is launched in mode 0 |
| `TestGenerateQwenVL_hybridResidentPrefillMatchesCPU` (tiny hybrid VL fixture, int4) | the resident branch ran (`ImgPrefillResident`), last-token logits cosine 0.999951 against the CPU, the 8-token continuation equal on two turns; with the WRONG layout mode launched the cosine reads 0.999552 and the test goes red |
| `TestQwen35VLReal_residentImagePrefillMatchesCPU` (real 0.8B, f32, 3 images) | **a gross-error gate only**: cosine 0.973 / 0.975 / 0.956 against the CPU's f32 (bar 0.9), same argmax, 16 of 16 generated tokens equal on all three images |
| the existing HF-golden tests and the Qwen2.5-VL / GLM-OCR resident gates | unchanged and green (mode 0, and the pairwise kernel) |

## Why the real-checkpoint gate is not tighter (the failed first runs, kept)

The real gate was first written at int4 with a cosine bar of 0.999 and read **0.951 / 0.975 / 0.946**, so it failed (`real-gate-first-int4-run.log`). Rather than move the bar, the controls (`controls_test.go.txt`, `controls-int4.log`) measured what the number can say on this model:

| prompt | CPU int4 vs CPU f32 | resident vs CPU f32 | resident vs CPU int4 |
|---|---|---|---|
| image, int4 resident | 0.8496 | 0.8492 | 0.9514 |
| text only (the existing, shipped resident prefill), int4 | 0.8439 | 0.8257 | 0.9804 |
| image, f32 resident | 0.8496 | 0.9728 | — |
| text only, f32 resident | 0.8439 | 0.9784 | — |
| image, f32 resident, **wrong layout mode injected** | — | 0.9699 | — |

On this small model int4 is 0.85 from f32, the shipped resident text prefill sits at 0.98 from the CPU even at f32, and injecting a wrong m-RoPE layout moves the image number by 0.003 (and, at int4, moved it from 0.951 UP to 0.960). So no real-checkpoint logits comparison can resolve the layout, and the bar was set from these controls as a floor for gross
failures (zeroed recurrent state, garbage KV), not as a precision claim. The layout is held by the kernel test and the tiny fixture, where it can be resolved. The 0.97 gap itself is the resident's pre-existing numerics for this family (the text control shows it without any image), not something this change adds; it is not investigated here.

## Not done

An MoE hybrid (the 35B-A3B) is not claimed on the GPU image path; the 9B was not run on CUDA. Image reuse on the resident (P9a) stays off for recurrent families (the state cannot be rewound to an image boundary), so a follow-up turn re-prefills, in a fraction of a second.
