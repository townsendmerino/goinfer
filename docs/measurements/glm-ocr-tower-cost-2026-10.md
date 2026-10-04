# GLM-OCR vision-tower cost on the Mac CPU (2026-10-03)

The record for the pre-registration in [`glm-ocr-tower-cost-2026-10/README.md`](glm-ocr-tower-cost-2026-10/README.md),
which was committed before any point ran and amended (the instant idle gate) before any point ran. **A record, not a
gate:** there is no pass or fail and no decision rule. O4 in [`../tasks/task-glm-ocr-2026-10.md`](../tasks/task-glm-ocr-2026-10.md)
reads it to choose a default pixel cap for laptops.

## Result

One `GlmOcrVisionEncoder.Forward`, the 24-block, 434 M-parameter GLM-OCR ViT, on the f32 reference path on the CPU, with
random `pixel_values`. The whole sweep ran 3 times, one fresh process per point:

| Point | grid_thw | Patches | Median | Range | All three | ms per patch |
|---|---|---|---|---|---|---|
| 1 MP | [1, 70, 72] | 5,040 | **28.77 s** | 28.71–29.00 | 28.77, 29.00, 28.71 | 5.708 |
| 2 MP | [1, 100, 102] | 10,200 | **92.26 s** | 92.02–92.78 | 92.26, 92.78, 92.02 | 9.045 |
| 4.8 MP | [1, 128, 192] | 24,576 | **427.12 s** | 426.73–438.68 | 426.73, 438.68, 427.12 | 17.380 |

**The attention-quadratic term dominates.** ms per patch at 4.8 MP is 3.04× its value at 1 MP, past the
pre-registration's 2× line. Cost per patch triples while the patch count grows 4.9×, so the processor's ceiling, about
7 minutes on this CPU, is the expensive end, not a linear extrapolation of 1 MP. The 4.8 MP point came in under the
pre-registration's 10–20 minute guess, and its 30-minute stop did not trigger.

## Provenance

- **Machine:** MacBookPro18,3, Apple M1 Pro (8 CPUs), 16 GB, macOS 26.6.2, `go1.27.0 darwin/arm64`. On AC power,
  `pmset -g therm` reported no thermal or performance warning before or after any point.
- **Run:** 2026-10-03, 11:28–12:10 PDT, on the Mac's night queue (job `glm-ocr-tower-cost`), started by day at the
  owner's word. The instant idle gate (at most 10% busy over 3 s, no foreign timed workload) before each process. The
  `sysctl vm.loadavg` lines before and after each process are in the logs.
- **Checkpoint:** `zai-org/GLM-OCR` at `2e85a62840ccac27daa451df36c736c4636b8628`, read from `~/models/glm-ocr` on the
  internal SSD. `model.safetensors` sha256 `a16eb0de98d199293371c560f95f83130d2a2c9612449df16839f08ff9498815`.
- **Code:** aikit `v1.52.0` = `01928d7af35bfc178bda4673882cd3201d455af4`, `TestGlmOcrVisionEncoder_costSweep`, built
  once into `vision-01928d7af35b.test` (sha256 prefix `a2171edd66a04a34`).
- **Path:** the f32 reference, `LoadGlmOcrVisionEncoder` with `quant=false`, on the CPU, with random `pixel_values`
  (rand seed 1). The cost does not depend on the values.
- **Logs:** [`glm-ocr-tower-cost-2026-10/night-2026-10-03/`](glm-ocr-tower-cost-2026-10/night-2026-10-03/): `provenance.txt`,
  the nine point logs and `results.txt`.

## What this is not

- **Not a speed claim against anything.** No peer was run, and the int8 tower (`quant=true`) was not measured.
- **Not the served end-to-end time.** Preprocessing, the text decoder's prefill of 6,144 image tokens at 4.8 MP, and
  decoding are all outside it.
