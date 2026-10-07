# Vision-tower int8 vs f32, on the real checkpoints (2026-10-02)

**A record, not a gate.** It exists to decide one thing: whether serve should keep forcing an int8 tower for the CPU-only towers under `--backend cuda|webgpu`
(it did, for Qwen2.5-VL, Qwen3.5+ and Gemma 4, until this date). It decided no: see `towerInt8` in `internal/serveapp/main.go`.

## What was measured

Each tower's output (the merged image features the decoder reads) with int8 matmul weights, against the SAME tower with f32 weights, on the same preprocessed image.
The reference is goinfer's own f32 tower, not Hugging Face: this isolates the cost of int8, it does not re-validate the tower (the towers' own parity to HF is gated elsewhere,
at f32). The int8 tower is built by the same call serve makes (`vision.LoadQwenVisionEncoder` / `LoadQwen3VisionEncoder` / `LoadGemma4Encoder` with `quant=true`).
The test source is `vision-tower-int8-fidelity-2026-10-02/towerq_test.go.txt`; it was a throwaway test and is not in the tree.

## Provenance

- Machine nobara-pc (amd64, CPU towers), 2026-10-02 (PDT), box otherwise idle by day, one run per row, so **no spread is claimed**.
- Checkpoints, all from `~/models` (the local bench set): `qwen2.5-vl-3b`, `qwen3.5-0.8b`, `gemma-4-26b-a4b-it` (its tower).
- Images: `testdata/glm_ocr/invoice.png` (a rendered invoice, 1000x1300) for all three; Qwen2.5-VL also on a 336x336 test image (a red square, a yellow circle, a blue background; the one the audit probes used).
- goinfer at `e135d9da` plus the working tree of the change; aikit v1.52.0. Deterministic (no sampling involved).

## Result

| tower | image | tokens | per-token cosine, mean | per-token cosine, worst | relative L2 | f32 s | int8 s |
|---|---|---|---|---|---|---|---|
| Qwen2.5-VL 3B | red square | 144 | 0.97473 | 0.67427 | 0.2282 | 2.0 | 1.7 |
| Qwen2.5-VL 3B | invoice | 1656 | 0.97499 | 0.28968 | 0.2085 | 26.3 | 31.5 |
| Qwen3.5 0.8B | invoice | 1271 | 0.99246 | 0.61994 | 0.1382 | not timed | not timed |
| Gemma 4 (26B-A4B tower) | invoice | 266 | 0.94998 | 0.39392 | 0.3129 | not timed | not timed |

The seconds are single exploratory runs and are not quotable; they say only that the int8 CPU tower is not obviously faster (the Qwen2.5-VL invoice was 20% slower), which matches the
earlier finding that int8 is a wash on plain AVX2 (`docs/completed/task-cpu-vision-prefill.md`).

## Reading it

An int8 tower that is faster would be worth some loss. This one is not faster and the loss is large: relative L2 of 0.14 to 0.31 against f32, and individual image tokens down to cosine
0.29. How much of that reaches the decoder's text was NOT measured here (one image per tower, no end-to-end comparison), so "garbles the answer" is not claimed; what is claimed is that
nothing gated these towers at int8 and this measurement gives no reason to default to it. GLM-OCR's int8 tower had already been excluded on the same ground.

## What changed because of it

`towerInt8`: Qwen2.5-VL, Qwen3.5+, Gemma 4 and GLM-OCR load f32 unless `-vision-quant int8` is given, on every backend. Gemma 3 keeps int8 under cuda/webgpu because its resident tower needs it.
Cost: the f32 tower holds more host memory than the int8 one.

## Not covered

Gemma 3's SigLIP tower, the one tower still int8 under cuda/webgpu: measured 2026-10-07 on the Mac in `siglip-int8-fidelity-2026-10-07.md` (relative L2 0.16-0.52, worst token 0.01-0.17). Other images; Qwen3.5-9B and the other Gemma 4 sizes; the text the decoder produces from int8 versus f32 features; the dense Qwen3.5 sizes at 1,024 tokens (the serve cap).
