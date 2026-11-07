- **Changes behavior: under `--backend webgpu` on a Mac the vision towers run on the GPU by default.** Gemma 3's, Gemma 4's, Qwen2.5-VL's,
  Qwen3.5+'s, Qwen3-VL's and GLM-OCR's towers were opt-in (`--vision-device gpu`) and are now the default, in float32; their speed gates read
  0.355 to 0.488 of the CPU towers' encode time on an M1 Pro (five interleaved pairs each). `--vision-device cpu` keeps the towers on the CPU and
  `--vision-quant int8` still picks Gemma 3's int8 tower. The tower is priced and an over-budget one refuses the load (see the pricing entry).
  Two cases keep the CPU unless `--vision-device gpu` (or an explicit `--vision-quant`) asks for the device: a discrete GPU, whose memory WebGPU cannot query and so cannot be priced, and a Qwen3.5+ `.gguf`
  mmproj tower, which has no `config.json` to price. Where the device was asked for on a discrete GPU, a tower that cannot start is warned about at startup and runs on the CPU unless `--require-backend`
  asks for a refusal; an image that needs more device memory than the decoder left makes the Qwen2.5-VL, Qwen3.5+, Qwen3-VL and GLM-OCR towers fall back to the CPU for good, and fails that one request
  for Gemma 3 and Gemma 4, with a hint to lower `-ctx` or `-kv-sessions`, resize the image or use `--vision-device cpu`.
