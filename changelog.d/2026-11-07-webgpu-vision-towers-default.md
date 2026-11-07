- **Changes behavior: under `--backend webgpu` the vision towers run on the GPU by default.** Gemma 3's, Gemma 4's, Qwen2.5-VL's,
  Qwen3.5+'s, Qwen3-VL's and GLM-OCR's towers were opt-in (`--vision-device gpu`) and are now the default, in float32; their speed gates read
  0.355 to 0.488 of the CPU towers' encode time on an M1 Pro (five interleaved pairs each). `--vision-device cpu` keeps the towers on the CPU,
  `--vision-device gpu` is now the same as the default, and `--vision-quant int8` still picks Gemma 3's int8 tower. A tower that cannot start (no device memory left after the decoder)
  is warned about at startup and runs on the CPU unless `--require-backend` asks for a refusal; an image that needs more device memory than the decoder
  left fails its request with a hint to lower `-ctx` or `-kv-sessions`, resize the image or use `--vision-device cpu`.
