- **Changes behavior: under `--backend webgpu` the vision towers run on the GPU by default.** Gemma 3's, Gemma 4's, Qwen2.5-VL's,
  Qwen3.5+'s, Qwen3-VL's and GLM-OCR's towers were opt-in (`--vision-device gpu`) and are now the default, in float32; their speed gates read
  0.355 to 0.488 of the CPU towers' encode time on an M1 Pro (five interleaved pairs each). `--vision-device cpu` keeps the towers on the CPU,
  `--vision-device gpu` is now the same as the default, and `--vision-quant int8` still picks Gemma 3's int8 tower. The tower's device memory is
  now priced: on a Mac, serve keeps the WebGPU tower on the CPU and says why when it exceeds the fit guard's budget of the memory left (`--require-backend`
  refuses), and the tower counts against RAM when `--kv-sessions` is priced. A discrete GPU has no free-memory figure under WebGPU, so there a tower that
  cannot start is warned about at startup and runs on the CPU (Gemma 3, Qwen2.5-VL, Gemma 4), the Qwen2.5-VL, Qwen3.5+, Qwen3-VL and GLM-OCR towers fall back to the CPU
  at the first image that runs them out of device memory, and Gemma 3's and Gemma 4's fail that one request with a hint to lower `-ctx` or `-kv-sessions`,
  resize the image or use `--vision-device cpu`.
