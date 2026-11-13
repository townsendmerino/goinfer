- **Under `--backend webgpu` a vision tower on the device is now priced.** On a Mac the tower counts against RAM when
  `--kv-sessions` and the context are sized (a single-slot build's context gives way, down to 4096, to fit it), and a tower that then needs more than the memory available refuses the load with its
  size instead of overcommitting (a CPU tower would hold the same RAM, so it is not a fallback; `--vision-device cpu` runs it anyway). The refusal advises `--vision-quant int8` only for Gemma 3.
  A discrete GPU has no free-memory figure under WebGPU, so there nothing changes. A Qwen3.5+ `.gguf` mmproj tower on the device is not priced.
