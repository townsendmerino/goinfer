- **On Metal, a Phi-3 or Phi-4 `.gguf` loads at `--quant q4k` by default and runs resident.** Metal's q4k lane keeps the file's
  Q4_K blocks exactly and quantizes no activation. Before, the default for these families on Metal was int8int8 with per-32
  activation scales, which Metal declines, so the model decoded on the CPU. Against Ollama on the same file (Phi-3 mini 4k q4,
  M1 Pro, depth 128, greedy) the q4k cell reads 42.4 tok/s to 41.7, LEVEL
  (`docs/measurements/peer-vetted-2026-10-07-macbook.md`). The load holds more of the process's memory than the CPU path did,
  because Metal keeps the weights and two KV slots resident: after a 64-token completion, 5.04 GB resident against 3.79 GB.
  An explicit `--quant` is kept, other families are unchanged, and WebGPU still loads these families at int8int8 with
  per-32 scales on the CPU. The load prints a note saying it took q4k.
