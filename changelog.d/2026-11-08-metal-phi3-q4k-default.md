- **On Metal, a Phi-3 or Phi-4 `.gguf` loads at `--quant q4k` by default and runs resident, under `-backend auto` too.** Metal's
  q4k lane keeps the file's Q4_K blocks exactly and quantizes no activation. Before, the default for these families on Metal was
  int8int8 with per-32 activation scales, which Metal declines, so the model decoded on the CPU. `-backend auto` now picks Metal
  for any q4k model, as it does for int4. Against Ollama on the same file (Phi-3 mini 4k q4, M1 Pro, depth 128, greedy) the q4k
  cell reads 42.4 tok/s to 41.7, LEVEL (`docs/measurements/peer-vetted-2026-10-07-macbook.md`). The load holds more of the
  process's memory than the CPU path did. Metal keeps the weights and two KV slots resident: at `-ctx 2048`, after a 64-token
  completion, the footprint is 4.0 GB (2.4 GB of weights, 1.5 GB of KV). The CPU path read its weights from a mapping macOS can
  reclaim, and its footprint was 0.2 GB. An explicit `--quant` is kept, other families are unchanged, and WebGPU still loads
  these families at int8int8 with per-32 scales on the CPU. The load prints a note saying it took q4k.
