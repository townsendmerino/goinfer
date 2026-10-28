- **The plain decode loop runs over a target of hooks** (SB5). `generateInto`'s loop is now `plainDecodeLoop`, which a
  scripted fake drives as well as the CPU, CUDA and WebGPU paths, so its contract is checked on every push without a
  model: the stop id, the processor and its gate, logprobs, the cancel and KV-pool exits, and the commit callback. No
  token stream changed: llama-tiny's 35 CUDA and 32 WebGPU decode streams, one per fast path, are byte for byte what
  they were.
