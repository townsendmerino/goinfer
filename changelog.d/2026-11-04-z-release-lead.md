- **v0.24.0: more models on the GPU, more conversations in the same memory.** On the model side, Gemma 4 E-models on Metal use a
  4-bit per-layer embedding sidecar (`.ple4`, 981 MB smaller on E2B), `--quant q4k` keeps `Q5_K` and `Q6_K` projection weights native on
  Metal and CUDA and writes a `.giw` sidecar on the first load of every `q4k` `.gguf`, Gemma 2 runs resident on Metal and CUDA, and
  Olmo Hybrid runs resident on CUDA, Metal (`int4` only) and WebGPU (read only on the M1 Pro's adapter). On the memory side, CUDA and
  Metal replace fixed per-slot KV reservations with a 64-position paged KV block pool and default-on copy-on-write prefix sharing
  (`--kv-share`), CUDA hybrid models hold two conversations warm on per-slot recurrent state, and `--kv-host-bytes` adds host-memory
  prefix snapshots with experimental opt-in CUDA host demotion.
