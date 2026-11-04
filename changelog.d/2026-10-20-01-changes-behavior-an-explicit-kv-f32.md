- **Changes behavior. An explicit `--kv f32`, `f16` or `i8` is delivered or the load refuses and says why.** Before, some
  combinations ran another precision silently. They now refuse when the backend cannot deliver what was asked: `--kv i8` on CUDA
  (it ran f32), `--kv f16` or `i8` on Metal for MLA (`deepseek_v2` and `deepseek_v3`, whose latent cache stays f32 on Metal),
  `--kv f16` with `--drafter` on CUDA, and `--kv f16` or `i8` on the CPU
  for MoE, MLA, Gated-DeltaNet, Mamba, LFM2 and Gemma 3n (they ran f32) or with CPU batched decode switched on. `--kv f32` on Metal
  is now delivered across every family instead of running f16. One resolver, `decoder.ResolveKV`, answers for every backend's build,
  the fit guard, the banner (which prints each cache's precision with the one requested) and the request log (a new `kv=` field).
  `docs/kv-precision-matrix.md` lists every backend, family and request; `TestKVPrecisionDelivered` checks it.
