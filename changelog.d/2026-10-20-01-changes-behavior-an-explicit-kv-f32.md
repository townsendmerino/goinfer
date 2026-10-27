- **Changes behavior. An explicit `--kv f32`, `f16` or `i8` is delivered or the load refuses and says why** (phase 0 of
  `docs/tasks/task-kv-precision-everywhere-2026-10.md`). Before, some combinations ran another precision silently. They now refuse:
  `--kv i8` on CUDA (it ran f32), `--kv f32` on Metal (it ran f16), `--kv f16` on a CUDA family the f16 kernels do not cover (it ran f32),
  `--kv f16` or `i8` on the CPU for MoE and own-loop families (they ran f32) or with CPU batched decode switched on, and `--kv f16` with
  `--drafter` on CUDA. `--kv auto` is unchanged. One resolver, `decoder.ResolveKV`, now answers for every backend's build, the fit guard,
  the banner (which prints each cache's precision with the one requested) and the request log (a new ` kv=` field).
  `docs/kv-precision-matrix.md` lists every backend, family and request; `TestKVPrecisionDelivered` checks it.
