- **`POST /admin/models/fit` says whether a load would fit, without loading.** It takes the load's body and returns the
  load's own memory check, priced now (`fits`, the context priced, need, budget, weight and KV bytes, the refusal), the
  file the load would open, and the decode path where that is known before the weights are opened. A `.gguf` whose sidecar
  `.giw` is not built yet answers `unknown`, naming the build (`docs/server.md`, "Admin: would this load fit?").
  For a `.giw` on Metal it also opens the bundle without building a resident and asks Metal's own checks before its build,
  so `decode_path` says `metal-resident (int4)` with the context the resident would hold (`resident_ctx`), or the CPU path
  with the decline the load would report. CUDA, a `.gguf` on a GPU backend, and a Metal int8 load still answer `unknown`.
