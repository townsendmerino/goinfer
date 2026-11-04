- **The `q4k` `.giw` sidecar is written on the first load of every `q4k` `.gguf`.** Loading a `.gguf` at `--quant q4k` (including Phi-3's
  default on CUDA) now writes a `.q4k.<target>.v17.giw` sidecar on first load and maps the sidecar on subsequent loads, storing `Q4_K`
  and native `Q5_K`/`Q6_K` tensors directly in the bundle so the GGUF is not parsed into the Go heap on each start (`--direct-load` or
  `GOINFER_GGUF_DIRECT` skips the sidecar).
