- **Fix: on CUDA, `serve` now runs a `--quant q4k` `.gguf`'s `Q5_K` and `Q6_K` tensors natively, as v0.25.0 said it did.** v0.25.0's
  native `Q5_K`/`Q6_K` reached CUDA only on a direct load. `serve` loads a q4k `.gguf` through its `.giw` sidecar, and the CUDA sidecar
  still widened those tensors to `int8`. So from a file's second load on, serve decoded them from `int8` copies. On Phi-3 mini the fix
  makes served decode 1.10× faster at depth 128 (126.3 against 114.6 tok/s) and holds 375 MB less of the card (342 MiB less for the
  whole server). The CUDA sidecar now carries
  the native blocks, under a new name: `<model>.q4k.k56.cuda.v17.giw` beside the `.gguf`. Served outputs change to the direct load's,
  bit for bit. A sidecar written before this fix, `<model>.q4k.cuda.v17.giw` in the same directory (with `.e4h` after `q4k` for an
  `--embed-int4` load), is no longer read: the first load after upgrading writes the new file beside it. You can delete the old file to
  free its disk space, about the model's size.
