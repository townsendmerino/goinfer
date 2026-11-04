- **Changes behavior. `Q5_K` and `Q6_K` run natively on Metal and CUDA under `--quant q4k`.** On Metal and CUDA, a `.gguf` loaded with
  `--quant q4k` keeps its `Q5_K` and `Q6_K` projection tensors in their native 256-weight super-block formats instead of widening them to
  `int8` copies (on Metal the LM head stays `int8`; on CUDA the head runs native too). Outputs change because the projections are read
  from their `Q5_K`/`Q6_K` super-blocks rather than re-quantized `int8` rows. On `Phi-3-mini-4k-instruct-q4.gguf`, device weight memory
  drops by 357 MB on Metal and 374.7 MB on CUDA.
