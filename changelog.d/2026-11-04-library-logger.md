- **Library messages can go to your logger.** `decoder.SetLogger` sends what the library and the GPU backends write (load decisions, declines,
  KV pool activity) to a `*slog.Logger`, at debug, info or warn. Nothing changes by default: each is still the same line on stderr. The `DECODE
  TIMING`, `DECODE SPLIT` and `DELTANET TIMING` diagnostics move from stdout to stderr, so they no longer mix with generated text in a pipe. One
  failure no longer prints twice on Metal's context guard or on CUDA's speculative recurrent rows.
