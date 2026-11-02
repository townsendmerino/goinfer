- **A `--quant q4k` load the fit guard refuses stays refused.** It used to retry with weight streaming, which pages nothing
  at q4k, so the retry loaded the model with no memory bound. The refusal no longer recommends `-stream-weights` for it either.
  This reaches `-direct-load`, and platforms with no default sidecar; the default sidecar route does not take that retry.
