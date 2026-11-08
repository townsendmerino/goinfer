- **On Metal, a tool or thinking-budget turn keeps the run-ahead chain while its gate is closed.** A turn with a gated logit processor (the
  lazy tool-call union, a thinking budget) used to leave Metal's greedy chain for the whole turn; the chain now runs on the prose part and
  stops at the step the gate opens, where the full-logits path takes over until the gate closes. Tokens are identical to before. On an
  M1 Pro the prose part of such a turn decodes 1.135x (Qwen2.5-Coder 1.5B) and 1.019x (Qwen2.5 7B) as fast, and a whole tool-call turn
  1.052x and 1.023x (`docs/measurements/hl-2026-11/README.md`). A turn whose gate opens and closes many times pays one wasted forward per
  opening and is not measured. Greedy and `top_k = 1` turns on a single slot only; MC3, CUDA and sampled turns are unchanged.
