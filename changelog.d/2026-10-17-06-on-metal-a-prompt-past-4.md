- **On Metal, a prompt past 4,096 tokens stays on the batched prefill for a model whose attention head dim the fused kernel does
  not take (Gemma 4 E2B and E4B: 256 and 512).** The exact attention kernel held 4,096 scores, and a longer prompt fell back to
  a pass about 4× slower (E2B at 8,192 tokens: 34 tok/s against 144 at 4,096). Past 4,096 keys it now tiles its scores. A prompt
  that fits runs the same kernel as before. When an E-model prompt does leave the batched pass, stderr says so once per
  request, and the `-log-requests` line carries the reason (`prefill_fallback_reason`).
