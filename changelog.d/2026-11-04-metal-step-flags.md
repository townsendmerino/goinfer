- **Metal multi-sequence batched decode flags (`--moe-step`, `--recurrent-step`, `--gemma4-step`).** `--moe-step` (on by default on
  Metal) batches concurrent MoE decode requests and short prompts under 16 tokens into one step (`1.49×` two-client aggregate throughput
  on Huihui-MoE-5B, bit-identical). `--recurrent-step` batches concurrent Qwen3.5-family and Nemotron-H decode requests on per-slot
  recurrent state (`1.58×` on Qwen3.5 0.8B, bit-identical). `--gemma4-step` batches concurrent Gemma 4 dense decode requests
  (`1.38×` on Gemma 4 E2B, bit-identical). In `goinfer-serve`, the two step flags are on when `--kv-sessions` is 2 or more and off at
  one session; `goinfer-chat` leaves them off. Either flag can be set explicitly (`--recurrent-step=false` overrides the default).
