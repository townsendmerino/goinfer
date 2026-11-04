- **On Metal, `serve` now batches Gemma 4 dense and, with `--kv-sessions` 2 or more, Qwen3.5-family and Nemotron-H models by default.**
  `--gemma4-step` and `--recurrent-step` are on where a load holds two or more sessions: 1.38× decode throughput measured in-process on Gemma 4
  E2B and 1.58× on Qwen3.5 0.8B at 4 concurrent sequences, with bit-identical output, for about 10 MB of step scratch. Plain `serve` keeps a
  recurrent model at one session, so it does not batch those without `--kv-sessions`, and `serve` runs a model with a vision tower attached
  one generation at a time, as before, which includes Qwen3.5 checkpoints that ship one. `goinfer-chat` stays off, since one session has
  nothing to batch. Either flag overrides, on or `=false`.
