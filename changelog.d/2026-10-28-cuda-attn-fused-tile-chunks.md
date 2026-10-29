- **`TestAttnFusedTile_defaultBitIdenticalWholeModel` derives its expected chunk count from `prefillPassRows()`** (tests
  only). It divided the prompt length by a literal `512` from before KS-B8 widened the default CUDA prefill pass to up to
  `2048` rows when scratch fits. On Qwen2.5-Coder-1.5B (`2048`-row pass, 28 layers) the default path launches the 128-row
  tile 28 times at `M = 1300` (1 chunk) and 56 times at `M = 3900` (2 chunks), with 0 of 151,936 logits differing from the
  forced 64x64 path.
