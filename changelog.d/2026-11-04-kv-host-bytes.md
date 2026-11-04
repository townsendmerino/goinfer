- **Host KV prefix snapshots (`--kv-host-bytes`) and experimental opt-in CUDA host demotion.** `--kv-host-bytes` (`auto` by default:
  10% of RAM up to 8 GiB; `0` disables it) snapshots a cold long prefill's KV blocks to pinned host memory on CUDA and Metal so a later
  conversation with the same prefix restores them from host RAM instead of re-prefilling. Passing an explicit byte size (such as
  `--kv-host-bytes 1G`) also opts into **experimental** CUDA host demotion (`Options.KVDemote`): when an active conversation borrows an
  idle slot's blocks, the idle slot's KV rows are copied to pinned host memory before release and promoted back to VRAM when that
  conversation returns, bit-identical (off under `auto` and `--kv-share off`).
