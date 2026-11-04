- **Prefix sharing is on by default (`--kv-share`).** `--kv-share` (`on|off`, default `on`) indexes completed full-attention 64-token
  KV blocks by token-span hash (scoped by model, adapter, and KV precision) so a new conversation that shares a system preamble or prefix
  with another slot reuses those blocks instead of re-prefilling them (by reference with copy on write on paged CUDA and Metal caches, or
  by device-to-device copy on non-paged caches). When `goinfer-serve` listens on a non-loopback address without `--api-key`, `--kv-share`
  is forced off so unauthenticated callers do not share cached prefixes across slots.
