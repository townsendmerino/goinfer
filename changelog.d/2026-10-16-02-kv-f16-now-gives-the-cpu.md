- **`--kv f16` now gives the CPU an f16 KV cache** (half the KV bytes; each stored row rounded to f16) for the generic families
  without MoE. Other families stay f32 and the banner says why. `--kv auto` and the library default stay f32, the reference.
  A CPU load given `--kv f16` was priced at f16 over an f32 cache before; it is now priced at what it builds.
