- **Each loader's output now has a per-tensor identity** (SB4). `decoder`'s `TestLoaderIdentity_table` hashes every
  tensor a load hands the model (its path, layout, shape and bytes) for each committed tiny fixture, safetensors and
  GGUF, in each quant mode that loads, and holds them to `testdata/loader_identity.json`; a `.giw` round trip must give
  the same identities. A loader refactor that leaves every tensor identical leaves the table unchanged, and a loader
  change names the fixtures it moved. No load changes because of it; the parity key does not read it yet.
