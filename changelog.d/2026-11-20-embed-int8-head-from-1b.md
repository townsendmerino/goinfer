- **`--embed-int4`'s default now depends on model size: a model of 1B parameters or more keeps the int8 output head.**
  Below 1B the default is still the int4 head, and on Metal still off. The load counts the parameters from the file's
  tensor directory before it picks the sidecar `.giw`, so a 1B+ `.gguf` opens (and builds once, if missing) the plain-head
  sidecar rather than the `e4h` one; a prebuilt `.giw` keeps the head its transcode baked. An explicit `--embed-int4`
  or `--embed-int4=false`, or serve's per-model `embed-int4=` key, wins at any size. Costs about 100 MB more VRAM on a
  1.5B and 240 MB on a 7B on CUDA. Library callers: `decoder.Options.EmbedInt4Default`, `decoder.EmbedInt4For`,
  `decoder.PeekParamCount`, `Model.EmbedInt4`. Issue #1160.
