- **Gemma 4 E-models on Metal at `int4` use a `.ple4` sidecar.** When a Gemma 4 E-model (E2B or E4B) is transcoded to its Metal `int4`
  sidecar (`.int4.ple4.metal.v17.giw`), its per-layer token embedding table (`per_layer_model_projection` / `ple_embed_tokens`) is stored
  at 4-bit instead of 8-bit, and `StreamTranscodeGGUF` streams the transcode one layer at a time so the full GGUF is not held in the Go
  heap. On Gemma 4 E2B on the M1 Pro, the sidecar is 981 MB smaller (`3,649.0 -> 2,668.0 MB`) with 94.25% top-1 agreement (`410/435`
  positions, mean KL `0.03257` nats) against the `int8` per-layer table.
