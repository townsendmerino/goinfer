- **New family (experimental, CPU only): Inkling-2-Mini (`inkling2_loop`), text only, from llama.cpp's GGUF.** Thinking Machines'
  26B mixture-of-experts runs its 24 layers as 48 layer visits a token (layers 6 to 17 three times), with attention-residual blocks,
  a relative-position bias and a per-token sink in attention, short convolutions, SITU experts and cross-pass message controllers.
  Its forward matches transformers at every visit on a tiny checkpoint converted by llama.cpp's own converter (cosine 1.00000).
  The tokenizer (`tokenizer.ggml.pre` `inkling2`) and the chat template (its effort levels, the thinking message and JSON tool
  calls) are new too, each held to the release's own files. Load the `.gguf` directly (`-direct-load`); there is no `.giw` sidecar
  for this family yet. No GPU backend runs it.
