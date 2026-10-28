- **`engine.Decoder`, the engine a loaded model generates on** (SB7b). It covers the GPU-resident paths and the CPU
  session path with its speculative variants, and it moves out of `serve` into the `engine` package behind a
  `SessionPool` the caller keeps. No behavior changes: the wire census reproduces byte for byte.
