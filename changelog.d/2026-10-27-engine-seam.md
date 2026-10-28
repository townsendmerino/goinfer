- **`serve`'s generation goes through an engine seam** (SB7, E1). The token stream comes from an engine: the decoder, with
  its resident, speculative and session paths, or the scripted seam that tests use. It is no longer chosen inline in the
  shared generation path. No behavior changes: the wire census reproduces byte for byte.
