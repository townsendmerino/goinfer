- **The serving shell carries a request's sampling as `internal/sampling`'s type** (SB7, E3), converting it only where it
  calls the decoder. No behavior changes: the wire census reproduces byte for byte.
