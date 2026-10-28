- **New package `engine` (Experimental)** (SB7b). It holds what the serving shell asks of a model: the `Engine`
  interface, its `Request` and `Report`, the `Model` handle and the block drafter, plus small helpers. `serve` uses its
  types; nothing a client sees changes.
