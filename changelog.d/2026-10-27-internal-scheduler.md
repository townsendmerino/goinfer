- **`serve`'s admission and KV session placement moved to `internal/scheduler`** (SB6, M1). This covers the FIFO turn granter
  and the choices of which resident session a prompt extends, which session donates rows by copy, and which is evicted.
  `internal/scheduler` imports only the standard library, a rule the layout test enforces. No behavior changes: the
  scheduling census reproduces byte for byte.
