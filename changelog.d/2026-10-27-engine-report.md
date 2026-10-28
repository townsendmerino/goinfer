- **The serving shell reads a generation's account through its own report type** (SB7, E2), not the decoder's
  `Generation`. One more `internal/serveapp` test file no longer imports `decoder`. No behavior changes: the wire
  census reproduces byte for byte.
