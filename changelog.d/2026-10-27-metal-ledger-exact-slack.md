- **The Metal ledger cell's lower bound is exact** (MM3c, tests only; Francis's decision of 2026-10-27). The cell reads
  its baseline when the resident's device is created again, and it requires measured ≥ U − S, where S is the bytes of the
  buffers of 4 KB or less that Metal counts by shared chunk. This replaces the earlier baseline move, which had let up to
  1.6 MB of over-charging pass. A planted 4 KB over-charge on a large buffer must fail the cell.
