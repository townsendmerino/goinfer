- **The Metal ledger cell reconciles every large buffer to the byte** (MM3c, tests only; Francis's decision of
  2026-10-27). Its baseline is read at the resident's device creation again. In this cell only, a test hook reads
  Metal's allocated-memory count around each buffer allocation. Every buffer of 16 KB or more must move it by exactly
  its ledger charge. Smaller buffers, which Metal counts by shared chunk, are measured and reported. This replaces the
  earlier baseline move, which had let up to 1.6 MB of over-charging pass. A planted 4 KB over-charge on one large buffer
  must fail the cell. Production allocation is unchanged.
