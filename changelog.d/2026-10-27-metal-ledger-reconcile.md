- **The Metal ledger cell reconciles the ledger itself** (MM3c, tests only; corrects the earlier per-buffer check, which
  compared the device with Metal's allocation size rather than with what the ledger recorded). Every buffer allocation and
  release is now traced. The ledger's own record, by line and tier, must equal the live buffers' charges. Each large buffer
  must match the device to the byte. Everything the device counts must be accounted for by traced allocations and frees.
  Plants for a ledger over-charge, a ledger under-charge and a buffer allocated around the constructors each fail the cell.
