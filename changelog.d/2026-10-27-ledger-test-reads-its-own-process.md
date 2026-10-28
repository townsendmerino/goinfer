- **The CUDA ledger test reads its own process's device memory** (MM3, tests only). `TestLedger_closesAgainstDevice` measured
  the device's free VRAM, which moves when another package's device tests allocate beside it. The CUDA device suite runs
  them side by side, and so it read the test red. The test now grades on the driver's per-process figure. Its bounds are
  unchanged.
