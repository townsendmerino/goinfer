- **Two CUDA tests no longer leave their device open, which had made `TestLedger_closesAgainstDevice` fail in full-package
  runs.** CUDA packs small buffers into a pool shared by every device in the process. `TestKernelLocalMemoryCensus` and
  the tests built on `actGroupHarness` never released their device, so a later model's small buffers landed in pages
  already counted, and its measured device memory read below the ledger's charge (4,194,304 against 4,479,240 bytes on
  llama-tiny). Both now release what they create, and the ledger test passes in the gate's order with its exact bound
  unchanged. New: `TestSmallAllocPool_sharedAcrossDevices` records the pool sharing.
