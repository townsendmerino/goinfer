- **Four more backend kernel tests no longer re-run on a decoder change** (SB1). `TestGemvRN_bitIdentical` and
  `TestGemvRNBandwidth` move to `cuda/kernels`. `TestMoEExpertGEMV_indexing`, `TestMoERoute_parity`,
  `TestMoERoute_pastOldCap` and `TestGumbelDeviceAgreesWithHost` move to `gpu/internal/kernels`. The Gumbel test now calls
  `internal/sampling`'s reference directly instead of through decoder's test wrapper, so it is held to the same code. The
  tests themselves are unchanged. These were the last cuda and gpu kernel tests that could move without moving production
  code first.
