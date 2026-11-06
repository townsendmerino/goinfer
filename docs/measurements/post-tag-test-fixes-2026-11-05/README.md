# Post-tag gate fixes, PR 2: the test reds v0.24.0's gates classified (2026-11-05, nobara-sb01)

The reds below were classified test-only on 2026-11-04 (v0.24.0's CUDA gate, `quick -full` and parity at `ccdfb3ad`) and accepted for that tag. Each item
says what was reproduced before the edit, what changed, and what ran after. Machine: nobara-pc, all gitignored fixtures placed, driver 595.112.07.

| item | red before | change | after |
|---|---|---|---|
| `TestAttnBatchedBandwidth` | `red-attn-bw.log` | passes the 13th kernel argument, the learned sinks (`gc.ArgDevicePtr(0)`), and checks every launch, upload and synchronize | `green-attn-bw.log`: 29.7 ms/launch, 4.8% of peak |
| `TestResidentWeightBytes_countsGemma4MoEExperts` | `red-decoder.log` | the expected delta is the experts plus the 1-D tensors MM8 prices (`denseVectorBytes`) | `green-decoder.log`: PASS |
| `TestSeams_subCaptureRefusedWhereGateSCannotRun` | `red-decoder.log` | `gemma4-moe-tiny-k3` joins the expected refusals | `green-decoder.log`: PASS |
| `TestGemma4EModel_realDeclinesResident` | `red-e2b.log` (heavy opt-in) | CUDA's decline and admit read `program.CUDA`'s operation table, as Metal's already do; the feature-set loops keep WebGPU only | `green-decoder.log`: PASS |
| `TestDecisionCensus`, `_secondProcess` | `red-decoder.log`: `.cache: is a directory` | the long-variant copy skips directories | **still red here**, see below |
| the `quick -full` decoder cells' 20 m timeouts | `red-timeout.log` | `-full` defaults to 45 m (`resolveQuickTimeout`); an explicit `-timeout` wins | `green-timeout.log` |

## TestAttnBatchedBandwidth: measured, where the 11-04 note was a reading

The test passed 12 arguments to `attn_batched`, which has had a 13th parameter (`sinks`) since the gpt-oss step; its sibling `TestAttnBatched_bitIdentical`
passes it. The test ignored every launch error. With the checks added and the 12 arguments left in, `red-attn-bw.log` reads
`cuLaunchKernel: CUDA_ERROR_INVALID_VALUE`; with the 13th argument, `green-attn-bw.log` reads 29.7 ms/launch (4.8% of the card's peak), the same as
the 10-30 gate's record (29.2 ms, 4.9%). Before the checks, the test printed 12 µs a launch and passed.

## The census is not fixed, and cannot be by this change

Skipping directories is necessary: `testdata/gemma-3-270m/.cache` made both tests fail at once. With it skipped, on this box the census then fails later:

- `green-touched.log` (`python3 scripts/touched.py`): both tests fail after about 50 s with `no space left on device`, copying the 77th fixture's weights into
  `/tmp`, a 32 GB tmpfs on another filesystem than the checkout, where the hard link the helper tries first is refused.
- A symlink fallback got past that, and the test then held 39 GB resident and 68 GB in swap (killed by hand at about 02:35 PST; `census-symlink-killed.log` holds only its first lines); with `TMPDIR` on the checkout's filesystem
  (hard links work) and a 24 GB cgroup cap, the test was killed by the cap (`census-capped.log` stops after phase D2). That fallback is not in this PR. `censusLoadAll`
  loads every fixture and long variant of `censusFamilies` at once, which is sound for the small fixture set CI has and not for this box's 21 large ones.

So `TestDecisionCensus` stays red on nobara with every fixture placed. It needs the census to leave large fixtures out or load them one at a time, which changes
what its golden covers; that is a decision for its owner, not made here.

## Other

- `green-touched.log`: `python3 scripts/touched.py` over this change. Everything passes except the two census tests above (`cmd/gate` ok, `backend/cuda/kernels` ok).
- `-full`'s default timeout was not run end to end here (a full `quick -full` is an hour); `TestResolveQuickTimeout` covers the rule.
