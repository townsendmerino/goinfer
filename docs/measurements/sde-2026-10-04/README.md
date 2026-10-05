# H1.2: the CPU suites under Intel SDE (2026-10-04)

`run-sde-goldens.sh` runs aikit's `linalg` suite and goinfer's tiny-fixture goldens and parity tests under `sde64 -hsw` (Haswell, AVX2 only: the control, which must match the native run), `-icx` (Ice Lake,
AVX-512 VNNI+VL) and `-spr` (Sapphire Rapids). SDE 10.13.1 (`sde-external-10.13.1-2026-07-28-lin.tar.xz`, unpacked at `~/tools/sde`, SHA-256 of the tarball
94e97d623fec54385686e1e7ba65ebc9941748c05ee451423948334892bf2b50; Intel's `.sig` was not checked).

**What emulation does and does not give.** The integer kernels (the VNNI dot products) are exact under SDE, so a numeric difference it shows on a VNNI CPU is the arithmetic, not emulation noise. Speed means nothing, so the FMA-peak
timing test is skipped and the LongPrompt tests (4+ minutes each) are skipped and named.

## Found by hand before queueing (the reason the job exists)

`TestDecodeParityInt4` (the amd64 greedy int4 golden, `decoder/parity_int4_test.go`) passes under `-hsw` and natively on the Ryzen 7 3700X, and **fails identically under `-icx` and `-spr`: drift at id 22, got 333,
want 8960.** Logits at that step, from the golden's own continuation (22 ids), same 0.5B int4 model:

| | top-1 | top-2 | cosine to f32 | relL2 to f32 |
|---|---|---|---|---|
| f32 (AVX2 and VNNI bit-identical) | 2 (18.61) | 333 (18.13) | | |
| int4, AVX2 | 8960 (17.821) | 333 (17.743) | 0.9581 | 0.303 |
| int4, VNNI | 333 (17.728) | 8960 (17.373) | 0.9553 | 0.313 |

The two int4 paths differ from each other by cosine 0.9974, relative L2 0.072 (max |diff| 0.86): that is a different quantised computation, not summation order, and the AVX2 near-tie (0.078) is what flips. Each path sits at
about the same distance from f32 (VNNI 3% further on this one step: one position, not a quality measurement). **No CI run can have shown it**: the test needs the 0.5B GGUF asset, which the runners do not have, so it skips there.
On the 2026-09-24 SDE run only `TestInt4_forwardParity` was executed, so there is no evidence this is new. The golden is per architecture, not per ISA: an amd64 host with AVX-512 VNNI fails it. **Resolved the same day (owner's decision: a VNNI-specific golden):** `parityWantInt4ByArch` gained an `"amd64-vnni"` list, captured under SDE `-icx` and `-spr` (identical 24 ids; it parts from the AVX2 list only at ids 22 and 23: 333, 1304 for 8960, 4136), selected by `int4GoldenKey()` from the ACTIVE kernels. The test passes under `-hsw`, `-icx`, `-spr` and natively. The list is integer-exact, but it has never been seen on a real VNNI CPU: the first run of this test with the asset on one is owed a look. Any failure in the night job is now unexpected.

The raw logits of that investigation are not committed (600 KB each); the table above is the record.

## Result of the first night run (2026-10-04, 22:31-23:12 PDT, 41 min; goinfer binary built at 37694114, aikit local checkout be7c35f = v1.55.0 + one commit)

| CPU model | aikit `linalg` | goinfer decoder (golden / parity / int4 / int8 / W4A8 / quant set) |
|---|---|---|
| `-hsw` (control: AVX2 only, must match native) | 160 passed, 12 skipped, 0 failed | 100 passed, 29 skipped, 0 failed |
| `-icx` (Ice Lake, AVX-512 VNNI+VL) | 151 passed, 21 skipped, 0 failed | 99 passed, 30 skipped, 0 failed |
| `-spr` (Sapphire Rapids) | 151 passed, 21 skipped, 0 failed | **partial**: 85 passed, 13 skipped, 0 failed, then the 25-minute cap |

`TestDecodeParityInt4` passed on the amd64-vnni golden under both `-icx` (15.3 s) and `-spr` (17.0 s) and on the AVX2 golden under `-hsw`: the golden captured under SDE holds on a second emulated VNNI core.
The skip counts differ across CPU models (aikit `linalg`: 12 under `-hsw`, 21 under the two VNNI models): tests that depend on a CPU feature skip themselves. The reasons are printed in the logs and I have not tabulated them.

**What is not established.** The `-spr` decoder leg did not complete: it hit its 25-minute cap inside `TestNgramSpeculativeGreedyParity`, with 14 tests that `-icx` completed never reached (TestNgramAdaptiveGreedyParity, TestSessionNgramSpecParity,
TestSpeculativeGreedyParity, TestW4A8DecodeParity, TestMatmulInt4_MConsistent and nine others). That test takes 15 s under `-hsw`, 99 s under `-icx` and **102 s run alone under `-spr`**, so the 22+ minutes is not a property of the test or of
SPR emulation alone, and I could not reproduce it. The goroutine dump at the cap shows no hang: the test was waiting on its token channel while a decode goroutine was runnable in `gatedMLP` and the suite's memory reclaimer was inside a GC.
The cause is **unexplained**. A `-spr`-only re-run with a 45-minute cap and its own output directory (`rerun-spr-1/`) is queued (`sde-spr-rerun`); a second stall there would be a second sample of the same dump. No `-spr` claim beyond "85 passed, 0 failed, incomplete" is made until it runs.
