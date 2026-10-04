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
On the 2026-09-24 SDE run only `TestInt4_forwardParity` was executed, so there is no evidence this is new. The golden is per architecture, not per ISA: an amd64 host with AVX-512 VNNI fails it. Decision for the owner:
a VNNI-specific golden (integer-exact, capturable under SDE) versus a closeness check on such hosts. Until then the script treats `TestDecodeParityInt4` failing on `icx`/`spr` as expected and anything else as unexpected.

The raw logits of that investigation are not committed (600 KB each); the table above is the record.
