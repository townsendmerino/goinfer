# OP1-o2: the CUDA flash-decode lane with Gemma 2's attention softcap (nobara-sb01, 2026-11-08)

**What.** `docs/completed/task-forward-architecture-2026-10.md` §4.6a, "Owed, not in OP1": the flash-decode lane with the cap.
- Until now a capped layer (Gemma 2's `attn_logit_softcapping`) was kept off the lane by `faEligible`, so Gemma 2 decoded at depth on
  the exact split-KV lane.
- **The change:** `decode_fa.cu` joins the modules built with `-DATTN_SOFTCAP`, as OP1 did for `glue`, `prefill_batched` and
  `decode_splitkv`:
  - Every `fa_partial*` kernel, the M=1 lane and the 24 multi-row verify kernels, takes `ATTN_CAP_PARAM` after its int8 scales, and
    both score sites call `attn_score(d, scale ATTN_CAP_ARG)`.
  - The capped variants are `decode_fa_cap.ptx` and `decode_fa_kv16_cap.ptx`, built at the ambient NVRTC 12.9.86 like `decode_fa.ptx`.
    There is no capped int8 build: `CUDAKVI8Decline` refuses every capped family first.
  - A capped model's resident loads them through `attnPTX`, `withCap` appends the cap to both `fa_partial` launches, and `faEligible`
    no longer declines a capped layer.
- **How the bars were set.** The gates below were written alongside the code, not registered before it. The coordinator's brief was
  "build plus correctness tests". Their bars are the lane's existing ones, not new choices: the f64-oracle bound of
  `TestFlashDecodeVsF64` (1e-4 of the largest output), the verify lane's bit identity (`TestFlashDecodeRowsBitIdentical`), and
  `TestFlashDecodeEndToEnd`'s wiring criterion.

## Gates and readings (tier 0 and one tier-1 correctness read; card tests under the shared timing lock)

- **G2-I, the plain modules unchanged.** Before any edit, `decode_fa.ptx`, `decode_fa_kv16.ptx` and `decode_fa_kv8.ptx` were rebuilt from
  the unchanged source and came back byte for byte. After the edit they come back byte for byte again: in the plain build both macros
  are empty. An uncapped model's lane is therefore the same PTX it was.
- **The capped lane against a capped f64 oracle** (`TestGemma2Softcap_flashDecodeVsF64`, gemma2-tiny's resident, cap 5).
  - **Setup:** the lane is driven on synthetic K/V at head dimensions 64, 128 and 256, the lane's (the fixture's 32 is not one), at 1 to
    1,000 keys, two window starts and split counts 1, 4 and 16.
  - **Result:** 90 cases, worst 4.64e-7 of the largest output against the 1e-4 bound, and two runs bit-identical each time.
  - **Control:** the scores are drawn at a few times the cap. In 72 of the 90 cases the cap moves the output by over 1 %, and in all 72
    the lane with its cap lifted to 1e30 is red against the capped oracle.
  - **Verify lane:** 104 rows of multi-row launches equal the M=1 capped lane bit for bit.
  - **Instrument note:** the lane's partial buffer is sized at load for the fixture's head dimension, so the test re-sizes it (test
    line) for the 256-wide, 16-split, full-batch geometry it drives. Without that, the first run wrote past it and the context faulted
    (`CUDA_ERROR_ILLEGAL_INSTRUCTION`).
- **G2-L, the lane selectors** (`TestGemma2Softcap_lanesCUDA`, updated): a capped layer is now eligible for the lane, on
  `decode_fa_cap`. The fused prefill kernels, the image prefill and the drafter still refuse it by name, and the uncapped control
  still sees both lanes.
- **The plant, in the kernel source:** the multi-row kernel's score site without the cap. After a rebuild, the verify-lane bit identity
  is red ("S=4 n0=1 M=3 row 1 elem 0: multi-row -0.73821896, M=1 -0.73723036"; `plant-rows-uncapped.log`). Restored, both capped
  variants rebuild to the same SHA-256.
- **End to end on a real Gemma 2** (`TestFlashDecodeEndToEnd`, `gemma-2-2b-it-Q8_0.gguf` from `~/models` added to its list;
  `e2e-gemma2-2b.log`, 17:49 to 17:50 PST, 36 s).
  - **Setup:** a 2,048-deep teacher-forced decode, 24 steps, with the lane against the exact lane at four split counts. The lane
    launched 2,496 times.
  - **Result:** the lane's worst cosine is 0.9738, inside the accepted V-sum reduction-order family's 0.9600 on the same chaotic
    synthetic input. This is the wiring criterion, not a fidelity gate.
- **Unchanged around it:** the Gemma 2 parity and prefill bit-identity tests pass; CUDA's admission and memory censuses, with their
  plants, pass; the PTX consistency and embed checks pass (67 modules); the kernel local-memory census passes; `TestKV16_exactInput*`
  passes.
- **Checks:** `touched.py` PASS (16); staticcheck 0.8.1 is clean on `backend/cuda` at the cuda tags.

## Owed: the timed cell (by day; nothing timed ran tonight)

Merging this changes what Gemma 2 runs on CUDA. The lane is default-on from 2,048 keys (`flashDecodeDefaultMinKeys`), so a Gemma 2
decode past 2,048 keys moves from the exact split-KV lane to the capped flash-decode lane. No timed read of that exists yet.

**The cell, registered now, before any timed data:**
- **Arms:** a goinfer-vs-goinfer two-arm A/B (TE5(a)). `goinfer` is a serve built at this branch's head; `goinfer_old` is one built at
  its merge-base with `main`, where the capped lane is off.
- **Model:** `gemma-2-2b-it-Q8_0.gguf` from `~/models`. `bench_peer.py` has no Gemma 2 key yet; one line adds `"gemma2-2b"` to
  `MODELS`.
- **Shape:** depths 128 and 2,048; `BENCH_RUNS=2`; two passes with the arm order reversed (A B B A); the timing lock, and the void rule
  per cell.
- **Rule:**
  - **SHIP:** the paired median new/old decode tok/s is ≥ 1.02 at depth 2,048 and ≥ 0.98 at depth 128.
  - **REGRESSION:** under 0.98 at depth 2,048. The lane is then kept off capped layers again, by `faEligible`, until a fix.
  - **Between:** parked, to Francis with the numbers.
- **Command** (after the key is added):

  ```sh
  GOINFER_SERVE_CUDA=<new serve> GOINFER_SERVE_CUDA_OLD=<old serve> BENCH_BACKENDS=cuda BENCH_ENGINES=goinfer,goinfer_old \
    BENCH_MODELS=gemma2-2b BENCH_DEPTHS=128,2048 BENCH_RUNS=2 python3 scripts/bench_peer.py <out>.json
  ```

  Then the same with `BENCH_ENGINES=goinfer_old,goinfer`.
- **Wall time:** 2 arms × 2 depths × 2 runs × 2 passes, about 1 min a cell for a 2B at these depths: about 15 min, plus idle-gate waits.
