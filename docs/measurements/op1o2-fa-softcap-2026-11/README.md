# OP1-o2: the CUDA flash-decode lane with Gemma 2's attention softcap, opt-in (nobara-sb01, 2026-11-08)

**What.** `docs/completed/task-forward-architecture-2026-10.md` §4.6a, "Owed, not in OP1": the flash-decode lane with the cap.
- `decode_fa.cu` joins the modules built with `-DATTN_SOFTCAP`, as OP1 did for `glue`, `prefill_batched` and `decode_splitkv`.
  - Every `fa_partial*` kernel, the M=1 lane and the 24 multi-row verify kernels, takes `ATTN_CAP_PARAM` after its int8 scales, and
    both score sites call `attn_score(d, scale ATTN_CAP_ARG)`.
  - The variants are `decode_fa_cap.ptx` and `decode_fa_kv16_cap.ptx`, built at the ambient NVRTC 12.9.86 like `decode_fa.ptx`. There
    is no capped int8 build: `CUDAKVI8Decline` refuses every capped family first.
- A capped model's resident loads them through `attnPTX`, and `withCap` appends the cap to both `fa_partial` launches.
- **Off by default.** A capped layer takes the lane only when the load sets `decoder.Options.FlashDecodeSoftcap`, read through
  `Model.FlashDecodeSoftcap()` (no environment read, no CLI flag). Without it, Gemma 2 decodes on the exact split-KV lane as before.
  - The default stays off because the lane's outputs differ from the exact lane's: on Gemma 2 2B, 24 of 24 steps differ and the argmax
    moves on 3 to 6 of 24. The end-to-end read below is a wiring criterion, not a fidelity gate.
  - Making the lane the default for capped models needs the fidelity gate registered below, and Francis's call. This follows the
    coordinator's hold on #806, from Gemini's review.
- **How the bars were set.** The correctness gates below were written alongside the code, not registered before it. The coordinator's
  brief was "build plus correctness tests", and their bars are the lane's and G2-K's existing ones.

## Correctness gates and readings (card tests under the shared timing lock; nothing timed)

- **G2-I, the plain modules unchanged.** `decode_fa.ptx`, `decode_fa_kv16.ptx` and `decode_fa_kv8.ptx` were rebuilt from the unchanged
  source first, and come back byte for byte before and after the edit: in the plain build both macros are empty.
  - G1-K, `TestAttnKernelDigests` (`backend/cuda/kernels`): 2,064 digests equal base's on this card and driver (595.112.07).
- **G2-K for the lane** (`TestAttnSoftcap_flashDecodeAgainstFloat64`, `backend/cuda/kernels`, beside OP1's three kernels).
  - **Coverage:** `fa_partial_{64,128,256}` with `fa_combine`, and every `fa_partial_rows_{hd}_g{G}` the matrix reaches with
    `fa_combine_rows`, from both capped modules (f32 and f16 KV), against the float64 reference, on inputs where the cap acts.
  - **Matrix:** caps 50 and 2; head dimensions 64, 128 and 256; G = 1, 2, 4 and 8; windows 0 and 100; peaked and not; 4 and 16 splits at
    one query; verify batches of 1, 5 and 16 rows at 16 splits.
  - **Result:** 1,344 cases (4,224 verify rows) inside G2-K's bar (cosine ≥ 0.9999, max abs ≤ 1e-2), none void.
  - **Plant:** the kernels handed a lifted cap (1e30) put 4,992 of 4,992 comparisons outside the bar, 2,496 at each KV type
    (`g2k-plant-cap-lifted.log.gz`; the gate run is `g2k-flash-decode.log`, the digests `g1k-digests.log`).
- **The resident path** (`TestGemma2Softcap_flashDecodeVsF64`, gemma2-tiny loaded with `FlashDecodeSoftcap`).
  - The opt-in reaches the resident, and its lane loads `decode_fa_cap`.
  - Through the resident's own launch path it matches a capped f64 oracle: 90 cases, worst 4.6e-7 of the largest output, bound 1e-4.
  - Lifting the cap reddens all 72 cases it moves, and 104 verify rows equal the M=1 lane bit for bit.
- **G2-L, the lane selectors** (`TestGemma2Softcap_lanesCUDA`): a capped layer on a default load is not eligible for the lane; opted in, it
  is eligible on `decode_fa_cap`. The fused prefill kernels, the image prefill and the drafter refuse it either way, and the uncapped
  control sees both lanes.
- **A kernel-source plant:** the rows kernel's score without the cap turns the verify bit identity red
  (`plant-rows-uncapped.log`). Restored, both variants rebuild to the same SHA-256.
- **End to end on a real Gemma 2** (`TestFlashDecodeEndToEnd`, `gemma-2-2b-it-Q8_0.gguf` from `~/models`, opted in; `e2e-gemma2-2b.log`).
  - 2,496 lane launches.
  - The worst cosine against the exact lane is 0.9738, inside the accepted V-sum reduction-order family's 0.9600: the wiring
    criterion holds.
- **Unchanged around it:** the Gemma 2 parity and prefill bit-identity tests; CUDA's admission and memory censuses; the PTX consistency
  and embed checks; the local-memory census; `TestKV16_exactInput*`.
- **Registries:**
  - The option grid gains `FlashDecodeSoftcap` (`docs/option-state-grid.md`, regenerated).
  - The resolve and serve options goldens gain its `false` key, and nothing else moved.

## The fidelity gate, registered before any run (by day; not run tonight)

| | |
|---|---|
| **Question** | Does Gemma 2 on the capped flash-decode lane decode as well as on the exact split-KV lane, both scored against the CPU reference? |
| **Instrument** | `TestFlashDecodeGateVsReference` (`backend/cuda`), the lane's registered gate, with its new `G22B` cell: `~/models/gemma-2-2b-it` (local NVMe), int4, `FlashDecodeSoftcap` on so the lane arm takes the lane, K = 3,900, held-out set B. Preconditions 1 to 3 as registered: the lane arm launched the lane, A/A determinism, the prefill seed row identical across arms. |
| **Reference** | Phase A: `TestPrefillGateReference`'s new `G22B` row, CPU f32 from the same directory, K = 3,900, into `~/goinfer-logs/prefill-ref-b/G22B-K3900-p*.bin`. |
| **Rule** | `docs/measurements/attn-decode-fa-fidelity-PREREGISTERED.md`'s criteria, unchanged: (a) hard flips ≤ exact + 2√exact; (b) paired agreement ≥ exact − 1.0 pt and lane ≥ exact on at least half the prompts; (c) mean KL ≤ 1.10 × exact's. AMBIGUOUS, PARKED: (b) within its last 0.2 pt, or (c) in 1.05 to 1.10 ×. |
| **Outcome** | PASS: the evidence Francis needs to decide whether capped models take the lane by default; this does not flip the default itself. FAIL: the lane stays opt-in, and the record names the failing criterion. |
| **Limit** | Gemma 2's window is 4,096, so K = 3,900 does not cross it. The windowed lane is held by G2-K at window 100, not by this gate. |
| **Fit** | The 2B at int4 with a 4,028-position cache; checked first by serve's own banner. |

**Commands** (in an OP1-o2 worktree, binaries built once from it; references first):

```sh
GOINFER_HEAVY_TESTS=1 GOINFER_PREFILL_GATE_PROMPTS=b GOINFER_CPU_REF_MODELS=G22B GOINFER_CPU_REF_KS=3900 \
  python3 scripts/timing_lock.py run --shared --label op1o2-refs -- <decoder.test> -test.run '^TestPrefillGateReference$' -test.v -test.timeout 3h
GOINFER_HEAVY_TESTS=1 GOINFER_PREFILL_GATE_PROMPTS=b GOINFER_CUDA_FLASH_DECODE=16 GOINFER_FA_GATE_CELLS=G22B:3900 \
  python3 scripts/timing_lock.py run --label op1o2-fidelity -- <cuda.test> -test.run '^TestFlashDecodeGateVsReference$' -test.v -test.timeout 2h
```

**Wall time:** the reference is about 40 to 60 min of CPU. The 1.5B took about 9.5 min a prompt at K = 3,900, computed in parallel;
the 2B has more layers and a 256k vocabulary. The gate itself is about 10 min. In all, 50 to 70 min, by day. The reference is
resumable and content-keyed, so an interrupted run keeps what it wrote.

## The timed cell (by day; after the fidelity gate, only if it passes)

A two-arm A/B (TE5(a)) of Gemma 2 2B decode on CUDA: `goinfer` is the opted-in lane and `goinfer_old` is the default exact lane. One
serve binary at this branch's head serves both arms, with the opt-in on one arm only; that needs a load flag for the option, which is not
built.
- **Shape:** depths 128 and 2,048; `BENCH_RUNS=2`; two passes in reversed arm order; the timing lock and the void rule per cell.
- **Rule:**
  - **SHIP:** the paired median ≥ 1.02 at depth 2,048 and ≥ 0.98 at depth 128.
  - **REGRESSION:** under 0.98 at depth 2,048.
  - **Between:** parked.
- **Wall time:** about 15 min plus idle-gate waits.
- `bench_peer.py` needs a `gemma2-2b` key.

## Addendum, 2026-11-09 03:18 PST, before the fidelity gate's first run: the cell's checkpoint, not its rule

- **What happened.** The registered checkpoint, the `~/models/gemma-2-2b-it` safetensors directory, cannot run in either step:
  - `TestFlashDecodeGateVsReference` tokenizes with `tokenizer.LoadGGUF`, which reads a `.gguf` only.
  - `TestPrefillGateReference` reads the directory's `tokenizer.json`, which goinfer cannot load: "byte_fallback set but \"<0x09>\"
    missing from vocab".
  - The fit check by serve's banner on the directory stopped at the same tokenizer error (`g22b-fit-hfdir.log`). It also wrote a
    1.4 GB int4 sidecar, `gemma-2-2b-it.int4.e4h.cuda.v17.giw`, into `~/models`, which is left there.
- **The change.** Both steps read `~/models/gemma-2-2b-it-Q8_0.gguf` (local NVMe), whose own tokenizer loads. This is the S cell's
  pattern: the CPU reference is f32 from the GGUF's weights, and the gate loads it at int4 on CUDA.
  - Both arms (exact and lane) load the same checkpoint, so the comparison is unchanged in kind.
  - The fit check on it passes (`g22b-fit.log`): "fit: 4 KV slots at a 4028-token cap hold 3.3 GB KV (weights 1.4 GB, budget
    41.3 GB …)", and the server reached "serving on".
- **Unchanged:** K = 3,900, held-out set B, the preconditions, criteria (a)/(b)/(c), the ambiguous band and the outcome rule.

## Reading, 2026-11-09 03:19 to 04:05 PST (nobara, RTX 2070 SUPER, driver 595.112.07): **PASS** by the registered rule

- **Run.** Main `45e3cce50` plus the addendum `123c6e2be`; binaries built once from it. Logs: `g22b-steps.log`, `g22b-refs.log.gz` and
  `g22b-gate.log`.
  - **The CPU reference:** 10 prompts at K = 3,900, 37 min, shared lock. A first launch built the decoder binary without its
    `goinfer_testhooks` tag, ran no test, and stopped at the script's own check; it is kept as `g22b-refs-attempt0-untagged.log`.
  - **The gate:** 196 s, under the exclusive lock.
  - **Void rule:** the gate's first try started at a 1-minute load of 12.44, the reference's own tail, and was **void**. The second
    started at 0.20, with 61 GB available, 5 min later.
- **Preconditions held.** A/A is bit-identical over 64 rows. The lane arm launched the lane, and the prefill seed row is identical across
  arms (the gate fails on either otherwise).

| | exact (split-KV) | lane (capped flash-decode) |
|---|---|---|
| mean teacher-forced agreement | 87.97% | 87.66% |
| hard flips | 12 / 640 | 17 / 640 |
| mean KL(reference ‖ arm) | 0.100507 | 0.097450 |

- **(a) hard flips: PASS.** The lane's 17 is within the amended bar of exact + 2√exact = 18.9. The strict reading (`lane ≤ exact`) fails,
  17 against 12, and is printed as such.
- **(b) agreement: PASS.** The lane is −0.31 pt from exact against a −1.0 pt bar, 0.69 pt clear of it and outside the last 0.2 pt
  (ambiguous) zone. Paired wins are 5/10, exactly the "at least half" line.
- **(c) KL: PASS.** The ratio is 0.970, under 1.10 and outside the 1.05 to 1.10 ambiguous band.
- **Verdict: PASSES.** The lane on Gemma 2 2B decodes as well as the exact lane at K = 3,900 by the lane gate's criteria. 617 of 630
  continuation rows differ from exact's, the change the opt-in exists to gate.
- **What it decides goes to Francis:** whether capped models may take the lane by default. This reading does not flip
  `FlashDecodeSoftcap`'s default.
  - Two points sit at a line, worth weighing: paired wins are exactly 5/10, and hard flips pass only the amended bar.
  - Gemma 2's window, 4,096, is not crossed at K = 3,900.
  - The registered timed cell (README above) is the other half of a default decision, and it has not run.
