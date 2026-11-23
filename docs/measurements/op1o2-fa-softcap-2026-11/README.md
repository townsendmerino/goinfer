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

## Addendum, 2026-11-15 11:53 PST (commit `20bb8e0f8`; first typed as 11:55, corrected from the commit time), before the timed cell's first run: the instrument, written down

The README above registers the cell's rule and shape. This fixes the parts it left to the harness, before any timed run.

- **Binary and flag.** One serve binary from `8f4273a32` (PR #998: `--flash-decode-softcap`, the load flag the cell needed), built
  `CGO_ENABLED=0 go build -tags cuda ./backend/cuda/cmd/serve`. `GOINFER_SERVE_CUDA` and `GOINFER_SERVE_CUDA_OLD` name that one file; the
  `goinfer` arm adds `-flash-decode-softcap` through `BENCH_GOINFER_ARGS` (the harness gives that to both engines) and the
  `goinfer_old` arm adds `-flash-decode-softcap=false` through `BENCH_GOINFER_OLD_ARGS`, after it, so the exact split-KV lane is
  `goinfer_old`'s. No peer arm.
- **Checkpoint.** `~/models/gemma-2-2b-it-Q8_0.gguf` (local NVMe), `-quant int4` (the harness's default), the checkpoint the fidelity
  gate read. `bench_peer.py`'s `gemma2-2b` key. The fit check by serve's banner, flag on, passes: "4 KV slots at a 8192-token cap hold
  5.0 GB KV (weights 1.4 GB, budget 42.0 GB)" (`cell-fit.log`).
- **Prompts.** `scripts/prompts.json` gains `gemma2-2b:128` (129 tokens) and `gemma2-2b:2048` (2,048), calibrated against this binary's
  `usage.prompt_tokens` by `scripts/bench_prompts_calibrate.py`.
- **Shape.** `BENCH_PHASES` default, `BENCH_DEPTHS=2048` (phase A is the depth-128 prompt, phase B depth 2,048), `BENCH_RUNS=2`,
  `BENCH_ABBA=1`: one counterbalanced quad per depth, the engine order the plan's in block 1 and reversed in block 2. That is the
  registered "two passes in reversed arm order" in the harness's own form (`abba_plan`), each cell a restarted server. Exclusive
  timing lock; the harness's idle gate (`BENCH_MAX_LOADAVG=1.0`) and the void rule stand, and a void cell is re-run in order and listed.
- **The statistic.** Per depth, per block, `goinfer`'s mean rate over its runs divided by `goinfer_old`'s; the "paired median" is the
  median of the blocks' ratios (two blocks, so their mean). The rule is the README's: **SHIP** at ≥ 1.02 at depth 2,048 and ≥ 0.98 at
  depth 128; **REGRESSION** under 0.98 at depth 2,048; otherwise parked. A block with an errored or missing arm is void.
  `grade_cell.py` prints each block's ratio beside the median.
- **Both arms must really differ.** The serve log has no line for the lane being taken, so `probe_arms.py` runs first, untimed: the
  same depth-2,048 prompt with an instruction that forces a long reply, greedy, 256 tokens, on the exact arm twice and on the lane arm
  once. The exact arm must repeat itself, both replies must run past 200 tokens, and the lane arm's text must differ from the exact
  arm's (the lane's outputs differ from the exact lane's: the README above reads 24 of 24 steps differing); identical text means the
  flag did not reach the lane and the cell does not run. Neither arm's serve log may carry a "flash-decode lane unavailable" warning.
  - **Amendments, before any timed run.** The probe first compared a 64-token greedy reply on the lane and the exact arm; the reply was
    a short generic sentence and read identical on both (`probe-first-run.log`), which cannot tell a lane not taken from an argmax that
    did not move. The second version compared the top-5 log-probabilities and read bit-identical on both arms and twice on the exact
    arm: serve scopes a request that asks for log-probabilities to exact attention (`EnterExactAttention`), so that probe could never
    show a difference. The version above asks for neither.
- **Wall time:** 8 restarted cells of about 2 to 3 minutes, plus idle-gate waits: about 25 min. Started and finished times are in the
  reading below.
- **A default change needs Francis.** A SHIP here, with the fidelity gate's PASS, is the evidence for that decision, not the decision.

## Reading, 2026-11-15 12:00 to 12:05 PST (nobara, RTX 2070 SUPER, driver 595.112.07): **SHIP** by the registered rule

- **Run.** Binary built from `8f4273a32` (PR #998), the registration above (`8e50ba4c2`) committed first; both arms on that one file, the lane
  arm `-flash-decode-softcap`, the exact arm `-flash-decode-softcap=false`. Exclusive timing lock, `BENCH_RUNS=2`, `BENCH_ABBA=1`: 8
  restarted cells, 4 a depth, blocks in A B / B A order. Records in `cell-2026-11-15/`: `decode-cuda-all.json`, `decode-cuda.log`,
  `provenance.txt`, `verdict.txt`, the serve logs; graded by `grade_cell.py`.
  - **Start state.** Load 0.20 and 60 GB available at the start; the harness's own idle gate held three cells for a 1-minute load of 1.03
    to 1.15 and let them go under 1.00. No cell started under the void rule's 8.0, none was voided, and none errored.
  - **Before it:** the untimed probe passed twice more inside the script (the arms' 256-token greedy texts differ, the exact arm repeats
    itself). A first launch (`first-launch-refused-no-go-on-path.log`) refused to measure, after its probe: `go` was not on the harness's
    PATH, so it had no free-memory reading, and it wrote no results. It was re-launched with `go` on PATH; nothing else changed.
  - **Wall time:** 4 min 41 s, against the 25 min the registration above estimated; a cell is about 30 s, not 2 to 3 minutes.

| depth | lane (capped flash-decode) | exact (split-KV) | lane / exact, per block | median |
|---|---|---|---|---|
| 128 | 177.23 tok/s | 177.40 tok/s | 0.9997, 0.9984 | **0.9990** |
| 2,048 | 143.17 tok/s | 84.58 tok/s | 1.6917, 1.6936 | **1.6927** |

- **Verdict: SHIP.** The median is at least 1.02 at depth 2,048 (1.693) and at least 0.98 at depth 128 (0.999).
- **Depth 128 is flat on purpose.** The lane acts only once a layer's attended span reaches 2,048 keys (`flashDecodeDefaultMinKeys`), so a
  129-token prompt with its decode never leaves the exact path on either arm; 0.999 is the noise between two identical paths.
- **The spread is small.** Each cell's two runs agree within 0.2 tok/s and the two blocks' ratios within 0.002, so the 2,048 reading does not
  rest on one lucky pair. There are two blocks, as registered; the paired median is their mean.
- **What it decides goes to Francis.** SHIP with the fidelity gate's PASS (above, 2026-11-09) is the evidence for making the lane the
  default for capped models; it does not flip `FlashDecodeSoftcap`'s default, and `--flash-decode-softcap` stays off.
  - Two points from the fidelity reading still stand: paired wins were exactly 5/10 and hard flips pass only the amended bar.
  - Gemma 2's window, 4,096, is not crossed at K = 3,900 or at this cell's depths. The windowed lane is held by G2-K, not by either gate.
  - Not measured: depths past 2,048, other capped models, a longer decode, int8 KV (refused for capped families), and TTFT.

## Default, 2026-11-22: on for capped models in chat and serve (Francis)

- **The decision.** Francis, 2026-11-22, relayed by the coordinator: the capped lane becomes the CUDA default for Gemma 2. It rests
  on the fidelity gate's PASS (2026-11-09) and the timed cell's SHIP (2026-11-15), both above.
- **Its shape.** `--flash-decode-softcap` defaults to true in goinfer-chat and serve (`internal/config`'s schema), and
  `--flash-decode-softcap=false` keeps the exact split-KV lane. It follows `--windowed-kv`'s pattern: `decoder.Options.FlashDecodeSoftcap`
  stays a plain bool whose zero value is off, so a library caller's zero `Options` are unchanged. There is no new environment read.
  `TestResolve_flashDecodeSoftcapDefaultOn` holds both apps' default and the opt-out.
- **Memory: none.** A capped model already loads the capped lane's module (`attnPTX` picks `decode_fa_cap` whenever the model has a
  cap) and allocates its partial buffer (`loadFlashDecode` does so for every model the lane is not declined for). The option only
  changes `faEligible`. Device and host bytes are the same either way, so the longest context that fits does not move. The served
  smoke below reads both arms' fit and KV plan banners.
- **Not covered,** as the readings above say: depths past 2,048 in the timed cell; Gemma 2's 4,096 window (G2-K holds the windowed
  lane at window 100); other capped models; and int8 KV, which is refused for capped families.

### Served smoke with the new default, 2026-11-22 14:24–14:26 PST (nobara, RTX 2070 SUPER, driver 595.112.07): PASS

- **What ran.** [`smoke_default.py`](smoke_default.py), under the exclusive timing lock, on one serve binary built from this branch
  (`e3480e58b`, `CGO_ENABLED=0 go build -tags cuda ./backend/cuda/cmd/serve`). Gemma 2 2B (`~/models/gemma-2-2b-it-Q8_0.gguf`,
  local NVMe, `-quant int4`) was served four times: the flag unset, `=false`, given, and `=false` again. Each server took
  `probe_arms.py`'s depth-2,048 greedy request for 256 tokens. kwin_wayland held 19 MiB of the card.
- **The default takes the lane.** The default's 256-token text equals the given flag's and differs from `=false`'s. `=false`
  repeats itself. No arm warned that the lane was unavailable.
- **No memory moves.** Every arm read the same plan: "4 KV slots at a 8192-token cap hold 5.13 GiB KV (weights 1.61 GiB …)",
  with the same KV pool ("a 32512-position pool for 4 conversations; each is guaranteed 8128, and one can use up to 8192"). Each
  server held 7,030 MiB of the card after load. The longest context that fits is unchanged.
- **Records.** [`default-2026-11-22/smoke.log`](default-2026-11-22/smoke.log) and the default arm's serve log,
  `serve-default.log.gz`. A first run is kept as `smoke-first-run-strict-fit.log`. It read FAIL on one check only: it compared
  the whole fit line, whose host budget (41.51–41.65 GiB) is the box's free memory at load. The plan in that line was equal on
  every arm. The script now compares the plan without the budget, and the re-run passed every check.

## Addendum, 2026-11-22, before the run: what the default reaches, and a reading past Gemma 2's window

Review of the default PR (#1230, relayed by the coordinator): the default is keyed on the model having an attention softcap, not on
Gemma 2 2B. The fidelity gate read K = 3,900 and the timed cell stopped at depth 2,048. Neither crossed Gemma 2's 4,096-position
sliding window, while serve's default plan for the 2B caps at 8,192.

- **What the default reaches, from the code.** `cudaResident.faEligible` refuses a capped layer unless `faSoftcap` is set, and the cap
  is `BackendView.AttnLogitSoftcap()`, which only two architectures fill:
  - `gemma2Architecture`: every Gemma 2 checkpoint, from `config.json`'s `attn_logit_softcapping` or the GGUF's
    `gemma2.attn_logit_softcapping`. The released 2B, 9B and 27B all set 50.
  - `gemma3Architecture` passes `config.json`'s `attn_logit_softcapping` through. The released Gemma 3 configs set it to null (0),
    and its GGUF path does not read the key, so no released Gemma 3 reaches the lane by this route.
  - The lane admits head dims 64, 128 and 256 only. Gemma 2 2B and 9B have head dim 256 and 27B has 128, so all three are eligible.
  - On this box's `~/models`, the reachable checkpoints are Gemma 2 2B: `gemma-2-2b-it-Q8_0.gguf`, and the `gemma-2-2b-it` directory,
    which goinfer cannot tokenize (addendum of 2026-11-09). There is no Gemma 2 9B or 27B, and `gemma-3-4b-it`'s config has a null cap.
- **The reading.** The registered gate, unchanged, on one more cell: `G22B:8000`, `gemma-2-2b-it-Q8_0.gguf` at int4 on CUDA, K = 8,000,
  held-out set B. Its CPU f32 reference is `TestPrefillGateReference`'s G22B row with `GOINFER_CPU_REF_KS=8000`. K = 8,000 plus the 64
  scored positions sits inside the model's 8,192, and every global layer attends past the 4,096 window. The local layers attend their
  4,096 window, so they run the windowed lane at full width.
  The gate loads with zero `Options`, so `ResidentWindowedKV` is off: the local layers keep the whole cache, and the window is applied
  by masking. Serve's default ring, window + 512 positions, is the same attention over a different storage layout. It is held by the
  windowed-KV tests, not by this gate.
- **Criteria, unchanged:** (a) hard flips ≤ exact + 2√exact, the bar as amended by Francis on 2026-09-13 (from the strict
  lane ≤ exact) and adopted in `attn-decode-fa-fidelity-PREREGISTERED.md` before the lane existed. The strict reading is printed too.
  (b) agreement ≥ exact − 1.0 pt, and lane ≥ exact on at least half the prompts. (c) mean KL ≤ 1.10 × exact's. The ambiguous band is
  the same. This is a reading, not a re-tune: nothing in the lane changes.
- **What it decides:** nothing by itself. Francis chooses the default's scope with it in hand: on for every capped model, on for
  Gemma 2 only, or off. A FAIL at K = 8,000 is evidence that the 3,900 PASS does not extend past the window.
- **Cost:** the reference is about 80–100 min of CPU, 10 prompts at 8,000 tokens; the 3,900 reference took 37 min. The gate is
  about 7 min under the exclusive lock. Both by day, finishing before 21:30 PST.

### Reading past the window, 2026-11-22 (nobara, RTX 2070 SUPER, driver 595.112.07): **AMBIGUOUS, PARKED** by the registered rule

- **Run.** This branch at `22f2ab3bf`, with both test binaries built once from it.
  - **The CPU reference,** `TestPrefillGateReference` G22B at `GOINFER_CPU_REF_KS=8000`, held the lock shared, 14:49–16:53 PST.
    The first 8 prompts took 1 h 41 m each, in parallel. The test's own deadline guard then held back the last two, because at that
    pace they would have ended past its 3-hour timeout. It kept the 8 it had stored, and a rerun resumed with them and computed the
    last two in 22 min (logs `g22b8k-refs-part1.log.gz` and `g22b8k-refs-resume.log`).
  - **The gate,** `TestFlashDecodeGateVsReference` `G22B:8000`, `GOINFER_CUDA_FLASH_DECODE=16`, held the lock exclusive,
    16:55:59–17:05:34 PST (`run-gate-8000.sh`, `g22b8k-gate.log`). It started at a 1-minute load of 0.80 with 61,589 MiB
    available; kwin_wayland held 19 MiB of the card.
- **Preconditions held:** A/A bit-identical over 64 rows, the lane launched, and the seed rows were identical. 621 of 630
  continuation rows differ between the arms.

| | exact (split-KV) | lane (capped flash-decode) |
|---|---|---|
| mean teacher-forced agreement | 84.69% | 83.75% |
| hard flips | 24 / 640 | 25 / 640 |
| mean KL(reference ‖ arm) | 0.160404 | 0.154706 |

- **(a) hard flips: PASS** on the amended bar, 25 against 24 + 2√24 = 33.8. The strict reading (lane ≤ exact) fails, 25 against 24.
- **(b) agreement: AMBIGUOUS.** The lane is −0.94 pt from exact. That is inside the −1.0 pt bar but within its last 0.2 pt, which the
  registration parks. Per prompt the lane is ahead on 1 and tied on 5, so lane ≥ exact on 6 of 10. It is behind on 4, by 1.56, 1.56,
  3.12 and 4.69 pt.
- **(c) KL: PASS.** The ratio is 0.9645, under 1.10 and outside the 1.05–1.10 band.
- **Verdict: AMBIGUOUS, PARKED.** At K = 8,000, every global layer attends past Gemma 2's 4,096 window, and the lane is not shown to
  decode as well as the exact lane there. Agreement sits in the parked zone on the bar. The 3,900 reading (PASS, −0.31 pt) does not
  extend past the window on this evidence. Per the registration, a re-run needs a named mechanism, not a re-roll.
- **What it leaves to Francis:** the default's scope. The options are on for every capped model, as built; on below the window only,
  which would need new code and its own gate; or off, the opt-in as before.

### Served smoke re-run with the reviewed script, 2026-11-22 17:05 PST: PASS

`smoke_default.py`, after the review's fixes, on the same binary (`e3480e58b`). It took the exclusive lock itself
(`smoke-rerun-fixed-script.log`), and all nine checks passed:
- the default's text equals the given flag's and differs from `=false`'s, and `=false` repeats itself;
- the fit plan and the KV plan are equal and present on every arm;
- device memory is 7,030 MiB on every arm, both at load and after the depth-2,048 request.
