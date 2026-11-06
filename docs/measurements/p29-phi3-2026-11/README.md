# P29 on Phi-3: why the one-slot step prefills 4.7× slower than the sequential loop (diagnosis, 2026-11-06)

Diagnosis only; no product change. Question (coordinator, from sb11m's reading of `TestOneSlotStep_phi3PrefillAB`, `~/goinfer-logs/pr709-metal-2026-11-06/README.md`): on Phi-3 mini q4k at one KV slot the step prefills 512 tokens at median 0.213× the sequential loop (about 110 ms a token against 24). Why?

**Machine and method.** MacBookPro18,3 (M1 Pro, 16 GB), macOS 26.6.2, on AC. Tree `origin/main` c1182f99 plus one uncommitted probe test (`p29_probe_test.go.txt`, not part of the product; it needs `goinfer_testhooks`). Tier 1, exclusive timing lock (`timing_lock.py run`), `caffeinate -i`, `GOMAXPROCS=4`, checkpoints from `~/models` only. The probe loads the file at `--quant q4k`, one resident KV slot, ctx 2048, behind the same fit and memory guards as the A/B (never bypassed), and times (a) `Forward` per token, (b) `forwardMultiOut` steps of B = 2, 4, 8 rows (the route `prefillByStep` uses; `multiNoHead`), each at four positions; (c) the dispatch, command-buffer and host-wait counts of one token and one step, by pipeline; (d) layer 0's four GEMV groups alone through `q4kStepGemv`, B = 1..8, 20 calls in one command buffer, GPU time from the buffer's own timestamps. Fit check first: Metal RESIDENT, dense 2.29 GB + host copy 2.62 GB + KV 0.77 GB against 8.31 GB free.

## Finding

**The loss is in the batched step's native K-quant GEMV kernels, `mc3_gemv_q4k_f32_rows` and `mc3_gemv_q5k_f32_rows` (`backend/metal/batch_rows.go`). They do not amortize the weight read across rows, and they fall off a cliff at B ≥ 5.** Nothing else in the step differs from the sequential token.

### What it is not (measured)
- **Not a fallback, a re-upload or a sync per row.** One token: 322 dispatches, 1 command buffer, 1 host wait. One 8-row step: 320 dispatches, 1 command buffer, 1 host wait. GPU-busy is 951.8 ms of a 952.9 ms wall (run 1): no host bubble.
- **Not per-row dequantization on the host, and not attention.** The same 8-row step costs 944 → 958 ms from position 0 to 448 (+1.5%), and `pAttnRows` is 32 dispatches of the 320. Phi-3's head dim 96 keeps it off flash-decode (`attnFAHeadDimOK` takes only 128, or 64 at group 7), but both arms then run the plain kernels and position barely moves the step.
- **Not the step's fixed cost.** The four GEMV groups at B = 1 sum to 24.4 ms over 32 layers (0.764 ms a layer, run 2), about all of `Forward`'s 23.0 ms token (run 1): the body GEMVs are essentially the whole token, and at B = 1 the rows kernels cost what `Forward` costs.

### What it is (measured)
Phi-3 (run 1, the valid cell; ms; the 5.1/4.9/4.8/4.6× column is step ÷ (B × Forward)):

| position | Forward /token | step B=2 | step B=4 | step B=8 | B=8 per row vs Forward |
|---|---|---|---|---|---|
| 0 | 23.0 | 62.9 | 142.2 | 943.6 | 5.12× |
| 128 | 23.8 | 63.9 | 143.6 | 946.2 | 4.97× |
| 256 | 24.4 | 65.2 | 147.0 | 953.4 | 4.88× |
| 448 | 25.5 | 66.9 | 150.1 | 958.0 | 4.70× |

The step is already slower per row at B = 2 (1.32–1.37×) and B = 4 (1.47–1.54×), then jumps 6.6× from B = 4 to B = 8. No B beats the sequential loop. Layer 0, GPU ms per call, by B (run 2 / run 3 agree within 12%; both are void by the rule, below, so read the shape):

| group (kernel, rows) | B=1 | B=2 | B=4 | B=5 | B=8 | B=8 ÷ B=1 |
|---|---|---|---|---|---|---|
| qkv (q5k, 9,216) | 0.192 | 0.458 | 1.040 | 1.764 | 7.647 | 40× |
| o + residual (q4k, 3,072) | 0.065 | 0.154 | 0.388 | 0.853 | 2.616 | 40× |
| gate\|up (q4k, 16,384) | 0.320 | 0.797 | 1.958 | 4.485 | 13.831 | 43× |
| down + residual (q6k, 3,072) | 0.187 | 0.381 | 0.528 | 0.622 | 1.028 | 5.5× |

At B = 8 the four groups cost 25.1 ms a layer, 804 ms over 32 layers, 84% of the 952 ms step. Eight rows through a kernel that read the weights once should cost about one row's time while memory-bound (gate\|up: 28.3 MB of Q4_K at 0.32 ms is 88 GB/s); it costs 43×. The q6k kernel is the only healthy one (5.5× for 8 rows). Per row, q4k gate\|up costs 0.32 → 0.40 → 0.49 → 1.73 ms at B = 1, 2, 4, 8: the work scales with rows, then worse.

### Why the 0.5B did not show it (and a third model that does)
Same probe, step ÷ (B × `Forward`); 0.5B and 1.5B from run 3 (valid cells, position 0), Phi-3 from run 1 (valid):

| model | quant mix in the step | Forward /token | step B=2 | B=4 | B=8 |
|---|---|---|---|---|---|
| qwen2.5-coder-0.5b q4_k_m (H 896, 24 layers, hd 64) | int8 rows in 3 of 4 groups, the down projection q6k | 5.4 ms | 0.52× | 0.44× | **0.62×** (faster) |
| qwen2.5-coder-1.5b q4_k_m (H 1536, 28 layers, hd 128) | native q4k (and q6k), no int8 | 14.3 ms | 0.88× | 1.05× | **3.15×** (slower) |
| Phi-3 mini 4k q4 (H 3072, 32 layers, hd 96) | native q4k, q5k, q6k, no int8 | 23.0 ms | 1.37× | 1.54× | **5.12×** (slower) |

The 0.5B's GEMVs run on the int8 rows kernel (`mc3_gemv_w8_f32_rows`), which scales 10× for 8 rows (gate\|up 0.033 → 0.347 ms), and its `Forward` is short and dispatch-heavy (242 dispatches in 5.2 ms of GPU time), so batching 8 rows into one command of the same 240 dispatches wins. On Phi-3 and the 1.5B every projection is a native K-quant GEMV, `Forward` is already weight-bandwidth-bound, and the step has nothing to win unless its kernels amortize the read, which they do not. **P29's ship (2.27× on the 0.5B) does not generalize: the 1.5B q4_k_m on the same route is 3× slower at B = 8, the same defect at a smaller size.** (1.5B gate\|up: q4k 0.206 ms at B=1, 8.572 at B=8, 42×.) The slow kernels belong to the opt-in `--quant q4k` lane (`r.q4kLane`, `q4kStepGemv`); other residents take the step on their own rows kernels, which this probe did not measure.

### Cause inside the kernel: a hypothesis, not established
Both slow kernels unroll the row loop fully (`SA_ROWS_UNROLL` is `clang loop unroll(full)`, `kernels/batch_msl.go`) around a body that holds `float4 al[4], ah[4]` per row and converts the weight nibbles to float once per row (`float(w & 15u)` inside the `r` loop), so register pressure grows with B, and the dequantization work repeats per row. The cliff between B = 4 and 5 (q4k gate\|up 1.96 → 4.49 ms for 1.25× the rows; q5k qkv 1.76 → 7.65 across B = 5 to 8) fits a register spill or an occupancy drop, and the flat-to-rising per-row cost below it fits the repeated dequantization. I did not take a compiler spill report or a GPU counter capture, so this is the reading to test first, not a finding. The q6k kernel, whose body differs (`q6k_blk` over `a[8]`), does not cliff.

## Options, each with its cost (Francis decides)

| option | what changes | cost | what it buys / risk |
|---|---|---|---|
| **1. Gate the step off for residents whose GEMVs are native K-quant** | `promptStepOK` (and `stepPrefill` where P29 needs it) returns false when the resident has q4k/q5k segments and no int8 rows, so Phi-3 and the 1.5B prefill sequentially | about 20 lines, a route-counter test of the `p29_route_test.go` shape, a P29 amendment; hours | Phi-3 512-token prefill from about 56 s back to about 12 s (24 ms × 512); keeps the 0.5B's 2.27×. Bit-identical (the sequential loop is the reference). Does not touch concurrent decode (see risks) |
| **2. Calibrate per model** | extend the build-time `calibrateRows` (today it serves `rowsQKV`/`rowsGU` and the non-q4k lane) to the q4k lane: per B, step rows kernel vs production's per-row GEMV, whichever is faster, none → sequential | about a day, tier 0 bit-identity tests, build time cost to be measured | keeps wins where they exist (the 1.5B at B = 2: 0.88×); no hard-coded rule |
| **3. Fix the kernels** | rows kernels dequantize each weight block once, then accumulate B rows; cap the unrolled rows per pass (4) to bound registers; same per-row arithmetic order | one to two days, bit-identity against decode (`TestOneSlotStep_q4kLaneBitIdentical` and the KS rows tests exist), A/B on all three models above | the only option with upside: a bandwidth-bound ideal is several times faster than sequential. Risk: a changed accumulation order changes bits, which G-Q1/G-Q2 and the step-vs-decode identity forbid, so the order must be kept |
| **4. Do nothing, amend P29** | `docs/queue-performance.md` P29 says "0.5B only" | minutes | the regression stays for `--quant q4k` on Phi-3 and the 1.5B |

Capping the step at 4 rows does not help: B = 4 is already 1.5× slower per row on Phi-3 and 1.0× on the 1.5B. My reading: 1 now, because it is small, reversible and restores the sequential speed, then 3 as its own tracked item with 2 as the fallback if the kernel's order cannot be kept.

## Risks outside this question (not measured)
- **Concurrent decode (MC3) and the n-gram verify use the same rows kernels** (`StepBatch`, `PrefillLastNArgmax`, `stepVerifyCost`). At two clients on Phi-3 the step costs 1.37× per row what the sequential token does (run 1), so batching two q4k clients may be slower than serving them one after the other; on the 1.5B B = 2 is 0.86×. I measured the kernels, not the served path.
- The registered P29 bands (`docs/queue-performance.md`) are written on the 0.5B; Phi-3's reading would be OFF (median under 1.00) by them, and its A/B passes only because it asserts no band.

## Runs, voids, and what is not clean
- **Run 1 (`raw/run1.log`, 07:13): valid.** Load 2.68, 8.19 GB reclaimable against a need of 5.68 GB + 2 GB. The probe then lacked the per-kernel section.
- **Run 2 (`raw/run2.log`, 07:28): VOID by the load rule** (1-minute load 5.02 > 4.0; the rule is half the Mac's 8 cores), kept because its shape matches runs 1 and 3 (steps within 3%, per-kernel within 12%); it recorded 6.11 GB reclaimable, also under the 7.68 GB needed, so it is void twice. Read its per-kernel table for shape, not magnitude.
- **Run 3 (`raw/run3.log`, 07:31): Phi-3 cell VOID by the memory rule** (6.84 GB reclaimable below the 7.68 GB needed; load 3.76). Its 0.5B and 1.5B cells are valid (load 3.79 and 3.65, 8.64 and 8.41 GB reclaimable). Listed, not averaged.
- **Run 4 (`raw/run4.sh`): aborted, no reading.** It was meant to re-take the Phi-3 cell once load was at most 4.0 and 7.7 GB were reclaimable. It polled for those inside the exclusive lock, so it held the lock for 14 minutes (07:34 to 07:48) while memory sat near 4.4 GB and other sessions' builds were refused; I stopped it. A re-take should poll outside the lock and take it only when both gates hold. Until then Phi-3's per-kernel magnitudes rest on voided cells and its step table on run 1.
- The pre-existing A/B (0.213) and this probe agree: 8 rows at 944–958 ms is 118–120 ms a row against 23–25 ms, 0.19–0.21.
- Other sessions' loaders ran beside the lock at times (touched.py, a Metal package test), as the lock status showed; none held it exclusive during a cell.

## Outcome: the kernel fix (Francis's option 3, 2026-11-06)

**Fixed, to parity or better.** Branch `p29-rows-kernels-2026-11`: the step's Q4_K and Q5_K rows kernels (`backend/metal/batch_rows.go`) now loop over the B activation rows (a loop the compiler is told not to unroll) and compute two output rows per simdgroup (`stepRowsNR`). Same arithmetic for every (output row, activation row), same lane map, so the bits are decode's. Q6_K was already healthy and is unchanged.

### End-to-end A/B (exclusive lock, 11:31–12:01 PST, order old new new old)
Sequential loop ÷ step on a 512-token prompt at one KV slot, `oneSlotPrefillAB` (the body of `TestOneSlotStep_phi3PrefillAB`, called through a scratch test with each model's `~/models` path), 5 pairs per run, two runs per arm; "old" is the kernels at `origin/main`. Above 1 the step is faster. Every cell started at load1 ≤ 2.8 and with ≥ 8.6 GB reclaimable (Phi-3 needs 7.68), listed in `fix/ab.log`; none void.

| model | old, run 1 / run 2 (medians) | new, run 1 / run 2 | new's 10 pairs |
|---|---|---|---|
| Phi-3 mini 4k q4 | 0.213 / 0.215 | **1.140 / 1.151** | 1.131 to 1.162, all above 1 |
| qwen2.5-coder-1.5b q4_k_m | 0.344 / 0.343 | **1.561 / 1.592** | 1.554 to 1.593 |
| qwen2.5-coder-0.5b q4_k_m | 1.632 / 1.636 | **2.373 / 2.395** | 2.260 to 2.409 |

Every new pair beats every old pair (the largest old Phi-3 pair is 0.248; the smallest new one 1.131). Phi-3 goes from 4.7× slower than the loop to 1.14–1.15× faster, about 5.3× the old step. **Caveat, like for like:** the step runs no head on its first 511 tokens and the loop runs the head on all of them, so each ratio includes the skipped head work. That is small on Phi-3 (32,064 × 3,072) and larger on the 1.5B and 0.5B (151,936-wide vocabulary); I did not measure the split, so read the 1.5B and 0.5B gains as "up to this much from kernels". Phi-3's 1.14× is the most conservative of the three.

### MC3 concurrent decode, two clients on Phi-3 (reported, not graded)
Two single-token `Forward`s ÷ one two-row step on two slots, both with the head, 10 pairs in alternating order, positions 64–73, `ctx 1024`:

| arm | median | range |
|---|---|---|
| old, run 1 / run 2 | 0.778 / 0.727 | 0.721 to 0.958 |
| new, run 1 / run 2 | **0.999 / 0.998** | 0.994 to 1.010 |

At two clients the step costs the same as serving the two requests one after the other; before, it cost about 1.3–1.4× as much. No gain at two clients, no loss.

### Bits and the registered bar
- `TestKSC1_stepMatchesDecodeKernels` (Q4_K and int8, B = 1..8, K = 256, 1536, 3072, 4864, three modes, byte-equal against decode's rows kernels), `TestKSO1_stepMatchesDecode` (Q5_K and Q6_K, B = 1..8), `TestKSO1a_q4kBitsUnchanged`, `TestOneSlotStep_*` and `TestQ4KStep_*` all pass. `TestKSC1_stepRowsPlantIsRed` rotates the lane groups in the Q4_K kernel and 116,499 outputs differ. (Reversing the groups is not a plant: `simd_sum` is symmetric under it; two other first plants, swapping two terms of a sum and regrouping it, did not change a bit either.)
- CLAUDE.md's bit-identical rule: bit-identical, yes; scratch and allocated memory not higher (the kernels add no buffer or threadgroup memory); faster on 10 of 10 timed pairs on Phi-3 against the loop (and against every old pair); it replaces the two kernels rather than adding a second path. All four hold.

### What the kernel work found (all GPU ms per call, Phi-3 gate|up, B = 8 unless noted; logs in `fix/`)
- The **full unroll** over the rows was the cliff: with the row loop not unrolled, 9.8 → 2.6 (`ablation.log`).
- It is **not** the weight conversion (hoisting it changed nothing), the **row stride** (padding the activation rows by 16 to 128 floats: 10.1, 10.2, 10.3, 10.3, 10.3), or the **device loads** (staging the activations in threadgroup memory, patch kept: 10.7).
- My first "all rows read the same activation row" run, 0.39, was invalid: the compiler merges identical rows.
- **The ceiling is the sequential loop's.** Run one row at a time, the new kernel's 8 rows cost 1.85 ms; the decode kernel run eight times costs 1.86 ms. Tiles of 2 or 4 rows with the weights converted once were no faster (2.2 to 7.4, `rowtile-hoisted.log`), and 3 to 4 output rows a simdgroup match 2 (6 and 8 are slower, 2.5 and 5.5 ms). The per-(activation row × super-block) work is the floor, and it cannot be cut without changing the arithmetic order. What the step wins over the loop on Phi-3 comes from the narrow projections (the o-projection 0.36 ms against 0.68–0.82 for eight decode calls: decode's 8 output rows a simdgroup leave 384 simdgroups) and from one command buffer for the whole prompt, not from sharing weight reads.
- That also says something about decode: at N = 3,072 the decode rows kernel at 8 rows a simdgroup is 1.8–2.2× slower than 2 rows a simdgroup (0.085–0.102 against 0.047 ms, one row). Not pursued here.

Experiment sources: `fix/rows_timing_test.go.txt`, `fix/rowpass_experiment_test.go.txt`, `fix/ab_test.go.txt` (scratch; not in the tree, which a layout ratchet protects).
