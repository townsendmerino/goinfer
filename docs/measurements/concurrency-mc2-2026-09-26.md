# MC2 — batched decode on CPU EARNS on the Mac (1.69–2.04× at B = 4, bit-identical); J8's independent workers do as well; MC3's Metal S0: bit-identical small-M kernels barely amortise (2026-09-26)

MC2 of [`task-concurrency-2026-09.md`](../tasks/task-concurrency-2026-09.md): does one CPU forward carrying B
sequences beat B sequential forwards by enough to fund MC3?

The registered gate is aggregate tok/s at B = 4 against B = 1, earn ≥ 1.25×, kill < 1.1× (loosened 2026-09-26, before
any run). Per-sequence output must meet the registered identity gate. Decision 2 folded J8's N-independent-workers
cell into the same session. S0: [`concurrency-mc2-s0-2026-09-26.md`](concurrency-mc2-s0-2026-09-26.md).

**Verdict on the Mac: EARN, on every cell, with every rep above the line.**
- Batched B = 4 against the production single-token forward: **1.945× and 1.690×** on the 0.5B (depth 128 / 512),
  **2.042× and 1.798×** on the 1.5B.
- The batched output is **bit-identical** to single-sequence decode.
- The Linux CPU cells (`nobara`) are registered and not yet run. The verdict is the Mac's until they are.

**The J8 cell is just as large.** Four independent decode workers on one model reach 2.00–2.48×, level with or above
batching. One sequence's decode does not fill the M1 Pro's cores, so on this CPU concurrency alone recovers most of
what batching does, without a batched forward.

## The prototype

`decodeMultiStep` (`decoder/batchdecode_mc2_test.go`, test-only, as MC2 registered) is one forward carrying B
sequences, each with its own `KVCache` at its own position, one token each:
- q/k/v, o, gate/up, down and the LM head run as M = B matmuls through the kernels `forwardN` uses;
- attention runs per sequence over its own cache, in the same `Append` + `attendBatchedHeads` (acc64) sequence that
  single-token decode's default case runs.

It covers plain dense families: generic forward, pre-norm, dense MLP, f32 append-forever KV, no adapter
(`mc2Eligible`).

**Identity gate (registered: "if [activation quantization] is per row, batched output should be bit-identical").**
It is per row (`MatmulBTW4A8Batch` quantises each of the M rows with its own scale), and `forwardN`'s contract already
makes M = B projections match M = 1 row for row. `TestMC2_decodeMultiStepBitIdentical` checks the whole step: 4
sequences at depths 5 / 23 / 40 / 11, stepped 12 times together, teacher-forced, against the production `forward` on
copies of the same caches. Every logit is bit-identical on the committed `llama-tiny` fixture (f32, runs in CI), and on
qwen2.5-coder-0.5b at int4 and int8int8.

## Measurement (Mac CPU)

`TestMC2_batchedDecodeThroughput`:
- M1 Pro 16 GB, macOS 26.6.2, CPU backend, goinfer `4786d879` plus the test file;
- qwen2.5-coder-0.5b and -1.5b instruct q4_k_m from `~/models`, int4;
- 8 caches prefilled to the depth, then 16 greedy steps per sequence per arm, every arm rewinding its caches to the
  depth first;
- 5 reps, arms interleaved rep by rep with rotating order, median;
- idle-gated (load1 1.87 at start), 2026-09-26 16:51–16:53 PDT.

The first launch ran nothing: `go` was not on the launchd job's PATH
([`run-mac-attempt1-nogo.log`](concurrency-mc2-2026-09-26/run-mac-attempt1-nogo.log)). Raw:
[`run-mac.log`](concurrency-mc2-2026-09-26/run-mac.log).

Aggregate decode tok/s, median, and × serial x1, the production forward on one sequence:

| model | depth | serial x1 | serial x4 | batched B=1 | B=2 | **B=4** | B=8 | J8 N=2 | **J8 N=4** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5B | 128 | 108.9 | 1.006× | 1.004× | 1.303× | **1.964×** | 2.296× | 1.799× | **2.478×** |
| 0.5B | 512 | 94.5 | 0.980× | 0.967× | 1.267× | **1.707×** | 1.920× | 1.760× | **2.445×** |
| 1.5B | 128 | 52.8 | 0.995× | 0.987× | 1.274× | **2.044×** | 2.424× | 1.551× | **2.002×** |
| 1.5B | 512 | 46.3 | 0.995× | 0.995× | 1.229× | **1.800×** | 2.076× | 1.633× | **2.155×** |

**The registered metric**, batched B = 4 ÷ serial x1, paired per rep (5 reps, sorted):

| model | depth | metric | per-rep range |
|---|---:|---:|---|
| 0.5B | 128 | **1.945×** | 1.869–1.997 |
| 0.5B | 512 | **1.690×** | 1.616–1.795 |
| 1.5B | 128 | **2.042×** | 2.026–2.056 |
| 1.5B | 512 | **1.798×** | 1.786–1.823 |

The results read as follows:
- **The prototype costs nothing at B = 1:** 0.97–1.00× the production forward. So the batched gain is batching, not
  a faster reimplementation.
- **Depth erodes it, as S0 predicted.** Attention is per sequence and does not amortise: at 512 the B = 4 gain falls
  to 1.69–1.80×, from 1.95–2.04× at 128. It sits below S0's single-sequence ceiling (2.1–2.3×) by about that share.
- **J8's workers match or beat batching here, which was not expected.** J8's registered band is ≥ 1.25× aggregate at
  4 concurrent requests, with p99 per-request latency ≤ 1.5× the single-worker figure. The aggregate clears it
  (2.00–2.48×).
  - The latency half is **not measured** as registered: the test times aggregates, not requests.
  - What the aggregates imply: with 4 running concurrently each sequence runs at 0.50–0.62× its alone rate, so a
    request takes 1.6–2.0× as long as alone.
  - For comparison, under serialisation the last of 4 queued requests waits about 4×.
  - Batching implies the same per-request slowdown (B = 4 at ~2× aggregate is 0.5× per sequence).
- **The two are probably not additive**, since both fill the same cores. They are not measured together.

## MC3 S0 — a bit-identical small-M W4A8 kernel on Metal (exploratory)

MC3's first new piece is a small-M W4 GEMM, and S0 of MC2 flagged that Metal's batched verify costs ~0.86–0.96 of a
decode per extra row (R12). `TestMC3SmallMProbe` (`metal/gemm_smallm_mc3_test.go`) extends R18's shipped
rows-per-simdgroup kernel to M activation rows. It is timed standalone per dispatch, with SLC-defeating rotation,
against the shipped M = 1 kernel run M times. Every variant's outputs are checked bit for bit against the shipped kernel
row by row, and all are identical. Logs: [`mc3-s0-probe.log`](concurrency-mc2-2026-09-26/mc3-s0-probe.log),
[`mc3-s0-probe2.log`](concurrency-mc2-2026-09-26/mc3-s0-probe2.log).

Best gain against M sequential dispatches:

| shape | int form M = 2 | **int form M = 4** | int form M = 8 | masked-half form (H) M = 2 | H M = 4 |
|---|---:|---:|---:|---:|---:|
| 1.5B qkv | 1.28× | **1.52×** (R=4) | 1.11× | 1.57× | 1.42× |
| 1.5B o | 1.28× | **1.48×** | 1.15× | 1.29× | 1.21× |
| 1.5B gate/up | 1.17× | **1.30×** | 0.81× | 1.10× | 0.91× |
| 7B qkv | 1.12× | **1.20×** | — | 1.15× | 0.75× |
| 7B o | 1.23× | **1.35×** | — | 1.20× | 0.76× |
| 7B gate/up | 1.09× | **1.15×** | — | 1.05× | 0.65× |

- **Bit-identical scalar small-M kernels barely amortise on Metal.** On the shape that dominates a token, gate/up, the
  best M = 4 gain is 1.30× (1.5B) and 1.15× (7B). The int form re-unpacks every nibble for each of the M rows, and R18
  found these kernels bound by per-weight work, so the arithmetic grows almost linearly with M.
- H, MLX's masked form, shares the unpack across the M rows but spills registers at R·M ≥ 8, collapsing to 0.24–0.45×.
- So MC3, batched decode on Metal, is unlikely to clear its ≥ 1.2× bar on a bit-identical integer kernel. The route
  that could pay is the matrix units (`simdgroup_matrix`, as fast prefill uses), with M padded to 8. That means f16
  products, so **MC3 would need a registered fidelity gate instead of bit-identity.** That design decision belongs to
  the owner before MC3 starts.
- *Superseded the same day (the fidelity-gate conclusion only).*
  [`concurrency-mc3-s0-2026-09-26.md`](concurrency-mc3-s0-2026-09-26.md) built the matrix-unit kernels with exact
  integer operands and production's reduction order. They are bit-identical, at 2.9–8.0× 8 sequential GEMVs, so no
  fidelity gate is needed.
- **A side finding worth its own brief.** At M = 1, H beats the shipped R18 kernel standalone, bit-identically:
  - gate/up 0.117 against 0.137 ms (1.17×, 1.5B) and 0.452 against 0.556 ms (1.23×, 7B);
  - qkv 1.32× (1.5B).

  R18 tested float forms, but staged activations as shorts and masked 32-bit words. The difference here is
  half-staged, pre-scaled activations and 16-bit masks. This is standalone only, and R18 showed standalone gains need
  in-sequence grading.

## Linux CPU cells (`nobara`, added 2026-09-26 evening)

Setup: amd64 with VNNI and split-half kernels, the same test at `1fce456b`, 2026-09-26 20:38–20:45 PDT, load 0.34 at
start. Raw: [`run-linux.log`](concurrency-mc2-2026-09-26/run-linux.log), [`run-linux.sh`](concurrency-mc2-2026-09-26/run-linux.sh).
The identity gate passes on amd64 too: every logit is bit-identical, on llama-tiny and on the 0.5B at int4 and int8int8.

| model | depth | serial x1 | batched B=1 | B=2 | **B=4 (metric, paired)** | B=8 | J8 N=2 | **J8 N=4** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5B | 128 | 46.1 | 0.814× | 1.103× | **1.230×** (1.18–1.24)* | 1.344× | 1.374× | **1.638×** |
| 0.5B | 512 | 42.6 | 0.763× | 1.010× | **1.113×** (1.10–1.13) | 1.232× | 1.374× | **1.728×** |
| 1.5B | 128 | 19.6 | 0.925× | 1.351× | **1.475×** (1.45–1.49) | 1.504× | 1.235× | **1.329×** |
| 1.5B | 512 | 18.3 | 0.933× | 1.264× | **1.396×** (1.36–1.40) | 1.410× | 1.245× | **1.364×** |

\* This cell overlapped a 1.2 GB rsync from this box (the set-A reference copy). Its reps 4–5 dip in every arm, so the
paired ratio holds, but it is the least clean cell here. **Re-run clean 2026-09-27 06:27 PDT** at `35feb843`, with load1
0.94 and nothing else running: batched B = 4 **1.185×** (1.16–1.20, paired), J8 N = 4 1.613×, serial ×1 47.7 tok/s
([`run-linux-0.5b-d128-clean-2026-09-27.log`](concurrency-mc2-2026-09-26/run-linux-0.5b-d128-clean-2026-09-27.log)).
The reading does not change: the 0.5B at depth 128 stays in the owner band on Linux, and the workers still win there.

Against the registered band (earn ≥ 1.25×, kill < 1.1×, the owner decides in between):
- **The 1.5B earns on Linux** (1.40–1.48×).
- **The 0.5B falls in the owner band** (1.11–1.23×).
- No cell kills. Over both machines, 6 of 8 cells earn and 2 go to the owner.

On amd64 the picture differs from the Mac:
- **The prototype's own M = 1 path costs 0.76–0.93×** production decode. amd64's decode kernels (split-half/VNNI GEMV)
  are tuned for M = 1, and `forwardN`'s M = B path is not. A tuned small-M kernel would lift every batched cell.
- **Workers win on the small model** (J8 1.64–1.73× against batching's 1.11–1.23× on the 0.5B). **Batching wins on the
  larger one** (1.40–1.48× against 1.33–1.36× on the 1.5B), where one sequence's decode already keeps the cores busier.
- That is the case for MC3c step 2 (batching behind the same admission) on larger models, next to the workers MC3c
  step 1 shipped. The two combined are not measured.

### Linux 7B cell (added 2026-09-27): the MC3c step-2 trigger

MC3c's step 2 (batched CPU decode behind serve's N-wide admission) was registered to start only if batching beats step
1's workers. The 1.5B left that open: batching led by 1.02–1.11×. This cell asks whether the lead grows on a larger
model.

Pre-registered in `2f5185c0` before any timing (the task doc's MC3c section):
- metric: batched B = 4 ÷ J8 N = 4, per rep, median of 5, at each depth;
- the lower depth's median ≥ 1.15× builds step 2; the higher < 1.05× parks it; anything else goes to the owner.

Setup: `nobara`, Ryzen 7 3700X (16 threads), qwen2.5-7b-instruct q4_k_m from `~/models`, int4, and the same test at
`2f5185c0`. Each depth ran in its own `go test`, idle-gated: load1 ≤ 1.0 with no other go, test or serve process.
Depth 128 ran 10:36–10:42 PDT (load1 0.43 at start), and depth 512 ran 10:44–10:53 (0.99). Raw:
[`run-linux-7b-2026-09-27.log`](concurrency-mc2-2026-09-26/run-linux-7b-2026-09-27.log) (driver),
[`run-linux-7b-d128-2026-09-27.log`](concurrency-mc2-2026-09-26/run-linux-7b-d128-2026-09-27.log),
[`run-linux-7b-d512-2026-09-27.log`](concurrency-mc2-2026-09-26/run-linux-7b-d512-2026-09-27.log),
[`run-linux-7b-2026-09-27.sh`](concurrency-mc2-2026-09-26/run-linux-7b-2026-09-27.sh).

**Identity passes on the 7B.** 4 sequences × 12 steps are bit-identical to the single-token forward at int4 and
int8int8 ([`identity-linux-7b-2026-09-27.log`](concurrency-mc2-2026-09-26/identity-linux-7b-2026-09-27.log)).

| model | depth | serial x1 | batched B=1 | B=2 | **B=4 (metric, paired)** | B=8 | J8 N=2 | **J8 N=4** | **B=4 ÷ J8 N=4 (trigger)** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7B | 128 | 5.1 | 0.965× | 1.691× | **2.734×** (2.73–2.74) | 3.341× | 1.092× | **1.133×** | **2.41×** |
| 7B | 512 | 5.0 | 0.965× | 1.650× | **2.586×** (2.58–2.59) | 3.117× | 1.106× | **1.148×** | **2.25×** |

The trigger column is computed per rep from the rep lines, which the harness prints to 0.1 tok/s. Every rep reads the
same: 14.0 / 5.8 at depth 128 and 12.8 / 5.7 at depth 512. Rounding bounds each ratio to 2.39–2.44 and 2.22–2.27.
The ratio of the harness's own medians (each against serial x1) agrees: 2.734 / 1.133 = 2.41 and 2.586 / 1.148 = 2.25.

**Decision, by the registered rule: build step 2.** The lower depth's median, 2.25×, clears 1.15× by a wide margin.

The results read as follows:
- **The trend continued, steeply.** Across 0.5B → 1.5B → 7B, batching's lead over the workers at B = N = 4 goes from
  0.64–0.75× to 1.02–1.11× to 2.25–2.41×.
- **On the 7B the workers barely scale.** 4 independent decodes reach 1.13–1.15× one decode. A single 7B decode already
  keeps this box's memory bandwidth busy, so running four of them splits it four ways.
- **Batching reads each weight once for B tokens.** It keeps scaling past B = 4, to 3.1–3.3× at B = 8.
- **So step 1's workers do almost nothing for a 7B on this CPU**, and step 2 is where the concurrency win is for
  larger models.
- **The batched numbers are a floor, as registered and not adjusted for.** The prototype's M = 1 batched path costs
  0.965× production decode on the 7B, much less than on the 0.5B (0.76–0.81×).
- Depth erodes batching, as on every other cell: 2.73× → 2.59× against serial from depth 128 to 512. Attention is per
  sequence and does not amortise.

### Mac 7B cell (added 2026-09-27): `-cpu-batch auto` on darwin

MC3c step 2 shipped with `-cpu-batch auto` batching models of ≥ 2 GiB of dense weights, but off on darwin until a Mac
7B cell showed batched B = 4 ÷ J8 N = 4 ≥ 1.15×. On the Mac's 0.5B and 1.5B the workers led; its 7B had not been
measured.

**Pre-registered** in `796ed628`, before any timing (the task doc's MC3c section):
- the metric is batched B = 4 ÷ J8 N = 4, per rep, median of 5, at each depth;
- both depths' medians ≥ 1.15× turn `auto` on for darwin; both below 1.05× leave it off; anything else goes to the
  owner.

**Setup:**
- the M1 Pro (8P + 2E cores), qwen2.5-7b-instruct q4_k_m, int4;
- loaded from its `.int4.cpu-arm64.giw`, transcoded by `cmd/prequant` for this cell
  ([`prequant-mac-7b-2026-09-27.log`](concurrency-mc2-2026-09-26/prequant-mac-7b-2026-09-27.log)). The `.gguf`'s CPU
  int4 load needs ~13.3 GB, which this Mac's fit guard refuses, and it was not bypassed;
- the test loads with `GOINFER_MC2_BACKEND=cpu`, new and test-only, because a row4-only bundle needs the CPU backend
  named;
- each depth in its own `go test`, idle-gated (load1 ≤ 2.0, no other go test or serve);
- depth 128 ran 19:13–19:15 PDT and depth 512 19:19–19:22.

**Two false starts.**
- The first launch (19:07) stopped on the backend error.
- Its identity step had also matched no subtest, which the script then took as a pass. The check was fixed to require
  the identity line before the relaunch at 19:09.
- No timing ran in the first attempt, and its logs are not kept in the record.

Raw: [`run-mac-7b-2026-09-27.log`](concurrency-mc2-2026-09-26/run-mac-7b-2026-09-27.log),
[`run-mac-7b-d128-2026-09-27.log`](concurrency-mc2-2026-09-26/run-mac-7b-d128-2026-09-27.log),
[`run-mac-7b-d512-2026-09-27.log`](concurrency-mc2-2026-09-26/run-mac-7b-d512-2026-09-27.log),
[`run-mac-7b-2026-09-27.sh`](concurrency-mc2-2026-09-26/run-mac-7b-2026-09-27.sh).

**Identity passes on the Mac's 7B:** 4 sequences × 12 steps are bit-identical to the single-token forward at int4
([`identity-mac-7b-2026-09-27.log`](concurrency-mc2-2026-09-26/identity-mac-7b-2026-09-27.log)).

| model | depth | serial x1 | batched B=1 | B=2 | **B=4** | B=8 | J8 N=2 | **J8 N=4** | **B=4 ÷ J8 N=4 (per rep → median)** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7B (Mac) | 128 | 17.8 | 0.996× | 1.305× | **2.182×** | 2.421× | 1.293× | **1.417×** | 1.540 · 1.538 · 1.542 · 1.554 · 1.532 → **1.540×** |
| 7B (Mac) | 512 | 16.4 | 1.006× | 1.276× | **2.019×** | 2.176× | 1.334× | **1.460×** | 1.430 · 1.395 · 1.383 · 1.340 · 1.238 → **1.383×** |

**Decision, by the registered rule: `auto` batches on darwin too.** Both medians are ≥ 1.15×. The `runtime.GOOS`
clause in `decoder/cpu_batch.go` is gone, and the 2 GiB threshold holds, so the Mac's 1.5B (1.23 GB), where the workers
lead, stays on the workers.

**Reading.**
- As on `nobara`, the workers barely scale on the 7B (1.42–1.46× serial at N = 4), while batching reaches 2.0–2.2×.
- The Mac's lead (1.38–1.54×) is smaller than the Linux box's (2.25–2.41×), but well past the bar.
- Depth 512's rep 5 (1.238) ran while load rose after the depth-128 cell; the median is unaffected.

## Owed
- J8's latency half, per request.
- ~~A re-run of the Linux 0.5B depth-128 cell without the concurrent rsync.~~ Done 2026-09-27: 1.185×.
- The H kernel at M = 1, graded in sequence as an R18 follow-on.
