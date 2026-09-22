# R9 step 1 — CPU decode per-component attribution, Linux half (the 0.5B anomaly)

`docs/tasks/red-october.md` R9, step 1, the half the Mac record
(`cpu-decode-attribution-2026-09-20.md`) could not do: the Ryzen box. Box: `nobara-pc`, AMD Ryzen 7 3700X
(8 cores / 16 threads; logical CPUs 0–7 are the eight cores, 8–15 their SMT siblings), 62 GB, goinfer `89ddc2d2`,
aikit v1.46.0, `GOAMD64=v1` (unpinned — the audit's note stands; the hot kernels are hand-written AVX2 assembly in
aikit `linalg/*_amd64.s`, so the pin question is about the Go-compiled glue, not the kernels). Models: the
qwen2.5-coder 0.5B and 1.5B instruct and the qwen2.5 7B instruct q4_K_M GGUFs under ~/models — the same three the standing
2026-08-26 §B8 CPU column measured (23.5 / 17.6 / 4.9 tok/s). Backend `cpu`, `Quant: "int4"`, greedy, 128-token
prompt (R9's decision-cell depth), 24 decode tokens, `TestR9_decodeAttribution`.

## What the brief asks this half to settle

The 2026-08-26 fit: ~13 ms fixed + 22 GB/s (Ollama ~5 ms + 27, against a 30.5 GB/s measured read ceiling). The
1.5B/7B fit predicts the 0.5B at 35.5 tok/s (28 ms/token); the cell reads 23.5 (42.5 ms) — **~14 ms/token
unaccounted**. The brief names two hypotheses to test explicitly — SMT siblings competing for the same load ports
(`GOMAXPROCS` 16 on 8 cores) and the 0.5B's K=896 LM head — and the resolution criterion: **one component accounts
for ≥10 of the ~14 ms, or the table says it is spread and names the spread.** Step 1 has no band; its output is
the table.

## Method (the Mac's, replicated exactly; one addition)

- Coarse split: `GOINFER_DECODE_TIMING=1` (shipped) — forward / sample / logitProc / embed.
- Fine split inside `forward`: the Mac's temporary diagnostic reconstructed as its record describes — `time.Now`
  + `atomic.AddInt64` around `causalAttention`, `mlp` (the non-parallel branch, qwen2.5's) and `logitsFromHidden`,
  gated by `GOINFER_R9_DIAG`, printed beside the `DECODE TIMING` line. **Verified a no-op before use:**
  `TestForwardN_matchesSequential` and `TestSpeculativeGreedyParity` pass with it present. **Reverted, not
  committed** — `decoder/model.go` is a parity-manifest core file, the same reason the Mac half gave.
- The addition: three thread configurations per model, because the SMT hypothesis is a configuration, not a
  component — (a) default `GOMAXPROCS=16`; (b) `GOMAXPROCS=8`, unpinned (the scheduler may still spread across
  siblings); (c) `taskset -c 0-7 GOMAXPROCS=8` (one thread per physical core, siblings idle).
- Fan-out shape: `TestR9_parWidthSweep` — matmul shard width 1/2/4/8/16 (`linalg.SetParallelWidth`, numerically
  inert by contract), interleaved up-then-down on one loaded model. Serial is the compute floor; width-8 vs
  serial/8 is the fork/join + imbalance cost; width-16 vs width-8 is the SMT contribution. The attention head
  fan-out is not swept: `decoder/scratch.go`'s `maxAttnWorkers = 6` is a compile-time constant (the M1 Pro's
  P-core count) and applies unchanged on this 8-core box — noted as a finding in its own right below.

## Pre-registered rule for anything step 1 names (written before either build)

Step 1 has no band; step 2's Linux band (0.5B ≥ 40 tok/s) was written against the 2026-08-26 row and — as the
data below shows — the 0.5B already reads 40.3 tok/s today, so that band is moot and is not what the builds are
judged by. Instead, per fix step 1 names: **bit-identical** (`TestForwardN_matchesSequential`, the golden parity
tests, the parity manifest re-baselined only after those pass) AND a **paired, in-process ABBA A/B ≥ 3% on the
1.5B at depth 128** ships; 1.5–3% parks; < 3%… kills. A fix that touches a depth-dependent path must also not
regress any other measured depth. A fix confined to one architecture by measurement (this box) is shipped for
that architecture only, and the other box's measurement is named as open, not assumed.

## Data 1 — the per-component table, three thread configurations (logs `…-linux-g16/g8/g8-pinned.log`)

ms/token, `forward` and its split (sample/logitProc/embed ≤ 0.10 ms everywhere, as on the Mac):

| model | config | forward | attention | MLP | LM head | tok/s |
|---|---|---:|---:|---:|---:|---:|
| 0.5B | 16 threads | 24.2 | 6.55 | 12.38 | 5.10 | 41.3 |
| 0.5B | GOMAXPROCS=8 | 21.8 | 5.28 | 11.32 | 5.04 | 45.9 |
| 0.5B | 8, pinned to cores 0–7 | 26.0 | 7.36 | 13.53 | 4.91 | 38.5 |
| 1.5B | 16 threads | 74.3 | 18.47 | 46.68 | 8.79 | 13.5 |
| 1.5B | GOMAXPROCS=8 | 73.9 | 16.42 | 48.80 | 8.40 | 13.5 |
| 1.5B | 8, pinned | 76.9 | 17.11 | 51.12 | 8.31 | 13.0 |
| 7B | 16 threads | 221.0 | 30.23 | 170.03 | 20.07 | 4.5 |
| 7B | GOMAXPROCS=8 | 226.1 | 29.10 | 177.12 | 19.25 | 4.4 |
| 7B | 8, pinned | 227.1 | 29.12 | 178.11 | 19.22 | 4.4 |

**SMT hypothesis: dead.** 16 vs 8 threads is inside run-to-run noise at every size, and pinning one thread per
physical core is slightly *worse* — the fixed cost is not sibling contention.

**The 0.5B anomaly does not exist any more.** The standing row (23.5 tok/s, 2026-08-26) is stale by 1.7×: the
0.5B reads 41 tok/s today, and its split is unremarkable — the K=896 LM head is the largest non-MLP item at 5.1 ms
(21% of the token) but streams at ~21 GB/s, i.e. it is bandwidth, not a defect. The 1.5B (13.5 tok/s, standing
17.6) is now the size that under-performs its bytes: effective rate 15 GB/s against 20–21 for the other two. Both
standing Linux CPU rows need a `bench_peer.py` re-anchor; that is the brief's own "Measure" item, not this step's.

## Data 2 — where the 1.5B's time goes (width sweep, isolated matmuls, the inner split)

**Fan-out width** (`TestR9_parWidthSweep`, `…-width.log`): 1.5B MLP at width 1/2/4/8/16 = 180 / 110 / 78 / 61 / 47
ms; 16 workers buy only 3.8× over serial. Serial MLP is 6.4 GMAC/s — while aikit's own AVX2 W4A8 dot runs at
15.5–18.8 GMAC/s hot and its M=8 tile at 27.6 (`linalg` benchmarks, this box). Neither compute- nor bandwidth-bound.

**The same matmuls in isolation** (`TestR9_mlpMatmulIsolated`, `…-isolated.log`): the exact `matmulInto` calls
`mlp()` makes, on the loaded 1.5B's own 84 MLP matrices, DRAM-streamed in order: width 16 → **29.5 ms** (39 GMAC/s,
19.5 GB/s of weights); width 1 → 82 ms. The token's MLP component was 47 / 180 ms. **Roughly 40% of "MLP" was
not the matmuls.**

**The inner split** (`…-full-split.log`, shipped 16-thread defaults before any fix):

| model | q/k/v | attn core (rope, KV, scores/softmax/AV) | o | gate+up | **activation** | down | LM head |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5B | 2.87 | 1.63 | 2.22 | 6.44 | 2.07 | 4.20 | 5.09 |
| 1.5B | 5.53 | **10.97** | 2.58 | 20.14 | **13.54** | 12.67 | 8.68 |
| 7B | 15.93 | 4.54 | 9.61 | 92.93 | **27.89** | 50.59 | 20.02 |

Two anomalies, each with a mechanism read from the code and then confirmed by switching it off:

1. **The SwiGLU activation costs as much as the down projection.** `silu` is a scalar `float64 math.Exp` per
   element; above `activationFanoutThreshold = 8192` elements `parallelElementwise` spawns 6 goroutines per layer
   (`activationFanoutWorkers = maxAttnWorkers`, the M1 Pro's P-core count). The 0.5B (inter 4864, serial) does it
   at 17.7 ns/element; the 1.5B and 7B (fanned out) at ~54 ns/element **effective** — the goroutine wake stagger
   (aikit S-02's own 92 µs finding) costs 3× what 9–19k scalar silu calls save. Serial build (`…-serialact.log`):
   1.5B 13.5 → 3.9 ms, 7B 27.9 → 8.0.
2. **The 1.5B's attention core is 2.4× the 7B's on a smaller problem.** The 1.5B is the one geometry with 6 query
   heads per KV head, and R13's grouped acc64 kernels (`GOINFER_ATTN_GROUPED`, default on) engage only for
   `group == 6`. aikit ships them as NEON assembly with a pure-Go `attn_acc64_group_other.go` fallback that returns
   0 blocks — so on amd64 the "grouped" path is a Go loop. `GOINFER_ATTN_GROUPED=0`: core 10.97 → 2.59 ms.
   **Across depth** (`TestR9_groupedDepthSweep`, `…-grouped-depth.log`, ABBA): grouped vs per-head core
   10.6/2.7 (128), 42.5/6.6 (512), 79/11 (1024), 142/21 (2048), **239/42 ms at depth 4096 — the whole token
   315 vs 110 ms, 2.86× slower.** Not a crossover: the amd64 fallback loses at every depth and the gap widens.

The fork/join term the brief expected is real but smaller than either: matmuls at width 16 reach 64% of the read
ceiling (19.5 of 30.5 GB/s) in isolation, ~3 ms/token over the isolated sweep inside the token. The per-worker
S-02 timestamps inside a real token are still not taken (no aikit probe hook; the width sweep bounds the term).

## Data 3 — the two fixes, built, gated, shipped (`…-ab.log`, `…-after.log`)

Both are architecture-conditional defaults in `decoder/cpu_tuning_{arm64,other}.go`, vars so a test can flip them:
`activationFanoutEnabled = false` and `attnGroupedKernels = false` on every non-arm64 platform; arm64 unchanged.
Both bit-identical by construction and by gate: `TestForwardN_matchesSequential`, `TestSpeculativeGreedyParity`,
the golden parity tests all pass; the parity manifest re-baselined after they did.

`TestR9_cpuTuningAB` — one loaded 1.5B, depth 128, each knob flipped in-process, ABBA, 3 pairs:

| knob | ON → OFF ms/token | paired | component ON → OFF |
|---|---:|---:|---:|
| activation fan-out | 65.8 → 55.3 | **1.189×** (1.178–1.206) | 12.97 → 3.67 |
| grouped attention (amd64 Go fallback) | 62.4 → 55.8 | **1.118×** (1.113–1.129) | 9.92 → 2.84 |

Both ship by the rule (≥ 3%). After, shipped defaults (`…-after.log`): **1.5B 74.5 → 54.5 ms/token (1.37×, 18.3
tok/s); 7B 222.2 → 201.9 (1.10×)**; 0.5B unchanged by construction (below the fan-out threshold; 7-head groups
never took the grouped path) — its 27.5 ms in that run against 24.2–24.8 earlier is the ~10% run-to-run drift a
back-to-back CPU run shows, not an effect, and is the reason the paired A/B is the number that stands.

The DECODE SPLIT lines are kept, gated by the shipped `GOINFER_DECODE_TIMING` (one bool check per site when off),
so the Mac can read its own activation and attention-core shares with no patch.

## What this establishes, and what it does not

- Established: the Linux per-component table at all three sizes; the SMT hypothesis dead; the 08-26 0.5B anomaly
  gone with the stale row; two named, mechanism-confirmed, bit-identical fixes shipped for non-arm64 with paired
  wins of 1.19× and 1.12× (1.37× stacked on the 1.5B, 2.86× at depth 4096).
- Not established: whether the activation fan-out also loses on the M1 Pro (its constant was tuned there with
  `silubench`; the DECODE SPLIT MLP line now answers it in one run — open for the Mac); the S-02 per-worker
  timestamps inside a real token; the served (`bench_peer.py`) re-anchor of the Linux CPU rows, which this record
  shows are stale in both directions (0.5B faster, 1.5B slower than 08-26).
- `GOAMD64` stays unpinned: the hot kernels are hand-written assembly dispatched on runtime feature detection, and
  nothing in this split pointed at compiler-generated scalar code on a hot path.
