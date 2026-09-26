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
- **A side finding worth its own brief.** At M = 1, H beats the shipped R18 kernel standalone, bit-identically:
  - gate/up 0.117 against 0.137 ms (1.17×, 1.5B) and 0.452 against 0.556 ms (1.23×, 7B);
  - qkv 1.32× (1.5B).

  R18 tested float forms, but staged activations as shorts and masked 32-bit words. The difference here is
  half-staged, pre-scaled activations and 16-bit masks. This is standalone only, and R18 showed standalone gains need
  in-sequence grading.

## Owed

- The Linux CPU cells (`nobara`: amd64, VNNI and split-half kernels) at the same four configurations. They wait for the
  set-A reference regeneration, which is already queued for that box.
- J8's latency half, per request.
- The H kernel at M = 1, graded in sequence as an R18 follow-on.
