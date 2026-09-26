# MC2 S0 — on CPU one forward carrying 4 rows costs 1.8–1.9 decodes, not 4: batching's ceiling is ≈ 2.1–2.3× at B=4 (2026-09-26)

S0 for MC2 of [`task-concurrency-2026-09.md`](../tasks/task-concurrency-2026-09.md). MC2 asks whether one CPU forward
that carries B sequences beats B sequential forwards by enough to fund MC3. Its registered gate is aggregate tok/s
at B = 4 against B = 1, earn ≥ 1.25× and kill < 1.1× (loosened 2026-09-26, before any run). That grade needs a
batched multi-sequence step function that does not exist yet. **This S0 bounds the answer with no new code**, using a
forward that already exists.

**Answer:** on the Mac's CPU a single forward carrying n rows costs:
- **1.77–1.93×** a one-row forward at n = 4 (int4, 0.5B and 1.5B, depth 128 and 512);
- 2.8–3.1× at n = 8.

That puts batching's throughput ceiling at **≈ 2.1–2.3× at B = 4** and ≈ 2.6–2.9× at B = 8, far above the earn line.
The prior-art line "on CPU the batched verify costs close to N decode steps" (`completed/task-laguna.md`, from the
2026-08-31 probe) no longer holds on this tree.

## Method

`TestThetaProbe_CPU` (`decoder/theta_probe_test.go`), unchanged:
- prefill a cache to depth D (128, 512);
- then time `forwardN` over n rows at positions D..D+n−1, for n = 1, 2, 3, 4, 6, 8, 12, 16;
- median of 7 after one warm-up, with the cache truncated back to D before every rep.

`forwardN` is bit-identical to n sequential forwards by contract (the speculative verifier depends on it), so its
projections run at M = n through the same `WeightMat.MatmulBTW4A8Into` / int8 kernels decode uses. It gives one LM
head at M = n, and n rows of attention over the one cache.

- M1 Pro, macOS 26.6.2, CPU backend, goinfer `a3b30f49`.
- qwen2.5-coder-0.5b and -1.5b instruct q4_k_m from `~/models`; `GOINFER_BENCH_QUANT` int4 and int8int8. The probe's
  loader defaults to int8int8, and the 2026-08-31 run was that default.
- Load1 1.58–1.97. Run 2026-09-26 16:41–16:42 PDT.
- Logs: [`concurrency-mc2-s0-2026-09-26/`](concurrency-mc2-s0-2026-09-26/).

## Results

T(n) / T(1), median µs in parentheses for n = 1:

| model | quant | depth | T(1) | n = 2 | n = 4 | n = 8 | n = 16 | ceiling at B = 4 (4 ÷ T(4)/T(1)) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5B | int4 | 128 | 8.96 ms | 1.42 | **1.78** | 2.82 | — | **2.25×** |
| 0.5B | int4 | 512 | 10.84 ms | 1.40 | **1.77** | 2.88 | — | **2.26×** |
| 0.5B | int8int8 | 128 | 9.65 ms | 1.37 | 1.52 | 2.07 | — | 2.63× |
| 0.5B | int8int8 | 512 | 11.00 ms | 1.37 | 1.60 | 2.34 | — | 2.50× |
| 1.5B | int4 | 128 | 18.67 ms | 1.59 | **1.93** | 3.06 | — | **2.07×** |
| 1.5B | int4 | 512 | 22.01 ms | 1.56 | **1.91** | 2.91 | — | **2.09×** |
| 1.5B | int8int8 | 128 | 23.44 ms | 1.20 | 1.25 | 1.67 | — | 3.20× |
| 1.5B | int8int8 | 512 | 24.25 ms | 1.27 | 1.46 | 1.97 | — | 2.74× |

(n = 16 is in the logs.) For comparison, the 2026-08-31 probe read T(4)/T(1) = 2.68 at depth 128 and 2.03 at 512.
Its checkpoint and quant are not named in the log, and the loader defaults to int8int8 and the 0.5B. aikit's S-01
int4 tile, which roughly doubled CPU prefill, shipped in between.

## Why this is an upper bound, not the grade

- **Attention does not amortise across sequences.** B separate sequences each attend their own cache, while `forwardN`'s
  n rows attend one. At depth 128–512 on these models attention is a small share of a token, so the gap should be
  small, but it is not measured here. At long context it grows.
- **The registered cell is a real multi-sequence step.** MC2's grade has to come from B distinct sequences, each with
  its own `Session` cache and position, through one batched forward. The J8 cell (N independent decode workers) runs
  beside it in the same session.
- **Linux CPU is a registered cell and is not run here** (`nobara`, amd64: VNNI and split-half kernels, a different
  batching profile).

## What it means for the rest of the doc

- MC2 should **earn** on CPU unless per-sequence attention eats most of the margin. The registered grade still decides.
- **It says little about MC3 (Metal).** R12 measured Metal's marginal verify cost at Θ ≈ 0.86–0.96
  ([`r12-p21-theta-post-batch-2026-09-20.md`](r12-p21-theta-post-batch-2026-09-20.md)): T(4)/T(1) ≈ 3.6–3.9, a B = 4
  ceiling of ≈ 1.03–1.12×. That is because each verify row still runs Metal's M = 1-shaped GEMV. MC3's first new piece,
  a small-M W4 GEMM, is exactly that gap. R18's rows-per-simdgroup kernels amortise activations across output rows,
  not across tokens, so they do not change this.
