# P33, first measurement: where a Metal decode token's time goes: record (2026-11-28)

<!-- citations-at: 8387e5723 -->

Registered in [`registration.md`](registration.md), which was on `main` at `8387e5723` before the run. Qwen2.5-Coder-1.5B at
`int4` on one Apple M1 Pro with 16 GB (macOS 26.7.3), by day, the timing lock held exclusively, the checkpoint the model's Metal
`.giw` on the internal SSD. One pass of each cell, 08:04:04 to 08:06:07 PST; all three cells counted and none was void.

## Result

**The split counts, and it is not ambiguous.** `decomp`'s categories plus its floor are 101.4% of the full token at depth 128
(the registered band is 90% to 115%) and the full token's spread is 2.3% (at most 5%).

The candidates, ranked as registered, in recoverable milliseconds a token at depth 128:

| rank | candidate | recoverable |
|---|---|---|
| 1 | the fixed cost above its dispatch floor (`rms`, `swiglu_quant`, `quant_vec`, the KV store, `rope2`) | **1.245 ms** |
| 2 | GEMV gate/up | 0.708 ms |
| 3 | GEMV down | 0.510 ms |
| 4 | the full-logits path's host time | 0.496 ms |
| 5 | GEMV qkv | 0.372 ms |
| 6 | GEMV o | 0.331 ms |

The largest is over the 0.5 ms line and 0.537 ms ahead of the second, outside the 0.2 ms tie band. **By outcome 3 the per-layer
norm and quantizer kernels get their own registration next.** No kernel work was done here and nothing ships from this run.

Read the first row with its limit in mind: the rule counts that candidate's whole in-sequence work as recoverable, where each GEMV
is counted only down to the streaming-read ceiling. 1.245 ms is what those kernels cost today, not what a better kernel would
save. See "What this does not say".

## The run

| cell | start (PST) | end | load at start | load at end | swap, start to end | test |
|---|---|---|---|---|---|---|
| `gemv-s0` | 08:04:35 | 08:04:42 | 3.60 | 3.47 | 2087.12 to 2087.12 MB | PASS, 5.93 s |
| `decomp` | 08:04:42 | 08:05:09 | 3.47 | 2.53 | 2087.12 to 2087.12 MB | PASS, 27.12 s |
| `chain-ab` | 08:05:10 | 08:06:07 | 2.53 | 7.50 | 2087.12 to 2039.12 MB | PASS, 56.91 s |

- **Build:** the tree at `8387e5723` with no local change, `backend/metal`'s test binary built with `-tags goinfer_testhooks`
  on go1.27.4 (sha256 `2e80b72c7673dbe0`). The run script is `run-p33-split.sh` beside this file, and the launcher is in the
  run folder.
- **Fit:** the `fit` command's plan for the load at context 4,096, read before the run: 387,711,232 bytes, of which
  119,275,776 are KV (`run-2026-11-28/fit-plan.txt`, re-read after the run for its text; the same answer). The void rule took
  370 MB as the need for the two cells that load the checkpoint and 1,024 MB for `gemv-s0`, which loads none. Available memory
  was 13,426 MB at each start.
- **The idle gate** held the first cell 31 s: the 1-minute load was 5.04 when the lock was taken and 3.60 at the cell's start,
  under the 4.0 cap.
- **The load rose during `chain-ab`,** from 2.53 to 7.50; what raised it was not identified. The void rule's end test is on
  swap, which fell, so the cell counts. Its nine pairs show no drift across the cell: the full-logits arm read 93.8 to 98.2
  tokens a second and the chain 97.7 to 100.9, the last pair no slower than the first.

## The split

`TestMetalDecodeDecomp` (`backend/metal/decode_decomp_test.go:TestMetalDecodeDecomp`): each category's in-sequence GPU work is the
full token less the token with that category's kernels made no-ops; medians of 5 reps of 20 tokens, in milliseconds a token.

| category | depth 128 | share | depth 192 | share |
|---|---|---|---|---|
| GEMV gate/up | 3.095 | 31.2% | 3.106 | 31.0% |
| GEMV down (+residual) | 1.704 | 17.2% | 1.719 | 17.2% |
| LM head (int8) | 1.291 | 13.0% | 1.347 | 13.4% |
| attention | 0.822 | 8.3% | 0.954 | 9.5% |
| dispatch floor (every pipeline a no-op) | 0.714 | 7.2% | 0.719 | 7.2% |
| GEMV qkv (+bias) | 0.645 | 6.5% | 0.691 | 6.9% |
| GEMV o (+residual) | 0.536 | 5.4% | 0.544 | 5.4% |
| act-quant (`swiglu_quant`) | 0.500 | 5.0% | 0.519 | 5.2% |
| norm+quant (`rms`) | 0.487 | 4.9% | 0.492 | 4.9% |
| ctx-quant (`quant_vec`) | 0.116 | 1.2% | 0.106 | 1.1% |
| KV store | 0.103 | 1.0% | 0.075 | 0.8% |
| rope (`rope2`) | 0.039 | 0.4% | -0.009 | -0.1% |
| categories + floor | 10.051 | 101.4% | 10.265 | 102.5% |
| **full token** | **9.911** (100.9 tok/s), spread 2.3% | | **10.019** (99.8 tok/s), spread 1.5% | |

The four GEMVs are 5.980 ms, 60.3% of the token at depth 128. The floor's own spread is 6.2% and 6.0%.

## The derived figures

By the registration's formulas and no others. The streaming-read ceiling is the best figure `gemv-s0` printed in this run,
**181.6 GB/s** (512 MB read by 1,048,576 threads; 159.9 and 177.3 at the two smaller thread counts).

| GEMV category | weight bytes a token | work, depth 128 | bandwidth | share of the ceiling | recoverable |
|---|---|---|---|---|---|
| qkv | 49,545,216 | 0.645 ms | 76.8 GB/s | 42.3% | 0.372 ms |
| o | 37,158,912 | 0.536 ms | 69.3 GB/s | 38.2% | 0.331 ms |
| gate/up | 433,520,640 | 3.095 ms | 140.1 GB/s | 77.1% | 0.708 ms |
| down | 216,760,320 | 1.704 ms | 127.2 GB/s | 70.0% | 0.510 ms |
| sum | 736,985,088 | 5.980 ms | | | 1.922 ms |

Bytes are N·K/2 of nibbles plus N·K/16 of f16 group scales, times 28 layers (hidden 1,536, intermediate 8,960; qkv's rows are
1,536 + 256 + 256).

- **The fixed cost:** the floor 0.714 + `rms` 0.487 + `swiglu_quant` 0.500 + `quant_vec` 0.116 + `rope2` 0.039 + the KV store
  0.103 = **1.959 ms** at depth 128, of which **1.245 ms** is above the floor.
- **The full-logits path's host time:** 1000 ÷ 95.1 = 10.515 ms a token, less `decomp`'s full token at depth 192 (10.019 ms) =
  **0.496 ms**.
- **The chain's token:** 1000 ÷ 99.2 = **10.081 ms**, beside `decomp`'s 10.019 ms at depth 192. The two are not the same
  window: a chained token ends with the fused argmax head and starts with an embedding gather on the device, where `decomp`
  times the full-logits forward.

`TestCB01ChainAB` (`backend/metal/cb01_grade_test.go:TestCB01ChainAB`), greedy, 9 interleaved pairs of 256 tokens after a
64-token prompt: chain ÷ full-logits **median 1.0381** (min 1.0248, max 1.0499), 9 of 9 pairs above 1, and the two arms' tokens
identical in every rep; medians 95.1 and 99.2 tokens a second.

## The registered outcomes

1. **The split counts** (101.4%, spread 2.3%).
2. **The ranking** is the table under "Result".
3. **Not ambiguous.** The largest candidate is 1.245 ms, over 0.5 ms; the second is 0.708 ms, 0.537 ms behind. The fixed cost
   above its dispatch floor gets its own registration next.
4. **No claim about either engine's quantisation** comes from this run: it has one arm.

Below first place the rule ranks nothing firmly: gate/up, down and the host time are within 0.2 ms of their neighbours.

## What this does not say

- **1.245 ms is a cost, not a saving.** `rms` computes a norm and a quantized activation, `swiglu_quant` an activation function
  and a quantize, and so on: some of that work stays in any design. The GEMV rows are bounded below by a measured ceiling; this
  row is not. The next registration has to say what part of it a design could remove before any code is written.
- **The two smallest categories are inside the method's spread.** `rope2` reads 0.039 and -0.009 ms at the two depths and the KV
  store 0.103 and 0.075; a small category's reading moves by about 0.2 ms between passes (its paired columns, in
  `run-2026-11-28/results.txt`). The candidate's sum rests on `rms` and `swiglu_quant`, 0.987 ms of it.
- **The host time is the full-logits path's,** the wait between one token's last kernel and the next's first. A greedy request
  no longer pays it: the chain is on by default and reads 1.038 times that path's rate here. It is a cost only of requests the
  chain does not serve, and it is a difference of two absolute figures from two cells.
- **`gemv-s0`'s per-kernel rows are not today's decode kernels.** Its `sa` and `coal` rows time the single-row kernels
  (`gemv_w4a8_sa`, `gemv_w4a8_resid`), and they reproduce the 2026-09-26 record to the digit (gate/up 0.172 ms, 90 GB/s). The
  dense `int4` decode path now runs the rows-per-simdgroup forms that shipped after that record, which is why gate/up reads 140
  GB/s in sequence. The registration's cell table called those rows "each production GEMV kernel": that was wrong for this tree.
  No registered figure reads them; the formulas take only the streaming ceiling from this cell.
- **A slice.** One model, one quantisation, one machine, in process, at depths 128 and 192. Not the 7B, not the served path, not
  the mlx-lm side.

## Context, not results

**The earlier reading of the same test** (2026-10-24, for the kernel survey, depth 128, the 1.5B's `.gguf` on its Metal `int4`
resident; another day's run, 308 commits to the backend's non-test Go files earlier, so the columns are not differenced):

| category | 2026-10-24 | 2026-11-28 |
|---|---|---|
| GEMV qkv / o / gate\|up / down | 0.570 / 0.407 / 3.223 / 1.864 (6.064) | 0.645 / 0.536 / 3.095 / 1.704 (5.980) |
| LM head | 1.418 | 1.291 |
| attention | 0.643 | 0.822 |
| `rms` | 0.599 | 0.487 |
| `swiglu_quant` / `quant_vec` | 0.644 / 0.231 | 0.500 / 0.116 |
| `rope2` / KV store | 0.153 / 0.187 | 0.039 / 0.103 |
| dispatch floor | 0.650 | 0.714 |
| full token | 9.708 | 9.911 |

**The LM head is at the read ceiling.** Its in-sequence work, 1.291 ms for 233,981,440 bytes (N·K + 4N, the 2026-09-26
record's count), is 181 GB/s, and `gemv-s0`'s standalone row for the same kernel reads 1.456 ms, 161 GB/s. This is not a
registered figure and ranks nothing. A faster head is a narrower head, not a better kernel.

**What has been tried on the first candidate.** The kernel survey folded `swiglu_quant` into the gate/up kernel's epilogue and
the down kernel's staging, and parked it on 2026-10-24: it was not bit-identical on real checkpoints (one activation on a
rounding tie lands on the other side under fast math) and it was 4.9% slower (median 1.0489, 1 of 10 pairs faster), because
every down threadgroup re-quantized the whole activation. The shape that record left untried is a quantizer spread over several
threadgroups (partial maxima, then a parallel quantize).

**What is known about the GEMVs.** `docs/measurements/metal-decode-gemv-s0-2026-09-26.md` found each `int4` GEMV bound by its
per-weight work, 1.5 to 2.0 times slower than a twin that reads the same bytes, and not by bandwidth. This run's shares of the
ceiling (38% to 77%) are consistent with that and add nothing to it.

**Scale.** `docs/benchmarks.md` §B11 puts the 1.5B's greedy gap to mlx-lm at 92.0 against 138.2 tokens a second on the served
path, about 3.6 ms a token. The ranked candidates here sum to 3.17 ms of GPU work (1.245 + 1.922) only if every GEMV ran at the
streaming ceiling and the fixed kernels cost nothing. That is another day's served figure beside this run's in-process one, so
it is a scale and not a ratio; it is the reason the mlx-lm side of the split is still owed.

## Next

- A registration for the per-layer norm and quantizer kernels (outcome 3), starting from what a design could remove and from
  the parked fold's two findings. Kernel changes under `backend/metal/` land after the release that is being gated.
- The mlx-lm side of the split, from its own trace, and the 7B once its checkpoint is on this machine in a form this build
  loads: both were left out of this run by the registration.

## Files

- `registration.md`, `run-p33-split.sh`: the registration and the runner, as they were at the run's commit.
- `run-2026-11-28/provenance.txt`: the build, the model, the machine, and each cell's start and end readings.
- `run-2026-11-28/results.txt`: every figure quoted above, as the tests printed it.
- `run-2026-11-28/<cell>.attempt-1.log`: each test's whole output; `.start.json`, `.end.json` and `.peak.json`: the void rule's
  readings and the swap samples taken once a second.
- `run-2026-11-28/launch.log`, `launch-p33.sh`, `fit-plan.txt`: the launcher, its output, and the fit plan.
