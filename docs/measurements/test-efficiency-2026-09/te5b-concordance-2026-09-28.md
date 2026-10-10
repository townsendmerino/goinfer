# TE5(b) calibration: does an in-process A/B predict the served A/B? (2026-09-28)

Item TE5(b) of [`task-test-efficiency-2026-09.md`](../../tasks/task-test-efficiency-2026-09.md). It is an analysis of
records already on disk. **Nothing was run or timed.** The data, the classification rule and every citation live in
[`te5b_concordance.py`](te5b_concordance.py), which is stdlib-only and read-only; `python3
docs/measurements/test-efficiency-2026-09/te5b_concordance.py` prints every number below.

## Verdict

TE5(b)'s rule, verbatim: *"If the in-process ratio predicts the served direction in every pair, and the magnitude
within a stated factor, 'in-process by day, served at night for claims' becomes the default flow. Kill (b): any sign
disagreement for a kernel-local change."*

The rule is graded on tier 1: the served A/B is same-session and differs by the change alone. That covers 40 rows over
17 kernel-local changes.

- **Strict reading (an unresolved side counts as a disagreement): KILL (b).** Three
  kernel-local rows fail, and all three fail because one side could not resolve a direction:
  - **L1 f16 scales, 0.5B, nobara.** In-process 0.968×, with 3 of 5 pairs below 1 and a range of 0.960–1.027. Served
    1.016×, with non-overlapping runs. This is the only kernel-local row whose point estimates point opposite ways.
  - **fused_rms_gu wave rule, D7.** In-process 1.028× with no spread recorded. Served 1.020×.
  - **MC3 S4, 1.5B, 2 clients.** In-process 1.020× with no spread recorded. Served 1.059×.
- **Resolved-only reading (only rows where both sides resolve a direction): no kill.**
  - All 37 of 37 resolved kernel-local rows agree in sign. So do all 45 of 45 across both tiers.
  - The only resolved sign disagreement in all 87 rows is outside the kernel. It is G26's sampler microbenchmark at
    152k vocab: 0.810× in-process against 1.085× served. Its own record names the mechanism: the tight loop keeps the
    scratch buffer hot and the allocator warm.
- **Magnitude depends on what the in-process instrument timed.**
  - **Whole-token readings.** On the agreeing kernel-local rows, k = ln(served)/ln(in-process) has a median of 0.97
    and a range of 0.60–1.08. All 23 of 23 are within a factor of 2, and 21 of 23 are within 1.25×. The two outside
    1.25× are WebGPU's G35 (0.70) and G36 (0.60). G36 also compares one token at position 512 with a 512-token
    generation.
  - **Kernel-only readings.** Here k has a median of 0.55 and a range of 0.05–1.72, and 16 of 30 are within 2×. A
    kernel bench gives the direction, not the size. That is Amdahl, as expected: the kernel is a fraction of the token.
    But the spread is too wide for any stated factor to hold.

**What the two readings mean for the flow.** They differ only on rows whose in-process effect is about 3% or less (3.2%,
2.8% and 2.0%). At that size the in-process arm on record did not resolve a direction: a split pair-sign count in one
case, a single number with no spread in the other two. Nothing measured contradicts the in-process lane on a kernel-local change. The lane also cannot settle a
2–3% question on its own.

The rule as written kills (b). Whether the owner re-reads it resolved-only is the owner's call, and so is the factor.
If it is re-read, the data support this conditional flow:
1. For a kernel-local change, an in-process A/B that times the **whole token** and **resolves a direction** predicts
   the served direction. Every such row on record agrees, and its magnitude lands within 1.25× (21 of 23) or 2× (23 of
   23).
2. A kernel-only bench predicts direction only.
3. An unresolved in-process result is not a prediction. It goes to the served gate at night.

## Method

**Sources.** Four read-only sweeps, each returning file:line or commit citations, with every figure re-read here
before it entered the data:
- CPU measurement records;
- GPU (Metal, CUDA, WebGPU) records;
- the ledgers: `benchmarks.md`, `QUEUE.md`, `queue-performance.md`, `ollama-chase.md`, `legacy-benchmarks.md`,
  `CHANGELOG.md`;
- all 3,493 commit messages.

Where a JSON of served runs exists, the bounds were recomputed from the runs:
- `gate5-served.json`, `arm64-fix-served.json` and `arm64-speed-served*.json`;
- the roofline, re-anchor, q4k-peer and G26 JSONs;
- the two CUDA `served-three-way.json` files.

The arm64 kernel-bench medians were recomputed from `arm64-fix-kernel-bench.log`.

**Candidates.** The sweeps turned up 142 candidates:
- **42 included**: 29 in tier 1 and 13 in tier 2, making 87 rows.
- **100 excluded**, each with its reason in the last section:
  - 72 with an in-process A/B only;
  - 18 with a served A/B only;
  - 10 that have both halves but are not an A/B of one change.

**A row** is one change × one cell: model, depth, client count or K.

**Direction.** Every ratio is a speedup, where > 1 means the new arm is faster:
- old ms ÷ new ms, or new tok/s ÷ old tok/s;
- old kernel time ÷ new kernel time;
- for the one latency metric (a stall), old ÷ new.

A record that states new ÷ old time is inverted in the script, which records the inversion in that change's
`ip_dir`. Examples are the arm64 kernel bench's "fused ÷ f32 0.904" and q4k's "q4k ÷ int4".

**Resolution rule.** A side resolves a direction when one of these holds:
- its recorded per-pair or per-rep range lies wholly on one side of 1.0, or the arms' run ranges do not overlap
  (bounds min(new)/max(old) .. max(new)/min(old));
- it carries no spread, and |ln r| ≥ ln 1.035, the between-session drift `CLAUDE.md` documents for nobara;
- a record's own reason settles it. Two examples: R9's unpaired before/after runs sit against the record's documented
  ~10% back-to-back drift, and R17's served cells have a 128-key control reading ±2%.

A split pair-sign count does not resolve.

**Verdict per row.** AGREE means both sides resolve in the same direction. DISAGREE means both resolve in opposite
directions. UNRESOLVED means either side does not resolve.

**Disclosed:** I fixed the rule after reading the seed pairs (L1) and before assembling the other 39 changes. The
3.5% threshold is `CLAUDE.md`'s figure, not tuned to this table. The two readings of the rule (strict and
resolved-only) were fixed with the task, before the table existed. The 2.2% floor in the sensitivity line was added
after the do-nothing cells turned up, and it changes no verdict in the grading; it is reported as sensitivity only.

**k = ln(served) / ln(in-process)** is the fraction of the in-process effect, in log terms, that shows served. k = 1 is
perfect transfer, 0 < k < 1 means a smaller served effect, and k < 0 means the signs disagree.
- A kernel-only bench times a fraction of a token, so k < 1 is the expected shape for it, not a failure.
- Many rows also carry a whole-token in-process reading of the same change. Examples: R18's full-token GPU time beside
  its GEMV-work metric, and R17's full token beside its attention time. k is summarised both ways.
- The ratio of ratios (served ÷ in-process) is in the script output too.

**Tier 1 vs tier 2.**
- **Tier 1:** the served A/B is same-session (interleaved, or adjacent arms in one sitting). Its builds or settings
  differ by the change alone, or by commits the record shows do not touch the measured path.
- **Tier 2:** the served leg is cross-session, cross-tree or cross-condition, or it spans more than the change.
- TE5(b) is graded on tier 1; tier 2 is reported beside it.

**Locality.**
- **kernel:** the change is inside a kernel, or is a different kernel or launch configuration for the same op.
- **outside:** scheduling, batching, fork/join and barrier count, dispatch count, allocation or cache state, or host
  overhead. These are the changes where an in-process kernel bench is expected to mislead.
- **mixed:** both in one change.

## The table

"k, whole-token" is k against the row's whole-token in-process reading. It reads "(same)" when the primary reading is
already whole-token, and is blank when none exists. "Unresolved (in-process)" names the side that did not resolve.
Change IDs are the script's keys; the sources table below maps each one to its commits and files.

#### Tier 1

| change | cell | loc | in-process (class) | served | k | k, whole-token | verdict |
|---|---|---|---:|---:|---:|---:|---|
| L1-f16-amd64 | 0.5B d128 | kernel | 0.968 (e2e) | 1.016 | -0.49 | (same) | unresolved (in-process); point estimates opposite |
| L1-f16-amd64 | 1.5B d128 | kernel | 1.059 (e2e) | 1.047 | 0.80 | (same) | agree |
| L1-f16-amd64 | 7B d128 | kernel | 1.090 (e2e) | 1.089 | 0.99 | (same) | agree |
| L1-v1500-arm64 | 0.5B d128 | kernel | 0.551 (kernel) | 0.544 | 1.02 |  | agree |
| L1-v1500-arm64 | 1.5B d128 | kernel | 0.709 (kernel) | 0.554 | 1.72 |  | agree |
| L1-v1500-arm64 | 7B d128 | kernel | 0.347 (kernel) | 0.435 | 0.79 |  | agree |
| L1-fix-arm64 | 0.5B d128 | kernel | 1.106 (kernel) | 1.041 | 0.40 |  | agree |
| L1-fix-arm64 | 1.5B d128 | kernel | 1.109 (kernel) | 1.053 | 0.50 |  | agree |
| L1-fix-arm64 | 7B d128 | kernel | 1.149 (kernel) | 1.080 | 0.55 |  | agree |
| R18-metal | 1.5B d128 | kernel | 1.171 (kernel) | 1.135 | 0.80 | 0.91 | agree |
| R18-metal | 1.5B d2048 | kernel | 1.218 (kernel) | 1.133 | 0.63 | 0.98 | agree |
| R18-metal | 1.5B d3900 | kernel | 1.221 (kernel) | 1.124 | 0.59 | 0.90 | agree |
| R18-metal | 7B d128 | kernel | 1.317 (kernel) | 1.265 | 0.85 | 0.98 | agree |
| R18-metal | 7B d2048 | kernel | 1.337 (kernel) | 1.240 | 0.74 | 0.97 | agree |
| R18-metal | 7B d3900 | kernel | 1.332 (kernel) | 1.226 | 0.71 | 0.98 | agree |
| R18b-metal | 1.5B d128 | kernel | 1.079 (kernel) | 1.073 | 0.93 | 1.03 | agree |
| R18b-metal | 1.5B d2048 | kernel | 1.121 (kernel) | 1.071 | 0.60 | 1.00 | agree |
| R18b-metal | 1.5B d3900 | kernel | 1.136 (kernel) | 1.064 | 0.49 | 0.89 | agree |
| R18b-metal | 7B d128 | kernel | 1.102 (kernel) | 1.085 | 0.84 | 0.99 | agree |
| R18b-metal | 7B d2048 | kernel | 1.106 (kernel) | 1.077 | 0.74 | 0.97 | agree |
| R18b-metal | 7B d3900 | kernel | 1.112 (kernel) | 1.069 | 0.63 | 0.94 | agree |
| R17-metal | 1.5B d2048 | kernel | 2.754 (kernel) | 1.220 | 0.20 |  | agree |
| R17-metal | 1.5B d3900 | kernel | 3.500 (kernel) | 1.410 | 0.27 | 0.99 | agree |
| R17-metal | 7B d2048 | kernel | 2.777 (kernel) | 1.180 | 0.16 |  | agree |
| R17-metal | 7B d3900 | kernel | 3.490 (kernel) | 1.330 | 0.23 | 0.99 | agree |
| R2-metal | 1.5B d4000 / d3900 | kernel | 1.188 (e2e) | 1.187 | 0.99 | (same) | agree |
| R6-cuda | 1.5B d3900 | kernel | 3.459 (kernel) | 1.557 | 0.36 |  | agree |
| R6-cuda | 7B (D7) d8000 | kernel | 4.080 (kernel) | 1.543 | 0.31 |  | agree |
| vsum-spike-cuda | 7B (D7) d8000 | kernel | 1.407 (kernel) | 1.162 | 0.44 |  | agree |
| fused-rms-qkv-cuda | 7B (D7) d128 | kernel | 1.554 (kernel) | 1.036 | 0.08 | 0.94 | agree |
| fused-rms-qkv-cuda | 1.5B d128 | kernel | 1.383 (kernel) | 1.019 | 0.06 |  | agree |
| glu-quant-cuda | 7B (D7) d128 | kernel | 2.625 (kernel) | 1.045 | 0.05 | 0.95 | agree |
| fused-rms-gu-wave-cuda | 7B (D7) d128 | kernel | 1.028 (e2e) | 1.020 | 0.71 | (same) | unresolved (in-process) |
| G35-webgpu | 1.5B, 128 tokens | kernel | 1.130 (e2e) | 1.089 | 0.70 | (same) | agree |
| G36-webgpu | 1.5B pos 512 / 512 tokens | kernel | 2.639 (e2e) | 1.798 | 0.60 | (same) | agree |
| R8-vision-cuda | gemma-3-4b tower / img0 | kernel | 6.362 (e2e) | 4.661 | 0.83 | (same) | agree |
| S01-prefill-arm64 | 1.5B K=512 (n=9) | kernel | 2.884 (kernel) | 1.788 | 0.55 |  | agree |
| S01-prefill-arm64 | 1.5B K=3900 | kernel | 2.884 (kernel) | 1.618 | 0.45 |  | agree |
| MC3-S4-metal | 7B B=2 / 2 clients | kernel | 1.111 (e2e) | 1.121 | 1.08 | (same) | agree |
| MC3-S4-metal | 1.5B B=2 / 2 clients | kernel | 1.020 (e2e) | 1.059 | 2.96 | (same) | unresolved (in-process) |
| q4k-vs-int4-cuda | 7B d128 | mixed | 0.890 (e2e) | 0.893 | 0.97 | (same) | agree |
| q4k-vs-int4-cuda | 1.5B d128 | mixed | 0.753 (e2e) | 0.766 | 0.94 | (same) | agree |
| R9-linux | 1.5B d128 | mixed | 1.367 (e2e) | 1.348 | 0.96 | (same) | agree |
| R9-linux | 7B d128 | mixed | 1.101 (e2e) | 1.089 | 0.89 | (same) | unresolved (in-process) |
| fused-gateup-amd64 | 0.5B d128 | outside | 1.113 (e2e) | 1.241 | 2.02 | (same) | agree |
| fused-gateup-amd64 | 1.5B d128 | outside | 1.066 (e2e) | 1.135 | 1.98 | (same) | agree |
| fused-gateup-amd64 | 7B d128 | outside | 1.029 (e2e) | 1.020 | 0.69 | (same) | agree |
| G26-sampler-cuda | 1.5B (vocab 151936), temp 1.0 | outside | 0.810 (kernel) | 1.085 | -0.39 |  | DISAGREE; point estimates opposite |
| G26-sampler-cuda | phi3-mini (vocab 32064), temp 1.0 | outside | 1.136 (kernel) | 1.030 | 0.23 |  | agree |
| MC3c-S1-cpu | 7B B=4 / 4 clients | outside | 1.058 (e2e) | 1.051 | 0.88 | (same) | agree |
| MC3-metal | 1.5B B=4 / 4 clients | outside | 1.840 (e2e) | 1.593 | 0.76 | (same) | agree |
| MC3-metal | 1.5B B=2 / 2 clients | outside | 1.100 (e2e) | 1.071 | 0.72 | (same) | agree |
| MC3-7B-stack-metal | 7B B=4 / 4 clients | outside | 1.998 (e2e) | 1.785 | 0.84 | (same) | agree |
| MC3-7B-stack-metal | 7B B=2 / 2 clients | outside | 1.016 (e2e) | 1.018 | 1.12 | (same) | unresolved (in-process) |
| MC3-S3-metal | 1.5B B=4 / 4 clients | outside | 1.277 (e2e) | 1.261 | 0.95 | (same) | agree |
| MC3-S3-metal | 1.5B B=2 / 2 clients | outside | 1.096 (e2e) | 1.086 | 0.90 | (same) | agree |
| MC3-cuda | 1.5B B=4 / 4 clients | outside | 1.751 (e2e) | 1.380 | 0.57 | (same) | agree |
| MC3-cuda | 1.5B B=2 / 2 clients | outside | 1.298 (e2e) | 1.085 | 0.31 | (same) | agree |
| MC3-cuda | 7B B=4 / 4 clients | outside | 2.042 (e2e) | 1.826 | 0.84 | (same) | agree |
| MC3c-step1-cpu | 1.5B N=4 / 4 clients | outside | 2.002 (e2e) | 1.910 | 0.93 | (same) | agree |
| MC3c-step1-cpu | 1.5B N=2 / 2 clients | outside | 1.551 (e2e) | 1.579 | 1.04 | (same) | agree |
| MC3c-step2-cpu | 7B B=4 / 4 clients | outside | 2.410 (e2e) | 2.187 | 0.89 | (same) | agree |
| MC3c-step2-cpu | 7B B=2 / 2 clients | outside | 1.549 (e2e) | 1.466 | 0.87 | (same) | agree |
| chunked-prefill-metal | 1.5B stall (max decoder gap) | outside | 4.475 (e2e) | 4.695 | 1.03 | (same) | agree |
| chunked-prefill-metal | 1.5B cost (prefill total / cell wall) | outside | 0.927 (e2e) | 0.951 | 0.67 | (same) | agree |

#### Tier 2

| change | cell | loc | in-process (class) | served | k | k, whole-token | verdict |
|---|---|---|---:|---:|---:|---:|---|
| row4-arm64 | 1.5B d128 | kernel | 1.085 (e2e) | 1.070 | 0.83 | (same) | agree |
| lmhead-w8a8-arm64 | 1.5B d128 | kernel | 7.700 (kernel) | 1.720 | 0.27 |  | agree |
| A1-attn-arm64 | 1.5B d128 | mixed | 3.858 (kernel) | 1.266 | 0.17 |  | agree |
| R06-amd64-xtree | 1.5B d128 | outside | 1.053 (e2e) | 1.066 | 1.24 | (same) | agree |
| splitkv-A2-cuda | 1.5B nKeys 256 | mixed | 1.000 (e2e) | 0.941 | n/a | (same) | unresolved (in-process) |
| splitkv-A2-cuda | 1.5B nKeys 512 | mixed | 1.020 (e2e) | 0.939 | -3.18 | (same) | unresolved (in-process); point estimates opposite |
| splitkv-A2-cuda | 1.5B nKeys 2048 | mixed | 1.200 (e2e) | 1.191 | 0.96 | (same) | agree |
| M26-batched-prefill-cuda | M26 prefill / W3 cell wall | outside | 1.085 (e2e) | 1.043 | 0.52 | (same) | agree |
| P2b-host-sampler-cuda | phi3-mini (32k vocab), temp 1.0 | outside | 1.620 (kernel) | 1.280 | 0.51 |  | agree |
| P2b-host-sampler-cuda | 0.5B (152k vocab), temp 1.0 | outside | 3.060 (kernel) | 2.256 | 0.73 |  | agree |
| P2b-host-sampler-cuda | gemma3-1b (262k vocab), temp 1.0 | outside | 4.720 (kernel) | 2.371 | 0.56 |  | agree |
| R1-W4F16-metal | 1.5B d128 | kernel | 1.036 (e2e) | 1.029 | 0.81 | (same) | agree |
| L2L3-prefill-cuda | 1.5B K=128 | kernel | 1.720 (e2e) | 1.921 | 1.20 | (same) | agree |
| L2L3-prefill-cuda | 1.5B K=512 | kernel | 4.104 (e2e) | 3.952 | 0.97 | (same) | agree |
| L2L3-prefill-cuda | 1.5B K=2048 | kernel | 4.058 (e2e) | 3.866 | 0.97 | (same) | agree |
| L2L3-prefill-cuda | 1.5B K=3900 | kernel | 3.913 (e2e) | 3.840 | 0.99 | (same) | agree |
| R3-prefill-floor-metal | 1.5B K=64 | mixed | 4.011 (e2e) | 3.831 | 0.97 | (same) | agree |
| R3-prefill-floor-metal | 1.5B K=128 | mixed | 4.130 (e2e) | 4.660 | 1.09 | (same) | agree |
| optfwd-cuda | T=0.2 (0.5B synthetic / phi3-mini) | outside | 1.214 (e2e) | 1.011 | 0.06 | (same) | unresolved (served) |
| optfwd-cuda | T=1.0 (0.5B synthetic / phi3-mini) | outside | 0.996 (e2e) | 0.948 | 14.45 | (same) | unresolved (in-process) |
| dflash-drafter | Qwen3-4B thinking | outside | 0.820 (e2e) | 0.830 | 0.94 | (same) | agree |
| R16-prefill-gemm-metal | 1.5B K=512 | kernel | 3.249 (kernel) | 2.582 | 0.81 | 0.93 | agree |

## Summary

| | rows / changes | point-estimate signs agree | agree | DISAGREE | unresolved | resolved-only agreement |
|---|---|---:|---:|---:|---:|---:|
| tier 1, all | 65 / 29 | 63 / 65 (97%) | 59 | 1 | 5 | 59 / 60 (98%) |
| tier 1, kernel-local | 40 / 17 | 39 / 40 | 37 | 0 | 3 | 37 / 37 |
| tier 1, mixed | 4 / 2 | 4 / 4 | 3 | 0 | 1 | 3 / 3 |
| tier 1, outside the kernel | 21 / 10 | 20 / 21 | 19 | 1 | 1 | 19 / 20 |
| tier 2, all | 22 / 13 | 21 / 22 | 18 | 0 | 4 | 18 / 18 |
| all | 87 / 42 | 84 / 87 (97%) | 77 | 1 | 9 | 77 / 78 (99%) |

Magnitude on AGREE rows:

| slice | k, kernel-class reading | k, whole-token reading | served ÷ in-process, whole-token |
|---|---|---|---|
| tier 1, all | median 0.55, 0.05–1.72 (n=31) | median 0.94, 0.31–2.02 (n=44) | median 0.99, 0.68–1.12 |
| tier 1, kernel-local | median 0.55, 0.05–1.72 (n=30) | **median 0.97, 0.60–1.08 (n=23)** | |
| tier 1, outside the kernel | | median 0.88, 0.31–2.02 (n=18) | |
| tier 2, all | median 0.53, 0.17–0.81 (n=6) | median 0.97, 0.52–1.24 (n=13) | median 0.99, 0.93–1.13 |

**Sensitivity.** The served harness at n=2 can "resolve" about 2% on identical code (see the instrument notes). If a
served side must also clear 2.2% to count as resolved, two rows move from agree to unresolved: fused_rms_qkv on the
1.5B (served 1.019×) and fused gate+up on the 7B (1.020×). Strict kernel-local failures then become four, and the
resolved-only verdict is unchanged.

## What the rows show

1. **A kernel-only number overstates the served gain every time. A whole-token in-process number lands within a few
   percent.** The same change, three readings:

   | change | kernel or sub-bucket | whole token, in-process | served |
   |---|---:|---:|---:|
   | R17 attention, 1.5B@3900 | 3.50× | 1.413× | 1.41× |
   | R18 GEMV, 7B@128 | 1.317× | 1.270× | 1.265× |
   | glu_quant, D7 | 2.625× | 1.048× | 1.045× |
   | fused_rms_qkv, D7 | 1.554× | 1.038× | 1.036× |

   R6 (3.46× → 1.557×), the V-sum spike (1.407× → 1.162×), the LM head (7.7× → 1.72×) and S-01 (2.88× → 1.62–1.79×)
   have only the kernel reading, and all four overstate the same way.
2. **The one resolved sign flip is outside the kernel: G26's sampler microbenchmark.**
   - At 152k vocab it read HEAD 23% slower; served, HEAD is 8.5% faster, and the in-situ sampling step went 1467 →
     1009 µs.
   - The record: "The loop does not merely mis-scale the number, it inverted the comparison."
   - At 32k vocab the same bench agreed (1.136× vs 1.030×).
   - This is the class the kill line was written around, and it shows up as expected, on a component microbench of a
     change whose effect is allocation and cache state.
3. **A kernel bench can also under-predict.** At the v1.50.0 scalar widen on the 1.5B, the kernel bench reads 0.709×
   and served reads 0.554× (k 1.72). No bench of the kernel alone should be able to regress less than the token it
   sits in. So the bench omitted some cost the real decode pays. The record does not explain it. This is the
   "synthetic reproduces shape, not pressure" lesson.
4. **Split-KV A2 (tier 2) is the recorded case of an in-process claim that served refuted.**
   - `TestSplitKVCrossover` said "breaks even at 256, wins from 384+". Served, the 1.5B loses at 256 (0.941) and 512
     (0.939).
   - Its record: the tight `ForwardArgmax` loop "hides the per-token CPU dispatch a real request exposes", and best-of-3
     minimum "favours the higher-variance arm".
   - Under this rule both rows are unresolved on the in-process side: no number with a spread was ever recorded for
     256 or 512. The rule would not have let that claim stand as a prediction.
5. **Outside the kernel, direction transfers and magnitude is looser (k 0.31–2.02).**
   - Fused gate+up's served gain is about twice the in-process one on the 0.5B and 1.5B (k 2.02, 1.98). Its record: "I
     have not found why."
   - Batching changes (MC3, MC3 CUDA, MC3c) land below their in-sequence step ratio (k 0.31–0.95). The records put the
     difference down to prefills between steps and the straggler window.
6. **Fixing a wrong baseline made in-process agree with served.** DFlash's first in-process reading (0.98×) used a
   slower-than-production baseline. Against `Model.Generate` it reads 0.82×, and served reads 0.83×.

## Instrument notes (observations, not rows)

- **The kernel bench's own reading drifts.** The same aikit scalar-widen arm, run twice on the same Mac about 40 s
  apart, gave:
  - -count 8 medians (scalar ÷ f32 time): 1.815 / 1.402 / 1.418 / 2.879 (`arm64-fix-kernel-bench.log`);
  - the interleaved driver: 1.694 / 1.193 / 1.391 / 3.117 (`arm64-fix-kernel-bench-interleaved.log`).

  The 1.5B gate/up shape moved from 1.40× to 1.19× slower. The fused arm moved far less: 7B 0.870 → 0.949, the rest
  within 0.007. The campaign doc's "0.898–0.949" quotes the interleaved run. A single reading of a kernel-bench arm is
  not fixed to within ~15% on this box by day.
- **The served harness at n=2 "resolves" −2% on identical code.** The 0.5B keeps the original kernels in both the
  fused_rms_qkv and fused_rms_gu builds, yet its served cells read:
  - 0.9855×, runs 0.9816–0.9893, non-overlapping;
  - 0.979×, runs 0.9746–0.9834.

  Non-overlapping n=2 runs are not a direction below ~2%; hence the sensitivity line above.
- **Mislabels in the records.**
  - R9's "1.37× / 1.10×" is called a "paired A/B … measured in-process" in `cpu-peer-reanchor-2026-09-22.md`. The
    attribution record shows two separate runs; only the per-knob 1.189× / 1.118× are paired.
  - R13's record calls `BenchmarkDecodeAtDepth` / `TestZZDiagGroupedFires` "served". They are in-process, and R13 has
    no served A/B.
- **Level is not ratio.**
  - G35 read 118.4 tok/s in-process and 104.5 through the server on the same kernel.
  - The retired "0.5B 1.78×" divided an in-process 476.1 tok/s by Ollama's HTTP 268; the same code serves 320.1.
  - The concordance here is about the new ÷ old ratio only. Absolute in-process levels do not transfer.

## Limits

- **Survivorship.** A served A/B exists mostly for changes whose in-process result was worth taking further. A change
  the in-process lane wrongly killed never got served, so this table cannot measure the lane's false negatives.
  - R13 is the warning. Three wiring designs read "no effect" or "slower" in-process because of a softmax
    serialization bug next to the kernel. A default flow that stops at a negative in-process read would have parked a
    real 1.32× at depth 8192.
- **Many records carry no spread,** and those rows rest on the 3.5% threshold rule.
- **Several in-process legs are not strictly one process,** and the script names each instrument:
  - `BenchmarkResidentDecode` and R2's depth bench run separate processes;
  - glu_quant and fused_rms_gu are ncu profiles taken per build;
  - the G26 microbench runs one build per tree.
- **Several served legs are one pair per cell:** the V-sum spike, R6, the MC3 CUDA 2-client and 7B cells.
- **Served runs here are mostly n=2 or n=3.**
- **Record-level caveats are carried in each change's `caveat` field.** Examples:
  - `d88ef960` (fused gate+up's served build) does not resolve in this clone. It is the pre-rebase `a79ef544`, with
    only docs commits between it and the old arm.
  - R9's served span also carries an arm64-only aikit bump and an unexplained 0.5B regression.

## Sources

| change | what | machine | tier | in-process instrument → source | served instrument → source |
|---|---|---|---|---|---|
| L1-f16-amd64 | L1 f16 group scales, CPU int4 (row-widening f16 kernel) (S0b 79555115 (in-process); build 2669bf11 vs 8940eaca (served); merged 5c85f7c0) | nobara, Ryzen 7 3700X, CPU int4 | 1 | cpuDecodeAB paired in-process ABBA, 5 pairs, depth 128 (TestS0F16Scales_AB, test-hook f16/f32 switch) → docs/measurements/cpu-decode-peer-gap-2026-09-27/s0b-gate2-ab.log:57,113,169; docs/tasks/task-cpu-decode-peer-gap-2026-09.md:240-244 | bench_peer.py goinfer vs goinfer_old vs Ollama, one session, depth 128, 3 runs → docs/measurements/cpu-decode-peer-gap-2026-09-27/gate5-served.json; docs/tasks/task-cpu-decode-peer-gap-2026-09.md:321-327 |
| L1-v1500-arm64 | L1 as merged on arm64 (aikit v1.50.0): scalar Go f16 widen per scale (new 5c85f7c0 vs old 3cd62e6d (served); the 'scalar' arm of aikit's arm64-f16-widen bench) | MacBook M1 Pro, CPU int4 (row4) | 1 | aikit BenchmarkW4A8Row4F16Fix, M=1 row4 matmul over a cold bank, -count 8, median, scalar vs f32 → docs/measurements/cpu-decode-peer-gap-2026-09-27/arm64-fix-kernel-bench.log; docs/tasks/task-cpu-decode-peer-gap-2026-09.md:515-520 | bench_peer.py, two order-reversed passes, geometric mean, depth 128, 3 runs → docs/measurements/cpu-decode-peer-gap-2026-09-27/arm64-speed-served.json, arm64-speed-served-pass2.json; docs/tasks/task-cpu-decode-peer-gap-2026-09.md:410-414 |
| L1-fix-arm64 | L1 arm64 fix B: NEON widen + fused f16 row4 decode kernel (aikit v1.50.1) (aikit 2fd6f59 via goinfer e351fad4 vs 3cd62e6d; 5d223da3) | MacBook M1 Pro, CPU int4 (row4) | 1 | aikit BenchmarkW4A8Row4F16Fix, -count 8, median, fused vs f32 → docs/measurements/cpu-decode-peer-gap-2026-09-27/arm64-fix-kernel-bench.log; docs/tasks/task-cpu-decode-peer-gap-2026-09.md:515-524 | bench_peer.py pass 1 (fix -> Ollama -> old), depth 128, 3 runs; pass 2 skipped by owner amendment → docs/measurements/cpu-decode-peer-gap-2026-09-27/arm64-fix-served.json; docs/tasks/task-cpu-decode-peer-gap-2026-09.md:530-534 |
| R18-metal | R18 Metal decode GEMV, MLX-shaped rows kernels (i2244), bit-identical (serve-metal-622b1f9b vs 65ffe328; confirmation afd77749) | MacBook M1 Pro, Metal W4A8 | 1 | TestR18InSequence (testhooks), int4-GEMV work per token, 7 paired reps → docs/measurements/metal-decode-gemv-r18-2026-09-26.md:125-130 (alt :219) | bench_peer.py, three engines interleaved cell by cell, 3 runs x 8 x 64 → docs/measurements/metal-decode-gemv-r18-2026-09-26.md:241-246 |
| R18b-metal | R18b Metal decode GEMV, masked half-staged form (h4244), bit-identical (serve-metal-c5d7e310 vs 9bafd1f3) | MacBook M1 Pro, Metal W4A8 | 1 | TestR18InSequence (testhooks), int4-GEMV work per token, 7 paired reps → docs/measurements/metal-decode-gemv-r18b-2026-09-26.md:61-66 | bench_peer.py, three engines interleaved cell by cell, 3 runs x 8 x 64 → docs/measurements/metal-decode-gemv-r18b-2026-09-26.md:108-113 |
| R17-metal | R17 Metal decode attention at depth, attention_fa_blk (GQA groups 6/7) (7df881f5 vs dde11d93) | MacBook M1 Pro, Metal | 1 | TestR17AttentionProto, in-sequence attention = full - no-op, arms interleaved rep by rep (confirmation 7 reps at 1.5B@3900; exploratory 5 reps elsewhere) → docs/measurements/metal-decode-attn-r17-2026-09-25.md:71-77, 433-439 | bench_peer.py, same session, three engines interleaved, 3 runs x 8 x 64 → docs/measurements/metal-decode-attn-r17-2026-09-25.md:517-522 |
| R2-metal | R2 Metal attention_fa as the default decode attention past 1536 keys (7135283f; measured at 344b9514) | MacBook M1 Pro, Metal, 1.5B | 1 | metal depth bench (TestZZ_metalDepthBench), 3 runs per arm interleaved, each its own process, best-of-5 → docs/measurements/r2-attn-fa-speed-2026-09-21.md:44-49 | bench_peer.py, one invocation per arm, Ollama inside each as the drift control → docs/measurements/r2-attn-fa-speed-2026-09-21.md:62-70 |
| R6-cuda | R6 CUDA flash-decode lane attn_decode_fa (S=16) (d08eefa1 / c0fda79b; default-on 643ea0ac) | nobara, RTX 2070 SUPER, driver 595.91.07 | 1 | TestFlashDecodeKernelLadder, ncu gpu__time, one layer, exact split-KV vs lane → docs/measurements/attn-decode-fa-ladder-2026-09-20.md:16-17 | bench_splitkv.py, fresh serve per arm, arms adjacent with alternating order, idle-gated → docs/measurements/attn-decode-fa-served-2026-09-20.md:18,21 |
| vsum-spike-cuda | CUDA flash-decode V-sum split spike (S=4) (spike binary 2026-09-13 (opt-in, never default)) | nobara, RTX 2070 SUPER; D7 (Qwen2.5-7B) at depth 8000 | 1 | ncu, attention kernels total, 32 launches median, spike vs bit-identical split path → docs/measurements/vsum-split-spike-2026-09-13.md:11,15 | two serve arms, the same 8000-token prompt, 48 greedy tokens, one pair → docs/measurements/vsum-split-spike-2026-09-13.md:65-72 |
| fused-rms-qkv-cuda | CUDA fused_rms_qkv rows-per-warp (fewer redundant prologues), bit-identical (74813033) | nobara, RTX 2070 SUPER | 1 | TestFusedQKVRowsBench, ncu, rows-per-warp as a runtime parameter, random int4 weights of each geometry → docs/measurements/fused-rms-qkv-2026-09-21.md:24,26 (alt :47) | bench_peer.py, same session, three engines, one sweep (n=2) → docs/measurements/fused-rms-qkv-2026-09-21/served-three-way.json; fused-rms-qkv-2026-09-21.md:38-44 |
| glu-quant-cuda | CUDA glu_quant launched at 1024 threads instead of 256 (bit-identical) (a5ac99d2) | nobara, RTX 2070 SUPER, D7 | 1 | ncu of a served decode window, before and after builds (kernel time) → docs/measurements/d7-decode-breakdown-2026-09-21.md:34 | bench_peer.py, same session, three engines, one sweep (n=2) → docs/measurements/d7-decode-breakdown-2026-09-21/served-three-way.json; d7-decode-breakdown-2026-09-21.md:43-51 |
| fused-rms-gu-wave-cuda | CUDA fused_rms_gu wave rule for rows-per-warp (bit-identical) (wave-rule follow-up to 74813033) | nobara, RTX 2070 SUPER, D7 | 1 | ncu in situ on D7, GPU time per token, per build → docs/measurements/fused-rms-gu-diagnosis-2026-09-21.md:60 | bench_peer.py, same session, one sweep (n=2) → docs/measurements/fused-rms-gu-diagnosis-2026-09-21/served-three-way.json; fused-rms-gu-diagnosis-2026-09-21.md:66-68 |
| G35-webgpu | WebGPU quantize row max-abs: serial lane-0 scan -> 64-lane tree reduce (bit-identical) (30008c03) | nobara, WebGPU on RTX 2070 SUPER, 1.5B int8int8 | 1 | real-model decode, in-process, interleaved A/B, best-of-6 x 48 tokens → docs/QUEUE.md:1579 | gpu/cmd/serve, streaming chat, best of 4, new arm started twice around the old → docs/QUEUE.md:1589-1596 |
| G36-webgpu | WebGPU decode attention split over keys instead of head dims (262d4d17) | nobara, WebGPU on RTX 2070 SUPER, 1.5B | 1 | ablation profile, same session, whole token at pos 512 → docs/QUEUE.md:1725-1727 | gpu/cmd/serve, server-to-server, best of 2, dim-split re-measured after key-split → docs/QUEUE.md:1694-1702 |
| R8-vision-cuda | R8 CUDA SigLIP tower fused non-causal attention (BM=128) (64a39a72 / 6baed301) | nobara, RTX 2070 SUPER, gemma-3-4b-it tower | 1 | cuda/vision_tower_timing_test.go, tower seconds, arms in separate processes alternating, 2 rounds → docs/measurements/vision-tower-mma-2026-09-21.md:14 | serve --backend cuda, per-image request time, arms N (fused) vs E (exact) in one pre-registered run → docs/measurements/vision-tower-downstream-2026-09-21.md:8 |
| S01-prefill-arm64 | S-01/S-01b register-blocked W4A8 tiles, CPU prefill (aikit v1.31.0 -> v1.32.0; the bump is the adoption) (897fb18d) | MacBook M1 Pro, CPU int4, 1.5B | 1 | aikit's own single-core kernel microbenchmark (24.1 -> 69.5 GMAC/s), as quoted in 897fb18d → commit 897fb18d (kernel figure from aikit's SIMD-audit task doc, S-01) | bench_peer_prefill.py, three arms interleaved within each cell, order rotated per depth, TTFT tok/s → commit 897fb18d |
| MC3-S4-metal | MC3 S4: qkv and gate\|up as per-row production GEMVs at B <= 2 (calibrated kernel choice) (7fa344b2 vs 915bf6fc) | MacBook M1 Pro, Metal | 1 | TestMC3Step_throughput, in sequence, depth 128, B=2 step before -> after → docs/measurements/concurrency-mc3-s4-2026-09-27.md:78,80 | bench_w7_plain.py, 2 clients, 3 interleaved pairs → docs/measurements/concurrency-mc3-s4-2026-09-27.md:9-10 |
| q4k-vs-int4-cuda | CUDA --quant q4k (native Q4_K GEMV + per-32 activations) vs today's int4 (gemv_q4k_g32 v2 53f1a395, PR #3 210e7307; served d5077650) | nobara, RTX 2070 SUPER, CUDA resident | 1 | BenchmarkResidentDecode (cuda), 6 paired rounds, alternating, separate processes (7B); same binary, same session, 2 reps (1.5B, lever-1 fusion check) → docs/tasks/task-int4-weight-quality-2026-09.md:351 (7B), :456 (1.5B); docs/measurements/q4k-lever1-2026-09-26/fusion-check.txt | bench_peer.py, S1 (q4k) and S2 (int4) back to back, each with its own peers, 3 runs → docs/measurements/q4k-peer-2026-09-26/s1-q4k.json, s2-int4.json; docs/tasks/task-int4-weight-quality-2026-09.md:398,401 |
| R9-linux | R9 Linux fixes: activation fan-out off + grouped-attention Go fallback off (non-arm64) (714ad587 in HEAD aecfc9be vs 3ea2f93d (served)) | nobara, Ryzen 7 3700X, CPU int4 | 1 | in-process before/after runs of the shipped defaults (unpaired) → docs/measurements/cpu-decode-attribution-2026-09-22-linux.md:123-129 | bench_peer.py, goinfer HEAD vs goinfer_old 3ea2f93d vs Ollama, one session, n=2 → docs/measurements/cpu-peer-reanchor-2026-09-22.json; cpu-peer-reanchor-2026-09-22.md:56-58 |
| fused-gateup-amd64 | Fused gate+up+SwiGLU decode fork/join (one barrier per layer, the activation inside it) (a79ef544 (served build d88ef960, its pre-rebase twin) vs ec6fec95; only docs commits between) | nobara, Ryzen 7 3700X, CPU int4 | 1 | TestCPURoofline_fusedGateUp, paired in-process ABBA, 5 pairs → docs/measurements/cpu-decode-roofline-2026-09-23.md:124-126 | bench_peer.py phase A, same session, interleaved with Ollama, depth 128, n=2 → docs/measurements/cpu-decode-roofline-served-2026-09-23.json; cpu-decode-roofline-2026-09-23.md:139-141 |
| G26-sampler-cuda | Sampler, anchor ca29d6c -> HEAD f9f8833 (746 commits; P10 scratch-buffer reuse among them) (ca29d6c vs f9f8833; record 28957403) | nobara, RTX 2070 SUPER, CUDA resident, optFwd off in both arms | 1 | BenchmarkG26SampleTemp1_{32k,152k}, go test -bench, 5 runs per tree, each tree its own build → docs/measurements/g26-sampler-bench.log; docs/QUEUE.md:1003-1008 | bench_peer.py temp1.0_notrunc cell, anchor then HEAD sessions back to back (n=8 at 152k, n=15 at 32k) → docs/measurements/g26-152k-{anchor,head}.json, g26-anchor-n15.json, g26-head-nooptfwd-n15.json; docs/QUEUE.md:1003-1019 |
| MC3c-S1-cpu | MC3c step 2 S1: fused q\|\|k\|\|v and gate\|\|up W4A8 batch in the batched CPU step (000efe2e vs 2c1d89ec; pre-registered 5f7480e0) | nobara, Ryzen 7 3700X, CPU int4, 7B | 1 | TestCPUBatchS1_fusedVsUnfused, test-hook flip in-process, 5 reps, B=4 → docs/measurements/concurrency-mc3c-s1-2026-09-27.md:50-51 | bench_w7_plain.py, 4 clients, 3 pairs old new new old old new → docs/measurements/concurrency-mc3c-s1-2026-09-27.md:74-78 |
| MC3-metal | MC3 batched decode on Metal (matrix-unit batched matmuls in a shared step) (MC3 build vs MC1 serialized) | MacBook M1 Pro, Metal, 1.5B | 1 | MC3 S1 step in sequence: aggregate = B x production / step(B), paired per rep → docs/measurements/concurrency-mc3-s1-2026-09-26.md:71 | W7 through serve, 3 interleaved pairs at 4 clients; 2 clients reported → docs/measurements/concurrency-mc3-2026-09-26.md:92,103 |
| MC3-7B-stack-metal | MC3 stack (MC3+S2+S3+chunked prefill) on the 7B vs the pre-MC3 serialized build (current stack vs serialized, 2026-09-27) | MacBook M1 Pro, Metal, 7B | 1 | TestMC3Step_throughput on the current step, in sequence, 7 reps, same session as the W7 run → docs/measurements/concurrency-mc3-7b-w7-2026-09-27.md:107-109 | W7 through serve, 3 interleaved pairs at 4 clients; 2 clients reported → docs/measurements/concurrency-mc3-7b-w7-2026-09-27.md:74,77 |
| MC3-S3-metal | MC3 S3 step polish: per-row kernels as one multi-row dispatch (731f4f4e vs 4954f978) | MacBook M1 Pro, Metal, 1.5B | 1 | GPU step time in sequence, before vs after S3a+S3b → docs/measurements/concurrency-mc3-s3-2026-09-27.md:56-58 | W7 through serve, 3 interleaved pairs at 4 and 2 clients → docs/measurements/concurrency-mc3-s3-2026-09-27.md:90,93 |
| MC3-cuda | MC3 on CUDA: batched multi-request decode (7a44a58e vs 01bcb640) | nobara, RTX 2070 SUPER | 1 | MC3 CUDA S0: batched M-row pass with every row's exact logits vs M sequential decodes, depth 1024, best of 5 → docs/measurements/concurrency-mc3-cuda-s0-2026-09-27.md:34-38 | bench_w7_plain.py, 3 interleaved pairs at 4 clients (1.5B); one pair each for 1.5B 2c and 7B 4c → docs/measurements/concurrency-mc3-cuda-2026-09-27.md:15,67-68 |
| MC3c-step1-cpu | MC3c step 1: -max-concurrent N independent CPU workers (MC3c build at -max-concurrent 4 vs 1) | MacBook M1 Pro, CPU int4, 1.5B | 1 | MC2 J8 cell: N independent decode workers on one model vs serial x1, in-process → docs/measurements/concurrency-mc2-2026-09-26.md:58 | bench_w7_plain.py, one session; 4 clients three times, 2 clients once → docs/measurements/concurrency-mc3c-2026-09-26.md:11,66-68 |
| MC3c-step2-cpu | MC3c step 2: batched CPU decode behind serve's admission (vs step 1's workers) (2c1d89ec vs 0bc06f90) | nobara, Ryzen 7 3700X, CPU int4, 7B | 1 | MC2 Linux 7B cell: batched B vs J8 N workers, per rep, median of 5 (test-only prototype) → docs/measurements/concurrency-mc2-2026-09-26.md:180 | bench_w7_plain.py, 3 pairs at 4 clients; one 2-client pair → docs/measurements/concurrency-mc3c-step2-2026-09-27.md:66,74 |
| chunked-prefill-metal | Chunked prefill under MC3 (256-token chunks) (d4293d41 vs 731f4f4e) | MacBook M1 Pro, Metal, 1.5B | 1 | TestMC5_chunkCost: a 3000-token prefill alone, whole vs chunked, 3 reps interleaved (after the grading) → docs/measurements/chunked-prefill-2026-09-27.md:92-97 | bench_prefill_stall.py, 3 interleaved pairs → docs/measurements/chunked-prefill-2026-09-27.md:76,78 |
| row4-arm64 | W4A8 row4 repack + kernel on arm64 (with its dispatch fix) (1c271213, 0bb63a14) | MacBook M1 Pro, CPU int4, 1.5B | 2 | stub probe, canonical vs row4 in one harness, 64 decode steps (total time per token) → docs/completed/task-w4a8-neon-bandwidth.md:896-901 | bench_peer, after vs the campaign's Step 0 before (another session) → docs/completed/task-w4a8-neon-bandwidth.md:966-968 |
| lmhead-w8a8-arm64 | int4-mode LM head: weight-only Q8 -> full W8A8 (kernel selection) (a11c56bf) | MacBook M1 Pro, CPU int4, 1.5B | 2 | isolated LM-head shape (K=1536, N=151936): MatmulBTQ8 vs MatmulBTW8A8Into → docs/completed/task-w4a8-neon-bandwidth.md:927-931 | bench_peer, two separate runs, vs the post-row4 before → docs/completed/task-w4a8-neon-bandwidth.md:1000-1004 |
| A1-attn-arm64 | A1 decode attention: acc64 AV + 8-wide QK kernels + head threading (task-attention-decode-cost A1) | MacBook M1 Pro, CPU, 1.5B depth 128 | 2 | attendBatchedHeads depth curve, before vs after (c)+(b)+(a), attention only → docs/completed/task-attention-decode-cost.md:465-470 | bench_peer method, after vs the 2026-08-22 diagnosis session → docs/completed/task-attention-decode-cost.md:478-484 |
| R06-amd64-xtree | R-06 W4A8 q/k/v (+gate/up) batch, one fork/join (e4e04992 / 4e933525 (served 09-04); in-process 09-23 tree) | nobara, Ryzen 7 3700X, CPU int4, 1.5B | 2 | TestCPURoofline_w4a8Batch paired ABBA, 5 pairs (plain -> batch) → docs/measurements/cpu-decode-roofline-2026-09-23.md:124 | bench_peer.py goinfer-only cell, GOINFER_W4A8_BATCH toggled, n=10 paired → docs/tasks/task-recompute-audit.md:532-543 |
| splitkv-A2-cuda | CUDA split-KV decode attention default-on at nKeys >= 256 (kernel choice + extra launches) (a4932832 / 26ae07da (in-process 08-04); served P6a 08-09 at 686c9f8, driver 595.58.03) | nobara, RTX 2070 SUPER, 1.5B | 2 | TestSplitKVCrossover (tight in-process ForwardArgmax loop, best-of-3 minimum); TestDecodeDepthThroughput at 2048 → docs/tasks/task-decode-splitkv-attention.md:174-176; docs/ollama-chase.md:315-317 | P6a: 48 cells, a fresh serve per cell, ON / OFF, decode-only client timing → docs/legacy-benchmarks.md:1062 |
| M26-batched-prefill-cuda | CUDA MoE (Gemma-4-26B) takes the batched prefill path (4ee59e15 / 654fa481 / 9453430e) | nobara, RTX 2070 SUPER, M26 | 2 | in-process paired within one process, M=512/2048 (not interleaved, not at depth 8000) → docs/queue-performance.md:166; docs/benchmarks.md:1915-1917 | bench_peer W3 cell wall-clock at depth 8000, re-run at 9453430e against an earlier cell → docs/benchmarks.md:1894,1903-1905 |
| P2b-host-sampler-cuda | P2b deterministic parallel host normalization for temperature sampling (686c9f8 (after) vs ed81e13 (P1, before)) | nobara, RTX 2070 SUPER, driver 595.58.03 | 2 | host-only sampler microbenchmark, 16 cores → docs/completed/plan-still-slow.md:277-279 | decode-only, client-timed from the first streamed token, 8 completions x 2 runs, temperature-only t=1.0 → docs/completed/plan-still-slow.md:281-288 |
| R1-W4F16-metal | R1 Metal W4F16 decode lane (killed) (ebe50556) | MacBook M1 Pro, Metal, 1.5B | 2 | TestZZ_metalDepthBench, once per lane setting (separate processes), min-of-5 → docs/measurements/w4f16-decode-speed-2026-09-21.md:74 | bench_peer.py, two separate invocations, Ollama inside each → docs/measurements/w4f16-decode-speed-2026-09-21.md:52-59 |
| L2L3-prefill-cuda | CUDA prefill: fused attention (L2) + tensor-core GEMM (L3) (f966fa0c) | nobara, RTX 2070 SUPER, 1.5B | 2 | TestPrefillTTFT, batched arm, per-lever env arms → docs/measurements/prefill-l2l3-phase2-2026-09-05.md:40-43 | bench_peer_prefill.py, exact arm and fast arm as two runs, TTFT tok/s → docs/measurements/prefill-l2l3-phase4-peer-2026-09-05.md:57-62 |
| R3-prefill-floor-metal | R3 Metal short-prompt prefill floor 256 -> 64 (batched vs per-token command buffers) (a9ec3452) | MacBook M1 Pro, Metal, 1.5B | 2 | TestR3_startPosSpeed, one resident, 6 interleaved reps, startPos 512 → docs/measurements/r3-startpos-speed-2026-09-21.md:50-51 | bench_peer_prefill.py, exact vs batched, interleaved per depth, startPos 0 → docs/measurements/metal-prefill-floor-2026-09-20.md:50-51 |
| optfwd-cuda | Optimistic forward for sampled decode (on vs off) (6a4e0ae; in-process ff809596; served 28957403) | nobara, RTX 2070 SUPER, CUDA resident | 2 | cuda optfwd end-to-end benchmark (port of metal/optfwd_bench_test.go), qwen2.5-coder-0.5b, synthetic token-id prompt → commit ff809596 | bench_peer.py, phi3-mini, depth 128, GOINFER_NO_OPTFWD toggled, n=6 temperature ladder → docs/QUEUE.md:606-615 |
| dflash-drafter | --drafter block speculation (DFlash), Qwen3-4B with the server's default thinking template (7c485d31) | GPU resident (the spec's Model.Generate / GPU-argmax path); machine not restated | 2 | gate 3 re-run against Model.Generate as the baseline → docs/spec/08-dspark-dflash.md:2241-2245 | a served A/B, cited as 0.83x → docs/spec/08-dspark-dflash.md:2247 |
| R16-prefill-gemm-metal | R16 Metal prefill GEMM redesign (prototype 4) (b4e1f2c0 / 1fd9d95e) | MacBook M1 Pro, Metal, 1.5B | 2 | GEMM category in sequence, 7 paired reps → docs/measurements/metal-prefill-gemm-s2-2026-09-25.md:193-200 | bench_peer_prefill.py after wiring vs the morning's cell h (another session), TTFT tok/s at K=512 → docs/measurements/metal-prefill-gemm-s2-2026-09-25.md:270-275 |

## Candidates examined: 142. Included: 42. Excluded: 100, by reason

**In-process A/B only: no served A/B of the change exists (72).** L1 S0 per-group f16 kernel (superseded by the S0b row kernel before any served run); L2 R-06 default-on 09-27 (r06-gate3-ab.log 1.016/1.030/1.018; both gate-5 served builds have R-06 on); R-06 on the 7B, 09-20 (two process blocks, 0.9997x); R13 group-major acc64 attention (kernel 1.53-2.39x vs in-process decode 0.99x@2048 / 1.32x@8192, which its record calls 'served'; no bench_peer run); S-05 centering fold (kernel 1.284-1.296x, in-process token 1.048-1.097x); P14 W4A8 split-half on amd64 (kernel 1.12x, BenchmarkDecode +2.10%); P-07 gatedDeltaNetStep loop interchange (isolated loop 1.53x); sampler scratch reuse P8, 08-12 (microbench 5-6% slower, reverted); sampler scratch reuse P10, 08-19 (microbench; inside G26's anchor->HEAD row, not isolated); audit #9 sampler-scratch branch (525 -> 392 us/op microbench); R15 sampler max-scan fan-out (microbench, never shipped); KV re-gather strided P1; CPU worker pool / spin-then-park; per-shape matmul width (MC3c); q4k Phase 1a on CPU, PR #2 (BenchmarkDecode 0.977x / 1.308x; the peer comparison ran CUDA only); q4k lever 1 narrow-row kernel; q4k lever 3 fused per-32, PR #4 (its ~212-215 tok/s is in-process or projected); Phi-3 CUDA q4k (1.257x / 1.166x; the peer comparison ran the 1.5B and 7B only); per-32 activations Track A / H2 actquant, PR #1; R19 Metal prefill attention (served TTFT owed); R10 WebGPU rb64 GEMM; R11 / P20 expert-major MoE prefill; C' DMA overlap (its bench_peer row was 'the next measurement'); CUDA graphs on the 26B; C' UploadBatch; R14 CUDA drafter argmax; theta A/Bs (WebGPU, CUDA); non-copy n-gram fix; CUDA batched DeltaNet prefill; Metal L2 fused prefill attention; WebGPU batched prefill TTFT; CUDA prefill chunk demotion; R5 CUDA tile-128; G24 A3 f32 attention; P19 fused prefill attention; P18 CPU MoE expert-major; A3 prefill head fan-out; P12 qwen35 projections; G16 prefill head threading; Metal A1 coalescing; CUDA A1 coalescing; softcap parallel-for; q8Span SIMD widen; PGO; Metal split-KV port (reverted); Metal ForwardArgmax wiring; M-10; aikit v1.17.0 / v1.17.1 bumps; int8 per-Workspace decode threshold (b9b215da); G30 CUDA g4x2 clear; .giw kind 4; bounded top-p selection; WebGPU DP4A; CUDA split-KV q-staging (ncu only); CUDA autoresearch rounds (ncu only); split-KV warp-shuffle softmax (ncu only); C' pin-in-place (load time); Metal pread staging; P2 lazy Z (refuted); G37 constrained-decode mask (computed, not measured served); CPU speculative decoding; perf-campaign phases 1 and 3 (the served side was the chat demo, not HTTP); serve decode-threshold fix (BenchmarkDecode); fork/join removal (go tool trace); moeMLP scratch (no A/B at all); int4 W4A8 Workspace pool (no A/B at all); embedResident scratch (no A/B at all); audit microbenches P-02 / P-04 / P-13; LoRA cache and grid P-10 / P-11; MC2 on the Mac, MC3c on darwin; July cgo-free spikes (Metal Stage B, CUDA); June aikit v0.5.0 (BenchmarkDecode vs the chat demo).

**Served A/B only: no in-process A/B of the change exists (18).** MC3 S2 sampled tokens join steps; MC1 multi-slot KV on Metal; MC1 on CUDA; MC1 on WebGPU; MC0 CPU LRU fix; S6 Metal weight aliasing; G35 + G36 combined; P6a split-KV re-gate as shipped; P1 top_k=1 routing; tool-union grammar; R6 default-on 09-23 (the lane itself is the R6 row); R6 multi-row verify lane; J6 prefix-aware admission (its own harness); session-LRU fix; Linux sidecar default (CUDA instrument unclear); R7 CUDA device top-K (served reads the new build only); R7b Gumbel draw (served reads the new build only); GOMEMLIMIT on M35.

**Both halves recorded, but not an A/B of one change, or not comparable (10).** CPU decode Phase 0 baseline (157196bb vs 411e7fc4, many changes); L1 arm64 fix vs the merged 5c85f7c0 (derived across two served sessions; not independent of the L1 arm64 rows); R-06 on the Mac 09-03/04 (in-process instrument not identified: 'a paired, interleaved before/after A/B', e4e04992); S-01 v1.31.0 vs v1.34.0 stack comparison, de0cc654 (three aikit releases; the S-01 row uses 897fb18d's single bump); S-03/S-04 NEON (served 1.172x spans 14 aikit releases; the only in-process number is a pure-Go AV step); P20 expert-major on the W3 cell, 1.57x (the served cell also carries C' DMA overlap, 'not decomposed'); R16 at K=3900 (served exists only as a ratio to Ollama across sessions); MC4 spec verify on MC3 step kernels (in-process is a cost curve, not a new-vs-old ratio); fit-by-default context vs the ctx-fit fix (in-process on dense models, served on MoE models); the retired '0.5B 1.78x' (an in-process 476.1 tok/s divided by Ollama's HTTP 268; not a change A/B).

