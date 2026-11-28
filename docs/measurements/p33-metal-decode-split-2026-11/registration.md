# P33, first measurement: where a Metal decode token's time goes: registration (2026-11-28, before any run)

**Question.** Metal decode runs at 0.67 to 0.76 of mlx-lm's single-request rate at depth 128 (`docs/benchmarks.md`, §B11). P33
starts from a measurement, not from kernel work: on today's build, how does one decode token of Qwen2.5-Coder-1.5B at `int4` split
between the four GEMV categories, the LM head, attention, the norm and the quantizers, the dispatch floor and the host time outside
the GPU window, and how far is each GEMV from this machine's streaming-read ceiling?

**Why it is registered.** The split chooses what P33 works on next, so it settles a direction. Nothing ships from this run and no
speed bar is being cleared.

**Earlier reading, context only.** The same decomposition was read on 2026-10-24 at depth 128, for the kernel survey: full token
9.708 ms, the four GEMVs 6.06 ms, the head 1.42, the two quantizers 0.875, the floor 0.65. Since then 308 commits have changed
the Metal backend's non-test Go files and 54 its kernels (the block-table KV and the attention score among them), so the arm is not
unchanged and is read again.

**Instruments.** Three tests in `backend/metal`, as they are at the run's commit, from one binary built with `-tags
goinfer_testhooks`. Tier 1, by day, one Apple M1 Pro with 16 GB, the timing lock held exclusively, the checkpoint the model's Metal
`.giw` on the internal SSD.

| cell | test | reads |
|---|---|---|
| `gemv-s0` | `backend/metal/gemv_s0_test.go:TestMetalGEMVS0` | the streaming-read ceiling, and each production GEMV kernel alone at the 1.5B's and the 7B's shapes (random weights, no checkpoint) |
| `decomp` | `backend/metal/decode_decomp_test.go:TestMetalDecodeDecomp`, depths 128 and 192, 5 reps of 20 tokens | each category's in-sequence GPU work (the full token less the token with that category's kernels made no-ops), the dispatch floor, the full token |
| `chain-ab` | `backend/metal/cb01_grade_test.go:TestCB01ChainAB`, greedy, 9 reps of 256 tokens after a 64-token prompt | the whole-token decode rate through `Generate`: the greedy chain against the full-logits path, interleaved |

Depth 192 is the mean depth of `chain-ab` (64 to 320 keys). On Metal the chain-off arm is the full-logits forward (the trunk and
head `decomp` times) with a host argmax.

**Derived, by these formulas and no others.**
- A GEMV category's bandwidth: its weight bytes a token (N·K/2 of nibbles plus N·K/16 of f16 group scales, times the layer count)
  over its in-sequence work at depth 128, also as a share of the best streaming-read figure `gemv-s0` prints in the same run.
- A GEMV category's recoverable time: its work × (1 − its bandwidth ÷ that ceiling).
- The fixed cost: the dispatch floor plus the `rms`, `swiglu_quant`, `quant_vec`, `rope2` and KV-store categories.
- The full-logits path's host time: 1000 ÷ the chain-off median rate, less `decomp`'s full token at depth 192.
- The chain's token: 1000 ÷ the chain median rate, beside `decomp`'s full token at depth 192.

**What each outcome means.**
1. The split counts only if `decomp`'s categories plus the floor sum to between 90% and 115% of its full token at depth 128, and
   the full token's spread is at most 5%. Otherwise the run is reported and nothing is ranked from it.
2. The candidates are ranked by recoverable milliseconds a token at depth 128: each GEMV category's recoverable time, the fixed
   cost above its dispatch floor, and the full-logits path's host time.
3. The largest gets its own registration next if it is at least 0.5 ms a token. The method's spread on a small category is about
   0.2 ms, so two candidates within 0.2 ms are a tie. **Ambiguous:** a tie for first, or a largest candidate under 0.5 ms. Then no
   kernel work is registered, and P33's next step is the mlx-lm side of the split.
4. No claim about which engine's quantisation is faster comes from this run: it has one arm.

**Stopping rule.** One pass of each cell, in the order of the table. A cell refused at its start by the void rule
(`scripts/void_rule.py`: load above 4.0, or available memory under the load's fit need), or one during which system swap grew, is
void, listed, and run again once; a second void ends the attempt for the day. No pass is repeated because of what it read.

**Not in this run.** The 7B's in-sequence split (its checkpoint is not on this machine in a form today's build loads); the
mlx-lm side; the served path's time outside `Generate` (HTTP, streaming, detokenizing); the sampled paths.

**Cost.** About 5 minutes under the lock: `gemv-s0` one to two minutes, `decomp` under a minute for two depths (the 2026-10-24
reading, also two depths, took 29 s), `chain-ab` about a minute for 21 generations of 256 tokens at about 100 tokens a second, plus
two loads. The start and the expected finish are stated before it runs.
