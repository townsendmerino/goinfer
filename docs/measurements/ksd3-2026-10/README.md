# KS-D3: Metal's DeltaNet prefill scan with its state on chip, 2026-10-22

Gates: `docs/tasks/task-kernel-survey-2026-10.md`, "KS-D3 gates" (D3-G), registered before the code (e83abaee). Machine: MacBook
Pro, M1 Pro, 16 GB, macOS 26.6.2, Go 1.27.2. D3-G3, the timed read, ran at night on the owner's word, 2026-10-22, under the timing lock.

**The finding, re-checked on main at 14eea5f8.** `delta_rule_seq` reads and writes each thread's `hk`-float state row in device
memory for every prompt row, in threadgroups of 128 with rows `hk` floats apart. `delta_conv_seq` shifts its window through device
memory for every row. Both are as the survey described.

**The build.**
- **`delta_rule_seq_tg`:** a 32-thread threadgroup holds its 32 state rows in threadgroup memory (16 KB at `hk` = 128; element `kd` of
  lane `l` at `kd*32 + l`, so a load's 32 lanes hit 32 banks). It loads once, runs the same loop in the same order, and stores once.
  No barrier is needed, since a lane reads only its own row.
- **`delta_conv_seq_reg`:** the window in registers for `K` ≤ 5, the same operations in the same order.
- `resident.dnSeqPipelines` takes them where the geometry fits (`hk` ≤ 128, `K` ≤ 5), else the device-memory kernels.
  `dnSeqOldForTest` keeps the old arm for the A/B.

| gate | reading | result |
|---|---|---|
| D3-G1 kernel identity | `TestDeltaNetSeqKernels_matchDecodeBitwise`, both forms against the decode kernels at Qwen3.5 geometry (`hk` = `hv` = 128, `nv` = 48), qwen35-tiny's (16, 16, 4) and olmo_hybrid-tiny's (8, 16, 4), M = 1, 7, 33: every stage, the window and the state equal bit for bit | **PASS** (6 cells) |
| D3-G2 model identity | `TestKSD3_deltaNetModelsBitIdentical`, Metal resident at int4, a 40-token batched prefill and 8 decode steps, new kernels against the old in one process: qwen35-tiny, qwen3_5-tiny, qwen3_5_moe-tiny 0 of 2,304 logits differ each. olmo_hybrid-tiny is not compared: Metal declines its batched prefill ("arch/geometry not supported by f16 MMA prefill kernel"), so no scan kernel runs for it; D3-G1 covers its geometry | **PASS** (3 fixtures) |
| D3-G3 speed | `TestKSD3_scanKernelTiming`, GPU time of each kernel alone at M = 512, Qwen3.5 geometry, A B B A over 10 pairs, run at night on the owner's word, 2026-10-22 (22:48, timing lock held, 1-minute load 1.73 at the start, 6.1 GB available): `delta_rule_seq` 27.0–27.7 ms → `delta_rule_seq_tg` 19.1–20.0 ms, **median new/old 0.7245, 10 of 10 pairs below 1** (1.38×); `delta_conv_seq` 0.99 → 0.53 ms, median 0.5345, 10 of 10 (read only). Log: `~/goinfer-logs/ksd3-g3-2026-10-22.log` | **SHIP** (bar: median ≤ 0.98 and 9 of 10) |

**Plant** (the new rule kernel's `kvdot` summed in reverse): D3-G1 red in all three new-kernel cells, as registered. D3-G2 stayed
green under it: on these fixtures the reversed sum gives the same bits, while D3-G1's random inputs do not. A debug count confirmed
the new kernel was running there (threadgroup 32, 9 mixer passes). A second, stronger plant, the decay gate dropped from the rule's
first loop, turned D3-G2 red (2,304 of 2,304 logits on each fixture), so the gate sees a kernel change.

**Against the survey's estimate.** The survey put the rule at ÷3 to 10, and the whole Qwen3.5/3.6 prefill at 1.1 to 1.4×. The rule
measured 1.38× (÷1.38), well under the low end of that range. The whole-prefill effect is not measured here, and the survey's
range for it rested on the larger figure.
