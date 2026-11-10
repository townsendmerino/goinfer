# G-RF3-S under addendum 1, nobara, 2026-11-10: the run and its VERDICT

Registered: the G-RF3-S entry and its addendum 1 in docs/tasks/task-forward-and-memory-2026-11.md (bars and outcomes unchanged by the addendum, which makes the sample larger). The run script and scorer are on main from PR #888 (the copies run here equal main's).

- Machine nobara: RTX 2070 SUPER 8 GB, driver 595.112.07, 16 cores. CUDA, by day, every cell under the exclusive timing lock with the idle gate at a 1-minute load under 0.9. 72 cells, 11:03 to 11:54 PST, none void; the highest start load was 0.89 (the void line is 8.0). No thermal reading.
- Models, from the local ~/models set: Qwen2.5-Coder-1.5B-Instruct Q4_K_M and Qwen2.5-7B-Instruct Q4_K_M. Cells: stall (3 decoders, 6 newcomers a run, 2400-word prompts) and solo (no decoders) on each.
- Arm A: serve-cuda-A built from df6393703 (the merge of #879, RF3-1's merge base), sha256 eaaa3b4dc2eb. Arm B: built from 2235b2036 (the merge of RF3-1), sha256 78df8496db8a. A to B differs by exactly decoder/mc3_batch.go and decoder/model.go. Every cell logs its binary's hash (cells.log): 40 cells on A, 32 on B.
- Not used: a first serve-cuda-B was built from 2cb919116, which carries all of main since df6393703 (P4a/P4b, the rope re-read), so it is not A plus RF3-1. No scored cell used it.
- The A/A (pooled over two ABBA blocks, all on A) read 1.000, inside 0.03. The first A/A of the day, with 3 newcomers a run, voided (pair ratios 1.020 and 1.070) and is kept in run-2026-11-10-void, not scored; addendum 1 raised the sample.
- 7-stall.json is gzipped (over 1 MB); the scorer reads the plain .json, so gunzip it into the directory to re-score.

## Result (score.txt)

- 1.5B stall: newcomer TTFT A/B 1.006, AMBIGUOUS (the band is 0.98 to 1.03). Decode-gap max A/B 0.994, PASS (guard: 0.95 or better). Decoder tokens inside the newcomer windows, B/A 1.0.
- 1.5B solo: TTFT A/B 1.004, PASS (0.95 or better).
- 7B stall: newcomer TTFT A/B 1.002, AMBIGUOUS. Decode-gap max A/B 1.000, PASS. Decoder tokens in windows B/A 1.007.
- 7B solo: TTFT A/B 0.999, PASS.
- The registered prediction for the 1.5B stall TTFT was about 1.05; it read 1.006. The stall guard prediction (0.97 to 1.00) and the solo guard (1.00) held.

**VERDICT** as the scorer prints it: NOT a clean pass. By the registered outcomes: TTFT ambiguous on both models, so the result is parked: the alternation stays the default, and the reading goes to Francis. No guard failed.

RF3-2 (G-RF3-I, the in-process identity test) passed separately at 2ca514def: ride passes 2, ride rows 4, no identity mismatch.
