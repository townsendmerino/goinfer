# F16-speed, nobara, 2026-11-09: the run and its VERDICT

Registered beforehand: `docs/measurements/f16-speed-2026-11/registration.md` (bands and outcomes unchanged). Graded with `docs/measurements/cuda-f16-kv-2026-10/grade-speed.py`, output in `grade.txt`.

- Machine nobara (RTX 2070 SUPER 8 GB, driver 595.112.07, 16 cores), CUDA. One serve binary built from main `f5f06f4b2` (aikit v1.76.0), arms `-kv f16` (goinfer) and `-kv f32` (goinfer_old). Greedy, five runs a cell, both arm orders, exclusive timing lock per pass (the night runner). Night job 23:12 to 00:01 PST, rc 0, `night-job.log`.
- Checkpoints, all from the local `~/models` set: Qwen2.5-Coder 0.5B and 1.5B Q4_K_M GGUF, Phi-3-mini-4k Q4 GGUF, Qwen2.5-7B-Instruct Q4_K_M GGUF (7B at 8000 keys, context 8448).
- 56 cells, 0 errored, none void; every cell started at a 1-minute load of at most 1.00 (median 0.83), recorded per cell in the json files. No thermal reading. Serve logs in `serve-logs.tar.gz`.
- The registration said the 128-key cells would not be re-run; the harness ran them anyway (0.5B, 1.5B, 7B and phi3-mini at 128 keys, all 1.001 to 1.011). They are graded with the rest.

## Result

**VERDICT: PARKED (worst cell 0.972 in 0.95-0.98).** The worst cell is the 7B at 8000 keys: f16 68.1 against f32 70.1 tok/s, 0.972 pooled, 0.972 and 0.971 for the two orders alone. Every other cell is at or above 1.0: 0.5B 1.010 to 1.019, 1.5B 1.011 to 1.035, phi3-mini 1.007 to 1.115 (the f16 arm is faster by 6.7% at 2048 keys and 11.5% at 3900), 7B at 128 keys 1.001.

The carried-over prediction (7B at 8000 reads at least 1.10, no cell under 0.98) missed on both counts: the 7B at 8000 read 0.972, below 0.98.

Registered outcome for PARKED: `auto` stays f16, the figures are published as a cost, and Francis decides whether any family's `auto` changes. The run is not repeated for a better read.
