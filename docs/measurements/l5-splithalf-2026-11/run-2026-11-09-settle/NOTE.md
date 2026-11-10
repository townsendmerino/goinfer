# L5 A/B, the re-run with the settle fix (README addendum), nobara, 2026-11-09

Registered rule and bars unchanged: `README.md` beside this directory. This is the run that replaces the void first run.

- Machine nobara (16 cores, 64 GB), CPU only, `BenchmarkDecode`, Qwen2.5-Coder-1.5B-Instruct Q4_K_M from `~/models` (local NVMe), `GOINFER_BENCH_QUANT=int4`, batch 1, greedy. Date 2026-11-09, 22:34 to 22:40 PST, exclusive timing lock.
- Build: main `f5f06f4b2` (includes #868's settle fix), clean tree, one test binary for both arms. Order ON OFF OFF ON, ten pairs.
- Fires check: `TestW4A8SplitHalfFires_onBenchModel` passed, 197 tensors repacked, 0 skipped, 624.8 MiB second copy.
- Every visit waited for the 1-minute load under 4.0 (start loads 3.73 to 3.99, recorded per visit in raw.jsonl with the waits); free memory 64.4 GB throughout, far over the fit need plus 2 GB. 0 void visits. No thermal reading was taken (CPU decode); the load and wait per visit are the record.

## Result (grade.txt)

Median gain -6.48%, 0 of 10 pairs positive (pairs -5.76% to -7.32%): ON reads about 21.8 tok/s and OFF about 23.3. Peak VmHWM, recorded and not graded: ON 2884 MiB, OFF 2259 MiB.

**VERDICT: KILL L5**, by the registered rule (median gain under +2.0%). The result is stable across pairs and agrees in sign and size with the first, void run's ungraded reading (ON about 21.8, OFF about 23.3).
