# KS-A3: where `topk_select`'s time goes, then a multi-block build, 2026-10-23

`docs/tasks/task-kernel-survey-2026-10.md` KS-A3. The survey's ncu read
(`docs/measurements/kernel-survey-2026-10/README.md` § KS-A3) found one block on one SM:
- 101.7 µs a launch at (vocab 151,936, k 256);
- 211.5 µs at (262,144, 1,024).

The read does not say which phase holds that time. The kernel's phases are: load 2,048 strided samples, bitonic-sort them,
pick a threshold, compact every entry above it, bitonic-sort the candidates and write the first k (or the exact radix slow
path instead), then the Z sum.

## Pre-registration: the phase read (tier 1, no bar)

Registered 2026-10-23 before it runs. It informs the build's design and decides nothing on its own.
- **Instrument:** `instruments/topk_phases.cu.txt` is `cuda/topk.cu` with thread 0 stamping `%globaltimer` at each phase
  boundary. `instruments/ks_a3_phases_test.go.txt` launches it over a real logits row: after 8 greedy tokens, k 256 on the
  Qwen2.5-Coder-0.5B Q4_K_M and k 1,024 on gemma3-1b Q4_K_M, both loaded as serve loads them, with Z on and temperature 1.
- **Readout:** 20 launches after one warm-up; the median µs a phase and in total; how often the fast path ran; the candidate
  count.
- **Check:** the stamped kernel's ids must equal the shipped kernel's on the same row. A mismatch voids the read.
- **Run:** `bash docs/measurements/ks-a3-2026-10/instruments/run-phases.sh`, under the timing lock. **Cost:** two loads and
  42 launches, under a minute.

The build's own gate (output byte-identical to the shipped kernel on the existing `topk_test.go` cases and a sweep of real
and adversarial rows, plus the speed bar) is registered here after this read and before the build is measured.

## Readings

(not yet run: the GPU is reserved for the owner's work until the coordinator releases it)
