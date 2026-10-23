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

Phase read, 2026-10-23 at about 14:35 PDT, origin/main + this record, under the timing lock (it waited for another session's
26B test). Logs: `~/goinfer-logs/ks-a3-2026-10/phases.log`. 20 launches a model after one warm-up. The fast path ran in every
launch, and the stamped kernel's ids equal the shipped kernel's on the same row (the check holds; the read stands).

| phase | 0.5B, k 256 | gemma3-1b, k 1,024 |
|---|---:|---:|
| load 2,048 samples | 1.9 µs | 1.9 µs |
| bitonic-sort the samples | 32.1 µs | 32.1 µs |
| pick the threshold | 0.0 µs | 0.0 µs |
| compact (candidates found) | 15.4 µs (1,076) | 26.4 µs (1,802) |
| bitonic-sort the candidates and write k | 58.0 µs | 57.6 µs |
| exact slow path | not taken | not taken |
| Z | 28.9 µs | 46.3 µs |
| **total** | **136.1 µs** | **164.1 µs** |

The totals sit above the survey's ncu medians (101.7 and 211.5 µs, with two levels at each shape and Z included). These
are wall-clock stamps inside one launch on a live resident; they rank the phases and are not a substitute for the ncu time.

**What it says for the build.** The two single-block bitonic sorts are 90 µs of the 0.5B's 136: 66 barrier stages each, over
2,048 elements. The survey's proposal, multi-block compaction and Z, reaches the other 44 µs. So the build cuts the sorts first,
keeping their output exact:
- the threshold needs only the R-th largest of 2,048 samples (R ≤ 64), an exact selection, not a full sort; any exact
  selection returns the same value;
- the candidate sort needs only the first k of the order (logit desc, id asc), which is a total order. Selecting the k
  first and sorting k gives the same k in the same order.

Then compaction and Z spread over many blocks, Z keeping the same 1,024 serial partial sums and the same double tree.
The build's gate is registered here before it is measured.
