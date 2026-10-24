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

## The build

`cuda/topk.cu` becomes three launches, in stream order, from `cuda/topk.go:cudaResident.topkReadback`:
- **`topk_compact`** (fast path only, 32 blocks of 1,024): each block finds the same threshold by an exact radix select of the
  R-th largest sample (`tk_kth`, the slow path's own passes) instead of a bitonic sort. It then appends every entry ≥ t0 of
  its slice of the row to a global candidate buffer.
- **`topk_finish`** (one block): selects the K-th largest candidate key and sorts only the K survivors. When more candidates
  tie at the boundary than are needed, it sorts the whole set as before; a count outside [K, CAP] takes the unchanged slow
  path. It writes ids, logit bits and the row max, and resets the candidate count.
- **`topk_z`** (32 blocks of 32): the same 1,024 virtual threads each sum the same entries in the same order in f32. The last
  block runs the same double tree over the partials.

Output is the first K of (logit desc, id asc), a total order, so it should equal the one-block kernel's word for word, Z
included. The one-block `topk_select` is removed; the old arm below is origin/main's committed PTX.

**Memory:** +36.9 KB of device scratch a resident: 4,096 candidate values and ids, Z's 1,024 partials, three counters. This
is an increase, so CLAUDE.md's small-gains rule (which requires none) does not cover the item, and the speed bar below is
stricter than its 9-of-10 alone.

## Pre-registration: the build gate

Registered 2026-10-23 before the build is measured. Instrument: `instruments/ks_a3_build_test.go.txt`, copied into `cuda/` for
the run. The old arm is origin/main's committed `topk.ptx` (`topk_select`), loaded beside the build on one resident: the
Qwen2.5-Coder-0.5B Q4_K_M loaded as serve loads it. Held under the timing lock.
- **G1, bit-identity (tier 0, on device):** every output word, old against new: ids, logit bits and (when asked) Z's two
  float words.
  - Vocabularies 151,936 (the model's), 32,000, 8,192, 5,000 and 1,000.
  - Row kinds: normal, coarse ties, all equal, signed zeros, ascending, descending, peaked.
  - K 1, 40, 256 and 1,024; temperature 1 and 0.7; Z on and off.
  - Plus 16 real rows of the 0.5B at K 40, 256 and 1,024.
  - One differing word stops the item.
  - Before merge also: `TestTopKSelect_matchesReference` and the CUDA device suite.
- **G2, speed (tier 1, in process):** 10 ABBA quads of 64 sampled tokens at depth 128. Each token is `ForwardTopK`'s work:
  the forward, the top-K at k 256 with Z, temperature 0.7, and the readback. Readout: the per-quad ratio (A1 + A2) / (B1 + B2)
  of old over new, its median, and quads above 1.
  - **ship:** at least 9 of 10 above 1 **and** a median of at least 1.02;
  - **parked, to the owner:** 6–8 of 10, or a median in [1.00, 1.02);
  - **killed:** 5 or fewer, or a median below 1.00.
- **Recorded, no bar:** the top-K alone (launch, sync and readback) at k 256 and 1,024, old and new interleaved.
- **Load rule:** a quad started with the 1-minute load above 8.0 is void and re-run in order.
- **Cost:** one load and about 2,500 G1 cases of under a millisecond, then 42 blocks of 64 tokens at about 2 ms: about 2
  minutes.

## Readings

Phase read, 2026-10-23 at about 14:29 PDT, origin/main + this record, under the timing lock (it waited for another session's
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

## Readings: the build gate

At 9b62b2cca (old arm: origin/main's committed `topk.ptx`), 2026-10-23 14:56 PDT, under the timing lock, the card clear (402 MiB
used) and the 1-minute load 2.67. Log: `~/goinfer-logs/ks-a3-2026-10/gate.log`.

| id | reading | verdict |
|---|---|---|
| G1 | **608 cases, 0 differ** in any output word (ids, logit bits, Z hi/lo) | **PASS** |
| G2 | sampled token 2,007.5 → 1,955.4 µs; new speedup **median 1.0263×** (1.0250–1.0284), **10 of 10** quads above 1, 0 void | **SHIP** (both conditions: 9 of 10, median ≥ 1.02) |

- **G1's count.** The pre-registration estimated "about 2,500" cases. Its own grid comes to 560 synthetic cases (5
  vocabularies × 7 row kinds × 4 K × 2 temperatures × Z on and off) plus 48 real (16 rows × 3 K), so 608. The estimate was
  wrong; the grid ran as registered.
- **The top-K alone** (no bar; launch, sync and the 2 KB readback, 50 a reading, old/new interleaved):
  - 151,936 × k 256: old 119.3 / 115.8 µs, new 101.9 / 95.4 µs;
  - k 1,024: old 200.0 / 206.3 µs, new 133.2 / 131.6 µs.

  The sampled token saves 52 µs at k 256, more than the 19 µs the isolated read moves. Both arms of the isolated read pay the
  same sync and readback, so that does not explain the gap; it is unexplained. Not chased: the token is the gate's instrument,
  and it decided.
- **Memory:** +36.9 KB of device scratch a resident, as registered.

**Device tests** at d123b4352, under the timing lock:
- **The CUDA device suite** (`scripts/cuda_device_suite.py`, the whole `cuda` module): **GREEN, 772 passed, 0 failed, 0 device skips,
  236 other skips**. Log: `~/goinfer-logs/ks-a3-2026-10/device-suite/`.
- **The heavy top-K gates.** Inside that run `TestTopKSelect_matchesReference`, `TestSampledTopKStreamIdentity` and
  `TestSampledGumbelStreamIdentity` skip without `GOINFER_HEAVY_TESTS=1`, so they ran on their own with it. **All three pass**:
  272 (vocab × shape × K) rows in the reference order with Z within 1e-6, and both sampled-stream identity tests. Log:
  `~/goinfer-logs/ks-a3-2026-10/heavy-topk.log`.
