# Split-KV for Phi-3 decode on CUDA at depth: record

Registered in `README.md` beside this file (commit b320df2a9, 13:48 PST; Amendment 1, commit b73097b7e, 13:59 PST, both before any f16
data). Review perf finding 1 (the 2026-11-19 perf review). Graded by `grade.py`; the grade is `raw/run/grade.txt`.

## Verdict: KILL by the registered bar (the 128 leg); the 3900 leg alone meets WIN

| depth | valid pairs | B faster | median B/A | range | A tok/s | B tok/s |
|--:|--:|--:|--:|--:|--:|--:|
| 128 | 10 | 0 / 10 | **0.9660** | 0.9651 – 0.9690 | 125.9 – 126.4 | 121.8 – 122.1 |
| 3900 | 10 | 10 / 10 | **1.0515** | 1.0485 – 1.0525 | 62.0 – 62.5 | 65.2 – 65.8 |

The bar needed the 128 median at 0.98 or more, and it is 0.966, so forcing split-KV at every depth is a KILL as registered. At 3900 the
WIN half of the bar is met (10 of 10 pairs, median 1.0515 against 1.05), with the pair spread (0.4 %) far below the effect.

What the bar does not separate: arm B forces split-KV at every depth (`GOINFER_SPLITKV_MIN_KEYS=0`). The option the finding names, re-keying
the never class on bytes per key, would put Phi-3 under a depth threshold (the conservative default is 3072 keys), so the 128 cost
would not reach a re-keyed build at 128. No depth between 128 and 3900 was measured, so this run does not place that threshold.

## Fidelity: bit-identical

The scratch test ran arm A, then arm B, in one process with the delivered KV checked as f16. Arm counters: A made 0 split-KV launches, B made
126,848 at 3900 and 6,144 at 128, and neither arm made a flash-decode launch.

| window | steps | bit-identical | argmax agree (teacher-forced) | max \|dlogit\| | min cosine | free greedy prefix |
|---|--:|--:|--:|--:|--:|--:|
| prose 3700–3899 + 64 free | 264 | 264 / 264 | 200 / 200 | 0 | 1.000000000 | 64 / 64 |
| prose 64–127 + 64 free | 128 | 128 / 128 | 64 / 64 | 0 | 1.000000000 | 64 / 64 |

The same holds at f32 (the void first run, `raw/void-f32/`): 264 / 264 and 128 / 128. A threshold change would alter no output, so no
fidelity gate is needed for it.

## Context the finding asked about

- The finding's premise holds for serve: `auto` gives Phi-3 f16 KV on CUDA. It does not hold for the `backend/cuda` test binary, whose
  suite pins `auto` to f32 (Amendment 1). Any CUDA bench of Phi-3 through that binary times f32 unless it restores serve's `auto`.
- The decode rate has moved since the 0.72 peer row (2026-09-29, served, 53.4 tok/s at 3900). This instrument reads arm A at 62.0–62.5 tok/s
  at 3900. That is in-process and cross-session, so it is not a peer ratio and is not one here.
- From A's own figures, the token is 7.93 ms at 128 and 16.1 ms at 3900, so roughly half the 3900 token is depth-dependent cost. Split-KV
  recovers 0.79 ms of that 8.1 ms. Step 3's attention share (ncu; nsys is not installed, ncu is at
  `/opt/nvidia/nsight-compute/2026.1.1/ncu`) was not run, because the brief makes it conditional on a WIN.

## Provenance

- **Machine:** nobara-pc, RTX 2070 SUPER 8 GB (40 SMs), NVIDIA driver 595.112.07, kernel 7.3.0-202.nobara.fc44, 16 cores, 62 GiB.
- **Build:** go1.27.2, aikit v1.76.0. One test binary (`go test -c -tags 'cuda goinfer_testhooks'` in `backend/cuda`) built at b73097b7e
  (= `origin/main` 31f97ca5a plus docs) with `scratch_test.go.txt` added as `zz_splitkv_phi3_scratch_test.go`; sha256 prefix e9efcfa1fefd971d.
  `BenchmarkResidentDecode` is the tree's, unchanged.
- **Checkpoint:** `~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf`, loaded at `q4k` (per-32 activations) through its
  `q4k cuda v17` .giw beside it, on local NVMe. KV f16 delivered (`auto`, serve's policy restored).
- **Decode:** greedy (`ForwardArgmax`, the serve fast path), 128 timed tokens after a prefill of `GOINFER_BENCH_DEPTH`, one process per cell.
- **Date and order:** 2026-11-19. Depth 128 ran 14:07–14:10 PST, depth 3900 blocks 1–3 ran 14:10–14:29, and blocks 4 (re-run), 5 and
  the 128 re-run ran 15:24–15:38 PST after the outage. Every block was A B B A under an exclusive hold of the timing lock.
- **Box at cell starts:** 1-minute load 1.17–6.08 (all under 8.0; the higher readings are the bench's own load and prefill at ~570 % CPU),
  MemAvailable at least 59.3 GiB (60,770 MiB), GPU 59–68 °C. The desktop was logged in, and kwin_wayland held its CUDA context (17–18 MiB) in every
  cell. No other CUDA process was present.

## Void cells (listed, re-run, not averaged)

- **d128 block 1, c2 and c3 (B):** a foreign `go` (and at c2 a `compile`) process was running at the cell start, under the exclusive lock.
  The whole block was re-run as block 101 (15:37 PST).
- **d3900 block 4:** power was lost at about 14:32 PST (the last log write). c1 (A, 62.07) completed, but its partner c2 never reached
  the TSV, and neither c2's nor c3's log holds a reading. The whole block was re-run as block 104 (15:24 PST). The orphan c1 is in
  `raw/run/logs`, ungraded.
- **The f32 run (13:55–13:58 PST):** void by instrument (Amendment 1): the identity run and six depth-128 cells, kept in `raw/void-f32/`.
  For reference only, its 128 pairs read B/A 0.96–0.97, the same sign as at f16.

Correction: Amendment 1's heading first gave 14:01 PST, a guessed time. Its commit is stamped 13:59:02 PST, the true time, and the heading now says so; it was still before any
f16 data (the f16 identity run took the lock at 14:06).

## Files

`README.md` (registration and Amendment 1), `run_ab.py`, `run-sweep.sh`, `grade.py`, `scratch_test.go.txt`; `raw/run/` holds
`cells.tsv` (one row per cell, with its start state), `grade.txt`, the per-cell logs, the identity logs, and the pipeline and recovery
logs and scripts; `raw/void-f32/` holds the void f32 run.
