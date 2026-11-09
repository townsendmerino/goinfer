# G-RF2-S, final build: read 2026-11-09 (nobara)

The RF2 speed gate and never-swap (`docs/tasks/task-forward-and-memory-2026-11.md`, "RF2: registration" and "RF2: addendum 1"),
read once on RF2's final build, after RF2-3b-2b (#825) and RF2-3b-3 (#826). The first-working-build read is
`docs/measurements/rf2-2026-11/g-rf2-s-2026-11-09/README.md`.

## Provenance

- **Machine:** nobara, RTX 2070 SUPER, NVIDIA driver 595.112.07, run by nobara-sb01.
- **When:** 2026-11-09, 13:33–14:35 PST, by day. Every cell ran under the exclusive timing lock, behind the load gate (1.0;
  cell start loads 0.76–0.89). No cell was void.
- **Arms**, A B B A per cell group in one session:
  - **A, old:** `g-rf2-s-old-final-2026-11-09` at 983282891, which is a2c276e3d with RF2's merges reverted
    (`docs/measurements/rf2-2026-11/make-old-arm.sh`; #825 came in through #826's merge, whose revert removes it).
    `serve-cuda` sha256 2889452c930c96ae.
  - **B, new:** a2c276e3d, `main`. sha256 29441d5a7cf29a87.
  - Old to new differs by exactly RF2's code. The MC3 cells log each server's binary and hash (`mc3-cells.log`).
- **Checkpoints,** from `~/models` on nobara: Qwen2.5-Coder-1.5B-Instruct Q4_K_M, Qwen2.5-7B-Instruct Q4_K_M, gemma3-1b
  (`bench_peer.py`'s entry).
- **Settings:** `bench_peer.py` defaults, int4, greedy (temperature 0, seed 1).
- **Thermal:** the GPU read 47–53 °C at the start of the 64 decode records.
- **Swap:** growth 0 on every record, and every record's `error` is null.
- **Script:** `docs/measurements/rf2-2026-11/run-g-rf2-s.sh`, all phases (decode, deep, gemma, ttft, mc3, verify).
- **Not here:** host-memory samples (`procmem.jsonl`), so never-swap's VmHWM half cannot be read. The binaries' 20-character names
  (`serve-cuda-983282891`) never matched `procmem_sampler.py`'s exact name test against `/proc`'s 15-character `Name`; #871
  matches by the first 15 and makes `run-g-rf2-s.sh` require `serve-cuda-old` and `serve-cuda-new`, the names `score_speed.py` reads.

## Readings

Decode tok/s, B/A per block, then the median per cell (KV4's `score_speed.py`):

| model | depth | blocks | median | verdict |
|---|---|---|---|---|
| 1.5B | 128 | 0.725 0.729 (and 0.725 0.727 in the deep group) | 0.727 | FAIL |
| 1.5B | 2,048 | 0.746 0.745 | 0.745 | FAIL |
| 1.5B | 3,900 | 0.756 0.755 | 0.755 | FAIL |
| 1.5B | 8,000 | 0.779 0.779 | 0.779 | FAIL |
| 7B | 128 | 0.857 0.856 (0.858 0.858) | 0.856 | FAIL |
| 7B | 2,048 | 0.863 0.864 | 0.864 | FAIL |
| 7B | 3,900 | 0.870 0.869 | 0.869 | FAIL |
| 7B | 8,000 | 0.881 0.878 | 0.879 | FAIL |
| gemma3-1b | 128 | 0.765 0.758 | 0.762 | FAIL |
| gemma3-1b | 2,048 | 0.785 0.779 | 0.782 | FAIL |
| gemma3-1b | 3,900 | 0.778 0.782 | 0.780 | FAIL |

The old arm reads what it read in the first-build session (1.5B at 128: 277.7 then, 277.0 now). The new arm moved from 277.4 to
201.8 tok/s: about 1.3 ms more per token on the 1.5B and 1.9 ms on the 7B (86.1 → 73.8).

Time to first token, A/B over three pairs (`ttft_ab.py`):

| model | K = 512 | K = 3,900 |
|---|---|---|
| 1.5B | 1.013 (1.013, 0.999, 1.023) | 1.000 (0.987, 1.003, 1.000) |
| 7B | 0.999 (0.997, 1.001, 0.999) | 0.997 (0.995, 0.997, 0.998) |

MC3 aggregate at 4 clients on the 1.5B, B/A per pair (`score_mc3.py`):

| pair | old tok/s | new tok/s | B/A |
|---|---|---|---|
| 1 | 356.3 | 433.8 | 1.217 |
| 2 | 354.2 | 435.2 | 1.229 |
| 3 | 353.6 | 432.2 | 1.223 |

The median is **1.223**. `content_sha` is equal in every turn of each pair.

Verify, `bench_spec_lane.py`'s exact-attention arm with `-spec ngram` on each build, B/A per pair:

| cell | pairs | median | A/A floor | verdict |
|---|---|---|---|---|
| 1.5B:3900 | 0.974, 0.977, 0.983 | 0.977 | 0.999 | AMBIGUOUS |
| 7B:7500 | 0.995, 0.992, 0.992 | 0.992 | 1.001 | PASS |

## Verdict, against the registered rule

- **Decode: FAIL** at every registered depth on all three models (0.727–0.879). Decode under 0.98 at 2,048 keys or more opens the
  tuning budget (addendum 1): at most two rounds, each one kernel-side change and a re-run of only the failing cells, within one
  working day. Still under 0.98 after it is the kill: today's decode call stays beside the ragged one, the other shapes ship on
  their own bands, and the reading goes to Francis. The kill is taken without a tuning round (#869, on Francis's word): the cost
  is the solo shapes' routing, below, not a kernel that one change would close.
- **Verify: AMBIGUOUS** on the 1.5B (0.977, in 0.97–0.98); PASS on the 7B. An ambiguous shape is parked, not shipped on this
  evidence: verify stays on today's `ForwardN` (#869), and the reading goes to Francis. A speed cell is read once on the final
  build, and the tuning budget is decode's only, so there is no re-run.
- **TTFT: PASS** (bar: at least 0.97); the worst cell is 0.997 (7B, K = 3,900).
- **MC3: PASS** at 1.223, with identical replies.
- **Never-swap (KV4's G4): NOT READ.** Peak VmSwap's half holds by the records (swap growth 0 everywhere); the VmHWM half is
  VOID, with no host-memory samples. The post-kill confirmation read carries the sampler.

## Mechanism (strong, not proven)

RF2-3b-3 (#826) routes `Forward`, `ForwardArgmax`, `ForwardSample` and `ForwardN` through `cudaResident.soloRagged`, a
one-sequence `StepRagged`, in place of today's per-token launches. The evidence:
- the first-working-build read (e6a43d77, before #823, #825 and #826) has decode at 0.999–1.001;
- before #826 no solo shape touched the ragged step, so #823's and #825's ragged twins reached only MC3's step, which passes here;
- the cost is a near-constant time per token, the shape of per-step overhead (the ragged row tables and class lists, prefillCore's
  rows-mode pass) rather than of attention, which would grow with depth; CUDA graphs are opt-in, so they are not the difference;
- verify, whose k rows share one step, loses about 2% where a decode token loses 12–27%.

It becomes evidence with a tier-1 in-process timing on nobara: `Forward` on the 1.5B with `soloRaggedOffForTest` on against off
at a2c276e3d, where the switch exists.

## Scoring

From this directory, with KV4's scripts at `docs/measurements/kv4-cuda-2026-10-21/`: `score_speed.py <empty procmem> decode-qwen.json
decode-deep.json decode-gemma3.json`; the `SUMMARY` lines of `ttft-1.5B.log` and `ttft-7B.log`; `score_mc3.py w7.json`; the tail of
`verify.log`.
