# x86 CPU peer cells at v0.25.0 on nobara-pc (registered 2026-11-22)

**Status: REGISTERED 2026-11-22 13:58 PST, before any timed run.** By day, under nobara-pc's exclusive timing lock.
Results go under *Results*. A bar that turns out wrong gets a dated amendment with its mechanism, not an edit.

**Why.** The three x86 CPU cells below were last read at `e29da447` ([`peer-reanchor-2026-11-19`](../peer-reanchor-2026-11-19/README.md)
cells e and v), before v0.25.0's CPU changes (#1121, #1129 and the aikit bump #1170) merged. Owner's go: Francis,
2026-11-22 ("continue goinfer ... and performance work"), relayed by the coordinator.

## Inherited unchanged

Everything in [`peer-reanchor-2026-11-19/README.md`](../peer-reanchor-2026-11-19/README.md) not named under *What differs*:
- the ratio r = goinfer ÷ peer for decode; the bands AHEAD (every pair > 1.03), LEVEL (every pair in 0.97–1.03), BEHIND
  (every pair < 0.97), AMBIGUOUS-HIGH or -LOW otherwise, every pair deciding, never the mean; the 5% spread cap;
- 3 pairs per cell (8 completions × 64 tokens a run), depth 128, greedy, ctx 4096;
- the token gate, the decode-path gate, servers restarted between engines;
- the void rule of 2026-11-08 (`noise-registry.md` §8: a cell starts only with load ≤ 8.0 and `MemAvailable` ≥ its fit need,
  and is VOID if system swap grows at all while it runs), `BENCH_SWAP_VOID_STOP=3`, the harness's idle gate
  (`BENCH_MAX_LOADAVG=1.0`), the script's settle under 0.8 and its page-cache eviction before each step;
- Ollama v0.32.5 at `~/ollama-0325` (CPU-forced, `num_gpu: 0`) with the same tags; llama.cpp `a37cc46` beside on the 0.5B;
- the grader, [`../peer-reanchor-2026-11-19/grade.py`](../peer-reanchor-2026-11-19/grade.py), run on this campaign's files.

## Cells

| cell | what | engines | last graded (e29da447) |
|---|---|---|---|
| e | x86 CPU greedy, 0.5B @ 128, `-quant int4` | goinfer, Ollama, llama.cpp | BEHIND 0.867 (49.9 vs 57.5 tok/s) |
| v1 | x86 CPU greedy, phi3-mini at its default (`q4k`) @ 128 | goinfer, Ollama | BEHIND 0.763 (8.2 vs 10.7) |
| v2 | x86 CPU greedy, gpt-oss 20B at its default (`int4`) @ 128 | goinfer, Ollama | BEHIND 0.923 (9.9 vs 10.7) |

The 1.5B and 7B are not in scope: their CPU rows were re-measured at the int8-head default on 2026-11-20 (`48d15271c`).

## What differs from 2026-11-19

- **goinfer is the v0.25.0 release asset** `goinfer-serve-linux-amd64` (`v0.25.0 (d0da204c4b7a)`, backends cpu and cuda,
  aikit v1.78.0), sha256 `e93cdf59…` as the release's `checksums.txt` gives it, run with `-backend cpu`. 11-19 used a local
  build of `e29da447`.
- **The harness is `bench_peer.py` at the tag**, from a clean detached worktree at `d0da204c`. Its only change since
  `e29da447` is the fit-banner regex accepting `GiB` as well as `GB`.
- **Cell e holds the 0.5B only.** [`run.sh`](run.sh) is the 11-19 script's cpu part with these pins; the script checks the
  binary's hash and version, the worktree's HEAD, the tag and its clean state.

## Fit checks (2026-11-22 13:42 PST, `FIT_ONLY=1`)

All three load: 0.5B `cpu (int4)`, phi3-mini `cpu (q4k)`, gpt-oss 20B `cpu (int4)`. System swap already held 519 MiB from
earlier work; the void rule reads growth during a cell, not the level.

## Decision rule

- Each graded cell's outcome is reported against its 11-19 label. A cell whose label moves replaces its row in
  `docs/benchmarks.md` (the TL;DR's x86 CPU decode row and the summary sentence that quotes it), citing this record. A cell
  whose label does not move leaves the page's label as it is.
- A VOID cell stays VOID, is reported, and its old row stands.
- No move is attributed to #1121, #1129 or #1170: this is a cross-build, cross-session comparison (TE5(a)); each lever has
  its own record.
- No early stop, and no re-run of a valid cell.

**Tier:** day. Testing tier 2: the numbers are published, and no smaller tier measures a peer ratio.

## Cost

From 11-19's cells (0.5B arms about 40 s each plus about 100 s of idle gate; phi3-mini about 5 min an arm; v2's two arms
791 s): fit checks 1 min, e about 8 min, v1 about 12 min, v2 about 14 min, evictions and settles about 5 min.
**Expected: start 14:00 PST, finish about 14:45 PST.** The run holds the lock under `timeout 90m`.

## Results

(pending)
