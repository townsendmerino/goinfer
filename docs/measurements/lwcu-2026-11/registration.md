# LW-cu registration: main against a build before the layer walk, decode and TTFT on nobara

Registered 2026-11-12 16:41 PST, before any LW-cu speed measurement. The row is LW-cu in `docs/tasks/task-forward-and-memory-2026-11.md`. This file is
docs only; the run is a separate step, by day, after this merges, and nothing here has been measured.

## Question

CUDA's decode segments and `backend/cuda/prefill.go:cudaResident.prefillCore` read a decoded `internal/runtime/program/walk.go:LayerWalk` in place
of the family flags since 90e4d44f8 (no PR, no switch back). `captureGraphs` chooses its layers differently, and the PLE and layer-scale launches left
the unfused branch. Did main lose decode speed or time to first token against the build from before the subsystem contracts and the walk?

Tier 1, by day. Tiers 0 (the layer-walk test, G-RF2-I's batch invariance, KV4's G1a, G3b, the window replay) and the identity read
(`go run ./cmd/gate identity -backend cuda`, 30 families IDENTICAL, none DIFFERENT) already ran at main on 2026-11-12, and are recorded in the
nobara-v1 read's PR (the record is `docs/measurements/cuda-read-2026-11-12`); none of them can answer a speed question.

## Arms

- **A, `goinfer_old`:** `serve-cuda-old`, built from `51ed2ae27`, the parent of the subsystem contracts (`0df3294dc`). sha256 `465227c366a1ce2f47b52a9e34193e92da4b59bd64493f2ebd147c156d0f06c3`.
- **B, `goinfer`:** `serve-cuda-new`, built from `4edbe93e9`. Main's code at `43e2d5e06` is the same (the two differ in `docs/tasks/task-forward-and-memory-2026-11.md` only). sha256 `1d671c1e483267d402e168e8f383d71f0f60d4e74c05555dcc99efb1c1136dfa`.
- Both: `cd backend/cuda && GOWORK=<tree>/go.work CGO_ENABLED=0 go build -trimpath -tags cuda -o <bins>/serve-cuda-{old,new} ./cmd/serve`, one toolchain
  (go1.27.2), one box, built under the shared timing lock. Pinned copies in `~/goinfer-bench/lwcu-2026-11-12/bins/`; the run reads them, not a tree.
- **What differs.** A to B is not the walk alone: it is every commit from `0df3294dc` to `4edbe93e9` that reaches the CUDA serve binary (the subsystem
  contracts, `BackendModelView`, `decoder/quant`, the layer walk, `SequenceState`, the CI fix, the wgpu bump). A band that passes clears all of them. A band
  that fails does not name the walk: `0df3294dc` is the bisect point, and bisecting is Francis's call, not part of this run.
- Settings: serve's defaults, both arms (`-kv-sessions 4`, f16 KV where the family takes it, windowed KV on), the harness's `-require-backend`.
- Do-nothing arm: A. A revert of the walk is B returning to A's numbers, so a failed band goes to Francis with the revert named.

## Cells (RF2's, which are KV4's G2)

Models from `~/models` on NVMe: `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`, `qwen2.5-7b-instruct-q4_k_m.gguf`, `gemma3-1b-q4_k_m.gguf`. Greedy (temperature 0, seed 1),
essay-v2 prompts, the harness's defaults otherwise.

- **Decode at depth:** `scripts/bench_peer.py`, `BENCH_ENGINES=goinfer,goinfer_old BENCH_BACKENDS=cuda BENCH_ABBA=1 BENCH_REQUIRE_BACKEND=1`. Qwen 1.5B and 7B at depth 2,048 and 3,900 (context 4,096) and at 8,000 (`BENCH_CTX=12288`); gemma3-1b at 2,048 and 3,900.
  The harness also runs each invocation's depth-128 cell, so depth 128 is read at both contexts for the Qwens and once for gemma3-1b. Each cell group is A B B A in one session; a block's ratio is B's mean
  decode tok/s over A's, and the cell reads the median over its blocks (`score_speed.py`).
- **Time to first token:** `docs/measurements/kv4-cuda-2026-10-21/ttft_ab.py` (`TTFT_MODEL=1.5B`, then `7B`), K = 512 and 3,900, three alternating pairs, a fresh server per run, a warm-up first.
  Ratio A/B per pair, median over pairs.
- **Never-swap:** `docs/measurements/kv4-cuda-2026-10-21/procmem_sampler.py` over both servers for the whole run.
- **Not read:** MC3's aggregate and verify (RF2's other two cells). They go onto the same two arms if any cell below is not a PASS, or if Francis asks.

## Bands (RF2's, unchanged)

Each is the median paired ratio, B against A for decode (higher is better), A against B for TTFT.

- **Decode: PASS at 0.98 or better at every depth on every model; FAIL below 0.98.** There is no parked band for decode (RF2 addendum 1), so the 0.97 to 0.98 that
  `score_speed.py` labels AMBIGUOUS is read as FAIL here. No tuning round: a fail is reported, the walk code is not touched in this row's read.
- **TTFT: PASS at 0.97 or better at both depths on both models; 0.96 to 0.97 AMBIGUOUS (parked, not shipped on this evidence); below 0.96 FAIL.**
- **Never-swap:** PASS when every B peak `VmSwap` is no more than its A peak's and every `VmHWM` is within 64 MB of its A's (G4).

## Void rule and records

- A cell whose start 1-minute load is above 8.0, or whose free memory is below the fit need plus 2 GB, is VOID: it is re-run in the same order, listed in the record, and never averaged in.
  Every cell records its start load average and free memory. The idle gate is the harness's (`BENCH_MAX_LOADAVG` 1.0); the script also waits for load under 0.9 before each invocation.
- A cell where an arm ran off the GPU is VOID. A cell where swap grew during it is VOID and re-run (the 2026-11-09 read had one).
- Driver 595.112.07, both arms in one session. A different driver on the day voids the comparison to RF2's records, not this one's A-B pair.
- Exclusive timing lock for every GPU step (`scripts/timing_lock.py`); no other session runs beside it.

## Outcomes, decided now

| reading | what happens |
|---|---|
| every decode cell PASS, TTFT PASS, never-swap PASS | LW-cu's speed half closes; the row says main holds the 51ed2ae27 numbers |
| any decode cell FAIL | STATE: decision for Francis with the cell table; the revert (A) is the do-nothing arm; no bisect without the coordinator's word |
| decode PASS, a TTFT cell AMBIGUOUS | parked, reported to Francis; MC3 and verify go on the same arms |
| a TTFT cell FAIL | as a decode fail |
| a void cell | re-run in order; two voids in one cell group are reported |

## Cost and schedule

Cost basis: KV4's G2, 3 models x up to 4 depths x 2 arms x 2 passes = 44 decode cells at about 45 s (load, prefill, 3 timed runs), about 35 minutes; TTFT about 6 minutes; idle-gate
waits on top. **About 45 minutes**, 60 at the outside. Planned start 2026-11-13 by day, once the box is idle and this registration has merged; expected finish 45 to 60 minutes after the start.
Run it with `bash docs/measurements/lwcu-2026-11/run-lwcu.sh <out dir>`; it holds the exclusive lock per invocation, and writes the records, which are then scored with
`docs/measurements/kv4-cuda-2026-10-21/score_speed.py`.
