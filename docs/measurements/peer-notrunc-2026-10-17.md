# No-truncation sampled cells, re-measured with every sampling field sent (2026-10-17)

**Why.** The published `temp1.0_notrunc` rows sent each arm `temperature` only (Ollama also `seed`). Each peer then
used its own `top_k` / `top_p` defaults, so the rows were not like for like (`docs/benchmarks.md`, "Sampling changes
that move comparability"). Since `0600855e`, `scripts/bench_peer.py` sends `temperature 1.0, top_p 1.0, top_k 0` to every
arm. This record re-measures the CUDA cells that the published figures came from. The Metal cells are queued for the
night of 2026-10-17 on the Mac and get their own record.

## Pre-registration (written before the run)

- **Machine:** nobara-pc, RTX 2070 SUPER, driver `595.91.07` (the run stops on any other driver).
- **Build:** goinfer `63038121` (main, tree clean), `cuda` serve built by the script.
- **Peers:** Ollama 0.32.5 and llama-server, as in `peer-sweep-2026-09-29` cell f.
- **Cells:** CUDA, depth 128, config `temp1.0_notrunc`, models `0.5B`, `gemma3-1b`, `phi3-mini` (phi3-mini at its default
  quant), engines goinfer / Ollama / llama.cpp, same session, interleaved by the harness.
  - Protocol: `BENCH_RUNS=3` × 8 completions × 64 tokens.
  - Checked before any timed run: the script asserts that the config sends all three fields to the goinfer, Ollama and
    llama.cpp arms.
- **Grading:** `peer-notrunc-2026-10-17/grade.py`, an unchanged copy of the 2026-09-29 grader.
  - Bands: pair i = goinfer run i / peer run i. AHEAD all > 1.03, LEVEL all in [0.97, 1.03], AMBIGUOUS-HIGH / -LOW on a
    straddle, BEHIND all < 0.97.
  - Spread cap: either engine's run-to-run spread over 5% caps the cell at AMBIGUOUS.
  - VOID: under the harness's token gate, or a GPU cell whose decode path is the CPU's.
- **Tier and cost:** tier 1, by day (owner, 2026-10-17), one timed run under the timing lock. The cells' recorded times
  (`peer-sweep-2026-09-29/nobara/f-sampled.json`, `b5-r7b-cbf2c25d.json`) put it at about 4 minutes, plus build and idle
  waits.
- **Use:** the readings replace the TL;DR and §B5.1 temperature-1.0 CUDA figures. The old rows stay, with their note. No
  stop rule beyond the fixed cells, and no re-run until a cell passes.
- **Matched quants (added 2026-10-17, owner, before any figure is published):** a cell is comparable only when goinfer and
  each peer load the same quant class, read from the files (the Ollama tag's manifest blob, the llama.cpp GGUF, mlx-lm's
  `config.json`) and from goinfer's reported decode path. goinfer has no mode that keeps a Q4_K_M file's Q5_K and Q6_K
  tensors native, so a match is by class (4-bit against 4-bit), never tensor for tensor. From `bench_peer.py` after this
  change, a run whose arms do not share a class is refused unless `BENCH_UNMATCHED=1`, and then every cell records
  `quant_matched: false`.

## Metal (the Mac, night of 2026-10-17): pre-registration

- **Machine and build:** MacBook Pro M1 Pro 16 GB, goinfer `63038121` (serve binary pinned under
  `~/goinfer-bench/peer-notrunc-2026-10-17`).
- **Peers:** Ollama 0.32.5 and mlx-lm 0.31.3, as on 2026-10-04.
- **Cells:** Metal, depth 128, `temp1.0_notrunc`, the 0.5B, 1.5B and 7B. goinfer / mlx-lm / Ollama, except no MLX 0.5B
  checkpoint, which makes 8 cells. Protocol: 3 runs × 8 × 64 tokens. `bench_peer.py` also runs its greedy cells at 128,
  which are same-session controls and not part of the claim.
  Script: `peer-notrunc-2026-10-17/run-mac.sh`, a night job queued after M1 and P29.
- **The 7B's load:**
  - The by-day fit check refused its Metal resident build: 4.65 GB against the memory guard's live budget of 4.43 GB.
    It would have run on the CPU.
  - The job checks again at night and measures the 7B only if it loads `metal-resident`; otherwise its cells are recorded as not
    measured. The guard is not bypassed.
  - The 0.5B and 1.5B fit-checked `metal-resident (int4)`.
- **Grading:** the same bands and void rules as the CUDA cells above; a cell not on `metal-resident` is VOID.
- **Use:** the readings replace the TL;DR Metal row's T = 1.0 figures. The old figures stay, with their note.

## The shared truncated preset: pre-registration (2026-10-17, before either run)

Publishing order (owner, 2026-10-17): greedy first; then one shared truncated preset sent to every arm as the main sampled
row; the untruncated rows last, labelled "no truncation (top_p 1, top_k 0)".

- **Preset:** `temp0.8_topk40_topp0.9` (`bench_peer.py` `19f265b6`), sent to every arm: temperature 0.8, top_k 40,
  top_p 0.9, seed 1. These are Ollama's own defaults, made explicit.
- **Metal:** added to tonight's Mac job. The script and cell set are as above, the configs are this preset and
  `temp1.0_notrunc`, and the serve binary is unchanged. 8 more cells; the job's estimate goes from 20 to 30 min.
- **CUDA:** the same three models against Ollama and llama.cpp at depth 128, 3 runs. These are for nobara's session to run
  by day, through its own session or queue.
- **Grading and use:** the bands and void rules above. The readings are the main sampled row of the TL;DR and §B5.1.

## Results: CUDA (nobara-pc, read 2026-10-17)

Run 2026-10-17 12:13–12:22 PDT (557 s), `run-nobara.sh` at `63038121`. Driver 595.91.07, Ollama 0.32.5, llama-server
0.4.0-dev `427291b`. Every arm was sent `temperature 1.0, top_p 1.0, top_k 0` (Ollama also `seed 1`); the script checked
this before the run. A first attempt was refused by the harness's idle gate (loadavg 1.16 after the build) and measured
nothing (`nobara/run-refused-1.log`). Records: `peer-notrunc-2026-10-17/nobara/` (`f-sampled.json`, `grade.jsonl`,
`provenance.txt`, `run.log`).

Ratio r = goinfer ÷ peer, the median of the 3 pairs (range in brackets), graded by the pre-registered bands:

| model | peer | goinfer tok/s | peer tok/s | r | verdict |
|---|---|---:|---:|---|---|
| qwen2.5-coder-0.5b | Ollama | 355.2 | 214.8 | 1.650 (1.646–1.667) | **AHEAD** |
| qwen2.5-coder-0.5b | llama.cpp | 355.2 | 365.5 | 0.978 (0.967–0.991) | AMBIGUOUS-LOW |
| gemma3-1b | Ollama | 202.2 | 123.2 | 1.641 (1.633–1.664) | **AHEAD** |
| gemma3-1b | llama.cpp | 202.2 | 208.7 | 0.972 (0.966–0.979) | AMBIGUOUS-LOW |
| phi3-mini | Ollama | 113.7 | 123.5 | 0.921 (0.921–0.921) | **BEHIND** |
| phi3-mini | llama.cpp | 113.7 | 128.2 | 0.888 (0.886–0.889) | **BEHIND** |

- **What changed is Ollama's side.** Untruncated, Ollama decodes 214.8 / 123.2 / 123.5 tok/s, against 264.3 / 151.3 / 125.9
  greedy in the same session. The old rows' Ollama figures (268.6 / 149.8 / 125.8) sat at its greedy rate, which fits
  the old rows sampling with Ollama's own `top_k` and `top_p` defaults. goinfer's untruncated rate is within 1% of its
  own greedy rate on all three models.
- **Greedy controls, same session (not part of the claim):**
  - 0.5B: AHEAD of Ollama 1.334, BEHIND llama.cpp 0.948.
  - gemma3-1b: AHEAD of Ollama 1.313, BEHIND llama.cpp 0.927.
  - phi3-mini: BEHIND both (0.905 / 0.890).
- **Not covered:** llama-server applies its own default `min_p` when none is sent, and goinfer has no `min_p` request
  field, so the llama.cpp arm may still truncate there (`docs/benchmarks.md`, "Sampling changes that move
  comparability").

### The quant each arm loaded

The same for both nobara runs, except goinfer on phi3-mini:

| model | goinfer, this run | goinfer, nobara's run (`peer-sampled-cuda-2026-10-17`) | Ollama (tag → manifest blob) | llama.cpp (file) |
|---|---|---|---|---|
| qwen2.5-coder-0.5b | `int4` (`cuda-resident (int4)`) | `int4` | Q4_K_M | Q4_K_M |
| gemma3-1b | `int4` | `int4` | Q4_K_M | Q4_K_M |
| phi3-mini | `q4k` (`cuda-resident (q4k)`: the file's 81 Q4_K tensors kept exact; its 32 Q5_K and 17 Q6_K tensors at int8) | `int8int8` (every weight int8; the harness's old default override) | `p3m-local` → `sha256-01ec9e…`, Q4_K_M | `Phi-3-mini-4k-instruct-q4.gguf`, Q4_K_M |

Ollama's blob and the llama.cpp file are the same file: 2,393,231,072 bytes, file type 15, and the same tensor census
(81 q4_K, 32 q5_K, 17 q6_K, 65 f32). By class, this run's phi3-mini matches (4-bit against 4-bit) and nobara's does not
(8-bit against 4-bit). By the owner's ruling (2026-10-17), phi3-mini's cells wait for a matched-quant reading on both boxes
before anything is published.

### nobara's record (`peer-sampled-cuda-2026-10-17`, `e96737b8`)

- **Its 0.5B and gemma3-1b cells feed the §B5.1 rows that sent one field.** Its arms were sent both fields explicitly: the
  top_p 0.95 cells sent `top_k 0`, and the top_k 40 cells sent `top_p 1.0`.
- **Its phi3-mini cells are excluded:** goinfer ran `int8int8` against Ollama's Q4_K_M `p3m-local`.
- **Its shared-preset run used temperature 1.0,** not the pre-registered 0.8. It is kept as a record only.
- **Its 0.5B goinfer `temp1.0_notrunc` cell is VOID:** one completion ended at 10 of 64 tokens, under the token gate's 32 (a
  sampled end-of-sequence). Its rate, 354.4 tok/s, agrees with this run's 355.2.
- **The temperature-1.0 runs agree where both arms matched.** gemma3-1b reads 1.641 here and 1.658 there; Ollama's rates
  agree within 0.1% on all three models. So this run is cited, with nobara's as the repeat.

## Results: the shared truncated preset, CUDA

Read by nobara's session at harness `e8e8c95d`, matched 4-bit quants on every arm (phi3-mini at `q4k`):
`docs/measurements/peer-sharedtrunc-cuda-2026-10-17/README.md` (`932f6e4c`). goinfer ÷ peer: 0.5B AHEAD of Ollama 1.312 and
BEHIND llama.cpp 0.917; gemma3-1b AHEAD 1.272 and BEHIND 0.910; phi3-mini BEHIND both, 0.895 and 0.881. No cell was void.
This is phi3-mini's matched-quant reading on CUDA; the Metal one is tonight's job. The llama.cpp gap is filed as P31 in
`docs/queue-performance.md`.

## Results: Metal

Queued for the night of 2026-10-17 (`notrunc-metal-remeasure`, after M1 and P29). Results to follow.

## MLX's two server paths: pre-registration (2026-10-17, evening, before the run)

**Why.** In the Metal run above, MLX's sampled rows ran faster than its greedy ones: on the 1.5B, 110.8 tok/s greedy, 122.2
on the shared preset and 136.2 untruncated. The harness times every arm the same way: the engine's reported completion
tokens less one, over the harness's own clock from the first stream event to the last. Every completion was 64 tokens,
tokens per streamed event were about 1.0, and MLX's greedy cells ran at the lowest load of the night.

The cause is in mlx-lm 0.31.3's `server.py`. `_is_batchable` sends a request with no seed to the batch engine
(`BatchGenerator`), and one with a seed to the one-request `stream_generate` loop. The greedy config sent MLX no seed, while
both sampled configs sent `seed 1`. `bench_peer.py` now sends every arm `seed 1` in every config (`e90e82b9`) and records
each MLX cell's `mlx_path`.

- **Job:** `peer-notrunc-2026-10-17/run-mlx-seed.sh`, a Mac night job on 2026-10-17 with `bench_peer.py` pinned at
  `e90e82b9`.
  - mlx-lm only, Metal, depth 128, the 1.5B and 7B, 3 runs × 8 completions × 64 tokens.
  - Cells:
    - greedy with seed 1 (`single-request`);
    - `temp0.8_topk40_topp0.9_unseeded`, the preset with no seed (`batch`);
    - two same-session controls with seed 1, the preset and `temp1.0_notrunc` (`single-request`).
  - 8 cells, about 10 minutes.
- **Prediction 1, the owner's:** seeded greedy ≥ the untruncated rate, per model. Argmax costs less than sampling the
  whole vocabulary, so on one server path greedy should not be the slower config.
  - Graded against the same-session untruncated control, as r = greedy run i ÷ untruncated run i for the 3 pairs.
  - **Holds:** every r ≥ 1.00.
  - **Ambiguous:** pairs straddle 1.00, or all fall in [0.97, 1.00).
  - **Fails:** every r < 0.97.
- **Prediction 2:** the unseeded preset runs slower than the seeded preset, by about what separated the batch-path greedy
  row from the one-request sampled rows above.
  - r = unseeded run i ÷ seeded run i.
  - **Holds:** every r < 0.97.
  - **Ambiguous:** a straddle of 0.97, or all r in [0.97, 1.03].
  - **Fails:** every r > 1.03. The seed would then not be what moved MLX's rate.
- **Spread cap and voids:** as above. A run-to-run spread over 5% caps a cell at ambiguous, and the harness's token gate
  voids.
- **Use (owner, 2026-10-17):**
  - 2d quotes MLX rows from the one-request path only: the run above's sampled rows and this job's seeded greedy.
  - Each row says which path it ran.
  - The run above's unseeded greedy row is kept only as a note.
  - No stop rule beyond the fixed cells.

## Results: Metal (the Mac, read 2026-10-17)

Run 2026-10-17 19:28–19:46 PDT, `run-mac.sh`: goinfer `63038121`, Ollama 0.32.5, mlx-lm 0.31.3. The 7B's fit check at the
job's start read `metal-resident (int4)`, so all three models ran. Records: `peer-notrunc-2026-10-17/mac/`
(`g-metal-decode.json`, `provenance.txt`, `fitcheck-7B.log`, `run.log`, `grade.jsonl`, `grade-mlx.jsonl`).

`grade.py`, unchanged, grades the Ollama pairs. `mac/grade_mlx.py` grades the mlx pairs with `grade.py`'s own outcome,
spread and token-gate rules, imported unchanged; the registered grader pairs with Ollama and llama.cpp only. There is no
MLX 0.5B checkpoint, so the three MLX 0.5B cells are MISSING, as registered. The runner started the job at a load average
of 2.37 after its settle gave up. Every cell's three runs agree within 1.5%, and no cell hit the spread cap.

goinfer ÷ peer, the median of 3 pairs, with its grade:

| config | model | goinfer | Ollama | ÷ Ollama | MLX (path) | ÷ MLX |
|---|---|---:|---:|---|---:|---|
| greedy | 0.5B | 198.5 | 145.2 | 1.366 **AHEAD** | — | — |
| greedy | 1.5B | 93.4 | 85.1 | 1.099 **AHEAD** | 110.8 (batch) | 0.845 BEHIND |
| greedy | 7B | 30.0 | 25.0 | 1.199 **AHEAD** | 36.2 (batch) | 0.827 BEHIND |
| shared preset | 0.5B | 172.9 | 146.0 | 1.185 **AHEAD** | — | — |
| shared preset | 1.5B | 86.5 | 85.1 | 1.017 LEVEL | 122.2 (single-request) | 0.711 **BEHIND** |
| shared preset | 7B | 29.3 | 25.0 | 1.172 **AHEAD** | 38.7 (single-request) | 0.757 **BEHIND** |
| untruncated | 0.5B | 204.0 | 136.0 | 1.498 **AHEAD** | — | — |
| untruncated | 1.5B | 94.2 | 81.9 | 1.150 **AHEAD** | 136.2 (single-request) | 0.693 **BEHIND** |
| untruncated | 7B | 30.1 | 24.7 | 1.221 **AHEAD** | 40.1 (single-request) | 0.752 **BEHIND** |

**The greedy MLX rows are kept as a note only** (owner, 2026-10-17). They ran mlx-lm's batch engine because the greedy config
sent no seed, so they do not compare with the single-request rows. "MLX's two server paths", below, measures that.

## Results: MLX's two server paths (read 2026-10-17)

Run 2026-10-17 21:37–21:43 PDT, `run-mlx-seed.sh` (`bench_peer.py` at `e90e82b9`). It was a Mac night job, started at a
load average of 1.3 to 2.2. Every cell records its `mlx_path`, every spread is under 1%, and every token gate passed.
Records: `peer-notrunc-2026-10-17/mac/mlx-seed/` (`mlx-seed.json`, `provenance.txt`, `run.log`, `grade.txt`).

| model | greedy, seeded | untruncated, seeded | preset, seeded | preset, unseeded |
|---|---:|---:|---:|---:|
| | single-request | single-request | single-request | batch |
| 1.5B | 138.9 / 138.3 / 138.9 | 136.2 / 136.3 / 136.2 | 121.8 ×3 | 99.8 / 99.7 / 99.7 |
| 7B | 40.2 / 40.2 / 40.1 | 40.1 / 39.9 / 39.8 | 38.6 ×3 | 35.4 / 35.5 / 35.4 |

- **Prediction 1 holds** for both models: seeded greedy ≥ untruncated. The pairs are 1.020 / 1.015 / 1.020 (1.5B) and
  1.004 / 1.007 / 1.008 (7B), every r ≥ 1.00.
- **Prediction 2 holds** for both models: the unseeded preset is slower than the seeded one. The pairs are 0.818–0.819 (1.5B)
  and 0.917–0.918 (7B), every r < 0.97.
- **Reading:** the seed alone moves mlx-lm between its two server paths, and its one-request path is the faster for one
  client. The single-request preset also matches the earlier run's within 0.3% (121.8 against 122.2, and 38.6 against
  38.7).
- **goinfer against MLX's seeded greedy, cross-session:** 93.4 ÷ 138.6 = 0.67 on the 1.5B, and 30.0 ÷ 40.2 = 0.75 on the 7B,
  where the median of the 3 MLX runs is 138.6 on the 1.5B and 40.2 on the 7B. goinfer's figures are from the earlier run.
  This is a cross-session ratio, so it is not graded, and the bands do not apply to it
  (`docs/measurements/noise-registry.md`, TE3). A same-session greedy cell, goinfer and seeded MLX interleaved, would
  grade it.
