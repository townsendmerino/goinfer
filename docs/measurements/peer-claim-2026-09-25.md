# Peer claim sweep — goinfer v0.20.0-candidate vs Ollama (2026-09-25)

The brief is `docs/tasks/task-peer-claim-2026-09.md`. The candidate claim is "goinfer is as fast as Ollama or faster".
This document says which claim the numbers support.

## Part 1 — pre-registration

> **Committed and pushed before any timed run, and not edited after it.** If a bar turns out to be wrong, a dated
> amendment goes in *Amendments* below and gives the mechanism; the text above it stays.

### The ratio and the bars

- **Decode cells (a, b, c, e, f, g, i).** For each cell, pair *i* is goinfer's run *i* against the peer's run *i*
  in the same cell. A run is the mean decode rate over 8 completions of 64 tokens, timed client-side from the first
  streamed token (prefill excluded). The ratio is **r = goinfer tok/s ÷ peer tok/s**, so r > 1 means goinfer is faster.
- **Prefill cells (d, h).** Pair *i* is request *i* on each engine: the same unique `Session NNNN.` prefix, from
  `scripts/bench_peer_prefill.py`. The ratio is **r = peer TTFT ÷ goinfer TTFT**, so again r > 1 means goinfer is
  faster. TTFT is the user-visible time to first token, request overhead included. No claim here rests on the
  marginal prefill-throughput slope.
- **The bars on r:**

  | label | rule |
  |---|---|
  | **level** | 0.97 ≤ r ≤ 1.03 |
  | **ahead** | r > 1.03 |
  | **behind** | r < 0.97 |

- **A cell's outcome is decided by ALL its pairs, never by the mean:**

  | outcome | rule | counts toward "level or ahead"? |
  |---|---|---|
  | **AHEAD** | every pair > 1.03 | yes |
  | **LEVEL** | every pair in [0.97, 1.03] | yes |
  | **AMBIGUOUS-HIGH** | pairs straddle 1.03, none below 0.97 | yes, as level — never as ahead |
  | **AMBIGUOUS-LOW** | pairs straddle 0.97 | **no**: it blocks any "level or ahead" wording covering it |
  | **BEHIND** | every pair < 0.97 | no |
  | **VOID** | a validity gate below fails | no; the cell is re-run or reported as void, never guessed |

  The median r and the pair count are reported beside every outcome.
- **Spread.** If either engine's run-to-run spread in a cell exceeds 5% of its mean, that cell's outcome is capped at
  AMBIGUOUS (HIGH or LOW by the same straddle rule), whatever its pairs say. That is this page's existing 5% threshold
  for an indicative cell, applied before the fact.
- **Pair counts.** Decode: 3 pairs per cell (`BENCH_RUNS=3`). Prefill: 6 pairs per cell (`--n 6`).

### Validity gates — any failure voids the cell

1. **The checkpoint is the same weights on both sides**, verified per tensor with `scripts/gguf_same_weights.py`
   (results in *Same weights* below), and read from `~/models`. A path under `/srv/models` or `/Volumes` voids the row.
2. **Every completion returned at least 95% of the requested tokens by the engine's own `usage`.** Ollama
   has terminated early before (576 of 3072 on one cell). `tokens_per_chunk` is recorded for every cell and read.
3. **The provenance header shows driver `595.91.07`** (nobara-pc) **and an idle box at start** (`bench_peer.py`'s
   preflight). A different driver stops the sweep: that is a re-anchor, not a carry-forward.
4. **Servers restart between cells, and engines are interleaved cell by cell** (the harness's own protocol).

### Configuration, fixed now

**nobara-pc**
- **Machine:** RTX 2070 SUPER 8 GB, driver `595.91.07`, Ryzen 7 3700X, Nobara 44.
- **goinfer:** built from **this pre-registration's commit** — `cuda/cmd/serve` under `-tags cuda`,
  `CGO_ENABLED=0`, and root `cmd/serve` for the CPU cells. The binary paths are named after the commit.
- **Ollama v0.32.5** at `~/ollama-0325`, **with its own defaults**. On this card that means flash attention on
  (its log reports `flash_attn = auto` → "Flash Attention enabled") and an f16 KV cache.
- **llama-server `0.4.0-dev (build 1, commit 427291b)`** at `~/mycode/peers/llama.cpp/build/bin`, flash attention
  at its default `auto`. It is a second column wherever `bench_peer.py` supports it: cells a, b, c, e and f. It is
  not in cell d, because `bench_peer_prefill.py` does not drive it.
- **Context:**
  - depths 128, 2048 and 3900: the harness default (goinfer unpinned; Ollama `num_ctx` 4096; llama-server
    `--ctx-size` 4096), as in the §B8 anchor;
  - depth 8000: `BENCH_CTX=8192`, a separate invocation, which pins 8192 on all three engines and changes nothing
    else. **Not `BENCH_DEEP_CTX`**, which would switch to the deep generation protocol and force both peers' flash
    attention off; a claim against the peer as shipped cannot use it. `BENCH_CTX` is new in this commit;
  - the 26B cell: `BENCH_CTX=2048`.

**MacBook** (cells g–i; the Mac session runs them and fills them in)
- **Machine:** M1 Pro, `BENCH_MAX_LOADAVG=2.0` (this page's macOS qualifier). Metal backend via `metal/cmd/serve`;
  Ollama v0.32.5 with its defaults; llama-server where the Mac has it.
- **Pair counts:** same as nobara-pc — 3 per decode cell, 6 per prefill cell. Raising the run count on the Mac to
  compensate for its load floor is allowed, and the count used is recorded.
- **Depth "4000":** it follows §B3's Metal depth curve. The Mac session names the calibrated prompt it used; if that
  is `prompts.json`'s 3900-token prompt, the row says 3900.

### The cells

| cell | machine | what | models × points |
|---|---|---|---|
| **a** | nobara | CUDA greedy decode | qwen2.5-coder 0.5B, 1.5B; qwen2.5-7B — depth 128 / 2048 / 3900 / 8000 (12 cells) |
| **b** | nobara | CUDA greedy decode, controls | gemma3-1b (windowed); phi3-mini (flash-decode declines) — depth 128 / 3900 (4 cells) |
| **c** | nobara | 26B MoE on 8 GB | Gemma 4 26B-A4B, depth 128, context 2048. goinfer: C′ expert streaming (`-moe-cache-experts`; this release's DMA overlap is default-on). Ollama and llama-server: their own placement (CPU offload) |
| **d** | nobara | CUDA prefill TTFT | qwen2.5-coder 1.5B — K = 512 / 3900 (2 cells) |
| **e** | nobara | CPU (amd64) greedy decode | 0.5B / 1.5B / 7B — depth 128 (3 cells). Expected about 0.82×; this confirms or refutes it |
| **f** | nobara | CUDA sampled decode | 0.5B and phi3-mini, depth 128 — `temp1.0_notrunc` and `temp0.8_topp0.95` (4 cells) |
| **g** | Mac | Metal greedy decode | 0.5B / 1.5B / 7B — depth 128 / 2048 / 4000 (9 cells) |
| **h** | Mac | Metal prefill TTFT | qwen2.5-coder 1.5B — K = 512 / 3900 (2 cells) |
| **i** | Mac | CPU (arm64) greedy decode | 0.5B / 1.5B — depth 128 (2 cells) |

**Cell c is an architecture comparison, not like-for-like.** goinfer keeps every expert on the GPU and streams them
host↔VRAM, running its own int4 `.giw` bundle (`~/models/gemma4-26b-int4.giw`). The peers offload layers to the
CPU and run the Q4_K_M GGUF (`~/models/gemma4-26b-q4_k_m.gguf`). Its ratio is reported, and its outcome labelled,
but it never enters a "level or ahead" family claim.

### Claim wording, written now

**Scope rule.** A claim covers only the cells measured. It names the hardware, driver/OS where relevant, the peer
version, the models and the depths. Nothing is extrapolated to other model sizes, families, quantizations,
hardware or peer versions.

**The claims against llama.cpp are separate.** Its ratios are reported as a second reading. A sentence may name
llama.cpp only where its own pairs meet the same bars.

For each family, the first wording whose condition holds is the one used:

**A — CUDA decode (cell a).**
- **A-ahead:** all 12 cells AHEAD → "On an RTX 2070 SUPER, goinfer's greedy decode is ahead of Ollama v0.32.5 on
  qwen2.5-coder 0.5B and 1.5B and qwen2.5-7B at every KV depth measured (128, 2048, 3900 and 8000 tokens)."
- **A-level:** all 12 cells AHEAD, LEVEL or AMBIGUOUS-HIGH → the same sentence with "level with or ahead of".
- **A-partial:** otherwise → "…level with or ahead of Ollama v0.32.5 at [each passing model × depth]; behind at
  [each failing cell, with its median ratio]". The words "every depth" and "every model" are not used.
- **A-behind:** no cell passes → "…behind Ollama v0.32.5 at every cell measured, by [min]–[max]×."

**B — CUDA controls (cell b).** One sentence per model, with the same outcome labels. It never widens A. In
particular, a passing gemma3-1b does not stand for "windowed models", and phi3-mini does not stand for "models where
flash-decode declines".

**C — 26B (cell c).** "Gemma 4 26B-A4B on an 8 GB card: goinfer, keeping every expert on the GPU (host↔VRAM
streaming), decodes at [r]× Ollama's rate with its CPU offload ([outcome]); different checkpoints (int4 `.giw`
vs Q4_K_M)." The words "faster than Ollama" never appear without the architecture clause.

**D — CUDA TTFT (cell d).** "On the 1.5B, goinfer's time to first token is [r₅₁₂]× / [r₃₉₀₀]× Ollama's at 512 /
3900 prompt tokens (>1 = goinfer faster; [outcomes])." This is not a prefill-throughput claim.

**E — CPU amd64 (cell e).** "On CPU (Ryzen 7 3700X), goinfer's greedy decode is [outcome] Ollama v0.32.5's:
[r]× / [r]× / [r]× at 0.5B / 1.5B / 7B."

**F — sampled (cell f).** One sentence per model and configuration, with the same labels, for example "with
temperature 0.8 and top_p 0.95 at depth 128, goinfer is [outcome] on the 0.5B ([r]×)".

**G/H/I — Mac.** The same structure as A, D and E, naming the M1 Pro and the backend (Metal or CPU).

**The headline.** The unqualified "goinfer is as fast as Ollama or faster" may be written **only if every graded
cell** (a, b, d, e, f, g, h, i) is AHEAD, LEVEL or AMBIGUOUS-HIGH. Otherwise the headline is the widest *family*
claim above that holds in full, stated with its scope — for example "on NVIDIA, level or ahead on greedy decode at
every depth measured" — and it is followed by the families that did not hold, with their numbers.

### Same weights (checked before timing, 2026-09-25)

`scripts/gguf_same_weights.py`, each `~/models` GGUF against the Ollama blob the harness loads:

| model | Ollama tag | verdict |
|---|---|---|
| qwen2.5-coder 0.5B q4_K_M | `q05` | SAME WEIGHTS (repacked container) |
| qwen2.5-coder 1.5B q4_K_M | `q15` | SAME WEIGHTS |
| qwen2.5-7B q4_K_M | `q7b` | SAME WEIGHTS |
| gemma3-1b q4_K_M | `g31b` | SAME WEIGHTS |
| Gemma 4 26B-A4B q4_K_M (the peers' file) | `m26q4km` | SAME WEIGHTS |
| phi3-mini q4 | `p3m` | **WEIGHTS DIFFER — 0/195 tensors identical, f32 norms included: a different revision** |
| phi3-mini q4 | **`p3m-local`** | SAME WEIGHTS, 195/195 |

**phi3-mini:** the owner chose to re-import, so `p3m-local` was created with `ollama create` from the `~/models`
file. The harness now points at it. `p3m`'s template closed turns with `</s>`; `p3m-local`'s uses Phi-3's
`<|end|>`. **Every phi3-mini peer row produced with the `p3m` tag compared different weights.** That includes the harness's rows before this date, such as §B5.1. STEP 2 marks
them.

### Amendments

*(none)*

## Part 2 — results

*(Filled after the runs. nobara-pc fills a–f; the Mac session fills g–i.)*

| cell | points | outcome | median r (pairs) | peer tok/s or ms | llama.cpp r |
|---|---|---|---|---|---|
| a | CUDA greedy, 0.5B / 1.5B / 7B × depth 128 / 2048 / 3900 / 8000 (12) | **9 AHEAD · 1 LEVEL** (7B @8000) · **2 VOID** (0.5B @2048, 7B @128: Ollama ended its reply early) · **0 BEHIND** | @128 1.272 / 1.297 / *void 1.098*; @2048 *void 1.131* / 1.269 / 1.076; @3900 1.167 / 1.229 / 1.052; @8000 1.153 / 1.130 / 0.986 (0.5B / 1.5B / 7B; 3 pairs each) | per point below | 1.5B AHEAD @128 / 2048 (1.114 / 1.044), LEVEL @3900; 7B LEVEL @2048 / 3900; BEHIND on the 0.5B @128 / 2048 / 8000 (0.83–0.92) and at 8000 on 1.5B / 7B (0.948 / 0.939); 2 VOID |
| b | CUDA controls, gemma3-1b / phi3-mini × depth 128 / 3900 (4) | gemma3-1b @3900 **AHEAD** · @128 **VOID** (goinfer ended at 34 of 64 tokens) · phi3-mini @128 **AHEAD** · @3900 **VOID** (Ollama reported no tokens) | gemma3-1b 1.262 (@3900), *void 1.266* (@128); phi3-mini 1.138 (@128), *void 0.799* (@3900) | per point below | gemma3-1b @3900 BEHIND (0.922); phi3-mini @128 AHEAD (1.121), @3900 BEHIND (0.794); 1 VOID |
| c | Gemma 4 26B-A4B on 8 GB, ctx 2048 (1) — **architecture comparison** | **AHEAD** (not like-for-like; never in a family claim) | 1.815 (3 pairs) | Ollama 22.2 vs goinfer 40.2 tok/s | AHEAD (1.453; llama.cpp 27.6) |
| d | CUDA TTFT, 1.5B, K = 512 / 3900 (2) | **AMBIGUOUS-HIGH** at both (spread cap from Ollama's own spread, 21.6% / 5.3%; all 12 pairs above 1.0) | 4.994 / 1.019 (6 pairs each) | Ollama 445 / 972 ms vs goinfer 88 / 955 ms (medians) | — |
| e | CPU amd64, 0.5B / 1.5B / 7B @128 (3) | 0.5B **BEHIND** · 1.5B **VOID** (goinfer ended at 54 of 64) · 7B **VOID** (Ollama ended at 56 of 64) | 0.797; *void 0.802*; *void 0.841* | Ollama 57.6 / 24.2 / 6.1 vs goinfer 45.9 / 19.4 / 5.1 | 0.5B BEHIND (0.734); 2 VOID (raw 0.712 / 0.776) |
| f | CUDA sampled @128, 0.5B / phi3-mini × temp 1.0 / temp 0.8 + top_p 0.95 (4) | 0.5B t0.8/p0.95 **AHEAD** · 0.5B t1.0 **VOID** (goinfer 1326 of 1536 tokens) · phi3-mini t1.0 **AHEAD** · phi3-mini t0.8/p0.95 **AMBIGUOUS-HIGH** | 1.194; *void 1.265*; 1.138; 1.027 | per point below | phi3-mini t1.0 AHEAD (1.118), t0.8/p0.95 LEVEL (1.015); 0.5B both VOID (raw 0.854 / 0.909) |
| g | Metal, 0.5B/1.5B/7B × depth 128/2048/3900 (9) | **1 AHEAD** (0.5B @128) · **6 BEHIND** (every 2048/3900 cell) · **2 VOID** (1.5B, 7B @128) | 0.5B @128 1.183; @2048 0.753 / 0.701 / 0.707; @3900 0.582 / 0.609 / 0.612 (0.5B / 1.5B / 7B; 3 pairs each) | per point below | 0.5B @128 AHEAD (1.094); BEHIND 0.536–0.700 at every 2048/3900 cell; 2 VOID |
| h | Metal TTFT, 1.5B, K = 512 / 3900 (2) | **AMBIGUOUS-LOW** at 512 (spread cap; all 6 pairs behind) · **BEHIND** at 3900 | 0.377 / 0.239 (6 pairs each) | Ollama 591 / 4184 ms vs goinfer 1580 / 17510 ms (medians) | — |
| i | CPU arm64, 0.5B / 1.5B @128 (2) | 0.5B **AMBIGUOUS-LOW** (peer spread cap) · 1.5B **VOID** | 0.5B 1.166 (pairs 0.874, 1.166, 1.168); 1.5B 0.804, void | Ollama 0.5B 125.2 / 93.7 / 93.1; 1.5B 61.8 / 60.3 / 62.0 | 0.5B BEHIND (0.842); 1.5B VOID |


### Cells a–f in detail (nobara-pc, 2026-09-25)

**Provenance.** RTX 2070 SUPER 8 GB, driver **`595.91.07`** in every results header (the anchor; no re-anchor),
Ryzen 7 3700X (16 threads), Nobara 44, kernel 7.2.0. goinfer built from this pre-registration's commit `411e7fc4`:
`~/bench-peer-claim/serve-cuda-411e7fc4` (`cuda/cmd/serve`, `-tags cuda`, `CGO_ENABLED=0`) and
`serve-cpu-411e7fc4` (root `cmd/serve`), both built 17:50Z. Ollama v0.32.5 at `~/ollama-0325` with its defaults (flash
attention on, f16 KV). llama-server `0.4.0-dev (build 1, commit 427291b)`. Every checkpoint read from `~/models` on NVMe;
the Ollama store is `~/ollama-0325/models`. Timed runs, UTC: step 1 (a @128/2048/3900 and e) 18:23–19:10, a @8000
19:11–19:20, b 19:20–19:28, f 19:28–19:36, c 19:36–19:44, d 19:46–19:47. Raw files, the runner, every log (including
the void attempts) and the grader: [`peer-claim-2026-09-25/`](peer-claim-2026-09-25/) (`python3 grade.py` reproduces
every outcome here; it is the Mac's grader with the cell key widened to include file, backend and config).

Four things in the headers to read correctly:
- **`goinfer_tree_dirty: true`** in every file. The only untracked paths were `test.json` and this results directory
  being written; no tracked file was modified.
- **`d-prefill.json` says commit `558c6cad`.** The checkout was fast-forwarded to pull the Mac's results while step 6
  ran, and the prefill harness reads `HEAD`. The binary it drove is the `411e7fc4` build (`serve_mtime` 10:50:23 −0700
  = 17:50:23Z), and `411e7fc4..558c6cad` changes no `.go`, `go.mod` or `go.sum`. Its `peer_version` line reads "could
  not connect": the harness probes `ollama -v` with no server up. The binary is the same v0.32.5.
- **`a-e-dense.json`'s header time, 19:11:23Z, is a resume, not the measurement.** The runner was restarted after step
  2 was refused at load 1.00, and step 1 re-opened its file, found 36 of 36 cells done, ran nothing and rewrote the
  header. Each cell's own `machine` record carries its load and GPU state.
- **Two void attempts precede this run and nothing from them is used.** Attempt 1 exported Ollama's library directory
  in `LD_LIBRARY_PATH`, so llama-server loaded Ollama's bundled libllama and never came up; its partial file is kept as
  `void-attempt1-a-e-dense.json`. Attempt 2 was refused by the harness preflight at load 4.56. The runner now waits for
  load1 < 0.8 before every step and aborts the sweep on any failed step.

**Protocol.** `BENCH_RUNS=3` (3 pairs per decode cell), `BENCH_ENGINES=goinfer,ollama,llamacpp`, interleaved cell by
cell with a server restart between cells (the harness's own protocol), its preflight idle gate before every cell, and 6
unique-prefix requests per prefill cell. Depth 8000 used `BENCH_CTX=8192` on all three engines, and the 26B
`BENCH_CTX=2048`, as registered.

**Same weights:** Part 1's table, checked before timing (`same-weights.log`). Every phi3-mini peer row here used
`p3m-local`, the re-import of the `~/models` file.

**Cell a — depths 128 / 2048 / 3900, harness default context (a-e-dense.json)**

| model | depth | goinfer tok/s | Ollama tok/s | r pairs vs Ollama | outcome (median r) | llama.cpp tok/s | vs llama.cpp (median r) | tokens returned (goinfer / Ollama / llama.cpp, of 1536) |
|---|---|---|---|---|---|---|---|---|
| 0.5B | 128 | 341.9 / 337.4 / 342.1 | 268.8 / 269.3 / 268.6 | 1.272 1.253 1.274 | **AHEAD** (1.272) | 367.6 / 365.5 / 370.1 | BEHIND (0.924) | 1536 / 1536 / 1536 |
| 0.5B | 2048 | 306.5 / 307.6 / 305.0 | 270.9 / 270.3 / 270.5 | 1.131 1.138 1.127 | **VOID** (1.131) | 366.4 / 369.2 / 367.8 | BEHIND (0.833) | 1536 / 1392 / 1536 |
| 0.5B | 3900 | 301.9 / 302.4 / 306.7 | 259.2 / 259.1 / 259.2 | 1.164 1.167 1.184 | **AHEAD** (1.167) | 471.4 / 479.3 / 477.2 | VOID (0.640) | 1536 / 1536 / 120 |
| 1.5B | 128 | 253.1 / 252.9 / 252.9 | 195.1 / 195.1 / 195.0 | 1.297 1.296 1.297 | **AHEAD** (1.297) | 223.2 / 228.4 / 226.9 | AHEAD (1.114) | 1536 / 1536 / 1536 |
| 1.5B | 2048 | 227.4 / 228.4 / 228.3 | 179.8 / 179.6 / 179.8 | 1.264 1.272 1.269 | **AHEAD** (1.269) | 219.3 / 218.8 / 217.2 | AHEAD (1.044) | 1536 / 1536 / 1536 |
| 1.5B | 3900 | 215.0 / 214.9 / 215.6 | 175.0 / 174.9 / 174.9 | 1.228 1.229 1.232 | **AHEAD** (1.229) | 212.0 / 211.5 / 212.2 | LEVEL (1.016) | 1536 / 1536 / 1536 |
| 7B | 128 | 81.4 / 81.4 / 81.4 | 74.2 / 74.1 / 74.1 | 1.098 1.098 1.099 | **VOID** (1.098) | 80.5 / 80.4 / 80.4 | VOID (1.012) | 1536 / 1248 / 888 |
| 7B | 2048 | 76.3 / 76.3 / 76.3 | 71.0 / 71.0 / 70.9 | 1.075 1.076 1.076 | **AHEAD** (1.076) | 76.1 / 76.0 / 75.9 | LEVEL (1.004) | 1536 / 1536 / 1536 |
| 7B | 3900 | 73.3 / 73.3 / 73.2 | 69.6 / 69.6 / 69.6 | 1.052 1.052 1.052 | **AHEAD** (1.052) | 74.4 / 74.4 / 74.3 | LEVEL (0.985) | 1536 / 1536 / 1536 |

**Cell a — depth 8000, BENCH_CTX=8192 (a-depth8000.json; its depth-128 rows are a same-context reference, not graded cells)**

| model | depth | goinfer tok/s | Ollama tok/s | r pairs vs Ollama | outcome (median r) | llama.cpp tok/s | vs llama.cpp (median r) | tokens returned (goinfer / Ollama / llama.cpp, of 1536) |
|---|---|---|---|---|---|---|---|---|
| 0.5B | 128 | 333.3 / 333.3 / 331.4 | 267.8 / 267.9 / 267.8 | 1.245 1.244 1.237 | **AHEAD** (1.244) | 376.4 / 375.1 / 377.6 | BEHIND (0.886) | 1536 / 1536 / 1536 |
| 0.5B | 8000 | 300.1 / 301.1 / 299.2 | 260.3 / 260.3 / 259.7 | 1.153 1.157 1.152 | **AHEAD** (1.153) | 328.2 / 329.0 / 329.4 | BEHIND (0.914) | 1536 / 1536 / 1536 |
| 1.5B | 128 | 251.6 / 250.9 / 251.5 | 194.7 / 194.1 / 194.3 | 1.293 1.293 1.294 | **AHEAD** (1.293) | 228.7 / 224.9 / 225.9 | AHEAD (1.113) | 1536 / 1536 / 1536 |
| 1.5B | 8000 | 186.0 / 186.0 / 185.8 | 164.4 / 164.6 / 164.6 | 1.131 1.130 1.128 | **AHEAD** (1.130) | 195.5 / 197.0 / 195.9 | BEHIND (0.948) | 1536 / 1536 / 1536 |
| 7B | 128 | 81.4 / 81.4 / 81.4 | 74.2 / 74.0 / 74.1 | 1.096 1.099 1.099 | **VOID** (1.099) | 80.4 / 80.4 / 80.2 | VOID (1.013) | 1536 / 1248 / 888 |
| 7B | 8000 | 67.0 / 67.1 / 67.1 | 68.1 / 68.0 / 68.0 | 0.985 0.986 0.988 | **LEVEL** (0.986) | 71.3 / 71.6 / 71.5 | BEHIND (0.939) | 1536 / 1536 / 1536 |

**Cell b (b-controls.json)**

| model | depth | goinfer tok/s | Ollama tok/s | r pairs vs Ollama | outcome (median r) | llama.cpp tok/s | vs llama.cpp (median r) | tokens returned (goinfer / Ollama / llama.cpp, of 1536) |
|---|---|---|---|---|---|---|---|---|
| gemma3-1b | 128 | 186.2 / 190.5 / 188.5 | 149.4 / 149.4 / 148.9 | 1.247 1.275 1.266 | **VOID** (1.266) | 214.3 / 210.7 / 215.0 | VOID (0.877) | 816 / 1536 / 1536 |
| gemma3-1b | 3900 | 187.4 / 187.8 / 186.4 | 148.6 / 148.8 / 148.7 | 1.262 1.262 1.254 | **AHEAD** (1.262) | 202.1 / 203.7 / 204.1 | BEHIND (0.922) | 1536 / 1536 / 1536 |
| phi3-mini | 128 | 143.1 / 143.2 / 143.3 | 126.0 / 125.9 / 125.8 | 1.135 1.138 1.139 | **AHEAD** (1.138) | 127.7 / 127.8 / 126.6 | AHEAD (1.121) | 1536 / 1536 / 1536 |
| phi3-mini | 3900 | 59.7 / 59.7 / 59.7 | 74.8 / 74.7 / 74.7 | 0.799 0.799 0.799 | **VOID** (0.799) | 75.3 / 75.2 / 75.1 | BEHIND (0.794) | 1536 / 0 / 1536 |

**Cell c — Gemma 4 26B-A4B, BENCH_CTX=2048 (c-26b.json)**

| model | depth | goinfer tok/s | Ollama tok/s | r pairs vs Ollama | outcome (median r) | llama.cpp tok/s | vs llama.cpp (median r) | tokens returned (goinfer / Ollama / llama.cpp, of 1536) |
|---|---|---|---|---|---|---|---|---|
| M26 | 128 | 40.2 / 40.2 / 40.1 | 22.2 / 22.2 / 22.2 | 1.815 1.815 1.810 | **AHEAD** (1.815) | 27.6 / 27.7 / 27.6 | AHEAD (1.453) | 1536 / 1536 / 1536 |

**Cell e — CPU, 16 threads (a-e-dense.json)**

| model | depth | goinfer tok/s | Ollama tok/s | r pairs vs Ollama | outcome (median r) | llama.cpp tok/s | vs llama.cpp (median r) | tokens returned (goinfer / Ollama / llama.cpp, of 1536) |
|---|---|---|---|---|---|---|---|---|
| 0.5B | 128 | 46.5 / 45.9 / 44.6 | 57.7 / 57.6 / 57.6 | 0.806 0.797 0.774 | **BEHIND** (0.797) | 62.8 / 62.5 / 62.2 | BEHIND (0.734) | 1536 / 1536 / 1536 |
| 1.5B | 128 | 19.3 / 19.4 / 19.4 | 24.2 / 24.2 / 24.1 | 0.800 0.802 0.804 | **VOID** (0.802) | 27.1 / 27.2 / 27.3 | VOID (0.712) | 1296 / 1536 / 1536 |
| 7B | 128 | 5.1 / 5.1 / 5.1 | 6.1 / 6.1 / 6.1 | 0.842 0.841 0.840 | **VOID** (0.841) | 6.6 / 6.6 / 6.6 | VOID (0.776) | 1536 / 1344 / 720 |

**Cell f — depth 128 (f-sampled.json; the greedy rows are the file's same-session control, not graded cells)**

| model | depth | config | goinfer tok/s | Ollama tok/s | r pairs vs Ollama | outcome (median r) | llama.cpp tok/s | vs llama.cpp (median r) | tokens returned (goinfer / Ollama / llama.cpp, of 1536) |
|---|---|---|---|---|---|---|---|---|---|
| 0.5B | 128 | greedy | 337.7 / 333.3 / 336.6 | 268.2 / 268.1 / 267.8 | 1.259 1.243 1.257 | **AHEAD** (1.257) | 372.1 / 371.0 / 372.9 | BEHIND (0.903) | 1536 / 1536 / 1536 |
| 0.5B | 128 | temp0.8_topp0.95 | 318.4 / 318.5 / 318.5 | 266.6 / 266.0 / 266.6 | 1.194 1.198 1.194 | **AHEAD** (1.194) | 372.7 / 365.3 / 375.8 | VOID (0.854) | 1536 / 1536 / 1368 |
| 0.5B | 128 | temp1.0_notrunc | 338.3 / 338.0 / 337.4 | 267.2 / 267.2 / 267.2 | 1.266 1.265 1.263 | **VOID** (1.265) | 374.7 / 359.6 / 371.1 | VOID (0.909) | 1326 / 1536 / 1490 |
| phi3-mini | 128 | greedy | 143.4 / 143.1 / 143.0 | 125.8 / 125.7 / 125.6 | 1.140 1.139 1.138 | **AHEAD** (1.139) | 127.3 / 127.8 / 127.3 | AHEAD (1.123) | 1536 / 1536 / 1536 |
| phi3-mini | 128 | temp0.8_topp0.95 | 129.3 / 128.9 / 129.7 | 125.9 / 125.8 / 125.8 | 1.027 1.024 1.031 | **AMBIGUOUS-HIGH** (1.027) | 127.3 / 127.5 / 127.8 | LEVEL (1.015) | 1536 / 1536 / 1536 |
| phi3-mini | 128 | temp1.0_notrunc | 142.9 / 142.9 / 143.0 | 125.8 / 125.6 / 125.6 | 1.136 1.138 1.139 | **AHEAD** (1.138) | 127.9 / 128.6 / 127.4 | AHEAD (1.118) | 1536 / 1536 / 1463 |

**Cell d** (TTFT; r = Ollama TTFT ÷ goinfer TTFT, one pair per unique-prefix request)

| K | goinfer TTFT ms (6 requests) | Ollama TTFT ms | r pairs | outcome (median r) | spread goinfer / Ollama |
|---|---|---|---|---|---|
| 512 | 96 88 89 87 89 88 | 453 471 441 376 448 442 | 4.717 5.384 4.979 4.307 5.062 5.010 | **AMBIGUOUS-HIGH** (4.994) | 9.6% / 21.6% |
| 3900 | 955 952 956 947 955 958 | 992 960 1012 965 974 970 | 1.039 1.009 1.058 1.019 1.019 1.012 | **AMBIGUOUS-HIGH** (1.019) | 1.1% / 5.3% |

**The VOID cells are early stops, and they are reported, not re-run.** Validity gate 2 needs ≥ 95% of the requested
tokens (1536 = 3 runs × 8 completions × 64). Seven of the 24 graded decode cells miss it against Ollama:

| cell | who stopped early | tokens | reading |
|---|---|---|---|
| a 0.5B @2048 | Ollama | 1392 = 24 × 58 | deterministic at temperature 0 |
| a 7B @128 | Ollama (and llama.cpp) | 1248 = 24 × 52 (888 = 24 × 37) | deterministic, and the same in the ctx-8192 reference row. The Mac's 7B @128 stopped at the same 52 and 37 |
| b gemma3-1b @128 | **goinfer** | 816 = 24 × 34 | deterministic. Both peers ran all 64 tokens on the same weights. goinfer ending its reply 30 tokens early is a lead worth a look, not investigated here |
| b phi3-mini @3900 | Ollama | 0 reported, 1200 chunks = 24 × 50 | Ollama's `usage` reported no tokens at all on this cell, and it streamed 50 chunks per completion. Either reading misses the gate |
| e 1.5B @128 | **goinfer** | 1296 = 24 × 54 | deterministic. The Mac's goinfer 1.5B stopped at 58 |
| e 7B @128 | Ollama (and llama.cpp) | 1344 = 24 × 56 (720 = 24 × 30) | deterministic |
| f 0.5B temp 1.0 | **goinfer** | 1326 | a sampled end-of-sequence. goinfer's sampled requests carry no seed, so a re-run might pass by chance, and re-running until one does would be choosing the result |

A re-run under the same protocol reproduces every temperature-0 stop, so none were re-run, as on the Mac. Their ratios
are recorded and not graded. Nothing in Part 1 was changed for them. **The token gate is read per cell** (≥ 1460 of
1536), as the Mac's grader reads it. Part 1's wording, "every completion", would be stricter. The only graded pair it
could change is llama.cpp's phi3-mini at temperature 1.0 (1463 of 1536, AHEAD 1.118), a second-column reading.

**What the depth curve says now.** This is the first peer measurement with flash-decode on by default. Here is what
each engine gives up from depth 128 to depth 3900, and from 128 to 8000 at context 8192:

| model | goinfer 128→3900 | Ollama 128→3900 | goinfer 128→8000 | Ollama 128→8000 |
|---|---|---|---|---|
| 0.5B | −11.6% | −3.6% | −10.0% | −2.8% |
| 1.5B | −15.0% | −10.4% | −26.0% | −15.3% |
| 7B | −10.0% | −6.1% | −17.6% | −8.2% |

goinfer still loses more per token of depth than Ollama. It now starts far enough ahead that it stays ahead through
3900 on all three models, and through 8000 on the 0.5B and 1.5B; the 7B meets Ollama at 8000 (LEVEL, 0.986).
**phi3-mini @3900 is the exception,** and it was registered as one: there flash-decode declines, and goinfer runs
59.7 tok/s against 74.7 (Ollama, void) and 75.2 (llama.cpp, BEHIND 0.794). This is the one CUDA decode point where
goinfer reads behind both peers.

**Cell c, read with care.** goinfer's 40.2 tok/s is this release's C′ path (the DMA overlap, `5ccba8de`, measured
30.4 → 38.7 on the same checkpoint), not the 16.1 / 17.6 in `benchmarks.md` §B4.1, which predate it. The peers offload to
CPU and run the Q4_K_M GGUF, while goinfer runs the int4 `.giw`. Two more cautions:
- Every completion repeats the same greedy prompt, so goinfer's expert cache sees the same routing each time. The flat
  per-completion rates (41.1, then 39.8–40.4) show no warm-up inside the timed runs, but a benefit carried over from
  the discarded warm-up completion cannot be excluded by this data.
- goinfer's peak RSS was 25.5 GB, against 17.2 GB for Ollama and 17.0 GB for llama.cpp.

**Cell e confirms the expected ~0.82×.** The one graded cell is 0.797. The two void cells read 0.802 and 0.841 raw.


### Cells g–i in detail (MacBook, 2026-09-25)

**Provenance.** Apple M1 Pro, 16 GB (8 CPU cores: 6 performance + 2 efficiency), macOS 26.6.2 (kernel 25.6.0).
goinfer `9c592095`, tree clean: `metal/cmd/serve` for g and h, root `cmd/serve` for i. `9c592095` has **no Go,
`go.mod` or `go.sum` difference from this pre-registration's commit `411e7fc4`**; the one commit between them is the
release-asset guard (a workflow, a script, docs). Ollama v0.32.5 (the Homebrew build, its own defaults, models in
`~/.ollama/models`). llama-server **`0.3.0 (build 10621, commit c1d0e7a00)`**, the Homebrew build — **not** nobara-pc's
`0.4.0-dev 427291b`, so the llama.cpp column here and nobara-pc's are different builds. Every checkpoint read from
`~/models` on the internal SSD. The Homebrew Ollama LaunchAgent (`:11434`) stayed up but idle with no model loaded;
the harnesses ran their own servers on their own ports. Raw files, the runner and the grader:
[`peer-claim-2026-09-25-mac/`](peer-claim-2026-09-25-mac/) (`python3 grade.py` reproduces every outcome here).

**Protocol.** `BENCH_RUNS=3` (3 decode pairs per cell) and 6 unique-prefix requests per prefill cell, as
pre-registered. `BENCH_MAX_LOADAVG=2.0`, which `bench_peer.py` applies before the sweep and before every cell. On macOS
its preflight **refuses** a busy box instead of waiting, and `bench_peer_prefill.py` has no macOS load reading, so the
runner waits for load1 ≤ 2.0 before each harness call (the first launch at 11:16 local was refused at load1 2.35 and
restarted; nothing had been timed). Timed runs, local (UTC−7): g 11:19–11:48, i 11:48–12:12, h 12:12–12:33. Before
any timing, every model × backend was loaded once, untimed, so no cell paid for a sidecar transcode (the 0.5B's Metal
and CPU sidecars were written then; the 1.5B/7B Metal sidecars are v14 from 2026-09-24, and all three Metal loads
aliased the file with 1 MB anonymous).

**Same weights** (`scripts/gguf_same_weights.py`, each `~/models` GGUF against the Ollama blob the harness loads):
`q05`, `q15` and `q7b` all **SAME WEIGHTS (repacked container)**. llama-server reads the same `~/models` files goinfer
does.

**phi3-mini's `p3m` tag is the same weights on the Mac.** It is not a Mac cell, but Part 1's finding bears on older Mac
rows, so it was checked: the Mac's `~/.ollama/models` `p3m` blob against `~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf`
is **SAME WEIGHTS, 0 differing tensors**. The "`p3m` compared different weights" result in Part 1 is a property of
nobara-pc's Ollama store, so it does not void the Mac's earlier phi3-mini rows.

**Depth "4000".** `prompts.json` has no 4000-token calibration. The rows use each model's `:3900` prompt (3912 tokens,
the §B3 Metal depth curve's deep point), so they say **3900**, as the pre-registration requires.

**Cell g.**

| model | depth | goinfer tok/s | Ollama tok/s | r pairs vs Ollama | outcome (median r) | llama.cpp tok/s | vs llama.cpp (median r) | tokens returned (goinfer / Ollama / llama.cpp, of 1536) |
|---|---|---|---|---|---|---|---|---|
| 0.5B | 128 | 172.7 / 171.2 / 167.7 | 144.5 / 144.7 / 144.4 | 1.195 1.183 1.161 | **AHEAD** (1.183) | 155.4 / 156.5 / 156.9 | AHEAD (1.094) | 1536 / 1536 / 1536 |
| 0.5B | 2048 | 104.3 / 104.4 / 104.0 | 138.4 / 138.7 / 138.8 | 0.754 0.753 0.749 | **BEHIND** (0.753) | 149.1 / 150.2 / 149.6 | BEHIND (0.695) | 1536 / 1536 / 1536 |
| 0.5B | 3900 | 77.1 / 77.0 / 75.7 | 131.2 / 132.3 / 132.6 | 0.588 0.582 0.571 | **BEHIND** (0.582) | 142.9 / 143.7 / 142.8 | BEHIND (0.536) | 1536 / 1536 / 1536 |
| 1.5B | 128 | 74.1 / 73.4 / 72.6 | 85.2 / 85.9 / 85.8 | 0.870 0.854 0.846 | **VOID** (0.854) | 89.0 / 88.4 / 88.8 | VOID (0.830) | 1392 / 1536 / 1536 |
| 1.5B | 2048 | 56.4 / 56.2 / 56.4 | 80.3 / 80.3 / 80.4 | 0.702 0.700 0.701 | **BEHIND** (0.701) | 82.0 / 83.7 / 83.4 | BEHIND (0.677) | 1536 / 1536 / 1536 |
| 1.5B | 3900 | 46.7 / 46.6 / 46.6 | 76.6 / 76.6 / 76.5 | 0.609 0.609 0.609 | **BEHIND** (0.609) | 79.4 / 79.3 / 79.6 | BEHIND (0.588) | 1536 / 1536 / 1536 |
| 7B | 128 | 22.0 / 21.9 / 21.9 | 25.5 / 25.5 / 25.5 | 0.861 0.857 0.858 | **VOID** (0.858) | 25.9 / 25.7 / 25.9 | VOID (0.846) | 1536 / 1248 / 888 |
| 7B | 2048 | 17.1 / 17.1 / 17.1 | 24.2 / 24.1 / 24.2 | 0.706 0.708 0.707 | **BEHIND** (0.707) | 24.4 / 24.5 / 24.4 | BEHIND (0.700) | 1536 / 1536 / 1536 |
| 7B | 3900 | 14.3 / 14.4 / 14.3 | 23.4 / 23.4 / 23.4 | 0.612 0.613 0.611 | **BEHIND** (0.612) | 23.7 / 23.7 / 23.8 | BEHIND (0.604) | 1536 / 1536 / 1536 |

**Cell i.**

| model | depth | goinfer tok/s | Ollama tok/s | r pairs vs Ollama | outcome (median r) | llama.cpp tok/s | vs llama.cpp (median r) | tokens returned (goinfer / Ollama / llama.cpp, of 1536) |
|---|---|---|---|---|---|---|---|---|
| 0.5B | 128 | 109.4 / 109.2 / 108.7 | 125.2 / 93.7 / 93.1 | 0.874 1.166 1.168 | **AMBIGUOUS-LOW** (1.166) | 129.9 / 129.9 / 128.8 | BEHIND (0.842) | 1536 / 1536 / 1536 |
| 1.5B | 128 | 49.5 / 49.9 / 49.8 | 61.8 / 60.3 / 62.0 | 0.801 0.826 0.804 | **VOID** (0.804) | 57.4 / 61.1 / 62.8 | VOID (0.815) | 1392 / 1536 / 1536 |

**Cell h** (TTFT; r = Ollama TTFT ÷ goinfer TTFT, one pair per unique-prefix request).

| K | goinfer TTFT ms (6 requests) | Ollama TTFT ms | r pairs | outcome (median r) | spread goinfer / Ollama | goinfer `--exact-prefill` median |
|---|---|---|---|---|---|---|
| 512 | 1581 1579 1639 1444 1417 1654 | 599 592 586 591 591 586 | 0.379 0.375 0.358 0.409 0.417 0.354 | **AMBIGUOUS-LOW** (0.377) | 15.2% / 2.1% | 6212 ms |
| 3900 | 17695 17515 17504 17424 17524 17441 | 4188 4180 4183 4184 4202 4182 | 0.237 0.239 0.239 0.240 0.240 0.240 | **BEHIND** (0.239) | 1.6% / 0.5% | 62574 ms |

**The four VOID cells are early stops, and a re-run would reproduce them.** Validity gate 2 needs ≥ 95% of the
requested tokens (1536 = 3 runs × 8 completions × 64). At temperature 0 an engine that ends its answer before 64 tokens
does so at the same token every time: goinfer's 1.5B returned 1392 = 24 × **58** tokens on both Metal and CPU, and at
7B Ollama returned 1248 = 24 × 52 and llama.cpp 888 = 24 × 37. Each engine ends the reply at a slightly different
point on the same weights; each is under the bar. Re-running the same protocol would give the same counts, so these
cells are **reported as VOID, not re-run**, and their ratios above are recorded but not graded. Nothing in Part 1 was
changed for them.

**Two spread caps.** At h K=512, goinfer's six TTFTs span 1417–1654 ms (15.2%), so the cell is capped at AMBIGUOUS
although every pair is below 0.97 (0.354–0.417). At i 0.5B, Ollama's CPU runs fell from 125.2 to 93.7 and 93.1 tok/s
(30.8%), which caps the cell at AMBIGUOUS-LOW and is why one pair reads 0.874 and two read 1.17.

**Reading h against the 2026-09-18 row.** That row (`benchmarks.md` §A, "Metal prefill … re-measured 2026-09-18") put
Ollama 3.47× ahead at K=3900; here r = 0.239, i.e. 4.18×. **goinfer did not move** — its fast path read 225.0 TTFT
tok/s then and 223.8 now — **the peer did**: the 2026-09-18 run set `OLLAMA_KV_CACHE_TYPE=q8_0`, and this
pre-registered run uses Ollama's defaults (f16 KV), under which its TTFT rate at K=3900 is 941.7 tok/s against 781.8.
Two sessions, so the attribution is a reading, not a paired result.

*STEP 3 is nobara-pc's; nothing above is the claim.*

### Post-run findings (2026-09-25, after STEP 3) — no grade above is changed

Chasing the early-stop voids turned up three things. They are recorded here instead of re-grading
anything, because Part 1's outcomes stand as registered.

1. **The early stops are noise, not engine behaviour.** On the `"Continue this text. the the …"` prompt,
   a reply's length depends on a near-tie and on prompt-cache state:
   - goinfer's gemma3-1b @128 stopped at 34 tokens on int4 and at 54 on f32, where f32 writes the same
     reply llama.cpp does.
   - llama-server's gemma3-1b stopped at 51 tokens cold and ran to 64 on every warm repeat. This
     sweep times only warm repeats, which is why llama.cpp showed 64 there.

   So **the gemma3-1b lead in cell b is closed: not a goinfer template or stop-token bug.** The
   harness now uses a prompt that runs to 64 tokens on all three engines, cold and warm, and applies
   a proportional token gate (`ff196886`; `benchmarks.md` Methodology, "A reply that ends early").
   The next pre-registration should adopt that gate in place of this one's 95%.
2. **goinfer had no Phi-3 chat template.** Every goinfer phi3-mini cell here (b @128 and @3900, and
   both f cells) decoded a **raw completion**, while Ollama and llama.cpp decoded a chat prompt. The
   prompts differed by the template tokens, and goinfer's replies were newline runs. Fixed in
   `ff196886`, together with the tokenizer's missing added-token rstrip, which Phi-3's turn markers
   need.
3. **goinfer's quantized phi3-mini output is junk** (int4, int8int8, int4mix, on CPU and CUDA), even
   with the template. goinfer's f32 forward matches Hugging Face exactly (cosine 1.000000 at every
   position). The cause is the per-row int8 activation scale, which Phi-3's activation outliers
   (max/rms ≈ 90) flush to zero (open: `queue-engineering.md` H2).

Together, 2 and 3 mean **the phi3-mini rows in b and f are not like-for-like**, even though each
passed every Part 1 gate. Per-token decode cost does not depend on which tokens are generated, so
the ratios may well hold, but no claim should rest on them until they are re-measured after H2.

## Part 3 — the claim

*(STEP 3, written by nobara-pc on 2026-09-25 after both halves were in main: g–i at `558c6cad`, a–f at `02d86285`.
Every sentence below uses the wording Part 1 fixed for its outcome.)*

### The claim

**The unqualified claim, "goinfer is as fast as Ollama or faster", is not earned.** Part 1 allowed it only if every
graded cell in a, b, d, e, f, g, h and i came out AHEAD, LEVEL or AMBIGUOUS-HIGH, and e, g, h and i did not. Only one
family holds in full, D, so under Part 1's rule it is the headline: **on an RTX 2070 SUPER, goinfer's time to first
token on qwen2.5-coder 1.5B is 4.99× / 1.02× Ollama v0.32.5's at 512 / 3900 prompt tokens (>1 = goinfer faster), and
both cells are AMBIGUOUS-HIGH: level, never "ahead", because Ollama's own run-to-run spread caps them.** This is not a
prefill-throughput claim. Next to it, family A holds in its partial form. On the same card, goinfer's greedy decode is
level with or ahead of Ollama v0.32.5 on:
- qwen2.5-coder 0.5B at depth 128, 3900 and 8000;
- qwen2.5-coder 1.5B at 128, 2048, 3900 and 8000;
- qwen2.5-7B at 2048, 3900 and 8000.

It is 1.05–1.30× where ahead, and 0.99× (level) on the 7B at 8000. It is behind at none of the twelve cells, and two
are void, because Ollama ended its reply early: 0.5B @2048 and 7B @128, raw 1.13× and 1.10×. The CUDA controls (B)
are ahead on gemma3-1b at 3900 (1.26×) and on phi3-mini at 128 (1.14×). gemma3-1b at 128 and phi3-mini at 3900 are
void. phi3-mini at 3900 is where flash-decode declines: it reads 0.80× raw against Ollama and is BEHIND llama.cpp
(0.79×). Neither control widens A to "windowed models" or to "models where flash-decode declines". *(Every phi3-mini
reading in this paragraph is provisional: see "Post-run findings" above.)* For sampled decode
(F) at depth 128:
- temperature 0.8 with top_p 0.95: ahead on the 0.5B (1.19×) and level on phi3-mini (1.03×, AMBIGUOUS-HIGH);
- temperature 1.0: ahead on phi3-mini (1.14×); the 0.5B cell is void.

Gemma 4 26B-A4B on an 8 GB card (C): goinfer, keeping every expert on the GPU (host↔VRAM streaming), decodes at
1.82× Ollama's rate with its CPU offload (AHEAD). The checkpoints differ (int4 `.giw` against Q4_K_M), so this is an
architecture comparison. Where the claim does not hold:
- **CPU, Ryzen 7 3700X (E):** goinfer's greedy decode is behind Ollama v0.32.5's, 0.80× at 0.5B; the 1.5B and 7B are
  void (raw 0.80× / 0.84×).
- **M1 Pro with Metal (G):** goinfer is ahead at 0.5B depth 128 (1.18×) and behind at every 2048 and 3900 cell
  (0.58–0.75× across 0.5B / 1.5B / 7B); 1.5B and 7B at 128 are void.
- **M1 Pro Metal time to first token (H), 1.5B:** 0.38× at K=512 (AMBIGUOUS-LOW) and 0.24× at K=3900 (BEHIND).
- **M1 Pro CPU decode (I):** AMBIGUOUS-LOW on the 0.5B and void on the 1.5B.

Against llama.cpp, read by its own pairs:
- **RTX 2070 SUPER:** goinfer is ahead on the 1.5B at 128 and 2048 (1.11× / 1.04×) and level at 3900. It is level on
  the 7B at 2048 and 3900. It is behind on the 0.5B at every graded depth (0.83–0.92×) and at 8000 on the 1.5B and 7B
  (0.95× / 0.94×).
- **M1 Pro:** goinfer is ahead only on the 0.5B at depth 128.
- **CPU, both machines:** goinfer is behind.

Each sentence covers only the cells it names: these models at q4_K_M (int4 on goinfer's side of the 26B), these
depths, Ollama v0.32.5, llama.cpp `427291b` on nobara-pc and `c1d0e7a00` on the Mac, driver `595.91.07`, macOS 26.6.2.
Nothing is extrapolated to other sizes, families, quantizations, hardware or peer versions.

### Features: where each has what the other lacks

Peer cells are from `benchmarks.md` Table 1, whose Ollama column was verified against Ollama's repo and docs on
2026-06-10 and not re-verified for this page; goinfer cells are from the tree at `411e7fc4`.

| area | goinfer has, Ollama lacks | Ollama has, goinfer lacks |
|---|---|---|
| **Model library and registry** | — | `ollama pull <name>` from Ollama's own library of models that are already quantized and templated. goinfer's `pull` fetches `hf:owner/repo[:quant\|:file.gguf]` references and a few curated `demo:` tiers, sha256-checked, and has no registry of its own |
| **Concurrent models and parallel requests** | — | parallel request slots through `llama-server` (Table 1: "~ parallel slots"), and several loaded models. goinfer is batch-1 by design: one decode worker per model behind a bounded queue, and no continuous batching |
| **AMD GPUs (ROCm) and GPU breadth** | a WebGPU backend, which Ollama has no build of (§B8 ⁱ) | CUDA, **ROCm**, Vulkan and Metal. goinfer has CUDA, Metal and WebGPU, and no ROCm backend. WebGPU is its only route to an AMD GPU, and no AMD GPU has been measured |
| **Model coverage** | — | broad coverage, against goinfer's 36 architectures |
| **Multimodal** | — | broad vision support. goinfer has vision input only (Gemma 3, Qwen2.5-VL) and no audio |
| **Native dependencies** | pure Go, `CGO_ENABLED=0`, one static binary. Ollama spawns a native `llama-server` alongside its own binary (Table 1 ᵇ) | — |
| **Model inside the binary** | a `.giw` mapped from the executable image | — |
| **Correctness contract** | a Hugging Face logit-parity gate per family, and bit-identical decode | — |
| **Checkpoint formats** | loads safetensors, GPTQ, AWQ, GGUF and `.giw` directly. Ollama's runner reads GGUF (plus MLX); its importers are not checked here | — |
| **A MoE larger than VRAM** | runs every expert on the GPU, streamed host↔VRAM (the architecture behind cell c's 1.82×) | offloads layers to the CPU instead. Different, not missing: both run the 26B on 8 GB |
| **Cold start** | download, pull and first reply in 25 s against Ollama's 33 s on the M1 Pro, with no daemon (`cold-user-2026-09-06.md`, scenario E) | — |

The two rows the brief named as Ollama's strengths are its real advantages: the model library, and serving several
models and parallel requests. goinfer has no answer to either, by design (single-user, batch-1). The ROCm row is a
plain gap.
