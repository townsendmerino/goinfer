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
| a | | | | | |
| b | | | | | |
| c | | | | | |
| d | | | | | — |
| e | | | | | |
| f | | | | | |
| g | *(Mac)* | | | | |
| h | *(Mac)* | | | | |
| i | *(Mac)* | | | | |

## Part 3 — the claim

*(STEP 3, written by nobara-pc after both halves are in main.)*
