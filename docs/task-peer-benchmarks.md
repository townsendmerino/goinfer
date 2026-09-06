# Task: the peer matrix, redone — what goinfer is measured against, on what, and how (2026-09-03)

> **BLUF.** The last peer rows were taken before prefix reuse, the W4A8 tile, the NEON attention
> kernels, C′ expert staging with CUDA graphs, Metal streaming at N=64, KV-only prefill, the DFlash
> drafter and the constrained-decode fix — and before llama.cpp 0.2.0's `--fit`/`-ncmoe`, Ollama's
> restore points, FreeToken and the current MLX. They were also almost all one number, decode at
> depth 128, which is the number the audience can least feel. This doc scopes a matrix that asks
> the questions a user of each mode would ask: **six models × two boxes × the peers people actually
> run on that box × eight workloads**, with a **fidelity column beside every speed cell**
> (teacher-forced agreement with an fp16 reference) so a faster row at a worse quant cannot pass as
> a win, and **one pass@1 row per model cell** for the quality a reader actually asks about. Tier 1
> is ~100 cells and two quiet days; tier 2 is everything with its own reason to exist. Method is `docs/benchmarks.md`'s, unchanged.
>
> **Status: SCOPED 2026-09-03, nothing run.** goinfer `66288c2`, aikit v1.34.0. Peer versions are
> pinned on the day each box runs, in the header record `bench_peer.py` already writes.

## 0. What must not change

- **`docs/benchmarks.md` § Methodology is the authority.** A number enters a table only with
  machine, checkpoint+quant, greedy/seed, pinned versions, date, thermal note and local-disk path.
  `~/models` only — never `/srv/models`, never `/Volumes/` (`CLAUDE.md`). CUDA rows are anchored
  to the driver version.
- **Same session, interleaved, rotating arm order, n = 5, spreads reported.** Drift between
  sessions is ~3.5% on the Linux box; a ratio across sessions is not a ratio.
- **Off is a competitor** wherever a feature is on (speculation, constrained output, prefix reuse).
- **One method for every engine.** Every engine is driven over its own HTTP server and timed
  client-side from the streaming timestamps; an engine's self-reported numbers are a secondary
  column, never the primary. `scripts/bench_peer.py` already does this for goinfer and Ollama.
- **Retractions reach every page** — this matrix supersedes the older peer rows in
  `docs/benchmarks.md`, `docs/ollama-chase.md` and the Ollama Chase Ledger artifact; each is
  repointed, not left quoting the old figure.

## 1. Peers

| box | peer | why it is in | mode(s) |
|---|---|---|---|
| both | **Ollama** | the incumbent every prior row was against; the anchor to the old numbers | default; CPU-forced (`num_gpu 0`) for the CPU lane |
| both | **llama.cpp** (`llama-server`) | Ollama's engine without Ollama's overhead — separates the two; and the `--fit on` / `-ncmoe` MoE-offload modes the fit doc borrows from | default; `--fit on`; `-ncmoe N` on the MoE cells; `-ngl 0` for the CPU lane |
| Mac | **MLX** (`mlx_lm.server`) | what Mac users compare against; never in a row yet | its own 4-bit quant (a different quant — see §4) |
| Mac, optional | Zeno, turbo-fieldfare | the closed/Swift references for the 35B-on-16-GB cell | only if installed on the day; vendor numbers are not rows |
| Linux | **FreeToken** | hybrid CPU/GPU expert execution — the technique goinfer has not built (audit L-01), on the exact cell the audience knows its number for | its supported checkpoint/quant for each MoE cell |
| both, CPU lane | **go-llama** (goccy) | the Go-lane comparison | one cell, stated as single-threaded in the row |

Left out on purpose: vLLM and ExLlama (an 8 GB Turing card and a 16 GB Mac are not their regime; a
row nobody's hardware matches is noise), LM Studio (a GUI over llama.cpp/MLX — the engines are
already in). If a peer's version changes mid-matrix, its rows restart.

## 2. Models — six cells, each from one checkpoint both sides load

| cell | model | checkpoint | why |
|---|---|---|---|
| S | Qwen2.5-Coder-1.5B | GGUF q4_k_m | every internal number is in this model; the CPU lane's headline |
| D7 | Qwen2.5-7B-Instruct | GGUF q4_k_m | resident on both GPUs (int4 ≈ 4.5 GB); the harness recipe's model |
| M35 | Qwen3.6-35B-A3B | GGUF q4_k_m (kind-4 `.giw` for goinfer) | bigger than the card; Gated-DeltaNet — the resident-reuse and offload paths in one |
| M26 | Gemma-4-26B-A4B | GGUF q4_k_m | bigger than the card; attention-only MoE — the Metal streaming cell |
| G20 | gpt-oss-20b | MXFP4 | fits the Mac, not the card: a resident cell on one box and an offload cell on the other |
| H27 | Qwen3.8-27B | GGUF q4_k_m | fits neither GPU; native MTP head — the speculative cell |

MLX runs its own 4-bit conversion of each (`mlx-community`), FreeToken its own supported format;
those rows carry the quality column with extra weight (§4). goinfer runs the **shipped default**
configuration in every peer row — what a user gets with no flags — and its own variants only in
§6.

## 3. Workloads — the axis that changes

| id | workload | reports | fixture |
|---|---|---|---|
| W1 | decode at depth 128 | tok/s | the existing bench prompts — kept as the anchor to the old rows only |
| W2 | prefill at 512 and 3900 tokens | TTFT, prefill tok/s | a real source file and a real document, not the four-unique-words prompts G28 caught |
| W3 | long-context decode at depth 2k / 8k / 32k | tok/s, TTFT | one document, truncated per depth; 32k needs `-kv f16` on the card (§4) |
| W4 | **the agent turn** | turn-N TTFT, N = 1…10; turn-3+ is the headline | a 10-turn Claude Code transcript replayed through the OpenAI endpoint as strict prefix extensions (`docs/integrations/claude-code.md`'s loop) |
| W5 | speculative decode | tok/s vs the same engine with speculation off | D7-class target + its drafter: goinfer `--drafter` (DFlash pairing), llama.cpp `--model-draft`, Ollama if it has one on the day; H27's MTP head where the model runs |
| W6 | structured output | tok/s constrained vs unconstrained, same prompt | one JSON-schema extraction; goinfer `response_format`, llama.cpp `json_schema`, Ollama `format` |
| W7 | concurrency | aggregate tok/s and per-stream TTFT at 1, 2, 4 clients | W4's transcript, four independent conversations |
| W8 | embeddings | inputs/s | Qwen3-Embedding-0.6B (or nomic-embed) over 1,000 × 256-token inputs |

Every cell also records **load time to first token**, **peak RSS and VRAM** (`nvidia-smi`, `ps`,
`vm_stat`), and **the exact command line** — the zero-flag question the fit doc asks is answered
by whether the peer needed one.

**W4 is the row that did not exist before and matters most.** It is what prefix reuse (3358e6b,
R-01 phase 0, R-02, R-03), Ollama's restore points and llama.cpp's cache reuse all compete on, and
it is the number a harness user feels on every turn. The transcript fixture must be strict prefix
extensions with a tool result appended each turn, and one variant with an edited last message at
turn 6, so the non-extending case is measured too.

## 4. The normaliser: quality beside every speed cell

A tok/s row at a different quality is not a comparison. goinfer's group-32 W4A8 is not
llama.cpp's Q4_K_M even from the same file; MLX's 4-bit and FreeToken's format are further away;
and a 32k cell with f32 KV on one side and f16 on the other is two different models. Two kinds of
quality are in play and the matrix keeps them apart: **fidelity** — how faithfully an engine
reproduces the model, which is the engine's whole job and the property a wrong kernel or a wrong
reuse path damages first, fluently and silently — and **task quality**, which is a property of
model + quant and belongs beside a model cell, not a workload.

- **Fidelity, primary: teacher-forced top-1 agreement.** For each cell, 20 prompts (10 code, 10
  prose) with a ≥256-token fp16 reference continuation each. Feed every engine the prompt plus
  the reference continuation and count, position by position, how often the engine's argmax at
  that position equals the reference's next token — the engine measured without the cascade a
  greedy run has, where one near-tie early makes the rest of the text diverge. goinfer computes it
  through the batched argmax path (`PrefillLastNArgmax`), llama.cpp through `/completion` with
  `n_probs`, MLX through `mlx_lm`'s logprobs; report the agreement rate and the position of the
  first disagreement.
- **Fidelity, fallback: greedy agreement.** For an engine that cannot be teacher-forced (Ollama),
  the same 20 prompts run greedy; report the **mean agreement length** (tokens until first
  divergence) and the **exact-match rate**, and mark the cell as the strict form. Where both exist,
  both are printed.
- **The reference** is HF fp16 greedy from the existing pin scripts where the model fits the Linux
  box's RAM (S, D7); llama.cpp's f16 GGUF where it does not (M35, M26, G20, H27) — stated per cell.
  Two engines within a few points of each other are comparable; a row that wins speed and loses
  fidelity says both.
- **Task quality: one pass@1 row per model cell.** HumanEval+ and MBPP+ pass@1, greedy, through
  each engine's OpenAI endpoint, once per model cell in tier 1 — not per workload. Engines at the
  same quant should land within noise of each other; when one does not, the fidelity column has
  already said why. It is the number a reader trusts more than an agreement length, and it is the
  harness the expert-pruning experiment needs anyway. No judge-scored quality: it is noisy and
  measures nothing an engine changes.
- **KV precision is part of the cell.** Peers default to f16 KV; goinfer's default is f32. Every
  long-context cell states both, and the 32k card cell runs goinfer at `-kv f16` beside a note
  that this is not its default. No silent precision trades (the fit doc's rule).
- **Sampling is greedy with a pinned seed everywhere.** A peer that cannot pin its sampler runs
  temperature 0 and says so.

## 5. Tiers

**Tier 1 — both boxes, two quiet days.** D7, M35, M26 × {Ollama, llama.cpp default, the platform
specialist: MLX on the Mac, FreeToken on Linux} × {W1, W2 at 3900, W3 at 8k, W4}. Roughly 100
cells, plus the pass@1 row for each of the three model cells (§4). This is the matrix that answers
"how does goinfer compare today" for mode 3 and mode 4 users.

**Tier 2 — each with its own reason.** S and the CPU lane (llama.cpp `-ngl 0`, Ollama CPU,
go-llama) on both boxes — the pure-Go story; G20 on both boxes; H27 with the MTP head (W5) on the
Mac CPU path if it runs at a usable rate, else omitted with the number that says why; llama.cpp
`--fit on` and `-ncmoe` on the MoE cells against goinfer's expert cache; W3 at 2k and 32k; W5 on
D7-class with the DFlash pairing; W6; W7; W8 once R-07 lands (it did: a843e58); Zeno and
turbo-fieldfare if installed.

## 6. goinfer variants — a different axis from peers, tier 2 only

These rows show which goinfer configuration the recommendation should be, not how goinfer
compares; they double the run count wherever they appear, so they are limited to one cell each.

- `--quant int8int8` vs int4 on S and D7 (the ordering flipped with the tile; confirm on the peer
  fixtures).
- `--cpu-fast-attention` on/off on S at depth 8k (parity-class: the agreement column applies).
- `--drafter` on/off is W5's own do-nothing arm.
- **`GOEXPERIMENT=simd` — not a row today.** goinfer calls none of aikit's `simd`-gated kernels
  (`linalg/exp_simd.go` is reachable only from aikit's own benches; no `GOEXPERIMENT` appears
  anywhere in goinfer), so a build with the experiment on runs the same instructions as one
  without and would measure a null. It becomes a variant row when task-simd-audit S-06 step 2 wires
  the f32 transcendentals — a parity-class change (≤4 ULP on logits, goldens regenerate), which is
  exactly why it belongs on this axis with the agreement column beside it, never in a peer row. Add
  the arm to the harness now, gated on the build tag, so the day it changes bits it is measured on
  the same fixtures.

## 7. Harness — `scripts/bench_peer.py`, extended, not replaced

- **A peer table**: name, launch command, endpoint, model mapping per cell, version probe.
  llama-server, `mlx_lm.server` and FreeToken all speak OpenAI-compatible chat completions, so one
  client covers them; Ollama keeps its `/api/chat` driver. The header record gains the peer's
  version string and the command line per row.
- **Transcript replay (W4)**: a fixture directory of N turns; the harness sends turn k as the full
  message list, times TTFT from the first streamed content token, appends the reply and the fixture's
  tool result, and proceeds. Records per-turn TTFT, tokens reused where the engine reports it
  (goinfer's `PrefillReused`), and whether the engine's cache survived.
- **Concurrency (W7)**: `asyncio` clients over the same transcript, four conversations.
- **Fidelity (§4)**: a `--reference` run that stores the fp16 reference continuations; a
  teacher-forced scorer per engine (goinfer's batched argmax, llama-server `/completion` with
  `n_probs`, `mlx_lm` logprobs) and the greedy-agreement fallback; both score offline against the
  reference file.
- **pass@1 (§4)**: HumanEval+ / MBPP+ driven through each engine's chat endpoint at temperature 0,
  scored by the reference harness's own sandbox; one run per model cell, cached by (engine
  version, checkpoint sha256).
- **Void rules, pre-registered**: a cell is void if any arm's spread exceeds 10%, if the Mac reports
  a thermal event mid-cell, if the model path is under a forbidden root, or if a peer's version
  changed mid-matrix. Void cells are re-run, not dropped and not averaged in.

## 8. Reporting

A dated section in `docs/benchmarks.md` — "Peer matrix 2026-09" — with one table per box per
workload, columns per engine, the goinfer/peer ratio with its spread, the agreement column, and the
command lines in a collapsed block; raw cells as `docs/measurements/peer-matrix-2026-09/*.json`.
The Ollama Chase Ledger artifact is refreshed from those tables and its standings rewritten from
W1 to W4. Rows that read badly are reported at full value with their mechanism named where it is
known (FreeToken's hybrid execution on M35, MLX's fused attention at 32k) — a peer row is a
measurement, not a verdict, and the ones goinfer loses are the roadmap.

**Report the cell's wall clock separately from its tok/s, because on the deep cells they say
different things.** W3 depth-8000 on `nobara-pc` measured goinfer at 21.4 (M35) and 15.5 (M26)
tok/s against llama.cpp's 30.8 and 22.7 — a DECODE deficit — while the same cells took 1528.8 s and
384.3 s of wall clock against llama.cpp's 61.7 s and 67.0 s, which is a PREFILL deficit and moves
none of those tok/s figures. Two causes, two items: the prefill half is queue-performance **P20**
(the MoE families never take a batched pass; the dense half of it landed 2026-09-04, see
`docs/measurements/prefill-chunking-d7-2026-09-04.md`), and the decode half belongs with the
attention-at-depth work P19 already characterizes — goinfer's marginal cost per token rises with
KV depth where the peers' stays flat. Quoting either number alone attributes the whole gap to
whichever mechanism the reader already had in mind.

## 9. Sources

`docs/benchmarks.md` (Methodology, B5/B10), `scripts/bench_peer.py`, `docs/ollama-chase.md`,
`docs/task-zeno-compare.md`, `docs/task-freetoken-techniques.md`, `docs/integrations/claude-code.md`
(the loop W4 replays), `docs/measurements/cpu-peer-prefill-2026-09-01.md`, `docs/task-fit-to-hardware.md`
(the zero-flag question, the never-silently rule), `docs/audit-2026-09-02.md` (G28, L-01, L-05),
`docs/task-recompute-audit.md` (R-01…R-03, R-07), aikit `CHANGELOG.md` 1.32.0–1.34.0 and
`docs/task-simd-audit.md` (S-06).
