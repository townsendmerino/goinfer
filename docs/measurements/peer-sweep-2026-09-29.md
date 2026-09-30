# Peer sweep, re-run at one commit (nobara-pc and MacBook, 2026-09-29)

**Status: nobara-pc half (cells a–f) RUN 2026-09-29 22:09–23:38 PDT and graded 2026-09-30, below. The MacBook half (g–i) has not
run** (the Mac's night queue was not started). Pre-registered 2026-09-29, before any timed run; the text above *Results* is unchanged,
and a bar that turns out wrong gets a dated amendment with its mechanism, not an edit.

**Question.** What does the peer table say now? [`peer-claim-2026-09-25.md`](peer-claim-2026-09-25.md) is the last full
sweep. It cannot be the table any longer:
1. **Many of its cells are VOID** (a: 2, b: 2, e: 2, f: 1, g: 2, i: 1), most from a reply that stopped early. That was a
   prompt and gate problem, fixed in `ff196886` (`benchmarks.md` Methodology, "A reply that ends early").
2. **Its Phi-3 rows are not like-for-like** (no chat template; a quantized forward that was junk). Both are fixed:
   per-32 activation scales, and `--quant q4k` as the CPU/CUDA `.gguf` default.
3. **The code under every cell has moved:** Metal decode kernels (R18b, 09-26), the CPU decode work (09-27 and 09-28),
   `--embed-int4` default ON (09-28), weights format v15, the CUDA batched head, and more. A table whose cells come from
   different builds is not a table, so the whole grid is re-run at one commit.

**What is unchanged from 09-25 Part 1, and is inherited by reference, not restated:** the nine cells (a–i) and their models
and points; the ratio r (goinfer ÷ peer for decode, peer ÷ goinfer for TTFT); the bars (level 0.97–1.03, ahead > 1.03, behind
< 0.97); every outcome label, decided by all pairs and never by the mean; the 5% spread cap; 3 pairs per decode cell and 6 per
prefill cell; the same-weights gate (`gguf_same_weights.py`, weights read from `~/models`); the idle-box start; servers restarted
between cells with the engines interleaved; contexts (depths 128, 2048, 3900 at the harness default, 8000 at `BENCH_CTX=8192`,
the 26B at `BENCH_CTX=2048`); Ollama v0.32.5 at its defaults; the peers' versions (below).

**What changes, fixed now:**

| # | change | why |
|---|---|---|
| 1 | The token gate is the harness's proportional one (`token_gate()`: a cell is VOID if any completion returns fewer than max(10, half the requested tokens) or the engine reports no count; a shorter reply above that is graded and its counts reported). It replaces 09-25's 95%. | 09-25's own post-run finding 1 asked the next pre-registration to adopt it |
| 2 | **New validity gate: the decode path.** Every goinfer cell's serve log is kept beside the results and the decode path its load resolved to is recorded in the cell. A GPU cell whose decode path is the CPU's is VOID, and a load that fails is recorded as goinfer declining that model on that machine, not worked around. The TTFT harness additionally passes `-require-backend`, since its number is the prefill; the decode harness does not, because that flag also refuses a load whose *prefill* declines to the sequential path (Phi-3 on CUDA at `q4k` does, about 9× slower TTFT), which does not affect a decode-only rate. | the harness discarded serve's output, so a silent CPU fallback timed the CPU and looked plausible (found on the Mac 2026-09-29, task-actquant-pergroup) |
| 3 | **Phi-3 runs at what a user gets:** no `-quant` flag (`BENCH_QUANT_OVERRIDE` `phi3-mini=default`), i.e. `--quant q4k` on the CPU and CUDA `.gguf` paths. Ollama's tag is `p3m-local` on both boxes (195/195 tensors identical). | the 09-25 rows compared different work |
| 4 | **Mac Metal cells (g and h) run with `--embed-int4=false`**, disclosed on their rows. | the default `--embed-int4` makes a Metal load decline residency and run on the CPU (`docs/quantization.md`, "Known issue"); with change 2 that would void every Metal cell. **If the issue is fixed before the run, the flag is dropped and this is amended.** Cell i (CPU) and every nobara cell run at the shipped default. |
| 5 | Binaries are built from the pinned commit's worktree, never a dirty tree, and the harness runs from the same worktree. | the two boxes' checkouts carry other sessions' edits |

**Pinned commit:** `754f12d3` (the harness change above; the serve code is identical to `dc0699d4` and later). Peers: nobara
Ollama v0.32.5 at `~/ollama-0325`, llama-server `0.4.0-dev (build 1, commit 427291b)`; Mac Ollama v0.32.5, llama-server
`0.3.0 (build 10621, commit c1d0e7a00)`. Driver `595.91.07` on nobara-pc; a different driver stops the sweep.

**Decision rule.** Nothing is decided by this run. Its cells replace 09-25's in the peer table and the site's `claims.json`,
each row stating this commit, with the outcome labels above; a cell that is VOID stays VOID and is reported. Nothing is
re-graded to look better, and a loss stays a loss (the sweep's 09-25 result had six).

**Cost.** nobara-pc: est 90 min (09-25 took 81 min with llama.cpp: 2836 + 527 + 455 + 464 + 491 + 59 s). MacBook: est 90 min
(09-25: g 29, i 24, h 21 min). By night only, one timed run per box.

**A property to read the Phi-3 CUDA rows with:** at its `q4k` default the batched prefill declines, so a Phi-3 request's prefill is
sequential. The decode rate is timed from the first token and is unaffected; the cells take longer, and no TTFT claim is made for Phi-3.

**Not in this sweep:** the Phi-3 Mac question (own pre-registration, `task-actquant-pergroup-2026-09.md`); WebGPU; batching or
speculation cells (`task-concurrency-2026-09.md`); cold-start and load-time (`sidecar-v15-2026-09-29.md`).

## Results — nobara-pc, cells a–f (graded 2026-09-30)

**Run.** Night queue, 2026-09-29 22:09:57–23:38:19 PDT, all six steps rc=0 (3054 + 518 + 501 + 427 + 524 + 55 s). Binaries built
from a worktree at `754f12d3`, `goinfer_tree_dirty: False` in every results file, NVIDIA driver 595.91.07, Nobara Linux 44, aikit
`d7436e6`, Ollama v0.32.5 at its defaults, llama-server `0.4.0-dev` build 427291b. Every goinfer GPU cell's decode path is recorded
and was the GPU's (`cuda-resident (int4)`, Phi-3 `cuda-resident (q4k)`), so the decode-path gate voided nothing. Graded with
[`grade.py`](peer-sweep-2026-09-29/grade.py): the 09-25 grader with only the two changes this page registers (the proportional token
gate; the decode-path gate). Raw: [`nobara/`](peer-sweep-2026-09-29/nobara/), graded rows in
[`nobara/grade.jsonl`](peer-sweep-2026-09-29/nobara/grade.jsonl), per-cell serve logs in `nobara/serve-logs/`.

**Against Ollama** (r = goinfer ÷ Ollama for decode, Ollama ÷ goinfer for TTFT; 3 pairs per decode cell, 6 per TTFT cell; the median
r is shown, the outcome is decided by every pair):

| cell | what | outcome | median r | 09-25 |
|---|---|---|---|---|
| a | CUDA greedy, 0.5B @128 / 2048 / 3900 / 8000 | **AHEAD** ×4 | 1.396 / 1.23 / 1.268 / 1.335 | 1.272 / void / 1.167 / 1.153 |
| a | CUDA greedy, 1.5B @128 / 2048 / 3900 / 8000 | **AHEAD** ×4 | 1.474 / 1.342 / 1.288 / 1.196 | 1.297 / 1.269 / 1.229 / 1.130 |
| a | CUDA greedy, 7B @128 / 2048 / 3900 / 8000 | **AHEAD** ×4 | 1.184 / 1.13 / 1.099 / 1.045 | void / 1.076 / 1.052 / 0.986 (LEVEL) |
| b | CUDA greedy, gemma3-1b @128 / 3900 | **AHEAD** ×2 | 1.385 / 1.364 | void / 1.262 |
| b | CUDA greedy, phi3-mini @128 / 3900 | **BEHIND** ×2 | 0.895 / 0.718 | 1.138 / void — not like-for-like then |
| c | Gemma 4 26B-A4B on 8 GB, ctx 2048 — architecture comparison | **AMBIGUOUS-HIGH** (spread cap: goinfer's runs 39.4 / 39.3 / 37.3, 5.3%) | 1.763 | AHEAD 1.815 |
| d | CUDA TTFT, 1.5B, K = 512 / 3900 | **AMBIGUOUS-HIGH** / **AMBIGUOUS-LOW** (spread caps: Ollama 17.9% / 6.6%) | 4.287 / 0.974 | AMBIG-HIGH 4.994 / AMBIG-HIGH 1.019 |
| e | CPU greedy, 0.5B / 1.5B / 7B @128 | 0.5B **BEHIND** · 1.5B **LEVEL** · 7B **AHEAD** | 0.905 / 1.01 / 1.043 | 0.797 / void / void |
| f | CUDA sampled @128, 0.5B, t=0.8 top-p 0.95 / t=1.0 | **BEHIND** / **VOID** (goinfer: one reply ended at 12 of 64 tokens, under the gate's 32) | 0.922 / (1.381) | AHEAD 1.194 / void |
| f | CUDA sampled @128, phi3-mini, t=0.8 top-p 0.95 / t=1.0 | **BEHIND** ×2 | 0.889 / 0.896 | AMBIG-HIGH 1.027 / AHEAD 1.138 — not like-for-like then |

**What it says, in order of weight.**
1. **Dense CUDA greedy decode: AHEAD of Ollama in all 12 cells**, 1.045× to 1.474×, none void (09-25: 9 ahead, 1 level, 2 void). The
   7B at 8000 moved from LEVEL to AHEAD.
2. **x86 CPU decode is no longer uniformly behind:** 0.5B still BEHIND (0.905, up from 0.797), the 1.5B LEVEL (1.01) and the 7B AHEAD
   (1.043), both previously void.
3. **Phi-3 is BEHIND, and this is the first like-for-like Phi-3 row.** At its real default (`q4k`, per-32 activations, a chat template)
   goinfer decodes 112.6 tok/s at 128 against Ollama's 125.8, and 53.4 against 74.4 at 3900 (0.718). The 09-25 "AHEAD 1.138" compared a
   raw completion on junk-quantized output and is withdrawn by this row. The Phi-3 prefill declines to the sequential path at `q4k`
   (disclosed before the run), which this decode-only rate does not include.
4. **A likely regression, found here: top-p sampling on CUDA.** goinfer's 0.5B at t=0.8 / top-p 0.95 decoded 239.3–242.9 tok/s, against
   318.4–318.5 at 09-25's `411e7fc4` (−24%), while greedy (362–371) and untruncated t=1.0 (359–363) got faster. So the cost is in the
   top-p path, somewhere in `411e7fc4..754f12d3`. Phi-3's top-p cell shows no such drop (111.7 vs 112.6 greedy), which points at a
   per-token cost that matters at 0.5B speed and hides at Phi-3's. **Not diagnosed**; owed: a goinfer-vs-goinfer A/B at the two commits
   (`BENCH_ENGINES=goinfer,goinfer_old`, one cell, by day it is minutes), then a bisect.
5. **The 26B cell dropped from AHEAD to AMBIGUOUS-HIGH on goinfer's own spread**, not on the ratio: its third run read 37.3 against
   39.4 / 39.3 (5.3% > the 5% cap). Every pair is still above 1.67. The median rate is 39.3 tok/s (09-25: 40.2).
6. **TTFT** is unchanged in substance: far ahead at K=512 (96 against 430 ms, capped by Ollama's own 17.9% spread), level at K=3900
   (965 against 954 ms), now capped LOW by Ollama's 6.6% spread.

**Against llama.cpp** (reported beside, as 09-25 did; the claims name Ollama): CUDA 1.5B ahead at 128–3900 (1.079–1.191) and level at
8000; 7B ahead at 128 / 2048 (1.093 / 1.052), AMBIGUOUS-HIGH at 3900 (1.029), level at 8000 (0.974); **0.5B behind at 2048 / 3900
(0.894 / 0.926)** and level at 128 / 8000; gemma3-1b level / ambiguous-low; Phi-3 behind (0.713–0.884); the 26B AMBIGUOUS-HIGH 1.415;
CPU behind at 0.5B / 1.5B (0.809 / 0.908), level at 7B (0.979); sampled top-p 0.5B behind 0.65.

**What moves because of this (per the rule above), and what waits.** The CUDA and CPU rows of the peer table and of the site's
`claims.json` are replaced by these cells. The Metal rows wait for the Mac half (g–i), which is re-queued. Nothing is re-graded to
look better: the Phi-3 losses, the CPU 0.5B loss, the llama.cpp losses and the top-p regression stay as measured.
