# Peer sweep, re-run at one commit (nobara-pc and MacBook, 2026-09-29)

**Status: PRE-REGISTERED 2026-09-29, before any timed run.** Queued on both boxes' night queues; each result section is
added below after its run, and a bar that turns out wrong gets a dated amendment with its mechanism, not an edit.

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
