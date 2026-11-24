# WG3-phi: Phi-3-mini resident on WebGPU against the CPU path it replaces (registered 2026-11-23)

**Status: REGISTERED 2026-11-23 09:30 PST, before any timed run.** Campaign: goinfer-notes `tasks/task-webgpu-2026-11.md`,
row WG3-phi. Results go in `README.md` beside this file.

**Question.** On nobara's RTX 2070 SUPER (WebGPU over Vulkan), does Phi-3-mini at `-backend webgpu`, now resident with
per-32 activation scales, decode faster than the path the same flags took before (the CPU under `-backend webgpu`), and
what does it cost in time to first token, now that its prompt runs token by token (the batched prefill takes no per-32
scales)?

**Arms.** Two serve builds, `CGO_ENABLED=1 -tags gpu ./backend/webgpu/cmd/serve`: `base` at the branch's merge base and
`head` at the branch head, each from a clean detached worktree; the only source difference is this branch. Both run
`-backend webgpu -quant int8int8 -ctx 4096` on `~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf` (the loader adds
per-32 activations, `activationSafeQuant`). The fit check requires `head`'s decode path to be `webgpu:vulkan-resident` and
`base`'s to be off the resident; either wrong ends the run (exit 75).

**Cells and instruments.**
- decode @128: `scripts/bench_peer.py`, engines `goinfer` (head) and `goinfer_old` (base), `BENCH_ABBA=2`, 8 completions
  × 64 tokens × 2 runs a cell, `BENCH_QUANT_OVERRIDE=phi3-mini=int8int8` (its default is q4k, which WebGPU has no kernel
  for): 4 pairs.
- TTFT @1024: [`ttft.py`](ttft.py), WG0's driver (`webgpu-baseline-2026-11/wg0_ttft.py`) with this cell's flags, prompt
  `phi3-mini:1024`, A B B A × 2 = 4 pairs, 6 timed requests a cell, median.
- Conditions: the exclusive timing lock; WG0's page-cache eviction, settle and void rule (a cell is void if system swap
  grows); the `kwin_wayland` context recorded.

**Metric and bars** (pair ratio = head ÷ base for decode tok/s, base ÷ head for TTFT, so > 1 favours head):
- decode: **win** if the median pair ratio ≥ 1.05 and all 4 pairs > 1; **loss** if the median < 1.0; else **ambiguous**.
- TTFT: **no regression** if the median pair ratio ≥ 0.8; **regression** below.

**Stopping rule.** One session, the 4 pairs a cell; no extension. A void re-runs in place, three in a row stop the cell.

**What each outcome means.** Decode win and no TTFT regression: the PR is armed. Decode win with a TTFT regression: the PR
stays unarmed and the trade goes to Francis (DECISION). Decode loss or ambiguous: parked, to Francis with the numbers.
