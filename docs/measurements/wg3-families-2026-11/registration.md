# WG3 families: resident on WebGPU against the CPU path each replaces (registered 2026-11-23)

**Status: REGISTERED 2026-11-23 11:20 PST, before any timed run.** Campaign: goinfer-notes `tasks/task-webgpu-2026-11.md`,
rows WG3-pair, WG3-temp and WG3-coh; WG3-phi has its own record (`../wg3-phi-2026-11/registration.md`), whose method this
one repeats. Results go in `README.md` beside this file.

**Question.** On nobara's RTX 2070 SUPER (WebGPU over Vulkan), does each family, now resident at `-backend webgpu`, decode
faster than the path the same flags took before (the CPU under `-backend webgpu`), and what does it cost in time to first
token, now that its prompt runs token by token (the batched prefill declines on each)?

| family | row | checkpoint (`~/models`) | head build |
|---|---|---|---|
| GLM-OCR (text) | WG3-pair | `glm-ocr` | the WG3-pair branch head |
| Ministral 3 | WG3-temp | `ministral3-3b-bf16` | the WG3-temp branch head |
| Command-R7B | WG3-coh | `command-r7b` | the WG3-coh branch head |

**Arms.** Per family, two serve builds, `CGO_ENABLED=1 -tags gpu ./backend/webgpu/cmd/serve`: `base` at the family's branch
merge base and `head` at its branch head, each from a clean detached worktree. Both run `-backend webgpu -quant int4
-ctx 4096`. The fit check requires `head`'s decode path to be WebGPU-resident and `base`'s to be off the resident; either
wrong ends that family (exit 75).

**Cells and instruments** (`run-nobara.sh`, `bench.py`):
- decode @128: `scripts/bench_peer.py`, engines `goinfer` (head) and `goinfer_old` (base), `BENCH_ABBA=2`, 8 completions ×
  64 tokens × 2 runs a cell: 4 pairs.
- TTFT @1024: WG0's driver (`../webgpu-baseline-2026-11/wg0_ttft.py`) with these flags, A B B A × 2 = 4 pairs, 6 timed
  requests a cell, median.
- Prompts: each family's depth-128 and depth-1024 prompts are calibrated first by `scripts/bench_prompts_calibrate.py`
  (on the base build) into a copy of `scripts/prompts.json` beside the results (`BENCH_PROMPTS`), never into the shared file.
- Conditions: the exclusive timing lock; WG0's page-cache eviction, settle and void rule; the `kwin_wayland` context recorded.

**Metric and bars** (pair ratio = head ÷ base for decode tok/s, base ÷ head for TTFT, so > 1 favours head), per family:
- decode: **win** if the median pair ratio ≥ 1.05 and all 4 pairs > 1; **loss** if the median < 1.0; else **ambiguous**.
- TTFT: **no regression** if the median pair ratio ≥ 0.8; **regression** below.

**Stopping rule.** One session, the 4 pairs a cell; no extension. A void re-runs in place, three in a row stop the cell.

**What each outcome means.** Decode win and no TTFT regression: that family's PR is armed. Decode win with a TTFT
regression: the PR stays unarmed and the trade goes to Francis (DECISION). Decode loss or ambiguous: parked, to Francis
with the numbers.
