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

## Results

Run: `run-nobara.sh` from `~/goinfer-bench/peer-notrunc-2026-10-17` on nobara-pc. Results to follow.
