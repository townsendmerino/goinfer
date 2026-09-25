# Peer claim sweep — goinfer v0.20.0-candidate vs Ollama (2026-09)

Task: run a same-session peer sweep of the goinfer v0.20.0 candidate against Ollama, to find
out which comparative claim we can publish. The candidate claim is "goinfer is as fast as
Ollama or faster". We might not earn it. The deliverable is the claim the numbers support,
whatever that turns out to be.

Two machines share this brief. Read the ROLES section and do only your share.

Read first: CLAUDE.md (Benchmarking, Measurement discipline), docs/benchmarks.md
(Methodology, Model storage, the §B2 re-anchor box, the summary table at the top),
docs/tasks/task-peer-benchmarks.md, and scripts/bench_peer.py's header.

## ROLES

- **nobara-pc session.** You own STEP 0 for ALL cells, including the Mac's cells g–i. Write
  the pre-registration, then commit and push it before any timed run. Then do cells a–f.
  Leave cells g–i blank in the doc; the Mac session fills them. Do STEP 3 only after the
  Mac's results are in main.
- **MacBook session.** Wait until nobara-pc has pushed the pre-registration. Then git pull
  and read docs/measurements/peer-claim-2026-09-<dd>.md; do not change its bars. Run cells
  g–i only. Fill them into that doc and into the Mac rows of docs/benchmarks.md, then rebase
  and push. Skip STEP 3.

## STEP 0 — pre-register before you time anything (nobara-pc)

Commit docs/measurements/peer-claim-2026-09-<dd>.md containing:

- Each cell below, with the ratio that counts as "level" (0.97–1.03×), "ahead" (>1.03×) or
  "behind" (<0.97×), and an explicit ambiguous band. A cell whose pairs straddle its bar is
  ambiguous, not a win.
- The claim wording for each outcome, written now. For example, "on NVIDIA, level or ahead
  on decode at every depth measured" is allowed only if EVERY CUDA decode cell comes out
  level or ahead. Otherwise the narrower wording names the cells that failed.
- The rule that a claim covers only the cells measured. Nothing is extrapolated to other
  model sizes, families or hardware.

Commit this before the first timed run and do not edit it afterwards. If a bar turns out to
be wrong, add a dated amendment below it that gives the mechanism.

## STEP 1 — cells

Use scripts/bench_peer.py only, NOT scripts/bench_compare.sh.

- Run interleaved, cell by cell, with a server restart between cells.
- Verify the same weights on both sides with scripts/gguf_same_weights.py.
- Every checkpoint must come from ~/models. A path under /srv/models or /Volumes voids the row.
- Stamp provenance.

**nobara-pc cells.** RTX 2070 SUPER on the driver 595.91.07 anchor. If the driver has
changed, stop and tell me: that is a re-anchor, not a carry-forward. Peer is Ollama v0.32.5
at ~/ollama-0325.

- a. **CUDA decode:** qwen2.5-coder 0.5B / 1.5B / 7B at depths 128 / 2048 / 3900 / 8000.
  This is the first peer measurement with flash-decode on by default. The published depth
  rows (0.71–0.95×) predate it, and this cell replaces them.
- b. **CUDA decode controls:** gemma3-1b and phi3-mini at depths 128 / 3900. gemma3-1b is
  the windowed-model control; phi3-mini is the case where flash-decode declines.
- c. **CUDA 26B MoE:** gemma4-26B-A4B at ctx 2048, goinfer C′ with this release's DMA overlap
  against Ollama's CPU offload. Report it as an architecture comparison, not like-for-like.
- d. **CUDA prefill time to first token:** 1.5B at K = 512 / 3900.
- e. **CPU amd64 decode:** 0.5B / 1.5B / 7B at depth 128. Expect about 0.82× and confirm it.
- f. **Sampled decode:** temperature 1.0, and temperature 0.8 with top_p 0.95, on the 0.5B and
  phi3-mini. This release changed both paths.

**MacBook cells.** M1 Pro, separate session, same protocol.

- g. **Metal decode:** 0.5B / 1.5B / 7B at depths 128 / 2048 / 4000.
- h. **Metal prefill time to first token:** K = 512 / 3900.
- i. **CPU decode:** 0.5B / 1.5B at depth 128.

Wherever bench_peer.py already supports it, add llama.cpp (llama-server) as a third column,
so every ratio can be read against a second peer.

## STEP 2 — record (each machine, its own cells)

- Fill the pre-registered doc with paired ratios, pair counts, spread, and each cell's
  outcome against its bar.
- Update the summary table at the top of docs/benchmarks.md, plus every row these cells
  supersede.
- Grep each superseded figure WITH its unit, so no page still quotes the old number.
- Commit the raw results files, gzipping anything over 1 MB.

## STEP 3 — the claim (nobara-pc, after both halves are in main)

At the end of the measurement doc, add two things:

- **The claim paragraph.** One paragraph with the comparative claim the pre-registration
  allows, going no wider than the cells that passed.
- **A features-parity table.** List where each engine has something the other lacks,
  including Ollama's strengths: the model library and registry, concurrent models and
  parallel requests, and AMD/ROCm support.

## Practicalities

- Run long jobs detached: setsid nohup on nobara-pc; on the Mac, launchctl submit … followed
  by ; launchctl remove.
- Keep logs somewhere durable, not /tmp.
- Commit in increments.
- Run the citation lint and check its exit code directly.
- Stop and ask me if the box won't go idle, the driver or Ollama version has changed, or a
  same-weights check fails.
