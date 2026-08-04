# Release qualification sweep — real models, every claimed cell, canonical timings

> **⚠ Peer numbers below predate the Ollama v0.32.5 re-anchor (2026-08-04).** Competitive figures
> in this doc (e.g. Ollama-CUDA ~149, Ollama-Metal 83.3, llama.cpp-CUDA 72.8, and any "×Ollama"
> multiple) were measured against **Ollama 0.5.7 (2025-01) / Ollama-Metal 0.32.0 / llama.cpp as of
> v0.5.0** — historical working records, not current claims. Current same-box numbers vs Ollama
> **v0.32.5** are in `docs/benchmarks.md` §B2 (CUDA) / §B3 (Metal).


> **When:** after the prefill-NaN fix (the actual v0.9.0 blocker) lands, before the tag.
> **Why it's a gate, not a chore:** this whole cycle's bugs — `Kwords%32`, the As-cap, the
> Gemma GELU-tanh — surfaced **only on real geometries**, never on tiny fixtures or
> synthetics. Tagging a release that claims "Gemma / Qwen / Mistral / MoE resident on
> CUDA / Metal" without having *run each real model on each backend* is a claim without a
> gate — the one thing the parity discipline forbids everywhere else. This run is the
> release-readiness evidence **and** the source of every benchmark number.

## The checklist is the generated matrix

`docs/hardware-matrix.md` (from `ResidentEligible`) is the exhaustive, self-maintaining
list of what to qualify. The sweep walks it:

- **Every `✅ resident` cell** → a real-model **qualification** + a **timing**.
- **Every `CPU fallback` cell** → confirm it *actually falls back* on a real checkpoint of
  that family with the backend enabled — declines to CPU, runs correct, **never crashes or
  emits garbage**. (The taxonomy promises this; prove it on real weights.)

Because the matrix is generated and freshness-gated, the checklist can't silently omit a
cell — a new resident cell shows up here as a new qualification obligation.

## Per-cell qualification bar

**Resident cell — two artifacts:**
1. **Correctness on a real checkpoint.** Preferred: argmax + logit-cosine vs the CPU path on
   the *same* real weights (the `backend-int4 vs CPU-int4` twin comparison, per the
   playbook — plus the ΔNLL and a norm-checked spot trace, not a bare cosine). Where a full
   CPU reference is impractical (large models), fall back to **coherent free-generation +
   ΔNLL over a real prompt set**. Record which bar was used.
2. **A timing.** Server-to-server, best-of-N warm, driven through the HTTP server (sampling
   + detokenize + JSON included) — the same discipline the current benchmark table already
   uses. Report tok/s **and** the reference peer on the same machine.

**Fallback cell — one artifact:** load a real family checkpoint with the backend built in,
confirm decline-to-CPU + correct output.

## Scope honestly — what you claim × real checkpoints × where hardware permits

- **One representative real model per family per backend**, not every checkpoint. A resident
  cell is qualified by a real member of that family, not all of them.
- **Hardware-blocked cells get proxied, and labeled as proxy.** Some cells can't run on the
  available boxes — Metal Mistral-7B > 16 GB unified memory, big-MoE unified-memory limits,
  the 8 GB CUDA card, license-gated weights. For those, use the existing proxies
  (assembly-equivalence: identical experts ≡ dense FFN at cosine 1.0; `weightDiff` vs the
  bit-exact loader; per-kernel parity) and **mark the cell "proxy-validated," never
  "real-e2e."** A cell you could only prove indirectly is not a cell you claim directly.
- The output records, per cell: `real-e2e` / `proxy` / `fallback-confirmed`, the metric, the
  checkpoint, and the timing — so the coverage claims and the benchmark numbers each carry
  provenance.

## Canonical timings — one run, cited everywhere

This sweep produces **the** set of numbers the README, `benchmarks.md`, and the release
announcement all cite — which retires the number-drift (218.6 vs 244.2 floating across
docs). Pin them versioned: hardware, driver, peer engine + version, goinfer commit, method.
Nothing quotes a tok/s that didn't come from this run.

## Make it repeatable, not a one-off

- A **real-model tier of `scripts/gpu_gate.sh`**, hand-run on the real boxes each release —
  GitHub CI cannot run device/checkpoint tests (the objc `msgSend` SIGSEGV on macos-latest;
  no GPU runner for CUDA). The device-free gates stay in CI; this is the hardware tier.
- **One process per cell.** `gpu_gate.sh §2` SIGSEGVs from cumulative model-load exhaustion
  when the whole suite runs in one process — run each family×backend qualification in its own
  process (and it's worth fixing that teardown leak, same family as the `Close()` leak).
- **Per-peer, same machine only.** Ratios compare an engine to its peer on the *same* box
  (CUDA vs Ollama-CUDA on the 2070 SUPER; Metal vs Ollama-Metal on the M1 Pro). Absolute
  tok/s do **not** compare across the CUDA and Metal columns — that's two graphics cards, not
  two engines. Keep that caveat in the output.

## Output = the release-readiness artifact

A qualification report keyed by matrix cell: `{cell → real-e2e|proxy|fallback-confirmed,
metric+value, tok/s+peer, checkpoint, provenance}`. Green across the claimed cells (real or
honestly-proxied) **is** the go/no-go for the tag; the timings **are** the benchmark table.
One pass, two deliverables.

## The sweep is itself break-it-first-able

Per the gate discipline: a deliberately-broken backend cell (mutate one kernel) must **fail
its qualification** — confirm the sweep catches it, revert. A qualification pass that can't
go red on a real bug is theater, not a gate.

## Sequence

Finish the prefill NaN → run the matrix-driven sweep on both boxes → the real-model greens
gate the tag and the timings become the published numbers. Don't tag on fixtures; tag on
real models that ran.
