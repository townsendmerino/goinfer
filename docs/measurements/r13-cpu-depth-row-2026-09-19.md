# R13 step 0(i) — first CPU peer depth row, and two real mechanisms found along the way

`docs/tasks/red-october.md` R13, step 0(i): "the peer depth row, both boxes: `scripts/bench_peer.py`
with `BENCH_DEPTH_BACKEND=cpu`, depths {128, 512, 2048, 3900}, greedy, the 1.5B and 0.5B int4 plus
phi3-mini as the MHA control — the first CPU depth cells `benchmarks.md` will have."

**Bottom line up front: the tooling now works end to end and produced the first real CPU depth
data, but this specific run is thermally contaminated from roughly its midpoint onward and its
absolute numbers must NOT be copied into `benchmarks.md` as-is. Two genuine, reproducible
mechanisms were found and are the real yield of this session: a hard memory-fit ceiling for
deep-context repeated sampling on a small-context model, and severe mid-run throughput drift with
no thermal telemetry to characterize it. A clean re-run (interleaved arms, cooldown pauses or a
cooler box) is needed before this row is fit for the provenance-gated table.**

## Machine / provenance

- MacBook Pro, Darwin 25.6.0 arm64, M1 Pro, 8 logical CPUs, 16 GB RAM.
- goinfer `efd7db0d` (tree dirty — unrelated uncommitted docs from another session were present;
  the binary was built clean from a `git stash`-free tree at that commit).
- Ollama 0.32.5, binary `/opt/homebrew/bin/ollama`, models `~/.ollama/models` (q05/q15/p3m tags —
  same GGUF files goinfer reads, per this harness's own "same weights on both sides" convention).
- `scripts/bench_peer.py`, `BENCH_ENGINES=goinfer,ollama BENCH_BACKENDS=cpu
  BENCH_DEPTH_BACKEND=cpu BENCH_DEPTHS=128,512,2048,3900 BENCH_MODELS=0.5B,1.5B,phi3-mini`.
- `BENCH_MAX_LOADAVG` raised from the script's own default (1.0) to 4.0: this machine's ordinary
  background baseline (VSCode, this Claude Code session, normal daemons) sat at 2.0–2.9 for the
  whole run, well above the conservative default. This is a real, deliberate widening of the
  "quiet box" bar the script's own comments argue for — flagged here rather than silently done, per
  the script's own "raise BENCH_MAX_LOADAVG deliberately" message.
- Run started 16:26:43 PDT, finished (with the one exclusion below) 17:43 PDT — **~77 minutes**,
  continuous, on a laptop with no active cooling beyond its own fans, running other apps throughout.
- Full raw output: `docs/measurements/r13-cpu-depth-row-2026-09-19.json` (resumable-format, one
  record per cell plus a provenance header).

## What this row is not

**Not a controlled, single-session-drift-free measurement.** The two mechanisms below were found
*because* this run was long, interleaved with heavy sustained load, and not deliberately paced —
exactly the conditions this repo's own measurement discipline (`CLAUDE.md` "Measurement
discipline") warns produce exactly this kind of contamination. This doc records the mechanisms
and the raw data for the record; it does not claim the numbers are benchmarks.md-quality.

## Finding 1 (real, reproducible): repeated deep-context requests exhaust session-slot memory

The `B goinfer phi3-mini K=3900` cell failed on its first attempt with HTTP 413 and then failed
identically on every retry, which would have kept the (resumable) sweep retrying it forever. The
plain 413 status obscured the real cause; the response *body* did not:

```
decoder: this request needs ~3.40 GB (KV 2.91 GB + prefill scratch 0.49 GB for 3907 prompt +
64 max_tokens positions) but only 1.92 GB of this machine's 2.75 GB currently-available memory
would be left as a safety margin — rejected before prefill rather than paging.
```

This is goinfer's memory fit-guard working correctly, not a bug — but the mechanism that gets it
there is worth recording. Reproduced directly, isolated from the sweep harness: a **fresh**
phi3-mini (4096 context) server accepts the exact calibrated K=3900 prompt fine on its first
request (200 OK, streams normally). Sending the *identical* request again, to the *same*,
still-running server, gets HTTP 413 — and every request after that does too:

```
request 1: HTTP 200
request 2: HTTP 413
request 3..16: HTTP 413
```

Why: two identical prompts never chain through `sessionLRU.bestExtend`'s strict prefix-containment
rule (a session's own full token history — prompt + the previous reply — is *longer* than the new,
identical-content prompt, so `n == commonPrefix` fails; see `internal/serveapp/sessions.go`). Every
repeated call therefore gets a **cold, fresh session** in a **new slot** (`-kv-sessions 4`, the
default). Each slot's KV cache for a ~3900-token f32 context on a 4096-ctx model costs ~2.9 GB —
so the *second* concurrent slot alone pushes the machine over its safety margin, and the guard
correctly refuses rather than let the machine page or (per this machine's own history this session
— see `docs/measurements/w4f16-decode-investigation-2026-09-19.md`'s sibling incidents) risk worse.

This is a genuine scoping fact for any harness that samples the *same* deep prompt repeatedly
(this one does: `NCOMP=8 × NRUNS=2` = 16 calls per cell) against a small-context model on a
memory-constrained box, not a defect to fix here. **`benchmarks.md|phi3-mini` depth cells above
whatever K makes `(session-slots) × (KV bytes at that K)` exceed available memory are out of scope
on this machine under the default `-kv-sessions`, full stop** — not just at K=3900; the ceiling is
a function of available RAM at run time, not a fixed K. `goinfer phi3-mini K=3900` and the paired
`ollama phi3-mini K=3900` cell (never attempted — the sweep never got past the goinfer side) are
excluded from the table below for this reason.

## Finding 2 (real, unresolved): severe throughput drift over the run's later half, no thermal telemetry

The *same* cell (goinfer, 0.5B, K=128), measured at two different points in this one run:

| when | mean tok/s | raw runs |
|---|---|---|
| Phase A, ~16:26 (cell 1 of 30) | **99.2** | [99.11, 99.25] |
| Phase B, ~16:44 (cell 8 of 30) | **48.9** | [49.16, 48.66] |

Same model, same K, same code, same machine, ~18 minutes apart under continuous load — decode
throughput roughly **halved**. `1.5B` (which started lower) stayed flat between its own two
K=128 readings (48.9 → 48.8), and `phi3-mini`'s own K=128 reading fell further still by the time
Phase B reached it, 40 minutes later (48.9 → 22.8 → its Phase-B K=2048 reading eventually reached
8.5 tok/s). The `1-min loadavg` recorded alongside every cell stayed in a narrow 3.1–4.0 band
throughout — **loadavg does not explain this drift**; it is flat while throughput is not. The
likeliest mechanism is sustained-load thermal/frequency throttling on the M1 Pro's performance
cores (no active cooling, ~77 minutes of near-continuous multi-core CPU saturation from this sweep
alone, run on top of an already-elevated background baseline) — but **this is not verified**: no
thermal telemetry (`powermetrics`, a temperature sensor read) was captured alongside the cells, so
this is a plausible mechanism, not a confirmed one. Recorded as a real open question, not a
resolved finding.

**Practical consequence for future long CPU sweeps on this machine:** interleave arms tightly (as
`bench_peer.py`'s own A/B/A/B cell ordering already does per depth) rather than trusting a number
taken late in a long run to mean what the same number taken early does; consider a cooldown pause
between depth steps, or splitting a sweep this long across multiple shorter invocations (the
harness is resumable specifically for this reason); capture `powermetrics --samplers smc` (or
equivalent) alongside the next attempt so a future run can *confirm* thermal throttling instead of
inferring it.

## Raw data (as measured — read Finding 2 before treating any single number as clean)

Ordered as measured. `Δphase` marks the two same-cell A/B comparisons discussed in Finding 2.

| phase | engine | model | K | tok/s | spread | wall (s) | loadavg(1m) |
|---|---|---|---|---|---|---|---|
| A | goinfer | 0.5B | 128 | 99.2 | 0.1 | 18.6 | 2.60 |
| A | ollama | 0.5B | 128 | 143.7 | 7.6 | 14.7 | 3.69 |
| A | goinfer | 1.5B | 128 | 48.9 | 0.1 | 34.3 | 3.80 |
| A | ollama | 1.5B | 128 | 71.9 | 1.3 | 23.6 | 3.52 |
| A | goinfer | phi3-mini | 128 | 48.9 | 0.6 | 35.2 | 3.16 |
| A | ollama | phi3-mini | 128 | 32.2 | 1.6 | 44.8 | 4.00 |
| B | goinfer | 0.5B | 128 (Δphase) | 48.9 | 0.5 | 33.2 | 3.16 |
| B | ollama | 0.5B | 128 | 144.7 | 4.7 | 14.5 | 3.96 |
| B | goinfer | 0.5B | 512 | 45.5 | 0.5 | 65.0 | 3.81 |
| B | ollama | 0.5B | 512 | 132.5 | 0.7 | 13.5 | 3.24 |
| B | goinfer | 0.5B | 2048 | 68.3 | 0.4 | 108.0 | 3.83 |
| B | ollama | 0.5B | 2048 | 81.8 | 11.2 | 26.4 | 3.72 |
| B | goinfer | 0.5B | 3900 | 53.2 | 0.2 | 369.1 | 3.45 |
| B | ollama | 0.5B | 3900 | 69.4 | 1.3 | 44.5 | 3.85 |
| B | goinfer | 1.5B | 128 (Δphase) | 48.8 | 0.2 | 43.5 | 3.89 |
| B | ollama | 1.5B | 128 | 49.0 | 5.3 | 41.2 | 3.35 |
| B | goinfer | 1.5B | 512 | 45.3 | 0.0 | 87.0 | 3.48 |
| B | ollama | 1.5B | 512 | 63.3 | 3.9 | 10.7 | 3.83 |
| B | goinfer | 1.5B | 2048 | 33.5 | 1.6 | 335.6 | 3.39 |
| B | ollama | 1.5B | 2048 | 35.7 | 0.0 | 58.7 | 3.74 |
| B | goinfer | 1.5B | 3900 | 25.1 | 0.2 | 729.5 | 3.10 |
| B | ollama | 1.5B | 3900 | 28.6 | 0.0 | 99.6 | 3.29 |
| B | goinfer | phi3-mini | 128 (Δphase) | 22.8 | 0.0 | 96.0 | 3.17 |
| B | ollama | phi3-mini | 128 | 21.3 | 0.1 | 61.2 | 3.55 |
| B | goinfer | phi3-mini | 512 | 18.8 | 0.6 | 217.7 | 3.98 |
| B | ollama | phi3-mini | 512 | 18.7 | 0.1 | 69.1 | 3.67 |
| B | goinfer | phi3-mini | 2048 | 8.5 | 1.0 | 908.3 | 3.93 |
| B | ollama | phi3-mini | 2048 | 12.6 | 0.2 | 134.6 | 3.75 |
| B | goinfer | phi3-mini | 3900 | **excluded — Finding 1** | | | |
| B | ollama | phi3-mini | 3900 | **not attempted — see Finding 1** | | | |

28 of 30 planned cells completed; the remaining 2 (both `phi3-mini K=3900`) are a documented scope
exclusion, not a failure to close.

## What is and isn't established

**Established:** the harness now genuinely drives a goinfer-vs-Ollama CPU depth sweep on the Mac
(`BENCH_DEPTH_BACKEND=cpu` plus a `GOINFER_SERVE_CPU`/`OLLAMA_BIN`/`OLLAMA_MODELS` config this doc
records) — R13 step 0(i)'s tooling gap is closed. The memory-fit-guard interaction with repeated
deep-context sampling on a small-context model (Finding 1) is established and reproduced in
isolation, independent of harness bugs. **Not established:** any single absolute tok/s number in
the table above as a clean, thermally-controlled reading — Finding 2 means the back half of this
run (roughly everything at 1.5B K≥2048 and all of phi3-mini's Phase B) may read slower than this
machine's true capability. The *shape* (goinfer behind Ollama at every depth measured, the gap
narrowing from 0.5B's ~3x to 1.5B's ~1.1-1.2x) is plausible but not to be quoted as a ratio without
a re-run.

## Next step

Re-run under conditions that let Finding 2 be either confirmed or ruled out: shorter, resumable
sub-sweeps (e.g. one model per invocation) with a cooldown between them, and `powermetrics`
captured alongside. Only then does this row belong in `benchmarks.md`. R13's remaining step-0
items — (ii) `BenchmarkDecodeAtDepth`'s category split (QK/softmax/AV) and (iii) the distinct-bytes
probe — are independent of this and not blocked by it.
