# Metal decode-at-depth ladder, same-session peer re-measurement — 2026-09-18

**R2 step 0** (`docs/tasks/red-october.md`): re-run the Metal decode depth bench and the peer's
depth curve in one session, replacing the 2026-08-09 table in `docs/benchmarks.md` §B3 and
settling whether the standing ratio is still current.

**Result: goinfer genuinely closed more than half the depth-decode gap since the last true
same-session comparison, and Ollama did not move — but only visible after catching and correcting
a methodology bug in this run itself.** The first pass used `OLLAMA_KV_CACHE_TYPE=q8_0`, carried
over unexamined from R4's prefill-sweep protocol, and it does not belong here: it cost Ollama
14–36% of its decode throughput, the loss growing with depth. A same-day isolation (below) removed
it and reproduced `metal-verdict.md`'s own 2026-08-04 peer row to within 1–4% at every depth —
Ollama's Metal decode-at-depth is flat across the six weeks between the two measurements. goinfer,
over the same window, more than doubled at the deep end (18.5→39.1 tok/s at 3900–4000) and grew
substantially at every depth. The true same-session ratio moved from **4.19× behind at 4000
(2026-08-04) to 1.96× behind at 3900 (today)**.

## Provenance

goinfer `a0924fb6` at launch time (working tree carried this run's own untracked log files, not a
code change). Ollama v0.32.5, `OLLAMA_FLASH_ATTENTION=1` (explicit; the archived row used
auto-detection, confirmed FA-on either way). Apple M1 Pro, 16 GB, macOS 26.6.2. Model:
`qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` (1.5B int4/q4_K_M), same file both sides.
`~/bench-cur/serve-metal` reused from R4 step 0 (already rebuilt fresh from the current tree that
session; no relevant commits landed between the two runs).

`scripts/bench_peer.py`, both engines, `BENCH_MODELS=1.5B BENCH_BACKENDS=metal
BENCH_DEPTH_BACKEND=metal BENCH_DEPTHS=128,512,1024,2048,3900` (3900 substituted for the brief's
nominal 4000 — `scripts/prompts.json` has no calibrated `1.5B:4000` entry, and 3900 is already the
calibrated depth R4 step 0 used, so it stays consistent with that record rather than adding a new
calibration). 2 runs × 8 completions × 64 generated tokens per cell (script defaults), decode-only
(client-timed from the first streamed token, prefill excluded on both sides by construction).

**Extended `scripts/bench_peer.py` itself**: Phase B (the depth sweep) was hardcoded to
`backend="cuda"`, which R2's brief anticipated ("extend it or use the depth bench as the
instrument of record and say so"). Added `BENCH_DEPTH_BACKEND` (defaults to `"cuda"`, so the
Linux release sweep is unchanged) following the same additive-override pattern every other axis
here already uses (`BENCH_MODELS`, `BENCH_DEPTHS`, `BENCH_ENGINES`, `BENCH_BACKENDS`), rather than
inventing a separate ad-hoc instrument.

**This run drives goinfer over its real HTTP server** (`/v1/chat/completions`), unlike the
superseded table's `TestZZ_metalDepthBench`, which called the resident `ForwardArgmax` path
directly. N-03 (`docs/audit-metal-2026-09-12.md`) flagged that internal path as not what
production serving actually calls (`generateInto`'s greedy case runs the full-logits
`ForwardEmbPipe` and argmaxes host-side) but "speed-neutral on UMA... so the depth shape should
still be representative." This run measures the real production path directly, closing that
caveat rather than relying on the argument for why it didn't matter.

## The KV-cache-quant confound, found and isolated same-day

First pass (`OLLAMA_KV_CACHE_TYPE=q8_0`, inherited unexamined from R4's prefill protocol):

| depth | goinfer | ollama (q8_0 KV) | ratio |
|---|---|---|---|
| 128 | 73.7 | 75.4 | 1.02× |
| 512 | 69.1 | 97.4 | 1.41× |
| 1024 | 61.9 | 69.7 | 1.13× |
| 2048 | 50.7 | 63.8 | 1.26× |
| 3900 | 39.1 | 56.4 | 1.44× |

This read as the gap to Ollama shrinking substantially against the archived comparison
(`docs/completed/metal-verdict.md` M0, 2026-08-04: 85.2/79.1/~80/77.5 at 128/1024/2048/4000, FA-on,
never mentioning KV-cache quantisation). Rather than accept "the gap narrowed" — which would
credit either goinfer's own decode kernel (unchanged this session; R4's M-03/M-04 are prefill-only
kernels, `gemm_w4f16_store`/`attention_prefill_fused`, and do not touch the decode path this
brief's `attention` kernel runs) or a real Ollama regression, neither of which this data supports
on its own — the KV-cache setting was isolated the same way `metal-verdict.md`'s own M0+FA-off
pass isolated Flash Attention: re-run with the single variable removed, same day, same box.

Isolation (Ollama only, `OLLAMA_KV_CACHE_TYPE` unset — Ollama's default, matching what the
archived row used):

| depth | ollama, q8_0 KV | ollama, default KV | delta | archived (2026-08-04) |
|---|---|---|---|---|
| 128 | 75.4 | 85.9 | +13.9% | 85.2 |
| 512 | 97.4 | 113.5 | +16.5% | — ¹ |
| 1024 | 69.7 | 82.6 | +18.5% | 79.1 |
| 2048 | 63.8 | 80.2 | +25.7% | ~80 |
| 3900 | 56.4 | 76.7 | +36.0% | 77.5 (@4000) |

¹ No archived point at 512; the archived curve used 128/1024/1953/3663≈4000.

**The delta grows monotonically with depth (14%→36%) — exactly the signature of a per-step
KV-dequantization cost that scales with context length**, not a flat overhead. Removing it
reproduces the archived row to within 1–4% at every depth that has one — inside this box's
documented ~3.5% ordinary session drift. **`OLLAMA_KV_CACHE_TYPE=q8_0` is a real, depth-growing
tax on Ollama's Metal decode and does not belong in a decode-at-depth peer comparison**, whatever
its place in R4's prefill protocol (prefill's KV-cache access pattern is different — mostly
written once, not re-read every step — so the same setting costing little there and a lot here is
not a contradiction, just a reason not to copy a peer-comparison env var across measurement
classes without checking).

## Corrected result (goinfer from the first pass; Ollama from the isolation)

| depth | goinfer | Ollama (default KV) | ratio (Ollama/goinfer) |
|---|---|---|---|
| 128 | 73.7 | 85.9 | 1.17× |
| 512 | 69.1 | 113.5 | **1.64×** ¹ |
| 1024 | 61.9 | 82.6 | 1.33× |
| 2048 | 50.7 | 80.2 | 1.58× |
| 3900 | 39.1 | 76.7 | 1.96× |

¹ K=512 is the one cell to treat with caution on the Ollama side: both the q8_0 run and the
isolation run show `tokens_per_chunk = 1.25` (80 tokens streamed over 64 chunks) at this depth
specifically, and only this depth — every other cell reads a clean 1.0. This looks like a
streaming/chunking artifact local to K=512, reproduced twice, not a one-off; it inflates Ollama's
apparent rate at that one point (both 97.4 and 113.5 may read a little high). Not investigated
further here — flagged for whoever next touches this depth curve.

**Against the one TRUE same-session comparison that exists — `metal-verdict.md`'s own M0 pair,
2026-08-04, goinfer 63.8/39.8/28.4/18.5 vs peer 85.2/79.1/~80/77.5 — the gap has genuinely
closed, and Ollama is the stable side, not goinfer.** (R2's own "Standing" section pairs a
*different*, later goinfer curve — 72.3/67.9/51.4/40.4, from the 2026-09-13 `TestZZ_metalDepthBench`
run — against this same 2026-08-04 peer row; that cross-session pairing happens to land close to
today's corrected ratio, but only because Ollama turned out to be stable across the gap, which
this run confirms rather than assumes.)

| depth | Ollama, 2026-08-04 | Ollama, today | Δ | goinfer, 2026-08-04 | goinfer, today | Δ |
|---|---|---|---|---|---|---|
| 128 | 85.2 | 85.9 | +0.8% | 63.8 | 73.7 | +15.5% |
| 1024 | 79.1 | 82.6 | +4.4% | 39.8 | 61.9 | **+55.5%** |
| 2048 | ~80 | 80.2 | +0.2% | 28.4 | 50.7 | **+78.5%** |
| 4000/3900 | 77.5 | 76.7 | −1.0% | 18.5 | 39.1 | **+111.4%** |

**Ollama's Metal decode-at-depth is flat within ordinary session noise (−1.0% to +4.4%) across
the six weeks between the two measurements. goinfer more than doubled at the deep end (18.5→39.1
tok/s) and improved substantially at every depth, growing with depth (+15.5%→+111.4%).** The
true same-session ratio at 3900–4000 moved from **4.19× behind (2026-08-04) to 1.96× behind
(today)** — goinfer closed more than half the gap, for a real reason, on the peer side held
constant. This matches red-october.md §3's already-recorded ceiling note that the 2026-08-09→
2026-09-13 `TestZZ_metalDepthBench` standing itself moved (62.0/47.8/27.2/18.2 → 72.3/67.9/51.4/
40.4) — this run is the first to attach a genuinely matched Ollama curve to that already-known
goinfer improvement, rather than the stale 2026-08-04 peer row it had been sitting next to.
goinfer's own numbers here are also stable against that 2026-09-13 standing itself (73.7/69.1/
50.7/39.1 vs 72.3/67.9/51.4/40.4, within 2% at every matching depth) — no further change since
then, consistent with M-03/M-04 (2026-09-13, both prefill-only kernels) having nothing to do with
this decode path, as expected.

## Decision rule

Step 0 has no band of its own — it is the "before" every later cell in R2 is measured against, per
the brief. No decision made here; this refreshes the standing goinfer's eventual `attention_fa`
kernel (R2's Build phase) will be compared to, and confirms the depth gap it is meant to close is
real and roughly 1.9–2× at 3900–4000, not overstated by a stale peer comparison.

## Out of scope

The kernel itself (R2's Build phase, gated on the lane decision), prefill (R4, already run), the
K=512 chunking anomaly (flagged, not chased).
