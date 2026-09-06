# CUDA prefill vs Ollama — the deficit at depth goes 12.1× → 3.16× (1.5B), 14.5× → 1.89× (0.5B)

**The peer row for L2+L3. Measured with the fast path ON, which is NOT the shipped default —
§3's gate returned DOES NOT SHIP, so every "fast" figure here describes what the kernels do when
explicitly enabled, not what a user gets today.** The exact arm in the same run *is* the shipped
default and reproduces the 2026-09-01 re-anchor, which is what makes the pair readable.

## Provenance

| | |
|---|---|
| box | `nobara-pc`, RTX 2070 SUPER, driver **595.91.07**, Nobara 44, kernel 7.2.0-202.fc44 |
| goinfer | `428a9433` + the pipeline-lint fix; serve built from this tree (`~/bench-l2l3/serve-cuda`, CGO_ENABLED=0, `./cuda/cmd/serve`) |
| peer | **Ollama v0.32.5** (`~/ollama-0325`) — the same peer build §B8 anchors decode against |
| models | qwen2.5-coder **0.5B** / **1.5B** instruct **q4_k_m** → int4, same GGUF both sides, from **`~/models` on local NVMe** |
| harness | `scripts/bench_peer_prefill.py`, **interleaved per cell** with a server restart between, 6 distinct prompts per cell, medians, every request carrying a unique `Session NNNN.` prefix |
| depths | K ∈ {128, 512, 1024, 2048, 3900} |
| thermal / load | GPU 52–55 °C at each arm's start, box otherwise idle |
| results | `~/goinfer-logs/prefill-l2l3/p4-peer-arm{0,1}.json` and `.log` |

**Arm structure and what is same-session.** Each arm interleaves goinfer against Ollama *within
every cell*, so the **goinfer-vs-peer ratio inside an arm is same-session**, which is the comparison
CLAUDE.md requires be interleaved. The **arm-vs-arm** comparison is across two runs (~3.5% drift on
this box); that is acceptable here because the same-session arm-vs-arm number already exists from
`TestPrefillTTFT` (Phase 2) and is not re-derived from this harness.

## 1. The baseline arm reproduces the 2026-09-01 re-anchor

Before reading any "after", the "before" is checked against the record:

| cell | re-anchor 2026-09-01 | this run (exact arm) |
|---|---|---|
| 1.5B K=512, TTFT rate | 1368 tok/s | **1381.6** |
| 1.5B K=3900 | 702 | **707.1** |
| 0.5B K=3900 | 1396 | **1458.7** |
| 1.5B marginal, deepest interval | 12.7× behind | **12.1×** |
| 0.5B marginal, deepest interval | 14.8× behind | **14.5×** |

Within a few percent on every cell, across a month. The instrument and the baseline are sound, so
the delta below is attributable to the kernels.

## 2. TTFT rate — what an interactive caller feels

`prompt_tokens / TTFT`, each engine's request overhead included. **Ratio = how many × behind
goinfer is; below 1.0 means goinfer is AHEAD.**

| model | K | goinfer exact | goinfer **fast** | Ollama | exact ratio | **fast ratio** |
|---|---|---|---|---|---|---|
| 0.5B | 128 | 3681.3 | **6711.5** | ~435 | 0.11 | **0.07** |
| | 512 | 2869.9 | **14543.4** | ~1418 | 0.49 | **0.10** |
| | 1024 | 2511.7 | **14165.3** | ~2470 | 1.00 | **0.17** |
| | 2048 | 1992.4 | **11992.4** | ~4457 | 2.21 | **0.38** |
| | **3900** | 1458.7 | **10057.7** | ~6582 | **4.45** | **0.66 (AHEAD)** |
| 1.5B | 128 | 1634.2 | **3140.1** | ~422 | 0.26 | **0.14** |
| | 512 | 1381.6 | **5459.9** | ~1242 | 0.92 | **0.22** |
| | 1024 | 1208.2 | **4973.7** | ~2128 | 1.77 | **0.43** |
| | 2048 | 967.6 | **3740.5** | ~3145 | 3.21 | **0.85** |
| | **3900** | 707.1 | **2715.0** | ~4161 | **5.92** | **1.52** |

**The crossover — the depth past which Ollama is faster to first token — moves from ~K=600 to past
K=2048 on the 1.5B, and off the measured ladder entirely on the 0.5B.**

## 3. Marginal cost per token — the honest throughput number

TTFT rate is not prefill throughput: Ollama carries a fitted **~345 ms** per-request floor that
goinfer does not, and that floor amortises over more tokens as K grows. The overhead-free quantity
is the marginal cost between adjacent depths, at the deepest interval (2067 → 3919):

| model | goinfer exact | goinfer **fast** | improvement | Ollama | exact vs peer | **fast vs peer** |
|---|---|---|---|---|---|---|
| 0.5B | 0.8905 ms/tok | **0.1173** | **7.59×** | 0.0616 | 14.5× behind | **1.89× behind** |
| 1.5B | 1.8391 ms/tok | **0.4810** | **3.82×** | 0.1521 | 12.1× behind | **3.16× behind** |

**Ollama's own marginal is unchanged between the two arms** — 1.5B 0.1520 vs 0.1521 ms/tok, 0.5B
0.0616 vs 0.0622 — which is the control that says the delta is goinfer's and not the harness's.

**goinfer's marginal cost still RISES with K** in both arms (1.5B fast: 0.0402 → 0.1173 ms/tok
across the ladder), where Ollama's is flat. Rising marginal cost is the O(K²) attention signature.
L2 flattened it substantially but did not remove it — the fused kernel is at **1.72% of tensor
peak** and **12.6% occupancy** (Phase 1 §4), so the remaining curvature is headroom, not a floor.

## 4. Against §6's projection

§6 projected, from measured category shares and labelled "counted, not measured": *"with both
landed, goinfer is inside ~1.3× of Ollama's overhead-free prefill at K=512 and ~2.5× at K=3900."*

Measured at the deepest interval: **3.16×** (1.5B). The projection was **optimistic at depth** by
about 25%, while Phase 2 found it accurate to 1.5% on the *internal* end-to-end cell. The
difference is that §6's peer column extrapolated Ollama's flat marginal against a goinfer curve
that is still rising; the internal projection had no such extrapolation. Recorded rather than
smoothed over: the Amdahl method held, the peer extrapolation on top of it did not, and those are
separable claims.

## 5. What this row does NOT say

- **It is not the shipped default.** `GOINFER_CUDA_FAST_PREFILL` is off unless set. §3's gate
  returned DOES NOT SHIP (`prefill-l2l3-phase3-2026-09-05.md`). Any table quoting the fast column
  must carry that, or it becomes a figure that outlives its caveat — which this repo has on record
  as happening to a `tok/s` number for months.
- **Two models, one card, one quant, greedy, dense.** No MoE (batched prefill declines statically,
  P20), no 7B peer cell (`prompts.json` has 7B entries but the peer sweep here is 0.5B/1.5B, as the
  re-anchor was).
- **TTFT includes one sampling step** on both sides; it is small and not subtracted.
- The fast arm's cache-check reads goinfer repeat/fresh ≈ 0.02 where the 2026-09-01 re-anchor read
  1.00 (no caching). The measurement is unaffected — every request carries a unique prefix, and
  TTFT rises monotonically with K in every cell, which a cache hit would not do — but the change
  is noted here because the re-anchor's "goinfer does not cache" line is now stale.
