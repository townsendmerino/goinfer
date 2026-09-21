# PRE-REGISTERED — is the split-KV sign flip explained by KV traffic per key?

Written 2026-09-12 **before profiling**. `docs/task-prefill-attention.md:55` warns this campaign
"has recorded three plausible-mechanism-as-conclusion attributions already; profile first" — this
file is that warning applied to my own mechanism.

## The claim

`splitkv-aa-floor-2026-09-12.md` establishes (floor 0.142%, effect 17-26x it, two runs) that
split-KV WINS for mistral-7b and LOSES for phi3-mini at the SAME nH=32. My proposed mechanism:

> split-KV buys occupancy, which only helps a LATENCY-bound kernel. phi3-mini (MHA, nKV=nH=32,
> 3072 KV floats/key) is nearer memory-pipe saturation, so more blocks cannot help it; mistral-7b
> (GQA 4:1, 1024 floats/key) is latency-bound, so they do.

## What is profiled

`attn_batched` at **M=1 (decode)**, split-KV OFF (`GOINFER_SPLITKV_ATTN=0`), depth **3900** on both
geometries. 3900 because it is the depth where both have a measured force-ratio and the signs differ,
and because phi3-mini is a 4k model and cannot go deeper. Decode launches are identified by grid
`(nH, 1, 1)` (`cuda/resident.go:3227`); prefill's are `(nH, M, 1)` and at 3900 tokens prefill attention
is `attn_fused` anyway.

## Prediction, and what would falsify it

Crucially, "saturated" here must NOT be assumed to mean DRAM. `task-prefill-attention.md:60-62`
measured this engine's attention as **L1/TEX 98.86% saturated with DRAM at 0.45% — idle**: the
re-reads are L1-served, not DRAM bandwidth. So the mechanism predicts a difference in the
**memory-pipe** metric that is actually saturated, whichever it is.

- **CONFIRMED** — phi3-mini shows a materially higher memory-pipe throughput (max of L1TEX / L2 /
  DRAM %) than mistral-7b, AND mistral shows the latency signature instead (higher no-eligible-warp
  %, lower throughput). A factor of >=1.5x on the saturated metric, in the predicted direction.
- **REFUTED** — phi3-mini is NOT nearer saturation than mistral (equal, or mistral higher). The
  mechanism is wrong and the sign flip needs a different explanation. The finding that
  `splitkvNever` is mis-keyed SURVIVES either way — it rests on the A/A-floored ratios, not on this.
- **AMBIGUOUS → PARKED** — both saturated, or both idle, or the difference is under 1.5x. Report the
  numbers and stop; do not reach for a third metric post hoc to rescue the story.

Second pre-registered thing that can disagree: **occupancy**. phi3-mini launches 32 decode blocks
against mistral's 32 — identical nH, so if the gate's own stated reason ("at/above this many query
heads the single-block kernel already fills the device") were right, their ACHIEVED OCCUPANCY should
be similar. If it is, the gate's reasoning is refuted independently of whether my mechanism is right;
if phi3-mini's occupancy is much higher, the gate has a point and my mechanism is the weaker story.

## Limitation accepted up front

The prompt is synthetic (a repeated token id) rather than calibrated prose. Valid here because the
kernel's grid, access pattern and data volume depend on nKeys and geometry, not on token VALUES —
but it means this run measures the kernel, not end-to-end tok/s, and must not be quoted as the latter.
