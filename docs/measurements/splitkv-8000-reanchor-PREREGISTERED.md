# PRE-REGISTERED — split-KV at 8000, and a re-anchor for the `never` class

Written 2026-09-12 **before any cell ran**. nobara-pc, CUDA. Opened from
`docs/task-decode-splitkv-attention.md` §OPEN and queue-performance's decode-depth-falloff P24.

## Why this run exists

Two separate defects in the shipped gate, one known and one found while setting this up.

1. **Known (the doc's own §OPEN):** every threshold in `splitkvThreshold`'s table was measured at
   2560–3900 attended keys at most, and the peer matrix's W3 cells run at **8000**. The verdict
   applied there is an extrapolation past the last data point.

2. **Found 2026-09-12, and it is structural:** the `splitkvNever` class is anchored on **phi3-mini**'s
   "monotone loss to 0.754 at 3900". `Phi-3-mini-4k` is a **4k-context** model. It *cannot be
   measured at 8000 at all*, so the extrapolation can never be checked on its own anchor, by anyone.
   Re-anchoring needs a geometry in the same class (nH ≥ `splitkvMaxHeads` = 24) that reaches 8000.
   **`mistral-7b-instruct-v0.1.Q4_K_M` qualifies and is already on disk**: nH=32, nKV=8, hd=128,
   L=32, context 32768. D7's own model (`qwen2.5-7b-instruct-q4_k_m.gguf`) is NOT on this box.

## Cells

Mode `force` (`GOINFER_SPLITKV_MIN_KEYS=0` = always split) ÷ off (`GOINFER_SPLITKV_ATTN=0`), which is
the question §B6 asked — *does the split path itself help here* — not what the shipped gate decides.

| geometry | nH | class | depths | role |
|---|--:|---|---|---|
| mistral-7b | 32 | `splitkvNever` (nH ≥ 24) | 3900, 8000 | **the decisive cells**; 3900 bridges to phi3-mini's anchor |
| 1.5B | 12 | measured winner | 8000 | positive control — split-KV must still win, or the run itself is suspect |

Paired per (geometry, depth), arms adjacent in time, arm order alternating, a freshly started `serve`
per arm, warm request discarded. Unchanged from the §B6.3 harness.

## Decision rule

Read on the **mean** of the paired per-cell ratios, not min-over-N.

- **`never` SURVIVES at depth** — mistral-7b at 8000 is ≤ 1.00 (split-KV no better than off). The
  extrapolation was right; the table gets a depth axis documented but no value change.
- **`never` IS WRONG at depth** — mistral-7b at 8000 is ≥ 1.05. The nH ≥ 24 rule is a
  shallow-depth artifact, the gate is costing real tok/s at the W3 cells, and `splitkvThreshold`
  needs a depth term. This is also the first real number on what the decode fork's occupancy is
  worth, because it prices the existing bit-identical split at the depth that matters.
- **AMBIGUOUS → PARKED** — strictly inside (1.00, 1.05). Too small to act on; record and stop.

Pre-registered second thing that can disagree, per the corollary: the **3900** cell. If mistral-7b at
3900 does NOT reproduce phi3-mini's direction (a loss), then mistral is not a valid stand-in for the
class and the 8000 cell says nothing about `never` — it says something about mistral. That check
decides whether the re-anchor is legitimate, and it is independent of the 8000 result.

## What this run does NOT decide

It does not decide the non-bit-identical decode fork. It prices the *upside* that fork chases, using
the bit-identical kernel already in the tree. The fork's quality objection was separately addressed
(`reduction-tree-accuracy-2026-09-12.md`); its cost — cross-M identity for spec-decode verify, and a
golden re-baseline — is untouched here.

## Guards

`bench_peer.preflight()` refuses above 1.0 one-minute load average or with more than one GPU compute
process. Both were tripped when this was written (loadavg 3.93; `gpu.test` holding 6811 of 8192 MiB),
so the run is queued behind an idle gate rather than started. A number measured on a contended GPU is
not distinguishable afterwards from one measured on a quiet box — which is the whole reason that
refusal exists.
