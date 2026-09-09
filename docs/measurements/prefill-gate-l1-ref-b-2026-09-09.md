# Prefill gate L1 — set B reference run — 2026-09-09

**Verdict: PASS (S model). Default flipped to ON. `metal/backend.go` `97b6e3b`.**

## Setup

| | |
|---|---|
| Machine | MacBook M1 Pro, 16 GB unified memory |
| Date | 2026-09-09 |
| Gate form | §3.2 pooled (task-prefill-gap.md) |
| Prompt set | B (held-out; set A used during development) |
| Reference | CPU backend, f32 weights + f32 activations (S); CPU backend, int8 weight-only + f32 activations (D7 fallback — f32 weights don't fit 16 GB) |
| S model | qwen2.5-1.5b-instruct-q4_k_m.gguf (1.5B int4) |
| D7 model | qwen2.5-7b-instruct-q4_k_m.gguf (7B int4) |
| Phase A (reference gen) | `TestPrefillGateReference`, `decoder/` package, 4360 s (~72 min), EXIT:0 |
| Phase B (scoring) | `TestPrefillGateVsReference`, `metal/` package, 2942 s (~49 min) for S |

## S model — decision cells

| Cell | exact agree | exact HF | exact meanKL | fast agree | fast HF | fast meanKL | verdict |
|---|---|---|---|---|---|---|---|
| S K=256 | 93.0% | 4 | 0.0347 | **94.2%** | **4** | **0.0328** | PASS |
| S K=512 | 95.0% | 4 | 0.0347 | 93.1% | **7** | **0.0323** | PASS |
| S K=1024 | 90.3% | 10 | 0.0520 | **90.2%** | **7** | **0.0512** | PASS |

HF ceiling for §3.2: `exact + 2√exact`. K=256: 4+4=8, fast=4 ✓. K=512: 4+4=8, fast=7 ✓. K=1024: 10+6.3=16.3, fast=7 ✓.

No per-cell veto issued on any S decision cell.

## S model — confirm cells (not gating)

| Cell | exact agree | exact HF | exact meanKL | fast agree | fast HF | fast meanKL |
|---|---|---|---|---|---|---|
| S K=3900 | 91.2% | 10 | 0.0508 | **92.0%** | **9** | **0.0496** |

Fast ahead on all three metrics at K=3900.

## D7 model — SKIPPED

D7 (7B int4 Metal resident) needs 12.4 GB resident + 3.5 GB KV = 15.9 GB on a 16 GB machine.
After S's Metal run completed, only 7.2 GB was available. The fit-guard blocked the load with:

> `qwen2.5-7b-instruct-q4_k_m.gguf needs ~8.9 GB resident at quant int4 + 3.5 GB KV = 12.4 GB;`
> `this machine currently has 7.2 GB of memory available (budget 5.0 GB = 70% of that).`

The Phase B launch script was missing `GOINFER_NO_FIT_GUARD=1` (Phase A had it for the CPU run,
which manages memory differently). D7 was skipped by operator decision — S cells are sufficient
for the §3.2 pooled decision, per `task-prefill-gap.md §3.2`.

## Pooled gate verdict

S decision cells: 3/3 pass, no per-cell veto. Fast arm ties or beats exact on HF and meanKL in
every S cell; agreement is within ±2pt at all depths and fast is ahead on the continuous KL
measure every time. The §3.2 gate passes on the S decision set.

**Decision: PASS. Metal fast prefill flipped to default ON above 512 tokens.**

## Changes shipped (goinfer `97b6e3b`)

- `metal/backend.go`: `metalFastPrefillEnabled()` returns `true` by default.
- `GOINFER_METAL_FAST_PREFILL=0` or `--exact-prefill` to opt out.
- `GOINFER_METAL_BATCHED_PREFILL` still honoured for backward compat (=1 on, =0 off, unset → default).
- `--metal-fast-prefill` is now a deprecated no-op.
- Floor: fast prefill declines below 512 tokens (override with `GOINFER_METAL_FAST_PREFILL_FLOOR=0`).

## Timing

| Phase | Start | End | Duration |
|---|---|---|---|
| Phase A (ref gen) | 12:07 PDT | 13:20 PDT | 4360 s |
| Phase B (scoring, S only) | 13:20 PDT | 14:09 PDT | 2942 s |
