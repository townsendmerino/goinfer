# v0.20.0 §C1: sweep run 2 and the scoped re-validation on aikit v1.51.1 (2026-10-01)

> **Status: complete, every blocker resolved.** nobara-pc (linux/amd64, Ryzen 7 3700X, RTX 2070 SUPER, 62 GB), go1.27.0,
> checkpoints from `~/models` (Qwen3.6-35B-A3B safetensors and Qwen3.8-27B from the `/srv/models` archive through the
> asset registry's env override; these are correctness gates, not timings).

## Sweep run 2 (`bcf50a49`, 06:24–08:48 PDT)

`go run ./cmd/gate parity`, `EMIT_MANIFEST=1` (`run-parity-sweep.sh`; log `sweep-run2/parity-sweep.log`). Untagged cell
794 pass / 59 skip / 0 fail; real-checkpoint cell 55 gates. Verdict: **2 blockers** (+2 coverage gaps, +1 first-run):

- `TestQwen35GGUF_gate`, argmax 57/80 against its 66 floor: bisected to aikit v1.50.2's amd64 int8 rounding and fixed in
  aikit v1.51.1 (`qwen35-gguf-bisect-2026-10-01.md`).
- `TestLagunaGGUF_gate`, refused at load by the fit guard: fixed with a per-model guard knob (`a6ce3d4a`).
- `TestOlmo3Real_gate` (first-run, not a blocker): a stale reference, diagnosed the same day; re-pinning it is the first
  job after the tag.

## The scoped re-validation (`70be7081`, aikit v1.51.1, 10:48–12:36 PDT)

The fix changes amd64 numerics wherever aikit's int8 quantizer runs (activations before W8A8/W4A8 matmuls, and int8
weights quantized at load), so `run-scoped-revalidation.sh` (pre-registered in `2d0b664f`; owner decision: scoped,
not a third full sweep) re-ran:

1. **The whole untagged cell** (`REALCKPT=0`): **794 pass / 59 skip / 0 fail**, 36m28s, ALL REQUIRED GATES GREEN, the
   same counts as sweep run 2 (`scoped-revalidation/cell1.log`).
2. **The 29 real-checkpoint gates that load a non-f32 quant**, through `GATE_RUN`: **21 pass / 7 skip / 1 fail**, 1h11m
   (`scoped-revalidation/realckpt.log`). The skips are the four Nemotron gates whose checkpoints are not on the box, two
   diagnostics, and `TestQwen3NextReal_oracle` (capacity), as in sweep run 2. `TestQwen35GGUF_gate` passed at **68/80,
   cosine 0.98740 / 0.99608**, the confirming run's numbers exactly.

The one failure, `TestQwen38GGUF_gate`, was not numeric. The fit guard refused the load in 0.29 s: 19.2 GB int4 + 32.0 GB
KV at the model's full context + 15.3 GB for the mapped `.gguf` = 66.6 GB, against a 29.5 GB budget left after the 35B
gates before it in the same process. It had passed in sweep run 2 (114.9 s). With the same per-model knob as the other two
GGUF gates (`1de75952`) it passes: **113.6 s** (`scoped-revalidation/qwen38-gguf-rerun.log`, `run-qwen38-gguf-rerun.sh`).

The 25 real-checkpoint gates that load f32 (55 in the cell, 30 quantized) never call the quantizer, so their sweep-run-2 results stand.

## What was stamped, and what was refused

- **Ledger:** 28 confirmations. 13 f32 real gates at `bcf50a49` (sweep run 2), 3 untagged-cell gates and 11 quantized
  real gates at `70be7081`, and `TestQwen38GGUF_gate` at `1de75952`. `TestQwen35Real_gate2FullModel` passed on both
  runs and left `neverConfirmed` (its 2026-09-18 entry said it could not fit this box).
- **Manifest:** sweep run 2's stamps for 32 families (`a5a4b6a5`), then the re-validation's for 12 families at the same
  method tier (`f0b9cf90`). The re-validation's own merge also **demoted eight families** (gemma, gemma2, granite, lfm2,
  mistral3, olmo_hybrid, qwen2_5_vl, smollm3) from validated real-oracle rows to tiny-golden ones (step 1 emits their
  tiny rows; their real-oracle gates are f32 and were not in step 2 to emit the stronger row afterwards, as a full sweep's
  real-checkpoint cell would). Those demotions were not committed; the eight keep sweep run 2's stamps.
  (`scoped-revalidation/parity_manifest.after.json` is the merge as the tool wrote it.)

## Follow-ups (after the release)

- **The manifest merge accepts a weaker method over a stronger one.** A scoped `EMIT_MANIFEST` run cannot stamp truthfully
  until the merge refuses a downgrade.
- **The gate writes its per-test JSON to fixed `/tmp` names**, so each run overwrites the previous one's
  (`/tmp/gate_parity_*.json`, `/tmp/gate_parity_rows.txt`). Sweep run 2's per-test JSON is gone; its text log is kept here.
- **Three GGUF loader gates needed the guard knob** because the guard prices the mapped checkpoint as resident and KV at
  the model's full context. A gate that pinned a small context would price KV realistically; the mapped-file term would
  still need the knob.
