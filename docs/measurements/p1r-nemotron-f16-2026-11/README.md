# P1r: CUDA `--kv f16` for nemotron_h, registered before any code (nobara-sb01, 2026-11-08 12:45 PST)

**Why.** `docs/tasks/task-kv-precision-everywhere-2026-10.md`, row P1r: "build CUDA `--kv f16` for nemotron_h, today refused (Francis,
2026-11-08: build it)". KV4b (`docs/measurements/kv4-cuda-2026-10-21.md`) delivered f16 KV for every other CUDA-resident hybrid. It left
Nemotron-H at f32 because it was not in `decoder/kvresolve.go`'s `cudaKVF16Delivered`, not because a kernel was missing.
- Nemotron-H is admitted to the CUDA resident by `program.CUDA` (Mamba-2 plus the non-gated ReLU² FFN, OP2b).
- Its attention layers read and write the cache through the same kernels as every dense family. Each of those has an f16 twin, gated by
  KV4b's G1 (`TestKV16_exactInputIdentity_decodeReaders`, `…prefillReaders`).
- The Mamba-2 layers keep no K/V; their recurrent state is not the KV cache.

**The change:**
- `nemotron_h` joins `cudaKVF16Delivered`, and the refusal text's family list changes with it.
- Nothing else: no kernel, no store and no pricing.
- `auto` stays f32 for it: `cudaKVF16AutoCleared` excludes every hybrid, and the hybrids' G4 did not pass. LFM2 is not CUDA-resident, so
  its cell stays the resolver's word only.

## Gates (as KV4b gave the other groups, `kv4-cuda-2026-10-21.md` §2, G5)

### Tier 0 (seconds, CI with a device)

A `mamba-hybrid` row in `kv4bGroups` (`backend/cuda/kv4b_groups_test.go`), fixture `nemotron-tiny`, so the table's two gates run on it:
- **G1, the store half** (`TestKV4b_layer0StoreIsTheRoundedF32Store`): the first layer that keeps K/V holds, at `-kv f16`, the f32 build's
  K and V rounded to nearest even, bit for bit. A hybrid's layer 0 is recurrent, and the test already takes the first cache layer.
- **G3, delivered bytes** (`TestKV4b_deliveredBytes`): each layer that keeps K/V holds f16 pools of exactly half the f32 build's bytes,
  and `decoder.ResidentKVBytes` prices exactly what the build allocates.
- **Red first:** both are red on `main` for this fixture, because the f16 load refuses. They go green with the change.
- **The matrix** (`docs/kv-precision-matrix.md`) is regenerated, and its CUDA `nemotron-tiny` `--kv f16` cell must read `f16`.

### Tier 1: G5, the range survey (by day, under the timing lock)

- **Instrument:** `TestP1r_nemotronRangeSurvey` (`backend/cuda`, `GOINFER_HEAVY_TESTS=1`).
  - It loads `~/models/nemotron/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf` (local NVMe) on CUDA at int4 with an f32 KV cache.
  - It prefills each of the ten held-out set-B gate prompts at 3,900 tokens, through `decoder.PrefillGatePromptSet` and the model's own
    tokenizer.
  - After each prompt it reads every attention layer's K and V rows back and records max |K| and max |V|.
  - The fit is checked first by serve's own banner.
- **Bar:** the worst max |K| or |V| over every attention layer and prompt is under 65504/2 = 32,752.
  - **PASS:** f16 is delivered as above.
  - **FAIL:** `nemotron_h` stays refused at f16, and the record names the layer and the value. A per-row scale is not built here.

**Prediction:** PASS (0.85). The other hybrids' worst values were 95 to 445 (`kv4-cuda-2026-10-21.md` §3).

**Cost:** tier 0 seconds. G5 about 5 to 10 minutes: one 9B load, ten 3,900-token prefills. A parity refresh follows, because a non-test
`decoder/*.go` file changes.

**Not gated here:** G4, the fidelity read. f16 is delivered on request only, and `auto` is f32, as for every hybrid. The coordinator's brief
was f16 identity and G5, as KV4b did for the others.

## Addendum, 2026-11-08 12:45 PST, before G5 runs: the cost; no bar or cell changed

- **Fit**, by serve's own banner (`fit.log`): `cuda-resident (int4)`, 6.3 GB of weights, a 4,028-token f32 cache at 0.1 GB; nothing refused.
  Serve exited only because `-require-backend` also refuses a sequential prefill: "the Mamba-2 mixer has no batched kernel".
- That same fact makes `PrefillLast` decline on this model, so the survey runs each prompt one forward per token, as serve does. That is
  ten prompts of 3,900 tokens, about 39,000 steps: **about 15 to 30 minutes**, not the 5 to 10 written above.
