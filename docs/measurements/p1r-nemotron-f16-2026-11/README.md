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

## Readings, 2026-11-08 (nobara, RTX 2070 SUPER, driver 595.112.07)

- **Tier 0: PASS** (`tier0-red.log` on `main`, `tier0-green.log` with the change).
  - On `main`, both table gates refused `nemotron-tiny`'s f16 load by name.
  - With the change, G1's store half holds: the first cache layer's K and V are the f32 store rounded to nearest even.
  - G3 holds too: 2 f16 layers. Their K/V pools are exactly half the f32 build's, and the whole slot is as priced: 1,065,472 B against
    f32's 2,130,432 B. That is half plus 256 B: the slot also holds fixed buffers that are not K/V and do not halve (Gemini's review of #790).
- **G5: PASS** (`g5.log`, 12:46 to 13:00 PST, the exclusive timing lock, 833 s). On Nemotron-Nano-9B-v2 (`cuda-resident (int4)`, f32
  cache), the ten set-B prompts at 3,900 tokens each, run one forward per token:

| attention layer | max \|K\| | max \|V\| |
|---|---|---|
| 14 | 32.77 | 10.49 |
| 21 | 30.96 | 20.39 |
| 30 | 41.78 | 17.95 |
| 39 | 38.94 | 44.67 |

  - The worst is 44.67, against the 32,752 limit.
  - Prediction "PASS (0.85)": right.
- **Delivered:** `--kv f16` on CUDA for `nemotron_h`, on request; `auto` keeps f32 (hybrid).
- **Registries updated with the change:**
  - `docs/kv-precision-matrix.md`: regenerated; the CUDA `nemotron-tiny` `--kv f16` cell reads `f16`.
  - MM1's decision census (`testdata/mm1_census.json.gz`): the Nemotron-H rows at `kv=f16` now plan an f16 cache, so both host classes
    are regenerated with `-census.update`. amd64 was run natively. arm64 was run as a linux/arm64 test binary under qemu-user on nobara
    (`census-arm64-qemu.log`), the class CI's linux-arm64 job checks.
  - `cmd/gate`'s heavy partition: the survey joins the `nemotron-h` group, and like that group's other test it skips without its
    checkpoint, so the group still grades as all-skip there.
- `touched.py`: PASS, 17 checks. The parity staleness gate is green, with nothing to refresh.

## Addendum, 2026-11-08 14:55 PST: Gemini's review of #790, answered (no bar changed)

- **The recurrent state between prompts ([test]).** The survey ran each prompt one forward per token on one resident without `Reset()`, so
  prompts 2 to 10 began from the previous prompt's Mamba-2 state.
  - It now resets before every prompt, and was re-run (`g5-rerun.log`, 14:41 to 14:55, the exclusive lock, 837 s).
  - Every per-layer maximum is identical to the first run: worst max |V| 44.67 at layer 39. **PASS** stands, now on ten independent prompts.
- **NaN ([test]).** A NaN in the readback would have been skipped by `max` and the `>` comparisons. The survey now counts NaNs and fails
  on any, requires a non-empty prompt set, and fails if the worst value is not positive. The re-run read no NaN.
- **The tokenizer ([risk]).** The GGUF's pre-tokenizer is `tekken`, which goinfer does not know. It falls back to cl100k with a one-digit
  cap, so the survey's 3,900-token sequences are not the canonical tokenization of the set-B text.
  - No HF tokenizer for Nano-9B-v2 is on the box.
  - The survey measures K/V magnitudes over real prose, which do not depend on the exact segmentation; the worst is about 1/730 of the limit.
  - Recorded as a limit of this reading, not re-run.
- **The `auto` note ([nit]).** Nemotron-H's `--kv auto` reason was the generic "until its quality gate passes". It now says what is true:
  "no quality gate has read a Mamba-2 hybrid yet (a passing G4 cell would clear it)". The DeltaNet hybrids' failed gate is not Nemotron-H's.
  The matrix and both censuses are regenerated; only Nemotron-H's rows moved.
- **The slot bytes ([nit]).** Corrected above: the pools are exactly half, and the slot total is half plus 256 B of fixed buffers.
