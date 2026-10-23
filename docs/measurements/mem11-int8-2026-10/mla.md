# MEM11-int8, MLA: int8 projections on a quantized load, 2026-10-22

Gates: `docs/tasks/task-memory-campaign-2026-10.md`, "MEM11-int8 gates, MLA first" (I8M-G), registered before the code (c65fe44c).
Machine: MacBook Pro, M1 Pro, 16 GB, macOS 26.6.2, Go 1.27.2. **Status: merged on D10 (b), owner, 2026-10-22: the three DeepSeek-family
manifest rows are pending until nobara re-runs their real-model gates at int8 (ledger row MEM11-int8-MLA-T3).**

**The build (B27's form for Mamba, applied to MLA).**
- **What goes int8:** on an integer-quantized load, q (or the q-LoRA pair), kv-A and o are held as per-row int8, weight-only:
  `linalg.QuantizeInt8` of the f32, the codes every GPU backend makes from them. Metal, CUDA and WebGPU now upload those codes
  as they are.
- **What stays f32:** kv-B, which the backends slice into W_UK and W_UV as f32. An f32, bf16 or f16 load keeps all six f32.
- **The `.giw`** carries the int8 as weight records, read by encoding byte, so an older bundle still loads.

| gate | reading | result |
|---|---|---|
| I8M-G1 holding | `TestMEM11_mlaInt8Holding`, deepseek-tiny: 4 MLA layers int8, kv-B f32; `ResidentWeightBytes` at int4 422,144 → 311,808 B and at int8int8 413,440 → 303,104 (both exactly the projections' saving, 110,336 B); the f32 load's logits hash `0xbd704d52…`, main's | **PASS** |
| I8M-G2 the upload | the codes and row scales, on load and after a bundle round trip, equal `linalg.QuantizeInt8` of main's f32 projections; Metal resident at int4 on deepseek-tiny: logits hash `0x7e63e089…`, unchanged from main (`TestD5_mlaOnMetal`); the Metal MLA tests and WebGPU's MLA tests pass; CUDA vetted and staticchecked from the Mac | **PASS** |
| I8M-G3 CPU fidelity, tiny | deepseek-tiny at int4, int8-MLA CPU against f32-MLA CPU, 32 steps of greedy lockstep: **min cosine 0.999887, argmax 31/32** (bar 0.999, 30) | **PASS** |
| I8M-G4 CPU fidelity, real slice | DeepSeek-V2-Lite-Chat Q4_K_M, layers 0–1, int4: **min cosine 0.997303, argmax 31/32** (bar 0.99, 28); resident weight bytes 1,248,837,632 → 1,178,890,752 for two layers, about 35 MB a layer (~0.94 GB over the model's 27) | **PASS** |
| I8M-G5 the family gate | `TestDeepseekV2LiteReal` and the Moonlight gate at int8int8, on nobara | open: MEM11-int8-MLA-T3 |

**Plant** (kv-B quantized too): red. kv-B differs on every layer, and Metal's logits hash moves (`0x621d19fb…`).

**The fit estimate, after CI's first run.** CI went red on `TestFitEstimate_f32PinnedMixersAgreeWithResident`'s Bailing case
(estimate over accounted 1.37 at int4, band 0.85–1.25). Two causes, both fixed here, no band moved:
- the estimators still priced MLA's q-LoRA pair and kv-A at f32; they now price them at int8 on an integer quant
  (`mlaInt8TensorName`), and kv-B stays f32;
- `ResidentWeightBytes` never counted a Kimi Delta Attention layer: Bailing's three KDA layers (six quantized projections, three
  conv kernels, the f32 gates) and the MLA output gate were missing. The ratio on main was 1.16 only because that missing part
  and the f32 MLA projections offset each other. Bailing now reads 0.96 at int4 and 0.92 at int8int8, deepseek-tiny 0.99 and 0.94.

The f32 logits hash in `TestMEM11_mlaInt8Holding` is arm64's (the amd64 jobs failed on it). The CPU reference is bit-identical
within an architecture only, so on amd64 the f32 path is held by the tiny forward golden's tolerance. The test's other checks pass
there, and under Rosetta on the Mac.

**Goldens moved, by this change only.** `TestInt4_forwardParity`'s `deepseek-tiny` and `kimi-tiny` entries are re-pinned (both
MLA). Their argmax is unchanged (120, 73), and their logit sums move 4.247686 → 4.241313 and 2.085561 → 1.977227. Every other
family's entry is unchanged.

**The parity manifest (D10 (b), owner, 2026-10-22).** `deepseek_v2`, `deepseek_v3` and `kimi_k2` were validated rows, a real-model
oracle at int8 residency, the load this change makes numeric. They are now `pending`, with `validated_at`, `date` and metrics
cleared, and their reference and method kept. That is the manifest's own mechanism: `TestParityManifest_fresh` does not
enforce a pending row's hash, and nothing in it was weakened. Their last validated readings stay on record here for the re-run
to compare:
- `deepseek_v2`: argmax 100.0%, cosine 0.99924 (7b50947a, 2026-10-05).
- `deepseek_v3`: argmax 100.0%, cosine 0.99951 (7b50947a).
- `kimi_k2`: shared-path via `deepseek_v3` (b067007).

The capability matrix (`docs/capability-matrix.md`, `.json`, `pull/capability-matrix.json`) now shows the three as pending; the
site module's tests and build pass with it. The other 39 rows' hashes are refreshed with forward goldens green (43 passed).
