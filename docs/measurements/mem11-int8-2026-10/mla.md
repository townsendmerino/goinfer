# MEM11-int8, MLA: int8 projections on a quantized load, 2026-10-22

Gates: `docs/tasks/task-memory-campaign-2026-10.md`, "MEM11-int8 gates, MLA first" (I8M-G), registered before the code (c65fe44c).
Machine: MacBook Pro, M1 Pro, 16 GB, macOS 26.6.2, Go 1.27.2. **Status: built, not merged; the merge waits on D10 (the parity
manifest).**

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
| I8M-G5 the family gate | `TestDeepseekV2LiteReal` at int8int8 on nobara | not run (D10) |

**Plant** (kv-B quantized too): red. kv-B differs on every layer, and Metal's logits hash moves (`0x621d19fb…`).

**Goldens moved, by this change only.** `TestInt4_forwardParity`'s `deepseek-tiny` and `kimi-tiny` entries are re-pinned (both
MLA). Their argmax is unchanged (120, 73), and their logit sums move 4.247686 → 4.241313 and 2.085561 → 1.977227. Every other
family's entry is unchanged.

**The parity manifest.** `deepseek_v2`, `deepseek_v3` and `kimi_k2` are validated rows: a real-model oracle at int8 residency,
the load this change makes numeric. Their `deps_hash` is left stale on purpose, so `TestParityManifest_fresh` stays red until
they are re-validated or the owner decides otherwise (D10). The other 39 rows' hashes are refreshed with forward goldens green
(43 passed).
