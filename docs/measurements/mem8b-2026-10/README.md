# MEM8(b): Gemma 4's per-layer table at int4 whatever the head pin says (D10 (a)), 2026-10-21/22

Gates: `docs/tasks/task-memory-campaign-2026-10.md`, §4 MEM8, "MEM8(b) gates" (PLE-G), registered before the code.
Machine: MacBook Pro, M1 Pro, 16 GB, macOS 26.6.2, Go 1.27.2, darwin/arm64; the owner's desktop open; load 2.4 at 07:00.
Checkpoint: `~/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf` and its two sidecars, both on local disk.

**Status: shipped on the owner's decision D12 (a), 2026-10-22** (the task doc's §7): PLE-G3 read AMBIGUOUS and the owner shipped it as built; `TestS9EModelBatched_tiny`'s bar moved from 0.98 to 0.975, this record being its mechanism (the tiny fixture's batched pass reads 0.97980 with the int4 table, unmoved at real size; the nearest planted defect reads 0.911). At 0.975 the test passes and every plant stays red.

| gate | reading | result |
|---|---|---|
| PLE-G1 table | the synthetic E2B GGUF and `gemma4-emodel-tiny`: the table int4 with the pin on and off, rows byte-equal; the token embedding int8 (pin on) and int4 (pin off) (`g1-g2.log`) | **PASS** |
| PLE-G2 sidecar | `.ple4` exactly for a per-layer source at int4 with the pin on; every other name unchanged; the transcoded table int4 and its rows byte-equal to the direct load's. **Deviation:** rows, not logits over 8 tokens, since neither synthetic source has the tokenizer a loadable bundle needs | **PASS** (deviation stated) |
| PLE-G3 quality | E2B on Metal, int8 table (`.int4.metal.v17.giw`, 3,649.0 MB) against int4 (`.int4.ple4.metal.v17.giw`, 2,668.0 MB): the eight G3 prompts (the registration said five; G3's set has eight), each 32 tokens free-run on the int8 arm and teacher-forced through the int4 arm: **top-1 agreement 410/435 = 94.25%, mean KL(int8 ‖ int4) 0.03257 nats (max 0.35015)** (`g3.log`) | **AMBIGUOUS** (agreement in 90–95%; KL inside the PASS band): parked |
| PLE-G4 goldens | `decoder` (20 PASS, 26 skips) and the CPU goldens through the parity refresh (33 passed, 0 failed) unchanged. `TestS9EModelBatched_tiny` (metal, generated fixture, so not in CI): the batched E-model pass against the layer-major pass on the same weights reads 0.97980 on the synthetic 96-token prompt against its owner-amended 0.98 bar (main 0.99158; the golden 12-token prompt 0.9905 against 0.9907). At real size the same comparison does not move: `TestS9EModelBatched_realE2B` reads the batched pass 0.39 points under the layer-major one with either table (int4 96.09% against 96.48%; int8, main 96.88% against 97.27%) (`s9-real-int4table.log`, `s9-real-int8table-main.log`) | one tiny-fixture bar missed: part of D12 |

Planted defect, run once and reverted: the table following the pin again (G1 and G2 red).

The size the change saves: the E2B Metal sidecar falls 981 MB (3,649.0 → 2,668.0 MB), the per-layer table's arithmetic
(2.35 → 1.32 GB in its own bytes, less the int8 scales).
