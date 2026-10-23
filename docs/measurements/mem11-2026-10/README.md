# MEM11-bf16: bf16 islands, 2026-10-22

Gates: `docs/tasks/task-memory-campaign-2026-10.md`, "MEM11-bf16 gates" (B16-G), registered before the code (8bc831f2), on D3
(bf16 first). Machine: MacBook Pro, M1 Pro, 16 GB, macOS 26.6.2, Go 1.27.2.

**The build.**
- `islandMat` holds an f32 island tensor as the bf16 `WeightMat` kind when every value is exactly a bf16 value (bits compared, so
  a NaN payload bf16 would drop keeps it f32), and as f32 otherwise.
- `stIsland` reads an island from a safetensors file the weights keep open: an F32 tensor as its zero-copy f32 view (unchanged),
  a BF16 tensor as its zero-copy bf16 bits, anything else through `islandMat`.
- The bf16 kind's matmul widens a tile and runs `MatmulBT`, so the forward is the f32 one's, bit for bit.
- The `.giw` format is unchanged: islands are still written as f32, and `islandMat` makes them bf16 on load.
- Families: LFM2's short-conv `inProj` and `outProj`; the DFlash and DSpark drafters' projections. CUDA plans a bf16 weight as
  an f32 one and widens its rows before the same int8 quantization.

| gate | reading | result |
|---|---|---|
| B16-G0 measure | LFM2.5-2.6B (`~/models/lfm2.5-2.6b`, BF16 safetensors) at int4 on the CPU, direct load, main against the branch: `ResidentWeightBytes` 4.063 → 3.325 GB; the 22 conv layers' projections 1.476 → 0.738 GB (36.3% of the model before, not the "over half" the doc's arithmetic gave); **Go heap in use after load 4.066 → 2.590 GB (−1.476 GB)**: the projections are now the checkpoint's mapped bf16 pages. Logs: `~/goinfer-logs/mem11-g0-heap.log` | read |
| B16-G1 identity | `TestMEM11_lfm2BF16Islands`: lfm2-tiny with its projections rounded to bf16, f32 against bf16 islands: 0 of the logits over 16 tokens differ; a serialization round trip reads them as bf16 and decodes the same logits. `TestMEM11_drafterBF16Islands`: a two-layer drafter trunk with bf16-exact weights drafts bit-identically (0 of 384 values), `fc` included; `stIsland` keeps an F32 tensor f32 and a BF16 tensor bf16 with the same rows | **PASS** |
| B16-G2 bytes and kinds | the islands are bf16 exactly when bf16-exact (`TestMEM11_islandMatRule`), their bytes halve (lfm2-tiny 196,608 → 98,304; drafter q+down 81,920 → 40,960), `ResidentWeightBytes` falls by exactly that, and the fit-estimate agreement tests pass | **PASS** |
| B16-G3 unchanged | the committed tiny fixtures are F32 safetensors, so their islands stay f32 views and their goldens are unchanged; `scripts/refresh_parity_hashes.sh` green (one `deps_hash` line, goldens 33) | **PASS** |

**Plant** (the exactness check skipped, so every island is rounded): red in `TestMEM11_islandMatRule`, and in the `.giw` round
trip (`TestLFM2_serializeRoundTripsTheConvMixer`: `-0.008598174` read back as `-0.008605957`). The registered expectation that
G3's goldens would go red did not hold: the committed fixtures are F32 on disk and never reach `islandMat`. The rule test and
the round trip are what guard it.

**A defect this build shipped, found and fixed 2026-10-22.** A direct load of a bf16 LFM2 checkpoint released the source mapping
its islands alias. `stIsland` keeps a BF16 tensor as a view of the file, and the end-of-load release (`mmapAliasRisk`) judges by
dtype alone, so on a checkpoint with no F32 tensor (LFM2.5-2.6B) it closed the mapping. The first read of an island then faulted
(found writing a `.giw` from the 2.6B for MEM11-giw-bf16: SIGSEGV in the writer's CRC). B16-G0 read byte counts and the heap but
ran no forward, and the tiny fixtures are F32, whose other tensors keep the mapping anyway. The fix: a load holding a bf16 island
keeps the mapping (`islandsMayAliasSource`). Its test, `TestMEM11_lfm2BF16CheckpointKeepsItsMapping`, loads an all-BF16 copy of
lfm2-tiny at int4 and f32. It is red without the fix, and its logits equal the same values stored as F32. On the real 2.6B at int4,
the mapping is retained and four decode steps run.

**Not run on the Mac:**
- The real LFM2 gate (`TestLFM2Real_gate`, realckpt) loads the 2.6B at f32, about 10 GB, which this Mac's free memory does not
  hold; G1's bit-identity covers the change, and the heavy tier runs the gate.
- The CUDA drafter upload with a bf16 drafter (the packer change) is vetted and staticchecked from the Mac but runs only on
  nobara: left open.

**MLA is not changed, and is a question for the owner.** The one MLA checkpoint in the bench set (DeepSeek-V2-Lite GGUF) stores
its MLA projections as Q4_K. Dequantized they are neither bf16- nor f16-exact, so a 16-bit island saves nothing there. Its
1.49 GB comes from widening Q4_K, and keeping those bytes is the int8/quant half of D3, behind a parity bar. A bf16 MLA island
would help only a BF16 safetensors DeepSeek or Kimi, at the cost of touching the CPU forward's row reads and three GPU builders.
