# MEM8(b′): the transcode streams its embedding-shaped tables, 2026-10-22

Gates: `docs/tasks/task-memory-campaign-2026-10.md`, "MEM8(b′) gates" (SQE-G), registered before the code (5cb71b3f).
Machine: MacBook Pro, M1 Pro, 16 GB, macOS 26.6.2, Go 1.27.2.

**The build.** In a streaming GGUF transcode, `token_embd`, `output` and Gemma 4's `per_layer_token_embd` are not built in
heap. The build registers each one's source with the writer (`giwWriter.tables`, `streamedTable`), and `writeHeadGlobals` writes
it in two passes over the source rows:

1. every row's scales, which the format puts first;
2. the rows quantized again, written a block at a time (32 MiB of codes). On a cpu-arm64 target each 4-row-aligned block is
   row4-repacked (`repackRow4CodesForEmit`).

The resident load is unchanged, and the safetensors transcode is out of scope.

| gate | reading | result |
|---|---|---|
| SQE-G1 bytes | `TestMEM8bp_streamedTablesAreTheWholeTableBytes`: gemma1-tiny, gemma2-hd-tiny, glm-tiny and the synthetic E2B-shaped GGUF × int4, int8int8, int8, f32 × `EmbedInt4` off/on × targets none, metal, cpu-arm64, cuda × the default block and a 200-byte block (a table across many blocks, int4 in 4-row groups): 256 cells, every `.giw` byte-identical to the whole-table write. `TestGemma4GGUF_streamedMatchesResident` (streamed against resident bundles) still passes | **PASS** |
| SQE-G2 peak heap | `TestMEM8bp_e2bTranscodeHeap`: E2B (`~/models/gemma-4-e2b-gguf/gemma-4-E2B_q4_0-it.gguf`), int4, metal target, main against the branch, one at a time, Go heap in use sampled every 20 ms. **Peak 2.835 → 0.589 GB (−2.25 GB)**; both outputs 2,781,780,206 B with sha256 `64cee222…a55e5`, byte-identical; the transcode took 5.6 → 3.0 s. Log: `~/goinfer-logs/mem8bp-g2.log` | **PASS** (bar: −1.0 GB and identical bytes) |
| SQE-G3 unchanged | `scripts/touched.py` green; the decoder's transcode, `.giw`, sidecar, split-GGUF, PLE and serialization tests pass | **PASS** |

**Plant** (pass two quantizing each int8 row into the previous row's slot): 160 cells red, every int8 cell.
