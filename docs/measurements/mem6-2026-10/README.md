# MEM6, the Metal routes: resident prefill embeds per chunk, 2026-10-22

Gates: `docs/tasks/task-memory-campaign-2026-10.md`, "MEM6 (Metal routes) gates" (MEM6-G), made concrete before the code (c2320410)
and G1's instrument refined before any build data (33993136). Machine: MacBook Pro, M1 Pro, 16 GB, macOS 26.6.2, Go 1.27.2.

**The build.**
- **Decoder:** a resident that implements `decoder.ResidentPrefillRows` reads a suffix's embeddings a range of rows at a time.
  The decoder embeds them into one reused slab, with one per-layer-input scratch for a Gemma 4 E-model
  (`embedResidentScratch`), so embedding row after row allocates nothing per row.
- **Metal** implements it on every route that took a whole suffix in one call:
  - the layer-major routes (paged Gemma 4, paged MoE, E-model), per 256-row chunk;
  - the step prefill, per piece;
  - the batched pass, which fills its f16 input rows and an E-model's per-layer inputs straight into their device buffers
    `rowChunk` (256) rows at a time. The host copy of every row's per-layer inputs (860 MB on E4B at 20K by arithmetic) is gone.
- The existing slice entry points are wrappers over the same code.

| gate | reading | result |
|---|---|---|
| MEM6-G1 bytes | `TestMEM6_prefillHostBytes`, `gemma4-emodel-tiny` with its window raised to 4,096, Metal-resident, prefills of 1,024 and 2,048 rows. (i) Most embedding rows the decoder held at once: **256 on both routes** (main: the whole suffix, 2,048). (ii) Batched route, Go bytes a prompt row adds: **−171 B** (flat; main 3,177 B, 1.77 embedding rows of 1,792 B). The layer-major route reads about 189 KB per row on main and the branch alike, command-encoding garbage that G1's refinement excludes | **PASS** |
| MEM6-G2 identity | the same prefills' last logits and 32 decode steps after them, hashed: equal to main's on both routes and both lengths (batched `0xd183d8b6…`, `0x0a71ddff…`; layer-major `0x88f6214a…`, `0xc40ab360…`) | **PASS** |
| MEM6-G3 unchanged | `touched.py` green; the Metal prefill, E-model, step, MEM7 and P3b tests; the decoder's Gemma 4, PLE and embedding tests; `scripts/refresh_parity_hashes.sh` green (forward goldens 43) | **PASS** |

**Plant** (the resident asking for the whole suffix in one rows call): red on both routes (2,048 rows held; 1,643 B per row on the
batched route).

**What the first bar missed, stated.** With the slab alone, the batched route read 592 B per prompt row, over the 224 B bar. The
profile put it in `gemma4PLEInputs`: each row allocated its own `[layers × per-layer dim]` context-aware buffer, 768 B a row here
and 43 KB a row on E4B (about 860 MB of churn at 20K). The embedding path now takes that buffer from a scratch. The bar was not
moved.

**Open:** the CUDA cells and the `RssAnon` return-the-memory gate (Linux, `mem0_linux.py`) are nobara's. CUDA's resident does not
implement `ResidentPrefillRows` yet, so it still takes the whole suffix.
