# KV0 — the tile census, the block size and the pool shape (2026-10-13)

`docs/tasks/task-kv-architecture-2026-10.md` §6 KV0 item 2. A desk reading: every kernel that reads or writes KV by absolute
position, read from source at main `dca88ad3`, nothing run (Metal could not be run on this box in any case). The per-kernel
tables are in two files beside this record, `kv0-tile-census-2026-10-13/cuda-kernels.md` and
`kv0-tile-census-2026-10-13/metal-webgpu-kernels.md`; the arithmetic is `kv0-tile-census-2026-10-13/block_arithmetic.py` and its
output. This page is the result. A claim in the appendices that its reader did not see in code is marked INFERRED there.

## 1. What the census found

**Addressing is already per key everywhere.** Every reader on every backend computes a key's row address from its own
position (`base + key*kvDim + kvh*hd`, or a lane-strided variant). 43 CUDA reader entry points, 14 live Metal readers and
every WebGPU shader do it. So a block table is one extra lookup per key (`table[k/B]*B + k%B`, hoisted per row or per block
where the loop allows), and no kernel has to change the grouping of its arithmetic.

**There is almost no contiguous key tile to align to.**

| backend | contiguous multi-key load | where it starts | needs |
|---|---|---|---|
| CUDA | `attn_fused*`: 64 keys per tile (32 in the experimental `bm32n32`), staged from per-key row pointers | full-attention layers: absolute multiples of 64 from key 0. Windowed layers: `startPos + qTile + 1 − window`, never aligned | B a multiple of 64 only if one base pointer per tile is ever used |
| CUDA | `splitkv_scores`: 128 keys per block, one thread per key, no reduction | full layers: multiples of 128; windowed: `winStart`, arbitrary | nothing (per-key) |
| CUDA | `fa_partial*` (flash-decode, **on by default from 2,048 keys**): 8-key warp blocks | `lo = winStart + sp·per`, `per = ceil(span/16)`: moves every ≤ 16 tokens | per-key lookup, never per-tile |
| CUDA | exact kernels (`attention`, `attn_batched`, `attn_block_full`, `attn_img_batched`, `mla_attn`) | none: thread t takes keys `winStart + t + 128j` | per-key |
| Metal | `attention_prefill_fused`: an 8-key `simdgroup_load` (one base, row stride `kvDim`) inside 32-key groups | absolute multiples of 8 and 32 | B % 8 == 0, and B % 32 == 0 to keep a group inside one block |
| Metal | `attention_prefill_steel`: 16-key blocks staged per key | absolute multiples of 16 | nothing |
| Metal | decode flash attention (`attention_fa*`, the MC3 row twins): 32-key blocks | `chunkStart + 32·(sgid + 4i)`, chunk length `ceil(nKeys/nSplit)`: moves with `nKeys` | per-key lookup; e.g. at nKeys = 3,900 the chunk-1 first block is keys 244..275, which crosses 256 |
| WebGPU | none: the 2,048-key `attn-keys` tile is a score-buffer tile; each lane reads one key's row | `p.start` | nothing |

So **B must be a multiple of 64 and nothing more** (CUDA's fused tile), and 64, 128 and 256 all satisfy every kernel. The census
does not choose among them.

**The design doc's bit-identity argument needs rewording.** §5.3 says tiles are aligned to blocks, so each kernel visits the
same keys in the same order. That is true for exactly one family, `attn_fused*` on full-attention layers. It is false for every
sliding-window reader, for flash-decode and the split-KV chunks on CUDA, for Metal's decode flash-attention family, and for
prefill writes at an arbitrary `startPos` (a reused prefix whose length is not a multiple of B). What keeps the bits is
different and simpler: **apply the table per key, and keep every tile, partition and thread-to-key map defined in absolute key
units**, as it is today. Re-cutting a windowed tile or a split chunk at a block edge would change the floating-point grouping:
that is a numerical change, not an addressing change, and would need re-baselining. A per-tile base-pointer optimisation is
bit-safe only for `attn_fused` on full layers.

## 2. B: recommend 64

| | 64 | 128 | 256 (the doc's working assumption) |
|---|---|---|---|
| allowed by every kernel | yes | yes | yes |
| Metal's prefill block (`PrefillBlockRows`: a multiple of 64, at least 256) | no change | needs rounding | needs rounding |
| CUDA prefill chunk 512 and windowed slack 512 | 8 blocks | 4 | 2 |
| Gemma-3-4B window group at 32K, f16: held / average waste | 123.2 MiB / 2.9% | 130.5 MiB / 5.6% | 145.0 MiB / 10.0% |
| tokens lost at a divergence or the end of a shared preamble | ≤ 63, average 32 | ≤ 127 | ≤ 255, average 128 |
| table entries and hash steps for a 32K sequence | 512 | 256 | 128 |
| one block of the 7B, f16 (a tier transfer unit) | 3.7 MB | 7.3 MB | 14.7 MB |

Reasons for 64: it is the smallest value every kernel allows, so it loses nothing to a kernel; it needs no change to Metal's
or CUDA's chunk sizes; it keeps the window group close to the window (a 1,024-token window holds 17 blocks of 64, 1,088
positions, against 5 blocks of 256, 1,280 positions); it shares more of a short preamble (opencode's 7,165 and Codex's 5,850
tokens: up to 3.6% and 4.4% lost at 256 against 0.9% and 1.1% at 64); and it sits inside the range the prior art measured
(PagedAttention's best was 16 to 128 tokens; see §11a of the task doc). What 64 costs is metadata, not memory: 512 table entries
a sequence and a hash step per block (about 82 µs for a 20,000-token prompt in an exploratory SHA-256 chain on this box's
3700X, one run, not a result) and more, smaller tier transfers (3.7 MB each at the 7B's f16, which a batched copy list absorbs).
**The one argument for a larger B is the host and disk tier's transfer size**, which KV7 can meet by moving runs of blocks.
128 is the fallback if KV1's index measures worse than this arithmetic says.

## 3. Pools per group (option A) or uniform pages (option B): recommend option A

The arithmetic, from the checkpoints' own geometries (`block_arithmetic.py`; f16 KV, B = 64 unless stated):

| model | groups | bytes per position | blocks per 32,768-token sequence | bytes held at 32K | average waste per sequence |
|---|---|---|---|---|---|
| Qwen2.5-7B (28 layers, 4 KV heads × 128) | one full group | 57,344 (f32: 114,688) | 512 | 1,792 MiB | 1.75 MiB (0.10%); 7.0 MiB (0.39%) at B = 256 |
| Gemma-3-4B (34 layers: 5 global, 29 local, window 1,024, 4 KV heads × 256) | global; local | 20,480; 118,784 | 512; **17** | 640 MiB; 123 MiB | 0.62 MiB (0.10%); 3.6 MiB (2.9%) |
| Qwen3.6-35B-A3B (40 layers: 10 full attention, 30 Gated DeltaNet; 2 KV heads × 256) | one full group + a state pool | 20,480 | 512 | 640 MiB | 0.62 MiB (0.10%); state 62.8 MiB per running sequence, independent of B |

Per-block bytes at B = 64: 3.67 MB (7B), 1.31 MB (Gemma global) and 7.60 MB (Gemma local), 1.31 MB (Qwen3.6). Waste is the
unfilled part of a sequence's last block per group: B/2 positions on average, B − 1 at worst.

**Why option A.**
- A window group's need is bounded per sequence (17 blocks for Gemma's 1,024 window), so the memory a uniform arena would make
  fungible between groups is small and predictable. Gemma-3-4B at 32K holds 123 MiB in its local group and 640 MiB in its
  global one; moving blocks between them is worth little.
- Option B pays a padding cost where layer counts do not divide evenly. Gemma-3's 29 local layers fill six groups of five, one
  slot of thirty padded: 3.3% of the local group. Gemma 4's layers differ in `kvDim` per layer, so a uniform page needs padding
  or an lcm page there (the CUDA census records the per-layer geometry), which the group-per-pool shape avoids.
- The recurrent state (62.8 MiB a sequence on the 35B-A3B) is a different size class from any block and needs its own pool in
  both options.
- It is also what TensorRT-LLM's hybrid manager does (task doc §11a).

**What both options leave open:** the planner's split between pools. It sizes the window and chunk groups for the running
sequences' needs and gives the rest to the full or latent group (§5.2).

## 4. The class map, per backend (KV0 item 3's "class map")

Which kernel class writes a row of a pass of m rows starting at s.
- **CUDA** (read from `cuda/prefill.go`, and **confirmed by every arm of `docs/measurements/kv0-invariance-2026-10-13.md`**):
  the fast kernels (`attn_fused`, `gemm_w4a8_mma`) when m ≥ 16 and max(promptLen, s + m) ≥ 512, where promptLen is the whole
  prompt's length for a pass inside a multi-chunk prefill and 0 otherwise; otherwise the exact kernels, which are bit-identical
  to decode. The prompt length is **the pass's end** for a prefill that arrives in two calls, so the first call of a split is
  judged on its own length.
- **Metal** (read, not run): below `metalFastPrefillFloor` a prompt runs as decode rows (`prefillByStep` over `mc3_*` kernels,
  or sequential `Forward`); above it the batched prefill kernels (fused, steel or exact by head dimension). Chunk invariance is
  the existing `TestMC5_prefillChunkInvariance`.
- **WebGPU**: no class depends on the prompt's length (per-row kernels; the one row-count selector, `w4MultiFor`, is
  bit-neutral and was pinned both ways in the invariance record).

## 5. What the census corrects in the task doc

- **§4's kernel counts.** CUDA has 8 writers and 43 reader entry points (about 19 distinct source bodies), not "about 30":
  `attn_fused*` is 8, not 10 (the other two of the doc's ten, `attn_vit_*`, read vision-tower scratch); 3 of the five `splitkv_*`
  read KV; `kv_store` is only ever launched at position 0, as a contiguous copy (windowed compaction, the K = V fill).
  Metal: `g4a_attn` is audio-tower scratch, `attention_fa_combine` and `mc3_attention_fa_combine_rows` read only split partials,
  `kv_store_f32` and `attention_f32` are compiled but never dispatched, and `rope2_kv` is unwired: 4 live stores and 14 live reader
  entry points. CUDA has no int8 KV and no scale tensors (f32, and f16 by default for cleared families).
- **§5.3's "Metal's `attention_fa_blk64` loads 64".** It is the head-dim-64 twin of `attention_fa_blk`; both load 32 keys a block.
- **§5.11's "Metal's 16-buffer bind limit".** The comments in `metal/batch_rows.go` and `metal/model.go` say 16; the pinned aikit
  `gpu` has `maxBindBuffers = 31`, and `mc3_attention_fa*_rows` binds exactly 16 today, so a table buffer (17) fits.
- **MC3 on CUDA is not a rows kernel.** `stepAttnRows` loops over rows and rebinds a slot per call; no CUDA kernel takes a
  per-row KV pointer or table, so §5.11's "a table per row" is new kernel API on CUDA, not an edit.
- **CUDA graphs capture no KV launch**, so a table pointer passed as a launch argument needs no recapture (read in
  `launchToken`).
- **WebGPU's limits matter more than its kernels.** Only three limits are raised; `maxStorageBuffersPerShaderStage` stays at
  8 (INFERRED from the WebGPU spec), and the int8 attention kernels already bind 7 storage buffers, so a table makes exactly 8. K and V
  must stay separate buffers (a writable and a read binding of one buffer in a dispatch is rejected, INFERRED). A slot is a
  whole `DecodeRunner` with baked bind groups, so paging means replacing the slot concept: this supports KV6's "keep slots".
- **The real cost of the kernel edit is PTX regeneration, not code.** CUDA kernels ship as precompiled `testdata/*.ptx`; the
  repository's convention is not to regenerate audited modules in place, and `prefill_batched.ptx` and `glue.ptx` carry the exact
  kernels every parity gate rests on. (INFERRED by the reader from `build_ptx.sh` and the comments in `attn_block.cu`.)
- **Hazards a pool introduces that the doc does not name.**
  - Metal's fused and steel prefill read up to 31 and 15 keys *below* a window's start and up to the next multiple of 8 past the
    last key (masked after the load): window-block reclaim must keep those mapped or point them at a finite scratch page, and a
    masked weight of 0 times a NaN is NaN.
  - CUDA's `attention` kernel computes `s*kvDim` as a 32-bit int, and the MLA kernels use an int32 `base`; they need `(long)`
    before a pool serves more than about two million positions of a wide layer.
  - WGSL's robust access silently clamps a wrong table index, so a paged gate must compare KV byte for byte.
  - The MC3 row kernels on Metal are string edits of the production kernel text; a signature change panics at init, which is a
    tripwire, not a drift.
- **A windowed fused prefill is chunk-invariant only for 64-aligned chunk starts (INFERRED by the CUDA reader).** The tile
  origin is `startPos + qTile + 1 − window`. The invariance record's arms ran with no windowed layer (the gate refuses a
  windowed load), so this is not measured: it is a class-map caveat for KV2/KV3 on windowed models, and KV1's windowed-layer
  tests should run the chunk and split arms at a start that is not a multiple of 64.
