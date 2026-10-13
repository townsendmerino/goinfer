> **Provenance.** Per-kernel reading for `docs/measurements/kv0-tile-census-2026-10-13.md`, made from source on 2026-10-13 at
> main `dca88ad3` by a read-only research agent working for the KV0 session, nothing executed. The session spot-checked the
> claims it relies on against the source: `BN`/`BM` in `attn_fused.cu`, `attention_fa_blk64` being the head-dim-64 block kernel,
> `maxBindBuffers = 31` in the pinned aikit. Everything else is the agent's reading; **INFERRED** marks what it did not read.
> Cited by declaration, never by line.

# KV0 tile census — CUDA

Tree: `/home/francis/wt/kv0-2026-10` (HEAD dca88ad3). Read-only; nothing in the repo was edited.
All paths are under `cuda/`. Citations are by declaration name. A claim marked **INFERRED** was reasoned, not read in code.

## 0. Method and scope

- Source read: every `.cu` that declares a `__global__` taking a K/V/latent cache pointer (`gemv_fwd.cu`, `glue.cu`,
  `prefill_batched.cu`, `attn_block.cu`, `attn_img_prefill.cu`, `attn_fused.cu`, `attn_fused_bm.cu`, `decode_splitkv.cu`,
  `decode_fa.cu`, `mla.cu`, `rope_pairwise.cu`, `rope_mrope_prefill.cu`, `kvtype.cuh`) and the launchers in `resident.go`
  (`decodeAttnGap`, `splitKVAttnDecode`, `segAMLA`, `UploadKV`, `copyF32`), `prefill.go` (`prefillChunked`, `prefillCore`,
  `useAttnFused`), `flash_decode.go`, `batchstep.go` (`stepAttnRows`), `kvwindow.go`, `drafter.go` (`DraftBlock`).
- CUDA KV today is **f32, or f16 under `--kv f16`/auto** for llama, mistral, qwen2, qwen3 and phi3 (`cudaKVF16Decline` in `cuda/kvpolicy.go`). There is **no int8 KV
  and no scale tensor on CUDA**, so the "scales" column is empty throughout. f16 is the same kernel source compiled with `-DKV_F16`
  (`kvtype.cuh`: `KV_T`, `KV_LD*`, `KV_ST`), so tile shape and addressing are identical and only the element is 2 bytes. f16 twins exist for
  `gemv_fwd`, `glue`, `prefill_batched`, `decode_splitkv`, `decode_fa`, `attn_fused`, `attn_fused_bm`. **No f16 twin** (so f32 only): `attn_block`,
  `attn_img_prefill`, `mla`, `rope_pairwise`, `rope_mrope_prefill`. f16 is declined for sliding-window, MLA, MoE, pairwise-RoPE, m-RoPE.
- **Counts differ from the doc's "about 30".** Real entry points that touch session KV by absolute position:
  - 8 writers: `kv_store` (only ever used as a contiguous device copy), `rope_kv`, `rope_kv_pw`, `rope_kv_batched`, `rope_kv_batched_pw`,
    `rope_kv_mrope_batched`, `rope_kv_mrope_batched_pw`, `mla_latent_store`.
  - 43 readers: `attention`, `attn_batched`, `attn_block_full`, `attn_img_batched`, **8** `attn_fused*` (hd64/hd128 x {64x64, bm32, bm32n32, bm128}; the doc's
    "ten" includes the two `attn_vit_*`, which do not read session KV), **3** of the doc's "five" `splitkv_*` (`splitkv_scores`, `splitkv_vsum`,
    `splitkv_vsum_partial`; `splitkv_softmax` and `splitkv_vsum_combine` touch only the score/partial scratch), 3 `fa_partial_{64,128,256}`,
    24 `fa_partial_rows_{64,128,256}_g{1..8}`, `mla_attn`. The two `fa_combine*` read only partials.
  - Distinct source bodies that need an edit: 7 writer bodies (kv_store needs none) and 12 reader bodies (13 counting both `attn_fused` files).
    The 24 + 8 + 3 entry points are one-line instantiations of `fa_partial_rows_impl`, `attn_fused_impl`, `fa_partial_impl`.
- **Not session KV, excluded:** `attn_vit_hd72_bm64/bm128` (`attn_fused_vit.cu`) and `tower_attn_hd64/72/80` (`tower_base.cu`) read vision-tower scratch; `attn_img_batched` is
  *also* launched by `vision_encoder.go` with imgStart=0, imgEnd=M over tower scratch (that use is not KV). The block drafter keeps **its own cache**
  (`residentDrafter.kc/vc`, `DraftBlock`, `ExtendContext`) with `rope_kv_batched` + `attn_block_full`; it is a second KV, not a slot of the target's.
- **CUDA graphs do not capture any KV launch** (`resident.go` comment above `KVSlots`: graphs capture only segA/B/C; rope_kv and attention run live in the gap), so a
  table pointer passed as a launch argument needs no graph re-capture. (Read in `launchToken` in `cuda/resident.go`, the `gSegA/B/C` replays.)
- **MC3 on CUDA is not a rows kernel.** `stepAttnRows` in `cuda/batchstep.go` loops over the rows and calls `decodeAttnGap` once per row with `r.kc/r.vc/r.kvBase`
  rebound to that row's slot buffers. There is no CUDA kernel that takes a per-row KV pointer or `[rows][maxBlocks]` table today; every launch gets one base
  pointer pair. (`flashVerifyAttn` in `cuda/flash_decode.go` is the only multi-row-per-launch reader, and it is one sequence.)

## 1. Addressing and tile table

Notation: `kvDim = nKV*hd` per layer (differs per layer on Gemma 4). "Row" = one key position's `kvDim` elements. "per-key" = every key row separately addressed from
the kernel's single base pointer + `s*kvDim`; "stride" = single base + stride. `winStart` = `nKeys-window` when `window>0 && nKeys>window`, else 0.

### 1a. Readers

| kernel (file : launcher) | reads | key address expression | per-key or stride | contiguous key-load tile (per iteration / block / split) | what a thread loads | tile origin and advance |
|---|---|---|---|---|---|---|
| `attention` (`glue.cu`; launched as `r.fAttn` in `decodeAttnGap` in `cuda/resident.go`, **fallback only** when `!prefillReady`) | K, V | `kc + s*kvDim + kvh*hd` (K), `vc + (s*kvDim + kvh*hd + d)` (V); `s*kvDim` is **int (32-bit)** | per-key | none: thread t takes keys `winStart+t, +128, ...`; one block per q-head, 128 threads | K: scalar `KV_LD(ks+d)` loop over d; V: thread d walks s ascending | origin `winStart`; stride 128 |
| `attn_batched` (`prefill_batched.cu`; `r.bAttn` in `decodeAttnGap` for M=1 and in `prefillCore` for M rows) | K, V | `kc + (long)s*kvDim + kvh*hd` (K float4 via `KV_LD4V`); `vc + ((long)s*kvDim + kvh*hd + d)` | per-key | none (as `attention`); grid (nH, M), 128 threads; row m has `nKeys=startPos+m+1` and its own `winStart` | K: float4 along hd (f16: 8-byte `kv_ld4`); V: scalar per (s,d); f16 build unrolls 16 keys but keeps ascending-s FMA order | origin = row's own `winStart` (absolute); thread stride 128 |
| `attn_block_full` (`attn_block.cu`; `d.attnBlock` in `DraftBlock` in `cuda/drafter.go`) | K, V (drafter cache, f32) | `kc + (long)s*kvDim + kvh*hd` | per-key | as `attn_batched`, but `nKeys = startPos+M` for every row | float4 K | origin `winStart` (drafter passes window 0, so 0); stride 128 |
| `attn_img_batched` (`attn_img_prefill.cu`; `r.bAttnImg` in `prefillCore` when image `blocks` exist) | K, V (f32) | `kc + (long)s*kvDim + kvh*hd` | per-key | as `attn_batched`; `nKeys` widened to the image block end, `winStart` from the *causal* count (decoupled) | float4 K | origin `winStart` from causal count; stride 128 |
| `attn_fused_hd64`, `attn_fused_hd128` (`attn_fused.cu`; `r.bAttnFused64/128` in `prefillCore`, gated by `useAttnFused`) | K, V (f32 or f16) | stage loop `kc + (long)(s0+kk)*kvDim + kvh*HD` for `kk<nk`, then `+dd` | **per-key row base, per-element copy** | **BN = 64 keys per tile** (`#define BN 64`), `nk = min(64, blockMaxKeys-s0)`; one block per (head, **BM=64** query rows), 128 threads | scalar `KV_LDH(src+dd)`: consecutive threads take consecutive `dd` of one key row; K staged `[BN][HD+8]` f16, V staged transposed `[HD][BN+8]`; mma.sync m16n8k8 reads smem | `s0` starts at `blockMinWinStart` and advances by 64 (below) |
| `attn_fused_bm32_hd64/128`, `attn_fused_bm32n32_hd64/128`, `attn_fused_bm128_hd64/128` (`attn_fused_bm.cu`; same launch site in `prefillCore`, `r.attnTile`) | K, V | identical to `attn_fused_impl`, template `<HD,BM,BN>` | same | BN = 64 (bm32, bm128), **BN = 32 (bm32n32)**; BM = 32 or 128 query rows (bm128 is the **default for hd128 layers with no window**, `tile128Launches`) | same | same rule; bm32n32 changes the bits (BN is part of the numerical contract) |
| `splitkv_scores` (`decode_splitkv.cu`; `r.skScores` in `splitKVAttnDecode` in `cuda/resident.go`) | K | `kc + (long)s*kvDim + (long)kvh*hd`, `s = winStart + blockIdx.y*128 + tid` | per-key | **128 keys per block**, grid (nH, ceil(nWin/128)); one thread = one whole key dot product; **no reduction, no cross-key state** | float4 K (`KV_LD4V`), q staged in smem | origin `winStart`, advance 128 (full layer: multiples of 128 from key 0) |
| `splitkv_vsum` (same file; `r.skVsum`) | V | `vc + ((long)s*kvDim + (long)kvh*hd + d)` | per-key | none: thread d owns output dim d and folds `s=winStart..nKeys-1` ascending (block 32 threads, grid (nH, ceil(hd/32))) | one float per (s,d) | origin `winStart` |
| `splitkv_vsum_partial` (opt-in, `GOINFER_SPLITKV_VSUM_SPLIT`, **not bit-identical**) | V | as `splitkv_vsum` | per-key | chunk `per = ceil((nKeys-winStart)/nSplit)`, `lo = winStart + sp*per` | one float per (s,d) | chunk origin `winStart + sp*per`; `per` changes with `nKeys` |
| `fa_partial_64/128/256` (`decode_fa.cu`; `r.faPartial` in `flashDecodeLaunch` in `cuda/flash_decode.go`; **default on** when `nWin >= 2048`, S=16) | K, V | K: `kc + (long)key*kvDim + (long)kvh*HD`, key = `b0 + (lane>>2)`; V: `vc + (long)(b0+kk)*kvDim + (long)kvh*HD` | per-key | CTA = 8 warps per (kv head, split sp) -> each warp takes **8-key blocks** `b0 = lo + 8*warp + 64*j`; split chunk `per = ceil(span/16)`, `lo = winStart + sp*per`; one 4-lane group per key for scores, all 32 lanes on one key for P.V | K: lane `sub` reads float4 #(j*4+sub), so 4 lanes read 64 contiguous bytes of one key row; V: float4 (hd>=128) or float2 per lane | origin `lo = winStart + sp*per`, **arbitrary**, and `per` moves every <=16 tokens as `nKeys` grows |
| `fa_partial_rows_{64,128,256}_g{1..8}` (24 entry points; `flashDecodeRowsLaunch` in `cuda/flash_decode.go` for spec-verify rows) | K, V | as `fa_partial_impl`; one K/V load serves R rows (`R = faRowsPerCTA`, 1..8) | per-key | same 8-key blocks from `lo = winStart + sp*per`; rows of a run share `(lo, per)`; a key past a row's own `hiRow` is masked | same | same as `fa_partial` |
| `mla_attn` (`mla.cu`; `r.fMlaAttn` in `segAMLA` in `cuda/resident.go`, M=1 only in production) | latent (in `r.kc[l]`, no V) | `latCache + (long)s*latDim` for the dot; `latCache[(long)s*latDim + d]` for the value fold over `d<rank` | per-key | none (thread-per-key like `attn_batched`, 128 threads) | float4 along latDim (576 for DeepSeek: 512 rank + 64 rope) | origin `winStart`; stride 128. Always f32, never windowed in the KV sense (`kvWindowSetup` in `cuda/kvwindow.go` skips `isMLA`) |

### 1b. Writers

| kernel (file : launcher) | writes | address | rows written per launch | start alignment |
|---|---|---|---|---|
| `kv_store` (`gemv_fwd.cu`; `r.kvCopy` via `copyF32` in `cuda/resident.go`) | K or V | `cache[(long)pos*n + i]`, **only ever launched with pos=0** (device copy: K=V layer's V fill, and the windowed compaction) | a contiguous run of n floats | n/a |
| `rope_kv` (`gemv_fwd.cu`; `r.ropeKV` in `decodeAttnGap`), `rope_kv_pw` (`rope_pairwise.cu`, f32 only) | K, V (rotated K in place + store) | `o = (long)pos*kvDim + h*hd`; `kc + o + d`, `+rhalf`, tail `+t` | **1 row** (`pos`); 1-D grid `g1cfg(nH*rhalf + nKV*rhalf + nKV*tail, 256)`. A KV-shared layer passes nKV=0 so nothing is stored | one `pos` per launch |
| `rope_kv_batched` (`prefill_batched.cu`; `r.bRopeKV` in `prefillCore`), `_pw` (`rope_pairwise.cu`), `rope_kv_mrope_batched` / `_pw` (`rope_mrope_prefill.cu`, `rope_pairwise.cu`; f32 only) | K, V | `pos = startPos + m` (grid.y = m), `(long)pos*kvDim + h*hd` | **M rows** `[startPos, startPos+M)`; grid.y = M. The m-RoPE twins keep row-sequential storage `pos = startPos+m` and only widen the rotation angle | first row = `startPos`: `0` for a cold prompt, the reused-prefix length otherwise (arbitrary), then `+512` per chunk (`prefillDefaultChunk` (= 512) in `cuda/prefill.go`; `prefillChunked` caps a windowed model's chunk at `kvWindowSlack+1 = 513`, i.e. still 512; OOM halves it down to `prefillMinChunk = 32`) |
| `mla_latent_store` (`mla.cu`; `r.fMlaStore` in `segAMLA`) | latent row | `latCache[base + i]`, `base = pos*mlaLatDim` passed as **int32** | 1 row, 1 block of 64 threads | `pos` |
| host `UploadKV` (`UploadKV` in `cuda/resident.go`) | K, V | one `gpu.Upload(r.kc[layer].At(base*kvDim*elem), keys)`; windowed layers go through `uploadKVWindowed` in `cuda/kvwindow.go` (tail only, to physical slot 0) | `n` contiguous rows from `base` (a CPU ring may start later than 0) | `base` arbitrary |
| spec-verify (`PrefillLastN*`, `StepBatch` rows) | via the same `rope_kv_batched` / `rope_kv` | as above | verify batch of M rows starting at the current position | arbitrary |

## 2. Alignment and straddle analysis

"Straddle" = the load unit is not contained in one block. For per-key rows the unit is one key row (`kvDim` contiguous elements), which a block
holds whole, so there is no straddle at any B.

### 2a. Kernels with no key tile (thread-per-key or thread-per-dim): `attention`, `attn_batched`, `attn_block_full`, `attn_img_batched`, `mla_attn`, `splitkv_vsum`

- **Straddle: no, for B = 64, 128, 256**, because nothing is loaded as a multi-key unit. Each key is an independent dot product in one thread; the V pass is one scalar load per (s,d).
- **Block-table change:** a lookup per key in the score loop (once per key, outside the d loop), and in the V loop either a per-iteration lookup or, better,
  an outer loop over blocks with the existing ascending-s loop inside it.
- **Bit identity must preserve:**
  - the thread-to-key assignment `s = winStart + t + j*nt` with `nt = 128` (each thread's `ls += e` runs over its strided keys in order, then a fixed tree over 128 partials). It is defined relative to `winStart`, an absolute position, **never relative to a buffer or block**;
  - the ascending-s FMA chain per output dim `d`;
  - the d-order of the dot (float4 with four separate fmas).
  The max reduction is order-free; the exp-sum reduction is not.
- **Reordering risk if a lookup restructure is careless:** moving to a block-outer loop in the V fold must keep a single accumulator walked in ascending s (that is the same order, so it is safe); splitting it into per-block accumulators changes bits.

### 2b. `attn_fused_*` (the one genuine key tile)

- Tile = `BN = 64` keys (32 for `bm32n32`), K rows staged per element from per-key row pointers.
- **Full-attention layers (`window == 0`):** `blockMinWinStart = 0`, so `s0 = 0, 64, 128, ...` in **absolute** key units, whatever `startPos` is (it only changes `blockMaxKeys`). Tiles are aligned to multiples of 64 from key 0.
  - B = 64: tile == block, never straddles. B = 128: 2 tiles per block, never straddles. B = 256: 4 per block, never straddles.
  - BN = 32 arm: aligned to 32, never straddles for any of the three.
  - Last tile is short (`nk < 64`); it still lies inside one block.
- **Windowed layers (`window > 0`):** `blockMinWinStart = max(0, startPos + qTile + 1 - window)` where `qTile = blockIdx.y*BM` and `startPos` is the pass's first row. Tiles are `blockMinWinStart + 64*j`.
  - This is **unaligned by construction**: `startPos + qTile` is the absolute position of the query tile's first row, and the "+1 - window" makes it `1 (mod 64)` even in the best case where both `startPos+qTile` and `window` are multiples of 64.
  - Origin differs per query tile and per pass (chunk) because it depends on `startPos + qTile`.
  - Straddle: **yes for every B (64, 128, 256)** if the kernel loaded a tile through one base pointer; **no** if it keeps addressing each key row separately, which the staging loop already does.
  - Early in a sequence (`startPos+qTile+1 <= window`) the origin collapses to 0 and the tile is aligned.
- **Block-table change:** replace `(s0+kk)` with a per-key lookup in the two staging loops (per-element today; the lookup can be hoisted per `kk`). Cost is staging-only; the mma loop reads shared memory.
- **Bit identity must preserve:** the tile partition in absolute key units, i.e. `BN = 64` and, for windowed layers, the origin `blockMinWinStart`. The online softmax rescale (`mRun`, `lRun`, `acc *= alpha` once per tile) is re-associated by tile, so **moving a tile edge to a block edge changes bits** for windowed layers. The shuffle order in the quad (`xor 1, 2`) and mma k-order are independent of addressing.
- Contract worth recording (**INFERRED** from the kernel plus `prefillChunked` in `cuda/prefill.go`): for windowed fast prefill the tile origin depends on `startPos+qTile`, so chunk-invariance of the fused windowed path holds only when every chunk starts at a multiple of 64 (the 512 default with `startPos = 0` does; a reused prefix of arbitrary length, or an OOM-halved chunk of 32, does not). The exact `attn_batched` path has no such dependence. This is relevant to KV0's chunk-invariance twin test.

### 2c. `splitkv_scores`

- One block covers 128 consecutive window indices; a thread owns one key, so there is **no reduction and no order to preserve**; the dot is the same d-order fma chain.
- Full layer (`winStart = 0`): block y covers keys `[128y, 128y+127]`. B = 64: spans two whole blocks (not contained in one); B = 128 or 256: contained, never straddles.
- Windowed layer: origin `winStart = nKeys - window`, arbitrary and moving every token; straddles for any B.
- Per-key row addressing already, so a per-key lookup is exact. `splitkv_softmax` and `splitkv_vsum_combine` read only scratch (`skScoreBuf` is `[nH*ctxCap]`, `backend.go`), not KV.

### 2d. Flash-decode (`fa_partial_*`, `fa_partial_rows_*`) — default on from 2048 attended keys

- Units: each warp's 8-key block (scores: one key per 4-lane group; P.V: one key at a time), CTA sweep 64 keys (8 warps x 8), split chunk `per = ceil(span/16)`.
- Origin `lo = winStart + sp*per`. For a full layer `lo = sp*per`, which is a multiple of 8 only when `per` is, and `per` changes as `nKeys` grows (every <=16 tokens). Windowed layers add the moving `winStart`.
- 8-key block straddle: **only-if `lo mod 8 != 0`** (then some block start has `b mod B > B-8`, for any B in 64/128/256); with `lo mod 8 == 0` and B a multiple of 8 an 8-key block never straddles. The split chunk (`per` keys) straddles block boundaries for every B except by accident.
- Per-key row pointers already (`kp = kc + key*kvDim + kvh*HD`; `vp` per `b0+kk`), so a per-key lookup is exact.
- **Bit identity must preserve:** partition `(lo, per, nSplit=16)` computed from `(nKeys, winStart)` in absolute units; 8-key block order `lo + 8*warp + 64*j`; the `2+3+3` shuffle ladders; `__expf`; the fixed ascending unit order in `fa_combine` (U = 128 units). None of this refers to a block grid, and a block table must not introduce one. Note this lane is already **not** bit-identical to the exact path (fidelity-gated).
- `fa_partial_rows` rows of a run share `(lo, per)` and differ only in `hiRow`; the host groups them by `faRowRuns` on `(winStart, per)`. A table lookup does not change that grouping.

### 2e. Writers

- `rope_kv`: one row per launch, so one block lookup per launch (or the host passes the row's base pointer, as the windowed view does today).
- `rope_kv_batched` family: **M rows per launch from `startPos`**, `startPos` arbitrary. A 512-row pass at `startPos = 0` covers exactly 8/4/2 blocks of 64/128/256; at an arbitrary `startPos` it starts mid-block. Chunk starts are `startPos + i*chunk`; with chunk=512 they are block-aligned only when `startPos mod B == 0`. Each row is already independently addressed (grid.y), so a per-row `table[pos/B]` is natural and bit-neutral (a store has no arithmetic, and rope's angle depends on `pos`, not on the buffer).
- `mla_latent_store`: one row; `base` is already an explicit int32 argument (`pos*latDim`).
- `UploadKV`: one contiguous host-to-device copy of `n` rows from `base`; paged storage turns it into a copy per block run.
- **Windowed view (`kvwindow.go`):** a windowed layer holds `window + 512` positions (`decoder.WindowedKVSlack`); kernels still compute `kc + pos*kvDim`, and `kview` hands them a pointer shifted back by `base*kvDim*4` bytes (**the `4` is f32 only**; f16 and windowed are mutually exclusive by policy). `base` is `max(0, p0-(window-1))` at the last compaction, arbitrary. `kvCompactSlot` moves the live tail `[newBase, p0)` to physical slot 0 through `kvScratch` using `copyF32` (the `kv_store` kernel) when the write would pass `base+cap`. A pool removes both the shifted view and the compaction (they are on the doc's "does not carry over" list).
  - Alignment consequence: because the view is shifted by whole rows (`base*kvDim*4` bytes, 16-byte multiple whenever `kvDim % 4 == 0`), float4 loads stay aligned; with a pool, each block base must stay 16-byte aligned (`B*kvDim*elem % 16 == 0`, true for B >= 4 and kvDim % 4 == 0).

## 3. What a pool would change that is keyed on `ctx` / padded length

- **Allocation per slot per layer is `ctxCap*kvDim` elements** (`resident.go`: `r.kc[l], r.vc[l] = r.akv(l, r.kvPositions(l)*kvDim)`; `allocKVSlot` repeats it per slot; `kvBytesForCap` is linear in `ctxCap`, piecewise for windowed layers). `ctxCap` is fit up to 16384 with a 4096 floor (`cudaCtxCapDefault`), `trimUnpinnedCtx`, `kvSlotsFit`, `checkKVFits`. A pool replaces all of that with a block budget.
- **`checkCap(pos, n)` / `ContextCap`** guard against `ctxCap`; **`UploadKV`** uses it too.
- **`skScoreBuf = [nH * ctxCap]`** (`backend.go`, split-KV scratch): sized by ctx, not by the pool. It would need `nH * maxAttendedSpan` per concurrent step.
- **Shared-memory cap:** single-block exact attention allocates `(nWin+128)*4` bytes; `singleBlockAttnShmemLimit = 48 KiB` gives the 12,160-key ceiling (`splitKVRequired`, `checkPrefillShmem`, `checkDrafterShmem`). It is keyed on the attended span, not on storage, so a pool does not change it.
- **Per-slot state:** `kvSlotBufs`, `kvBases[slot]`, `kvBase`, `UseKVSlot`, `ReusableKV` (windowed history lost below `base+window-1`). `kvRoomFor`/`kvEnsureSlot` bound rows per pass by the slack. All replaced by a table plus lifetime rules.
- **Int32 position arithmetic that a pool should not inherit:** `attention` (`glue.cu`) computes `s*kvDim` as int; `mla_latent_store`/`mla_attn` use int `base+i` and an int32 `base` argument. Fine at 16k-32k contexts, wrong at a few million positions. The other readers use `(long)`.
- **Per-layer geometry:** `kvDim` differs by layer on Gemma 4 (`kvPositions`, `kvBytesForCap`), and a KV-shared layer aliases its source's buffers (`allocKVSlot`, `kvShared`). A block is `B*kvDim_l*elem` bytes, so layers with different `kvDim` need separate pools (or a common page of `B*lcm`-sized byte pages, the doc's option B).
- **MC3:** one base pointer pair per launch; per-row launches (see section 0). A table-per-row interface would be new kernel API, not an edit of an existing one.

## 4. Bit-identity notes by kernel class (what must not move)

| class | grouping that fixes the floating-point result | does a per-key table lookup change it? | does per-tile base-pointer addressing (B-aligned) change it? |
|---|---|---|---|
| exact thread-per-key (`attention`, `attn_batched`, `attn_block_full`, `attn_img_batched`, `mla_attn`) | key s -> thread `(s-winStart) mod 128`; per-thread partial sum in ascending s; 128-wide tree; V fold ascending s per d | no | no tile exists; a B-aligned restructure of the loop is safe only if each thread's key sequence and the V accumulator order are kept |
| `splitkv_scores`/`splitkv_vsum` | per-key dot (no reduction); V fold ascending s | no | n/a |
| `attn_fused*` | tile = 64 (or 32) keys in absolute units from key 0 (full layers) or from `startPos+qTile+1-window` (windowed); online softmax per tile | no | **full layers: no (B multiple of 64). windowed layers: yes — would re-associate** |
| `fa_partial*` (default decode >= 2048 keys) | partition `(lo, per, 16)` and per-warp 8-key block order; fixed-order combine | no | would break the partition (it is not B-aligned) |
| writers | none (stores; rope math depends on `pos`) | no | n/a |

## 5. Synthesis

- **Contiguous key-load tiles that exist:** 64 (`attn_fused*` BN, the only key tile aligned from absolute 0 and only on full layers), 128 (`splitkv_scores` block, trivially per-key), 8 (a `fa_partial` warp block, origin arbitrary), 32 (experimental `bm32n32` BN), plus non-memory strides 128 (exact kernels' thread stride, relative to `winStart`) and 16 (the f16 V unroll, inside the s loop). **LCM of the aligned-from-0 tiles = 64; LCM including the 128-key `splitkv_scores` block = 128.** Any of B = 64, 128, 256 is a multiple of every one of them, so the doc's constraint ("B multiple of every contiguous key-load tile") **does not rule out any of the three** and does not choose between them on CUDA.
- **The constraining kernel for B** is `attn_fused*` on full-attention layers (BN = 64): B must be a multiple of 64 if it is ever to use one base pointer per tile. B = 32 would break it. Nothing in CUDA forces B above 64.
- **The doc's bit-identity argument ("tiles are aligned to blocks, so each kernel visits the same keys in the same order") is true on CUDA only for `attn_fused` on full layers.** It is false as stated for:
  - **windowed layers**: the tile origin (`attn_fused`), the thread-key assignment (exact kernels), the chunk `lo` (flash-decode), the 128-key blocks (`splitkv_scores`) all key on `winStart`, which is `nKeys - window` or `startPos + qTile + 1 - window`: arbitrary, and moving by one every token or every query tile;
  - **flash-decode and split-KV chunks**: `lo = winStart + sp*per`, `per = ceil(span/16)`, which changes every <=16 tokens;
  - **prefill writes at an arbitrary `startPos`**: a pass is not block-aligned unless the prefix length is a multiple of B.
- **What that implies:** the CUDA kernels keep their bits only if the block table is applied **per key** (`table[k/B]*B + k%B`, hoisted per row/block where the loop allows) and the **tile or partition definition stays in absolute key units**. Per-key addressing is already the code's shape in all 43 readers (`attn_fused` staging loops copy one element at a time from a per-key row pointer; the exact kernels, `splitkv_*`, `fa_*` compute `base + key*kvDim` per key). So no kernel needs a different grouping for a table; the risk is in a per-tile pointer optimisation, which is bit-safe only for `attn_fused` full layers. Do not realign windowed tiles or split chunks to block edges without re-baselining: that is a numerical change, not an addressing change.
- **Kernels that are unaligned by construction (list):** every sliding-window reader (`attention`, `attn_batched`, `attn_fused*` windowed, `splitkv_scores`, `splitkv_vsum*`, `fa_partial*`, `fa_partial_rows*`); both `fa_partial*` lanes even for full layers; `splitkv_vsum_partial`; and the batched writers at a non-multiple-of-B `startPos`.
- **Kernels aligned from key 0:** `attn_fused*` full layers (64), `splitkv_scores` full layers (128). The exact kernels have no block structure. `mla_attn` is thread-per-key on a latent row of `latDim` floats.
- **Cheap changes per kernel family:** 12 reader bodies + 7 writer bodies, each a one-line index change. The cost is not the code but the PTX: kernels ship as precompiled `testdata/*.ptx` regenerated by `build_ptx.sh`, with a convention (stated in `attn_block.cu`, `attn_fused_bm.cu`) of not regenerating audited modules in place because codegen shifts. `prefill_batched.ptx` and `glue.ptx` carry the exact kernels every parity gate rests on; editing them is the most expensive part of KV4.
- **Chunk-invariance observation for KV0 item 3 (INFERRED):** exact attention (`attn_batched`) is chunk-invariant by construction (per-row, absolute positions), consistent with `TestPrefillChunked_bitIdentical`. Fused windowed attention is chunk-invariant only for 64-aligned chunk starts (section 2b), and the fused path runs only for the final or ordinary chunks of a prompt of at least 512 tokens (`useAttnFused`, `forceExactKernels`, `fastPrefillFloor = 512`). The class map should record that for windowed layers a reused prefix of non-multiple-of-64 length changes the kernel's tile grouping.
- **Doc corrections to carry into KV0:** "about 30 CUDA kernels" is 8 writers + 43 reader entry points (about 19 distinct kernel bodies); `attn_fused_*` is 8 not 10 (the other two are vision-tower kernels); `splitkv_*` touching KV is 3 not 5; `kv_store` is used only as a contiguous copy (compaction, K=V fill), not as a per-position store; the windowed compaction copies via `copyF32`/`kv_store`, and CUDA has no int8 KV or scales.
