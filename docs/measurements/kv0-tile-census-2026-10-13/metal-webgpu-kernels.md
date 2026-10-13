> **Provenance.** Per-kernel reading for `docs/measurements/kv0-tile-census-2026-10-13.md`, made from source on 2026-10-13 at
> main `dca88ad3` by a read-only research agent working for the KV0 session, nothing executed. The session spot-checked the
> claims it relies on against the source: `BN`/`BM` in `attn_fused.cu`, `attention_fa_blk64` being the head-dim-64 block kernel,
> `maxBindBuffers = 31` in the pinned aikit. Everything else is the agent's reading; **INFERRED** marks what it did not read.
> Cited by declaration, never by line.

# KV0 tile census: Metal and WebGPU

Worktree read: `/home/francis/wt/kv0-2026-10` (HEAD dca88ad3). Read-only. Linux box: nothing here was run; every claim is from source unless labelled **INFERRED**.
Cites are by declaration name. MSL is embedded in Go string constants, not `.metal` files:
`metal/kernels/kernels_msl.go:AllKernels`, `metal/kernels/prefill_msl.go:PrefillKernels`, and the MC3 row kernels, which are *derived by string edit at init* from the first two in `metal/batch_rows.go:deriveRowsKernels`.

## 0. Headline findings (read these first)

1. **The doc's premise "attention_fa_blk64 loads 64 keys" is wrong.** `attention_fa_blk64` is the *head-dim-64* twin of `attention_fa_blk` (the 0.5B, G=7). Both load **32 keys per simdgroup block**, one key row per load (blk: 32 lanes x half4 = 256 B per key; blk64: 32 lanes x half2 = 128 B per key). There is no 64-key load tile anywhere on Metal.
2. **Almost no Metal or WebGPU kernel loads a contiguous multi-key tile.** Every decode kernel, the steel prefill kernel and every WGSL kernel address **one key row at a time** (`s*kvDim + kvh*hd + d`, or a lane-strided variant). The only true multi-key contiguous load is `attention_prefill_fused`'s 8-key `simdgroup_load` (one base pointer + row stride `kvDim`). So the "B is a multiple of every kernel's contiguous key-load tile" rule binds exactly one kernel family on Metal (needs `B % 8 == 0`; `B % 32 == 0` if you want its 32-key group inside one block) and none on WebGPU. A per-key lookup `table[k/B]*B + k%B` is correct everywhere without any alignment condition.
3. **Metal decode flash-attention blocks are unaligned by construction.** `attention_fa*` split chunks start at `winStart + split*ceil(nWin/nSplit)` and each simdgroup's 32-key blocks start at `chunkStart + 32*(sgid + 4i)`. Chunk length depends on `nKeys`, so block starts shift every token (e.g. nKeys=3900, nSplit=16: chunkLen=244; the chunk-1 first block is keys 244..275, which crosses key 256 and so straddles at B=64, 128 and 256). These need a per-key lookup (or a two-segment fast path), never a per-tile one.
4. **g4a_attn, tower_attn_hd*, eg2_* are not KV kernels** (audio/vision tower and encoder self-attention over per-call scratch). Remove `g4a_attn` from §4's list. `attention_fa_combine` and `mc3_attention_fa_combine_rows` read only the split partials, not KV, so they need no table either.
5. **Dead or unwired Metal kernels in the §4 list:** `kv_store_f32`/`attention_f32` (compiled, never dispatched: `r.kvF32 = false` is a constant in `resident` build), `rope2_kv` (comment says "NOT wired into any production dispatch site"; test-only). Live Metal KV kernels: 4 stores + 14 reader entry points (§1).
6. **Stale 16-buffer limit.** `metal/batch_rows.go` and `metal/model.go` (kvContig comment) say "aikit binds at most 16 buffers". The pinned `github.com/townsendmerino/aikit/gpu v0.33.5` (`metal/go.mod`; read in the module cache `gpu@v0.33.5/metal.go`) has `const maxBindBuffers = 31` (Metal's argument-table limit) and panics only above 31. `mc3_attention_fa*_rows` binds exactly 16 today; adding a table buffer (17) is legal. The 16 is a stale comment, not a constraint (the comment may still be true for some older aikit pin; the pin in this tree is 31).
7. **WebGPU has no multi-sequence row kernel at all.** MC1 only; the "rows" WGSL kernels (`attnKeysRows`, `attnBatched*`) are M rows of ONE sequence. A WebGPU "slot" is a whole `DecodeRunner` with bind groups baked over that slot's K/V buffers (`gpu/kv_slots.go:webgpuKVSlot`, `newKVSlot`).
8. **Only three of WebGPU's non-default limits are raised** (`gpu.go` device request: `MaxStorageBufferBindingSize`, `MaxBufferSize`, `MaxComputeWorkgroupsPerDimension`). Everything else stays at WebGPU defaults, notably `maxStorageBuffersPerShaderStage` (**8**, WebGPU spec default; INFERRED from spec, not read in this repo). The int8 attention kernels already bind **7 storage buffers** (q, keys, vals, kScale, vScale, ctx, sinks); a page-table binding makes **8, exactly the default cap**.

## 1. Metal: per-kernel census

Conventions: `kvDim = nKV*hd`. K and V are **separate buffers per layer** (`r.kc[l]`, `r.vc[l]`; int8 adds `r.ks[l]`, `r.vs[l]` = f32 scale per (position, KV head), index `pos*nKV + kvh`). A layer's slots are **one allocation** (`kvContig`) only when `kvSlots > 1 && !kvI8`: slot s at byte `s*kvSlotBytes[l]`, `kvSlotBytes = paddedCtxCap*kvDim*2`, `paddedCtxCap = ceil8(ctxCap)`; int8 slots are separate allocations. Windowed layers allocate the **full** ctxCap (no ring on Metal). KV-shared layers (Gemma 4 E) alias the source layer's buffers. Single-token dispatch is `e.Dispatch(p, nThreads, tg, ...)` (n = total threads).

Summary matrix (details per kernel below):

| kernel (host name) | live? | per-key or multi-key load | load width | tile / loop unit | tile start | straddle B=64/128/256 |
|---|---|---|---|---|---|---|
| kv_store | yes | per-pos write | scalar half | 1 row | pos | no (single pos) |
| kv_store_i8 | yes | per-pos write | char | 1 row + nKV scales | pos | no |
| kv_store_f16 | yes | M-row scatter, `positions[m]` | scalar half | 1 row each | startPos+m | no |
| mc3_kv_store_rows | yes | per-row pos + slotOff | scalar half | 1 row each | per-row | no |
| kv_store_f32, rope2_kv | dead / unwired | | | | | n/a |
| attention | yes | 1 key per thread | K half4 (8 B), V half scalar | 128-key sweep; V 8-key unroll; score tile 4096 | winStart | no (per-key) |
| attention_i8 | yes | same | K char4 + f32 scale/key; V char + scale | same | winStart | no |
| attention_f32 | dead | same | K float4 | same, V serial | winStart | n/a |
| attention_fa (legacy, G=1) | yes (G not in 2..8, hd 128) | 1 key per simdgroup iter, sgid-strided | half4/lane = 256 B row | 1 key; chunk = ceil(nKeys/nSplit) | chunkStart | no (per-key) |
| attention_fa_blk g2..g8 | yes (default for dense GQA, nKeys >= 1024) | 32 keys per block, one row per load | half4/lane = 256 B | **32-key block, chunk-relative** | `chunkStart + 32*(sgid+4i)` | **YES, unaligned** |
| attention_fa_blk64 g7 | yes (hd 64, G=7) | same | half2/lane = 128 B | same 32-key block | same | **YES** |
| attention_prefill | yes (hd>128 or fused off) | 1 key per thread | K half4 | 128-key sweep; sc[s] absolute, max 4096 keys | winStart | no |
| attention_prefill_img | yes (image turns) | same | K half4 | same | winStart | no |
| attention_prefill_fused | yes (hd%8==0, hd<=128, not steel) | **8-key simdgroup_load** (stride kvDim), 4 per 32-key group | 8x8 half tile, row stride kvDim | 8-key sub-tile; 32-key group | `floor(winStart0/32)*32`, abs-aligned | **no for B%32==0**; needs B%8==0 |
| attention_prefill_steel | yes (hd 128) | per-key half4 staging into threadgroup | half4 | 16-key block | `floor(winStart/16)*16`, abs-aligned | no (per-key) |
| mc3_attention_rows | yes | as attention | | | | no |
| mc3_attention_fa_rows, mc3_attention_fa_blk_rows g2..g8, mc3_attention_fa_blk64_rows_g7 | yes | as fa/fa_blk/blk64 | | | | as their base kernels |

### 1.1 Stores

**kv_store** (`AllKernels`; launcher `resident.encodeAttentionResidualWith` -> `e.Dispatch(r.pKv, g.kvDim, 64, qkv.At(kOff), qkv.At(vOff), r.kc[l], r.vc[l], g.uKvDim, uPos)`)
- Writes K, V (f16). K was already rotated by the separate `rope2` into the qkv buffer; `kv_store` copies. One thread per element of the row: `kc[pos*kvDim + i] = half(k[i])`.
- Address: `base + pos*kvDim + i`, single base + stride. 6 buffers (k, v, kc, vc, kvDim, pos).
- Tile: none (one position per launch). Paging: `pos*kvDim` -> `(table[pos/B]*B + pos%B)*kvDim`; `pos` is uniform per dispatch so the lookup is one scalar load.

**kv_store_i8** (launcher same site, `r.kvI8`; `e.Dispatch(r.pKvI8, g.nKV, 1, ...)`)
- One *thread* per KV head (tg=1): loops hd for absmax -> scale, then quantizes. Writes `kc/vc[pos*kvDim + kvh*hd + d]` and `ks/vs[pos*nKV + kvh]`. 9 buffers.
- Paging must translate both the data row and the scale row (scale block `[numBlocks][B][nKV]`, same table).

**kv_store_f16** (`PrefillKernels`; launcher `resident.prefillLastImg`: `e.Dispatch(pf.pKv, M*lu.kvDim, 128, qkvF, r.kc[l], r.vc[l], posB, ...)`)
- Writes M rows at `positions[m]` (`posv[m] = startPos + m`, strictly contiguous). `kc[pos*kvDim + i] = qkv[m*stride + kOff + i]`. 8 buffers. K here is already rotated (`rope_f16` before).
- `startPos` is whatever the caller passes: `PrefillBlockRows` multiples (see §1.6) from 0, but a reuse tail continues at an arbitrary common-prefix length. So the written run starts at an arbitrary offset inside a block and spans `ceil((startPos%B + M)/B)` blocks. Per-row lookup via `positions[m]` already exists; the run can also be split per block.
- Not run for int8 KV (`prefillOK` false when `kvI8`; such prompts take the sequential decode kernels).

**mc3_kv_store_rows** (derived in `deriveRowsKernels`; launcher `resident.forwardMultiOut`: `e.Dispatch(r.pKvRows, B*g.kvDim, 64, qkvB.At(kOff), qkvB.At(vOff), kcAll, vcAll, g.uKvDim, posB, uQKV, uM, slotOffB)`)
- Body is `kv_store` byte for byte plus a prologue: `kc = kc_all + slotOff[row]` (element offset = `slot*kvSlotBytes[0]/2`), `pos = pos_rows[row]`. 9 buffers (0..8). Rows may be consecutive positions of ONE slot (`prefillByStep`, spec verify) or different slots.
- Paging: replace `slotOff[row]` by a per-row table row `table + row*maxBlocks`; the kernel is derived by string edit so a signature change in `kv_store` panics at init (`edit` counts matches), which is a built-in tripwire.

**kv_store_f32** (dead) and **rope2_kv** (unwired, 13 buffers): both write `[pos*kvDim + ...]`; same translation if ever revived. Not in the product path.

### 1.2 attention (the shipped per-head decode kernel) and its int8/f32 siblings

**attention** (`AllKernels`; launcher `encodeAttentionResidualWith`: `e.Dispatch(r.pAttn, r.nH*tgReduceAttn, tgReduceAttn, r.qkv, r.kc[l], r.vc[l], r.ctx, uNH, uNKV, uHd, uNKeys, uScale, L.uWindow, L.attnSinks, L.uHasSink)`)
- Reads K, V (f16). nH threadgroups of tgs=128 threads. 12 buffers.
- Address, per key: `kb = kc + kvh*hd`; K row `kb + s*kvDim` read as `half4` along hd (8 B loads, `hd%4==0`, scalar tail); V `vb[s*kvDim + d]` scalar half, one thread per output dim (`d = tid + slot*128`).
- Key loop: thread `tid` owns keys `winStart + tid + 128*j` (uncoalesced across lanes by design; the comment records the half4 fix as 1.79x). Scores into `threadgroup float sc[4096]` indexed `s - winStart`.
- Tiles: if `nWin <= 4096` single pass; else online-softmax tiles of 4096 keys starting at `tileBase = winStart`, step 4096. V accumulation loop is an order-preserving 8-way load unroll starting at `winStart`/`tileBase` (loads batched, adds retired in the same sequential order, so the 8 is scheduling only).
- Alignment: `winStart = max(0, nKeys - window)` when `window > 0`, else 0. Global layer: tiles start at absolute multiples of 4096. Windowed layer with `nWin <= 4096` (all current windows): no tile structure; keys `[winStart, nKeys)` exactly. Windowed with window > 4096 (none shipped): tiles shift every token.
- Straddle: **no**. Every load is one key's row slice (or one scalar of one key), so a block boundary is crossed between loads, never inside one. Address can be computed per key; the 8-way V unroll is 8 independent scalar loads, each may use its own page.
- Paging change: `s*kvDim` -> `(table[s/B]*B + s%B)*kvDim` in the K read and both V loops (single-tile and deep path), 3 sites, same for `sc[]` untouched.
- Bit identity to preserve: thread-to-key map (`winStart + tid + 128j`), `tgs == tgReduceAttn == 128` (the denominator tree reduction is width-coupled; comment in kernel and `metal/model.go:tgReduceAttn`), the 4096 tile boundaries from `winStart`, the alpha rescale `acc*alpha`, the sink seeding, sequential V add order. None of these depend on addresses.
- Also used for gpt-oss (sinks) and windowed (Gemma local, Mistral) layers.

**attention_i8** (launcher same, `r.kvI8`: `e.Dispatch(r.pAttnI8, ..., r.qkv, kc, vc, r.ks[l], r.vs[l], r.ctx, uNH, uNKV, uHd, uNKeys, uScale, L.uWindow, sinks, hasSink)`; 14 buffers)
- Same structure as `attention`. K read `char4` (4 B) along hd, plus one f32 `ks[s*nKV + kvh]` per key (loaded once per key); V `char` scalar and `vs[s*nKV + kvh]` per key (the 8-way unroll loads 8 scales ahead).
- Paging: translate the data row AND the scale index. Scale blocks are 4*nKV*B bytes. No MC3, no prefill kernel for int8, no FA: this is the *only* int8 reader (plus `kv_store_i8`), which makes int8 the simplest case.

**attention_f32** (dead, `r.kvF32 = false`): same as `attention` with float KV, V loop serial; ignore.

### 1.3 attention_fa family (decode flash attention, first pass)

Launcher (`encodeAttentionResidualWith`, gated by `canUseAttnFA(l)`): `e.DispatchTG(r.pAttnFA, g.nKV*nSplit*128, 128, shmBytes=128*6*G*4, r.qkv, r.kc[l], r.vc[l], r.attnFAPartial, uNKV, uAttnFAG, uNKeys, uScale, L.uWindow, fanSplitBuf)`, then the combine (no KV). 10 buffers + threadgroup(0). Grid: `nKV*nSplit` threadgroups, each 128 threads = 4 simdgroups, one threadgroup per (kvHead, split).
Eligibility (`canUseAttnFA`/`attnFAHeadDimOK`/`attnFAGroupOK`): dense non-windowed (`L.window == 0`), no sinks, not int8 KV, not sandwich/postOnly/parallelBlock, no LoRA, no MoE/DeltaNet/qGate, hd 128 (any G <= 8) or hd 64 with G == 7, and `nKeys >= attnFADepthFloor (1024)`. Split count `attnFASplitFor`: 16 for the block kernels (`attnFABlkSplit`), capped at `max(nKeys/32,1)` and `attnFAMaxSplit = 32`. Per-key addressing identical in all three: `kb + s*kvDim + lane*4` (hd 128) or `+ lane*2` (hd 64), `kb = kc + kvh*hd`.

Split rule (all three): `winStart = (window>0 && nKeys>window) ? nKeys-window : 0` (always 0 in practice: dispatched with window 0), `chunkLen = ceil(nWin/nSplit)`, `chunkStart = winStart + split*chunkLen`, `chunkEnd = min(chunkStart+chunkLen, nKeys)`. Partition is **a function of nKeys alone**, recomputed each token.

- **attention_fa** (legacy; reached when G is outside 2..8 with hd 128, i.e. MHA G=1): simdgroup `sgid` takes keys `chunkStart + sgid + 4i` (sgid-strided, 1 key per iteration, `half4` per lane = one 256 B row per simdgroup). Per-key online softmax with `simd_sum` per head per key. Tile = 1 key. 4 simdgroups' partials merged sg0->sg1->sg2->sg3.
- **attention_fa_blk** (template G, hosts `attention_fa_blk_g2..g8`; default for dense GQA 2..8): per simdgroup, blocks of 32 consecutive keys at `b0 = chunkStart + sgid*32 + 128*i`, `nb = min(32, chunkEnd-b0)`. Inside a block: 32 separate K-row loads (`kb + (b0+j)*kvDim`, `float4(*(half4*)...)`), per head `simd_sum(dot)` so lane j holds key j's score; one online-softmax step per block (`simd_max`, `simd_sum`); then 32 separate V-row loads weighted by `simd_shuffle(p,j)`. Loop `for j<32` is fully unrolled (`ATTN_FA_BLK_UNROLL`); ragged tail `nb<32` uses a non-unrolled loop.
- **attention_fa_blk64** (hd 64, G=7 only, host `attention_fa_blk64_g7`): identical but `half2`/`float2` per lane (128 B row). Not 64 keys.
- Tile: **32-key block (chunk-relative)**, one key row per load. Alignment: block starts are `chunkStart + 32*(sgid + 4i)`, where `chunkStart = split*ceil(nKeys/nSplit)`. Multiple-of-32 only by coincidence (e.g. nKeys = 1024 gives chunkLen 64). Not a multiple of any B from absolute key 0.
- Straddle: **yes for B=64, 128, 256** (the nKeys=3900 case above; generally any block start `b0` with `b0 % B > B-32`). It is harmless to correctness if the lookup is per key inside the `j` loop (the key index `b0+j` is simdgroup-uniform, so the lookup is a uniform scalar load, or hoisted: compute `page(b0)` and `page(b0+31)`, fast path when equal).
- Bit identity to preserve: the partition (`chunkLen`, `nSplit` rule), block starts, unrolled j order (the block reductions `simd_sum`, `simd_max`, `simd_shuffle` on 32 lanes), merge order sg0..3 then split-ascending in the combine. All independent of addresses. Do NOT "fix" a straddle by splitting a 32-block at the page edge: that would change the softmax grouping and break identity.
- hd 128 G=7 and hd 64 G=7 instantiations are the graded ones; other G bit-agree with g7 head for head (`TestAttnFABlk_anyGMatchesG7`).

### 1.4 mc3_* row kernels (derived from the above; multi-sequence decode step)

Launcher `resident.forwardMultiOut` (batch.go). Eligibility `batchIneligible`: >= 2 resident slots, `kvContig`, f16 only (`!kvF32`, `!kvI8`), **no windowed layer** (`L.window != 0` -> ineligible, so Gemma/Mistral never batch), no MoE, no sink, uniform geometry across layers.
Common addressing: `kc_all = kvSlotBufs[0].kc[l]` (slot-0 base, offset 0), per-row `slotOff[row] = slot*kvSlotBytes[0]/2` elements; K and V are separate buffers, 'slots' laid out back to back inside each; row r of the step may be any slot at any position (also several rows of the same slot at consecutive positions, e.g. `prefillByStep` and verify).

- **mc3_attention_rows** (`e.Dispatch(r.pAttnRows, nPlain*nH*128, 128, qkvB, kcAll, vcAll, ctxB, uNH, uNKV, uHd, nKeysB, uScale, L.uWindow, L.attnSinks, L.uHasSink, uCount[nPlain], rowmapB, slotOffB)`): body = `attention`, prologue `kc = kc_all + slotOff[mc3_row]`, `nKeys = nKeys_rows[row]`. **15 buffers.** Rows below depth 1024 (the rows' `canUseAttnFAAt` is false).
- **mc3_attention_fa_rows / mc3_attention_fa_blk_rows (g2..g8) / mc3_attention_fa_blk64_rows_g7**: bodies = the three FA kernels; prologue maps `tg_rows -> (row, kvHead, split)` and returns for `split >= nSplit_rows[row]`; per-row `nKeys`, `nSplit`, partial region, slot offset. **16 buffers** (q_rows, kc_all, vc_all, partial_rows, nKV, G, nKeys_rows, scale, window, nSplit_rows, M, rowmap, slotOff, qStride, partStride, maxS) + threadgroup mem. The comment says "aikit binds at most 16"; the pinned aikit allows 31, see §0.6.
- **mc3_attention_fa_combine_rows**: partials only (10 buffers); no KV.
- Per-row fallback when `mc3FARowsOn` is false: the production kernels run once a row against `kvSlotBufs[q.slot].kc[l]`, a `Buffer.At(slot*kvSlotBytes)` view.
- Paging change: `slotOff[row]` is replaced by a per-row table pointer (a `[rows][maxBlocks]` int32 buffer). Because the rows kernels are string edits of production text keyed on exact signature fragments in `deriveRowsKernels` (`edit` panics if a fragment stops matching exactly once), the production kernels' signatures must change first and each `edit(...)` pair be updated together: expect init-time panics, not silent drift.
- Preserve for bit identity: bodies unchanged except the address; per-row identity with the single-row kernels is what `TestMC3*` check.

### 1.5 Prefill attention kernels

Launcher: `resident.prefillLastImg`. Selection: `imgB != nil` -> `attention_prefill_img`; else `lu.steel` (hd 128, fused on, `!prefillSteelAttnOff`) -> `attention_prefill_steel`; else `lu.fused` (hd%8==0 && hd<=128) -> `attention_prefill_fused`; else `attention_prefill`. All read the **bound slot's** `kc, vc` (f16 only: `prefillOK` excludes int8 KV), and the pass writes K/V with `kv_store_f16` first, then attends to keys `[winStart, startPos+m]`.

- **attention_prefill / attention_prefill_img** (`e.Dispatch(pf.pAttn, M*nH*128, 128, qkvF, kc, vc, ctxF, uNH, uNKV, uHd, uStartPos, uScale, uStride, L.uWindow)` (+ `imgB, imgN` for `_img`); 11 and 13 buffers): one threadgroup per (row, head), 128 threads; thread key map `winStart + tid + 128j`, K half4 loads, `sc[s]` indexed by the **absolute** key (not `s - winStart`), so keys must be `< prefillExactAttnMaxKeys = 4096` (the pass declines above). V loop serial per thread over `[winStart, nKeys)`. `_img`: `nKeys = blocks[b].y` for a query inside an image block (bidirectional within the block), window lower bound causal.
  - Tile 1 key/thread; no straddle. Alignment `winStart`. Paging: same per-key translation as `attention`; note `sc[s]` absolute index stays valid (it is not an address).
- **attention_prefill_fused** (`e.Dispatch(pf.pAttnFused, attnFusedTotal, attnFusedTg=128, ..., uMReal)`; 12 buffers): one *simdgroup* per (query head, 8-row query tile), 4 per threadgroup. Per simdgroup: keys in groups of `ATTN_KTILE = 32` = 4 sub-tiles of 8. K^T load: `simdgroup_load(kT, kBase + j0*kvDim + kk*8, kvDim, 0, transpose=true)` — an 8x8 half tile whose rows (8 consecutive keys) are `kvDim` apart from ONE base pointer. V: `simdgroup_load(vTile, vBase + (j0s + sub*8)*kvDim + cc*8, kvDim)`. Online softmax per 32-key group via threadgroup scratch (`sScr`, `pScr` half, `oScr`).
  - **Tile alignment: absolute.** `j0superStart = (winStart0/32)*32` with `winStart0` the first row of the tile's window start (0 for global layers), loop `j0s += 32`, `j0 = j0s + sub*8`. So every 8-key sub-tile starts at an absolute multiple of 8 and every 32-group at an absolute multiple of 32, independent of `startPos`.
  - **Straddle: no, for any B that is a multiple of 8 (sub-tile) and, if the 32-group is to stay inside one block, of 32. B=64, 128, 256 all qualify.** It is the only Metal kernel for which "tile is block-aligned" is a real design requirement (the 8 rows must be uniformly strided, so a straddling sub-tile would need a gather).
  - **Reads past the logical end.** The last sub-tile covers keys up to `ceil8(nKeysMax)`; keys `>= nKeysMax` are read but masked (`p = 0`) *after* the load. `metal/model.go` rounds the allocation to `ceil8(ctxCap)` rows for exactly this ("C-01"). Under paging the over-read stays inside the last mapped block iff `B % 8 == 0` (true), so no extra page; but those cells must hold *finite* values: `0 * NaN/Inf = NaN` in the PV MMA. Fresh pool blocks need zero fill (Metal zero-fills new buffers; a recycled block holds old finite KV, which is fine).  **INFERRED** that nothing else can write Inf there; check KV f16 overflow is impossible.
  - **Reads before the window.** Window layers start at `floor32(winStart0)`, so it reads up to 31 keys below the first row's window start (masked, `v = -inf`) and the 8-row tile reads rows' own windows differ per row. A window-group block-reclaim policy must keep at least `floor32(winStart)` mapped, or point out-of-window table entries at a zero/finite scratch page.
  - Preserve for bit identity: the 32-key grouping from absolute multiples of 32, the sub-tile order, the half rounding of P (`pScr` half), per-row alpha rescale in f32 scratch.
- **attention_prefill_steel** (hd 128 only; `e.Dispatch(pf.pAttnSteel, nH*ceil(M/32)*128, 128, ..., uMReal)`; 12 buffers): one threadgroup per (head, 32 rows), 4 simdgroups x 8 rows. K and V are staged **per key** into threadgroup memory 16 keys at a time: `kv4 = *(device const half4*)(kc + j*kvDim + kvh*128 + d4)` for `j = j0 + key`, `key = idx>>5` over 16 keys, `idx = tid + i*128`, i<4 (so each thread loads 4 half4, 128 threads x 4 = 16 keys x 32 half4). Keys `>= jEnd` are zero-filled, never read.
  - Tile 16 keys, **absolute alignment**: `jStart = (windowStart(firstRow) / 16) * 16`, `j0 += 16`. Straddle: **no** (per-key loads; aligned to 16, and 16 divides B). Reads up to 15 keys below the window start (masked).
  - Preserve: 16-key blocks from absolute multiples of 16, `exp2` with log2e folded into the scale, fragment lane map.

### 1.6 Prefill chunking, offsets and padded-ctx keys

- `PrefillLast(ctx, embeddings, startPos)` -> `batchedPrefill` -> `prefillLastImg`: M rows at `startPos..startPos+M-1`; the pass is a **single command buffer over the whole `embeddings` slice**. The decoder splits long prompts into blocks of `PrefillBlockRows` rows (`metal/backend.go:metalResident.PrefillBlockRows`): `max(int(512 MiB / per-row scratch)/64*64, 256)`, so **a multiple of 64, at least 256, not a multiple of 128 or 256 in general**; it returns 0 (no splitting) for layer-major MoE/E-model routes, int8 KV, sinks, exact-attention models, etc. The chunk edges are therefore multiples of 64 only (KV3 can round to B). Tail continuation of a reused prefix starts at an arbitrary `startPos`.
- Chunk invariance: `TestMC5_prefillChunkInvariance` (0 of 14,336,000 KV elements differ at 64..384). Fused/steel absolute tile alignment is why `startPos` does not move the tiles.
- Below `metalFastPrefillFloor` (and while below `metalStepPrefillCeiling`) a prompt runs as decode rows: `prefillByStep` = `forwardMultiOut` pieces of up to `batchMaxSeqs` rows on the **same slot** at consecutive positions (the `mc3_*` kernels above), else sequential `Forward`. So one prompt can be written by two kernel classes (decode-class `kv_store`/`mc3_kv_store_rows` vs `kv_store_f16`) depending on length; that is the class map KV0 step 3 wants, and its Metal half is: n tokens from `s`: `s+n < floor` -> decode class (step/Forward); else prefill class (fused/steel/exact by head dim).
- Anything keyed on padded ctx: `paddedCtxCap = ceil8(ctxCap)` (allocation + `kvSlotBytes` + `ks/vs` length + `slotOff`); `checkCap(pos, n)` against `ctxCap` (unpadded); `metalCtxCapMax = 32768` (`decoder.MetalCtxCeiling`) with `attnScoreTileBound = 4096` (decode tiles; not a true limit since tiled); `prefillExactAttnMaxKeys = 4096` (exact prefill kernel only); `attnFAMaxSplit = 32` (partial buffer length is `maxAttnFAPartialElems*32`, independent of depth). Pre-encoded command buffers (`attnPlan{fa, nSplit, f16Lane}`) bake buffer *bindings* (`stopExec` on slot switch); a table buffer bound once and mutated in place survives pre-encoding, but the grid (`nKV*nSplit*128`) is baked per plan and per `nKeys`.

### 1.7 Not KV kernels (checked and excluded)

`g4a_attn` (audio tower, reads `a.k`/`a.v` scratch, indexed `j*H + off` into a per-call buffer), `tower_attn_hd*` / `tower_attn_body` (vision towers), `eg2_*` (EmbeddingGemma encoder), `attention_fa_combine`, `mc3_attention_fa_combine_rows` (partials), `rope`, `rope2`, `mc3_rope2_rows`, `qk_norm*`, `rope_f16*` (write to the qkv scratch, not to the cache). DeltaNet/Mamba layers have recurrent state, not KV, and no cache buffers (`r.kc[l]` zero-value).

## 2. Metal: synthesis

**Key tiles in play:** 1 (per-key rows: attention*, fa legacy, fa_blk loads, steel staging); 8 (fused `simdgroup_load`, the only true contiguous multi-key tile); 16 (steel block); 32 (fused group; fa_blk block, which is chunk-relative); 4096 (decode score tile, `winStart`-relative).
**LCM of the absolute-aligned tiles** = lcm(8, 16, 32, 4096) = **4096**. B in {64, 128, 256} divides it; every B that is a power of two from 32 to 4096 satisfies the alignment rules.
**Constraining kernels for B:** only `attention_prefill_fused` (`B % 8 == 0` strictly; `B % 32 == 0` for a per-group fast path) and, weakly, `attention_prefill_steel` (`B % 16`). Nothing on Metal pushes B above 32 or below 256. B=256 is not blocked by any Metal kernel.
**Unaligned by construction:** `attention_fa*` split partition and 32-key blocks (chunk-relative, `nKeys`-dependent); decode `attention*` 4096 tiles for windowed layers (`winStart`-relative; moot because current windows <= 4096 take the single-tile path); the per-thread key map `winStart + tid + 128j`. Windowed masks start at `winStart` and mask; nothing needs block alignment. Implication: **per-key table lookup is the required form** for fa*, `attention*`, steel staging; the aligned form (one lookup per tile) is available only for the fused kernel's 8-key sub-tiles. Cost: one uniform table load per key for fa_blk (simdgroup-uniform index), per thread-key for `attention` (index varies by lane, 128 distinct table loads per sweep, cached).
**Bit identity argument:** all arithmetic grouping (tile boundaries, split partition, simd reductions, merge order) is a function of `nKeys`/`winStart`/`startPos`, never of addresses, so a pure indirection is identity-preserving provided (a) loaded values are unchanged, (b) tiles/chunks/blocks are not re-cut to respect page edges, (c) vector load alignment is kept (`half4` needs 8 B alignment: `B*kvDim*2 B` blocks are 16 B aligned), (d) the KV write is the same bits. The compiler could still re-schedule or re-associate under fast-math when the address code changes (**INFERRED**; hence KV5's byte gates).
**Buffer binds (limit 31 in the pinned aikit, comments say 16):** attention 12, attention_i8 14, attention_f32 10, attention_fa* 10 (+ threadgroup(0)), mc3_attention_rows 15, mc3 fa rows 16, kv_store 6, kv_store_i8 9, mc3_kv_store_rows 9, kv_store_f16 8, prefill kernels 11/12/12/13. Every kernel can take a table buffer without hitting 31; only `mc3_*fa_rows` would reach 17. If one reads "16" as a real limit, the fix is to fold `nKV`, `G`, `qStride`, `partStride`, `maxS` into one `constant struct` buffer.
**Layout facts that change under paging:** K and V already separate buffers per layer (a K pool and V pool per layer, or one pool per K/V with layer offset); slot s today = a byte offset into one allocation per layer (`kvContig`) or a separate allocation (int8, one slot); `Buffer.At` views ignore host `U16s()` offsets (`kvHostOff`); the host-side `UploadKV` writes through `kvHostOff` and a table-aware version must write per block.

## 3. WebGPU: per-kernel census

Common facts. KV is `array<f32>` (f32), `array<u32>` with 2 f16 per word (f16, `unpack2x16float`, element `ki` at word `ki>>1`, half `ki&1`), or `array<u32>` with 4 int8 per word (`ki>>2`, `unpacki8`) plus `array<f32>` scales indexed `s*nKV + kvh`. **K and V are separate buffers per layer**: `runLayer.kCache`, `vCache` (+ `kScale`, `vScale` for int8; MLA: one `latCache`), each `ctxCap*kvDim` elements (all layers, windowed too). KV-shared layers alias the source's buffers. All indices are **u32 element indices**. Each KV buffer is bound as a **whole-buffer storage binding** (`Size: buf.GetSize()`), and all KV bind groups are baked once in `newDecodeRunner`. The per-token state (`pos`, `nKeys`, `start`, `base = pos*kvDim`) lives in uniform buffers rewritten each token (`r.posUnis`). The ceilings per precision are fixed: f32 16384, f16 32768, i8 65536 (`decoder.WebGPUCtxCeiling`). One workgroup per query head; workgroup 128 lanes (256 for wide heads).

### 3.1 Stores (writers)

| shader (const) / launcher | writes | address | notes |
|---|---|---|---|
| `qkvFinalizeShaderWGSL` / `DecodeRunner` `qkvFinalize` (f32, fused rope(q)+store k+store v) | K, V f32 | `kCache[p.base + off + d]`, `vCache[p.base + i]`, `p.base = pos*kvDim` (uniform) | 7 bindings; one dispatch of `max(kvDim, nH*half)` threads |
| `ropeStoreShaderWGSL` / `kvStoreShaderWGSL` | K / V f32 | `dst[p.base + ...]` | only when not fused (not the default f32 path) |
| `ropeStoreF16ShaderWGSL` / `kvStoreF16ShaderWGSL` | K / V f16 | `dst[(p.base>>1) + idx]` (word) | one thread per word; `base` even |
| `ropeStoreI8ShaderWGSL` / `kvStoreI8ShaderWGSL` | K / V i8 + scale | `dst[(p.base + off)/4 + d/4]`, `scales[p.spos*p.nKV + h]` | one thread per KV head |
| `mlaLatentStoreWGSL` / `mlaStore` | latent | `latCache[p.base + i]`, `base = pos*latDim` | one workgroup |
| prefill bulk (`Context.prefillLastRows`): `cpy(k,0,kCache,kvRow*lkvDim*4, kvRows*lkvDim*4)` (f32 `CopyBufferToBuffer`) or `kvStoreF16Pipeline` with `uniform{totalElems, kvRow*lkvDim}` (f16) | K, V rows | **one contiguous run** `[positions[0], +M)` | i8 prefill and MLA prefill are declined (`runModelToPrefill`: `rm.kvI8`, `rm.mla`), so those prompts run sequential decode stores |
| `residentDecoder.UploadKV` | K, V rows | `queue.WriteBuffer(kCache, base*kvDim*bytes, ...)` | host-side contiguous range; i8 also writes `scOff = base*nKV*4` scales |

Paging change for writers: `p.base = pos*kvDim` becomes a table lookup done on the host (the uniform is rewritten per token, so the host can write the translated base) *or* in-shader; a bulk run of M rows would be split per block (`B` rows each) into several `CopyBufferToBuffer`/`WriteBuffer` calls, or a shader with a per-row lookup.

### 3.2 Readers

All are one workgroup per (query head [, row]). K row of key s for KV head kvh: element offset `s*kvDim + kvh*hd`.

| shader (const, pipeline label) | where used | KV | per-key addressing | key loop / tile | tile start | vec width |
|---|---|---|---|---|---|---|
| `attnShaderWGSL` ("attn") | decode, f32, hd<=128 not key-split eligible; also library paths `decodetoken_fused/batched`, `attnDevice` | f32 | `keys[s*kvDim + kvbase + d]` | **serial over keys** (1 key/iter), lane per dim, 7-step tree reduce + barrier per key | `p.start` (window start) | scalar f32 |
| `attnKeysShaderWGSL` ("attn-keys") | decode default for f32 hd<=128 hd%4==0 kvDim%4==0 | f32 | `k4[s*kvDim4 + kvb4 + i]`; V `vals[s*kvDim + kvbase + t]` | `TILE = 2048` keys; lane t owns keys `tileStart + t + 128j` (own dot, no cross-lane traffic); V pass: lanes own dims, serial over tile keys | `p.start` (0 global; `pos+1-W` local) | **vec4 f32 (16 B)** along hd for K; scalar V |
| wide key-split (`attnKeysWideSource`, "attn-keys-wide") | decode f32 hd 129..256 | f32 | as above, 2 dims per lane for V | same, TILE 2048 | `p.start` | vec4 |
| `attnF16ShaderWGSL` ("attn-f16") | decode f16, hd<=128 | f16 words | `keys[ki>>1]`, `ki = s*kvDim + kvbase + d` | serial per key (as "attn") | `p.start` | 1 f16 |
| `attnI8ShaderWGSL` / `attnI8DP4AShaderWGSL` ("attn-i8"/"attn-i8-dp4a", the latter when `c.hasDP4A`) | decode i8 | i8 words + scales | `keys[kw_base + d]` (dp4a: `kw_base = (s*kvDim+kvbase)>>2`, word per lane, 4 int8 per word) | serial per key | `p.start` | 4 int8 per word (dp4a) |
| `attnWideTemplateWGSL` x {f32, f16, i8} ("attn-wide", "attn-f16-wide", "attn-i8-wide") | head_dim > 128 not key-split eligible | per variant | `ki = s*kvDim + kvbase + d0 + j*256` | serial per key, 256 lanes, strided dims | `p.start` | 1 element |
| `attnKeysRowsSource(false/true)` ("attn-keys-rows", "attn-keys-wide-rows") | batched prompt pass on a sliding-window layer or a wide head (`prefillLastRows`, f32 only) | f32 | as attn-keys, row `wid.y` | TILE 2048 from `rowStart`; row's `rowKeys = p.nKeys + row`; window start per row | `rowStart = max(p.start, rowKeys-window)` | vec4 |
| `attnBatchedShaderWGSL` ("attn-batched"), `attnBatchedF16ShaderWGSL`, `attnKeysBatchedShaderWGSL` ("attn-keys-batched") | batched prompt pass, global layers | f32 / f16 | as attn / attn-f16 / attn-keys | per-key serial (plain, f16) or TILE 2048 from key 0 (keys) | **0** (absolute multiples of 2048 for the keys kernel) | scalar / vec4 |
| `mlaAttnWGSL` | MLA decode | latent f32 | `lat[j*latDim + d]`, `d = t, t+128, ...` | serial per key; K = whole latent, V = first `rank` of the same row | 0 | scalar |

(`ForwardN` is not a row kernel: it records K single-row decode plans, one `DecodeRunner` each, against the shared KV into one command buffer.)

### 3.3 WebGPU tile/straddle verdicts

- **Contiguous key-load tile: none.** Even `attn-keys`' TILE=2048 is a *score-buffer* tile; each lane reads one key's row (`vec4` along hd, 16 B), the V pass reads one key row across lanes (coalesced 512 B for hd=128 f32). Keys are never loaded as a multi-key contiguous run.
- **Alignment:** decode `attn-keys` tiles start at `p.start`: absolute multiples of 2048 for global layers (start 0); window layers start at `pos+1-W` (unaligned, moves each token). Batched/rows variants: from 0 (or `rowStart`). 128-key lane stride inside a tile.
- **Straddle at B=64/128/256:** not applicable; no kernel's correctness depends on it. A page-table lookup per key (or per lane-key) works. B | 2048 for every B <= 2048, so a per-tile lookup is also available for the keys kernels if wanted.
- **Bit identity to preserve:** the tile boundaries (2048 from `start`), lane-key map (`tileStart + t + 128j`), 128-wide reductions, V serial order within tile, `sc[]` size 2048 (WGSL needs a compile-time array; `attnKeysTile` must equal `TILE`). Attention is not claimed bit-identical across kernel pairs (gated on cosine), but each kernel is deterministic.

### 3.4 WebGPU hazards for paging (limits and structure)

1. **Bind groups are baked per runner** over the slot's K/V buffers (`kv_slots.go` header). A shared pool buffer works with the baked scheme only if every runner binds the same pool plus its own small *table* buffer; the table would be a new storage (or uniform array) binding in each attention bind group (the KV bindings are `read_write` for stores, `read` for attention: **K/V pool buffers cannot be both bound writable and read in the same dispatch**, which holds today because stores and attention are separate dispatches; keep them separate).
2. **maxStorageBuffersPerShaderStage = 8 (default, not raised).** Today's counts: f32 narrow/keys attention 5 storage (q, keys, vals, ctx, sinks) + 2 uniform; f16 same; **i8: 7 storage** (q, keys, vals, kScale, vScale, ctx, sinks) + 2 uniform; i8-wide same. A table buffer brings i8 to exactly 8. Any second per-sequence binding (separate scale-table, group tables for a model with several KV groups, e.g. Gemma local/global have the *same* kvDim only in some families) exceeds it. Mitigation: fold sinks/table/scales into fewer buffers, or request a higher limit (adapter-dependent, a new device request in `gpu.go`).
3. **maxStorageBufferBindingSize / maxBufferSize** are requested at the adapter's value (`gpu.go`), so a pool buffer per layer may be as large as the adapter reports; the actual number is adapter-specific (printed by `hwinfo.go`) and was **not measured here**. Today a slot's per-layer buffer is `ctxCap*kvDim*bytes` (e.g. 16384 x 512 x 4 = 32 MiB f32). A shared pool per layer must hold `numBlocks*B*kvDim*bytes` under that cap, so the pool's total capacity per layer is bounded by the binding size, which on a default-limit adapter (128 MiB, **INFERRED** from the WebGPU default) is about 4 slots' worth at that example. K and V pools are two bindings, never one buffer with offsets (writable/read conflict, see 1).
4. **u32 index arithmetic.** All indices are u32 elements; `pos*kvDim` is computed on the host and passed in uniforms (`uint32(pos * g.kvDim)`), and `s*kvDim` in-shader. A pool index `table[s/B]*B*kvDim` must stay < 2^32 elements (a 4 GiB f32 pool); the binding-size cap makes that unreachable on most adapters.
5. **Alignment:** `bindOff` (per-row q/ctx sub-ranges in `prefillLastRows`) requires 256-byte offsets (`qRow%64`); not a KV issue unless blocks are bound by offset rather than by an in-shader table. f16 needs `kvDim` even (a token's words align); i8 needs `hd%4 == 0`; a block must be `B*kvDim` even / multiple of 4 elements. The vec4 K loads need `hd%4` and `kvDim%4` (`attnKeysEligible`).
6. **Robust access hides bugs.** `residentDecoder.checkCap` in `gpu/residency.go` comment: WGSL robust-access silently clamps out-of-range reads and writes. A wrong table entry reads a wrong (clamped) row with no error: gates must compare KV byte for byte.
7. **A slot is a runner; slots are separate allocations.** `newKVSlot` clones `kCache/vCache/kScale/vScale/latCache` and rebuilds the runner (plan, uniforms, bind groups, ForwardN pool grown lazily per slot, rebuilt per slot). Paging on WebGPU therefore means replacing the slot concept, not editing a kernel; this supports KV6's recommendation to keep slots unless a WebGPU use case exists. The WGSL kernel edit itself is small (one lookup in ~12 shaders + stores).
8. **Prefill and rows use one contiguous run + `CopyBufferToBuffer`** (`prefillLastRows`) for f32: with paging this becomes a per-block copy loop; the batched shaders compute causal bounds from `basePos + row` and start at key 0, so they need only the lookup.
9. **MLA** has a single latent buffer per layer with row width `rank + qkRope` (e.g. 576): its own group; K and V are one row, both `mlaAttnWGSL` loops read the same address `j*latDim`.
10. **No row-batched multi-sequence kernel exists**, so "pass a table per row" (§5.3's MC3 form) is a Metal/CUDA feature; on WebGPU a table per runner is enough.

## 4. WebGPU: synthesis

- Tile LCM: 1 (per-key rows); the only structural "tile" is TILE=2048 (arithmetic grouping from `p.start` / 0). LCM relevant for alignment = 2048 (B | 2048 for B in {64, 128, 256}).
- Constraining kernels for B: none for correctness. The binding-size cap and the 8-storage-buffer cap constrain pool design, not B.
- Unaligned by construction: windowed layers' `p.start`; no straddling issue since loads are per-key.
- Windowed layers allocate the full ctxCap today; the window group's reclaim needs nothing beyond keys `>= p.start` (the key-split kernels read exactly `[start, nKeys)`; no pre-window over-read as on Metal fused/steel).

## 5. What I did not read or verify (labelled)

- No Metal or WebGPU kernel was run. All behaviour is from source.
- `maxStorageBuffersPerShaderStage = 8`, `maxBindGroups`, `minStorageBufferOffsetAlignment = 256` are WebGPU spec defaults (**INFERRED**); `gpu.go` requests only the three raised limits listed in §0.8. Actual adapter binding-size values were not looked up.
- The claim that the MC3 FA rows kernels can bind 17 buffers rests on `aikit/gpu@v0.33.5` `maxBindBuffers = 31` plus Apple's table limit quoted in its comment; I did not run it.
- Fast-math/FMA scheduling effects of changing address code were not examined (**INFERRED**).
- I read `metal/kernels/*_msl.go` for every KV kernel, all MC3 derivations in `batch_rows.go`, launch code in `metal/model.go`, `metal/prefill.go`, `metal/batch.go`, `metal/backend.go`; WebGPU `gpu/attention.go`, `attn_keys_rows.go`, `attn_keys_wide.go`, `mla.go`, `kv_slots.go`, the attention paths of `decoderunner.go` and `prefillrunner.go`, `residency.go` (allocation and `UploadKV`). The legacy library entry points (`DecodeTokenFused`, `DecodeTokenFusedBatched`, `attnDevice`) reuse `attnPipeline` against caller-supplied buffers and have no non-test callers outside `gpu/`; I treated them as equal to "attn".
