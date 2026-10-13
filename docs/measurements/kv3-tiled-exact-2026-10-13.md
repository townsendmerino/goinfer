# KV3: a tiled exact attention for CUDA's batched prefill, past 12,160 keys (registered 2026-10-13, before any code)

Owner decision 2026-10-13 (`docs/tasks/task-kv-architecture-2026-10.md` §8 Q7): build the tiled exact kernel the census recommended.
Branch `kv3-tiled-2026-10`, from `kv3-cuda-2026-10` (it needs that branch's guard and admission figure). nobara-pc, RTX 2070 SUPER.

## 1. Facts this rests on

1. **The census** (`docs/measurements/kv3-long-prompt-census-2026-10-13.md`): Gemma-3-4B and Gemma-4-E4B decline past 12,160 attended keys on the first
   global layer and fall to the per-token path: 9.17 and 13.78 ms a token, so at least 183 s and 276 s for a 20,000-token prompt, against 0.87 and 2.43 ms a
   token batched at 1.5k. The paged-expert models do not decline (their cost is expert paging).
2. **Why they decline.** The exact batched attention (`attn_batched`, grid `nH × M`, block 128) keeps one row's scores in shared memory,
   `(nWin + 128)·4` bytes, so it cannot pass 48 KiB (12,160 keys); the fused attention serves head dim 64 and 128 only. Gemma-3 is head dim 256; Gemma-4's
   global layers are 512 (local 256).
3. **Decode already has the kernel shape.** `splitKVAttnDecode` (`cuda/resident.go`) is three launches (`splitkv_scores`, `splitkv_softmax`, `splitkv_vsum` in
   `cuda/decode_splitkv.cu`) that hold the scores in a global buffer and are **bit-identical to `attn_batched(M=1)`**, held by `TestSplitKV_bitIdentical`.
   It splits only the independent axes (keys for the scores, output dims for the V fold); the softmax keeps `attn_batched`'s 128-wide partition and
   tree, and each output dim keeps the whole ascending-key fold. Decode takes it exactly where the single block cannot run (`splitKVRequired`).
4. **The exact class is one class.** `attn_batched`, `gemv_w4a8_rn` and decode's split-KV agree bit for bit (KV0's record); a batched prefill that
   runs only exact-class kernels produces the bits of the per-token path.

## 2. The change

- **Three row-batched kernels** in a new `cuda/prefill_splitkv.cu` and its own `.ptx` (the `attn_block.cu` isolation pattern: no shipped PTX changes):
  `splitkv_scores_rows`, `splitkv_softmax_rows`, `splitkv_vsum_rows`. Each is the decode kernel with a row index in the grid's Z axis. Row `m` of a pass at
  `startPos` attends `nKeys = startPos + m + 1`, window start as `attn_batched` derives it from that `nKeys`, and its own `nWin`; the instruction sequence
  per (row, head) is the decode kernel's, so identity is by construction; only the addressing (a row's `q`, score slice and `ctx` row) is new.
  The sink is passed as decode passes it. Head dims 64 to 512 (the q staging is `hd·4` bytes of shared; the float4 K read needs `hd` a multiple of 4,
  which the families here satisfy and the host checks).
- **Rows are chunked** so the score scratch stays bounded: `R = max(1, budget / (nH · maxNWin · 4))` rows a launch group, with `budget = 64 MiB`. The
  scratch is one device buffer allocated with the pass's other scratch.
- **Routing.** A layer whose exact attention would exceed the shared-memory bound, and that the fused attention does not serve, takes the row-batched
  kernels. `checkPrefillShmem` stops declining for it and keeps declining for what these kernels do not serve (image blocks, `attn_img_batched`; the verify
  lane). A layer under the bound is untouched: `attn_batched` as today. So **no existing prefill's bits change**.
- **Admission.** `PrefillScratchBytes` includes the score scratch for the pass, and `TestKV3_prefillScratchBytes_matchesThePass` is extended to hold it.
- **Regeneration.** `./build_ptx.sh prefill_splitkv` with the pinned `nvidia-cuda-nvrtc-cu12==12.6.85` environment, so the new `.ptx` is reproducible and no other
  `.ptx` is rebuilt.

## 3. Gates (registered)

| | gate | tier | pass |
|---|---|---|---|
| G1 | the row kernels equal `attn_batched` row by row: random q, K and V, head dim 64, 128, 256 and 512, MHA and GQA, sliding window, sink, `startPos` 0 and not 0, f32 and f16 KV, spans below the bound (a test seam forces the new route there) | day | every output float bit-identical |
| G2 | no change below the bound: the existing digest runs at 300, 3,900 and 12,000 tokens against `kv3-cuda-2026-10`; the `cuda` identity suites the branch already runs | day | bit-identical |
| G3 | batched equals per-token past the bound, on a tiny head-dim-256 fixture at 12,400 to 16,384 tokens (the fixture's context raised as `kv0LoadCtx` does), logits and every row's KV | day | bit-identical |
| G4 | the same on the real models: Gemma-3-4B at 12,400 tokens and Gemma-4-E4B at 12,400 tokens (global layers head dim 512), batched against per-token, last-row logits and a KV digest | night (the per-token side is at least 2 and 3 minutes a model at 12,400 tokens) | bit-identical |
| G5 | speed: time to first token on Gemma-3-4B and Gemma-4-E4B at 12,400 and 20,000 tokens, batched against the per-token path, same session, interleaved | night; one exploratory cell by day, never quoted | **PASS** at 3x or more at 12,400 tokens on both; **AMBIGUOUS** 1.5x to 3x (parked: tune the V-sum's occupancy, a rows axis on the dims split, then re-read once); below 1.5x the kernel is not kept |
| G6 | no regression where it does not route: TTFT at K = 512 and 3,900 on Gemma-3-4B against `kv3-cuda-2026-10`, and the Qwen3-4B pair the branch already runs | night | 0.97 or better |
| G7 | admission: the pass's own scratch (a test seam reports it) is within the registered 5% of `PrefillScratchBytes` at 16 and at 512 rows, tiny and real | day | within the margin, over-pricing only |

**Expectation, stated so it can be wrong.** Each (row, head) reads its whole key range from the cache with no reuse across rows beyond the GQA group, so the cost is
cache bandwidth, not the fused kernel's 64-key tiles. A 20,000-token prompt reads about 10 TB of K and as much of V across the six global layers of Gemma-3-4B (8 heads, head dim 256, f32) before L2 reuse:
tens of seconds at this card's bandwidth, not the 17 s of the batched rate at 1.5k. G5 measures it; this note does not predict a number.

## 4. Not in this work

Gated-DeltaNet hybrids (no decline was logged for Qwen3.6 at 12,242 keys), MLA, the image-block prefill, the all-logits, embedding and residual tails, and a fused
attention for head dim 256 or 512 (a larger numerics project: it is not bit-identical to decode). An unresolved G5 is not a reason to re-register the bar.

## 5. Estimate

Build and G1 to G3 and G7 by day: one session. G4 to G6 on the night queue under the timing lock, about 40 minutes estimated (two real models, per-token sides at
2 to 5 minutes each, interleaved with the batched side, two lengths).

## 6. Amendment 1 (2026-10-13, before any code or run)

Three facts read from the tree while preparing the build; none comes from a run of the new kernels.

1. **No tiny fixture has head dim 256.** G3 as registered names one. G3 runs on `llama-attnfa-tiny` (head dim 128, GQA 4:2, two layers) with
   `GOINFER_CUDA_FAST_PREFILL=0`, at a context raised past the bound: with the levers off the pass is exact class and its attention takes the row kernels
   past 12,160 keys, so batched against per-token is still a bit-identity. Head dims 256 and 512 are held at the kernel level by G1 and on the real models by G4.
2. **Gemma-3's batched pass is not exact class as a whole.** Above the 512-token floor its projections run the tensor-core GEMM (the fast lever); only its
   attention is exact, because no fused attention serves head dim 256. So "batched equals per-token, bit for bit" is true for Gemma-4-E4B (an E-model forces the
   exact kernels) and for Gemma-3 only with `GOINFER_CUDA_FAST_PREFILL=0`. **G4 becomes:** Gemma-4-E4B at 12,400 tokens, default settings, bit-identical; Gemma-3-4B at
   12,400 tokens with the levers off, bit-identical; and Gemma-3-4B at default settings, a fidelity reading against the per-token path (last-row logits cosine
   0.99 or better and equal argmax: the fast GEMM's own class past the bound, which the lever's existing gate covers only to 3,900 tokens). The kernel's own
   identity at default settings is G1 plus a new **G2b (day):** on a full model pass below the bound, forcing the row kernels (a test seam) against
   `attn_batched` with every other kernel unchanged, logits and KV bit-identical.
3. **The score scratch depends on the attended span, and `PrefillScratchBytes(rows)` does not see it.** A pass allocates the score scratch only when a layer
   routes to the row kernels, sized `R·nH·maxNWin` floats plus `R·nH` inverses. Admission reads it through a second optional method,
   `PrefillSpanScratchBytes(rows, keys)`, added to the request's price with `keys` the request's prompt length; `PrefillScratchBytes` is unchanged, so its
   existing equality test stands. **G7 becomes:** the pass's allocations equal `PrefillScratchBytes(M) + PrefillSpanScratchBytes(M, keys)` exactly, at spans
   below and past the bound.

Any pass can route, not only a Gemma's: a suffix of 15 rows or fewer past 12,160 keys is exact class on every family (it declined to the per-token path before, a
few tokens). The routing rule is the registered one; this is its consequence.

## 7. Amendment 2 (2026-10-13, after the build, before the night jobs)

1. **The module is built with this box's ambient NVRTC, 12.9.86, not 12.6.85.** §2 named the pin of the audited set (`glue`, `moe`, `gemv_fwd` and their kin). Every
   other production module, `decode_splitkv.ptx` and `prefill_batched.ptx` among them, is built at 12.9.86, and a new module follows its siblings:
   `./build_ptx.sh prefill_splitkv`, no other `.ptx` rebuilt. The kernels have no bare multiply-add (they are in `lintedKernels`), so the compiler version is not a
   numerics choice, and G1 measures the result.
2. **Head dim must be a multiple of 4**, as `attn_batched`'s float4 read of `q` already requires (a head dim of 6 faults in `attn_batched` itself). G1's smallest head dim is 20.
3. **The image-block pass still declines past the bound, by name** (an MLA model does not batch at any length, a static decline). An E-model's layers and a Gated-DeltaNet
   hybrid's attention layers go through the same launch site as any exact layer, so they route too: the E-model is read by the Gemma-4-E4B smoke below, the hybrid is read
   from the code only, and the census's inference that Qwen3.6 did not decline is still unmeasured.

## 8. Day readings (2026-10-13)

Branch `kv3-tiled-2026-10`. Logs: `kv3-tiled-exact-2026-10-13/` (`day-gates.log`, `g3-tiny.log`, `digest-tiled.txt`, `gemma3-forced-day.log`, the two smoke JSONs).

| gate | reading | verdict |
|---|---|---|
| G1 | 360 cases (180 on each KV type: head dim 20, 64, 128, 256, 512; MHA and GQA; a causal past; a window opening partway; with and without a sink; groups of 1, 7 and every row), every float bit-identical to `attn_batched`. A softmax launched at block 64 turns 180 of them red. | **PASS** |
| G2 | digests at 300, 3,900, 4,100 and 12,000 tokens on the 1.5B equal `kv3-cuda-2026-10`'s (`digest-tiled.txt` against that branch's `digest-new.txt`) | **PASS** |
| G2b | tiny, f32 and f16 KV, levers off and tensor-core GEMM alone, 300, 1,000 and 3,900 tokens: forced route against `attn_batched`, 0 of 256 logits and 0 of up to 3,993,600 KV elements differ, 12 of 12 cells | **PASS** |
| G3 | tiny, levers off, 12,400 and 16,384 tokens, f32 and f16 KV: batched against per-token, 0 logits and 0 of up to 16,777,216 KV elements differ, 4 of 4 cells; without the module the same pass declines by name | **PASS** |
| G7 | the pass's allocations equal `PrefillScratchBytes(M) + PrefillSpanScratchBytes(M, keys)` in 16 of 16 cells (levers on and off, start 0 and 12,300, 8 to 512 rows); the span term is present exactly when the pass is exact past the bound | **PASS** |

**Not gates (measurements and smokes, one run each, exploratory, not to be quoted as results):**

- *Real weights, below the bound:* on Gemma-3-4B one 512-row pass whose last row attends 12,032 keys, `attn_batched` against the forced row kernels, three interleaved pairs:
  0 of 262,144 logits and 0 of 35,651,584 KV elements differ in every pass; 1.25 s against 1.40 s a pass (all 34 layers forced, three pairs, the same sign each time).
- *Served smoke, one request each:* Gemma-3-4B, 12,241 tokens: 25.8 s, no decline (the census: unfinished at 150 s). Gemma-4-E4B, 12,215 tokens: 55.9 s, no decline (the census:
  unfinished at 150 s). G5 measures the ratio.

**A harness defect found and fixed in G1 before it was read.** The first runs failed in about half the invocations, always on the reference's side: the row kernels matched a
float64 host computation and `attn_batched` did not. The test uploads `K`, `V` and `q` and launches the reference at once on its own stream; a host-to-device copy from pageable
memory returns once it is staged, so the reference read the tail of `q` before it landed. A context synchronize and a read-back of `q` after the uploads fixed it: 12 of 12 runs
clean. `TestAttnBatched_bitIdentical` has the same shape with buffers small enough that it has not shown; it is not changed here.

