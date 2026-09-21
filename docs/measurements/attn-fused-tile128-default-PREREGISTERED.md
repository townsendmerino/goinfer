# PRE-REGISTERED — making the 128-row `attn_fused` tile the default (R5 / P24)

**Written 2026-09-21 BEFORE any of the measurements below were taken. Not edited after any result was seen.** Record: `attn-fused-tile128-default-2026-09-21.md`. Builds on `attn-fused-tile-2026-09-21.md`
(1.5B hd128 K=3900 attention 807 -> 337 ms, bit-identical to the shipped kernel, window=0). The owner asked for it to be default; this registers what must hold for that, including the cases not yet timed.

## The change under test
The shipped selector launches `attn_fused_hd{64,128}` (64x64). Proposed: launch the 128-row-tile kernel (`attn_fused_bm128_hd{64,128}`) **when the layer has no sliding window (window == 0) and M >= T**; otherwise the 64x64 kernel exactly as today.
Windowed layers are excluded because their key-tile grouping starts at the block's first row, so a taller block is NOT bit-identical there (never tested for it). `GOINFER_CUDA_ATTN_FUSED_TILE=64x64` forces the old kernel everywhere (kill switch, kept).

## Gates (all must hold; a failed precondition voids, it is not a pass)
1. **Bit-identity, kernel level** (extend `TestAttnFusedTile_*`): 128x64 == 64x64 bit for bit for window=0 on the existing 162 shapes PLUS M in {16,17,48,96,127,128,129,255,256,257} and startPos in {0, 5, 384, 1024}.
2. **Bit-identity, whole model** (new): a full chunked prefill (logits + the KV cache written) with the default selector equals the same prefill with `64x64` forced, `math.Float32bits` equal, on: qwen2.5-coder-1.5B (hd128) at 1300 tokens (3 chunks) and 3900; qwen2.5-0.5B (hd64) at 1300; **gemma3-1b (windowed + full layers, if the checkpoint is on this box)** at 1300. The test must show the 128 kernel was launched (launch count) on the models where window==0 layers exist, and NOT on the windowed layers (count).
3. `TestAttnFused_vsF16Reference` all arms, `TestPrefillChunked_bitIdentical`, `TestPrefillChunked_fastKernelsOnEveryChunk`, `go test -short ./cuda/...`, vet, staticcheck, gofmt: green.
4. **No new fidelity gate is run**, because gates 1-2 show the default's outputs are bit-identical to the shipped kernel's whose §3 gate already passed; if gate 2 finds ANY differing bit, the change does not ship as default and the finding is recorded.

## Small-M threshold T (chosen by a pre-registered rule, not by looking at a curve)
Micro-sweep, kernel launches only (attention only, hd128 nH=12 nKV=2 and hd64 nH=14 nKV=2), M in {16,32,48,64,96,128,192,256,384,512}, startPos in {0, 1024, 3400}, both arms interleaved, median of >= 20 timed launches after warm-up, per-(hd,M,startPos) paired ratio t128/t64.
- **T = the smallest M in the sweep such that for EVERY sweep M >= T and every startPos, t128/t64 <= 1.03**; below T the 64x64 kernel is used. If no such T exists below 512, the change is **parked**, not shipped with a hand-picked T.
- The chunk shapes production produces are 512 (default chunk) and the final remainder (any M); the sweep's rows M<128 exist for the remainder.

## Performance decision rule (served path = `TestPrefillDecomp` categories, arms interleaved as separate processes, 3 rounds, idle box, best-of-3 inside; the 64x64 arm is the do-nothing arm and the per-round A/A spread is reported)
- **Ships as default:** 1.5B K=3900 attention >= 2.0x faster (<= 404 ms; measured diagnostic was 2.40x) AND 0.5B (hd64) K=3900 attention >= 1.5x faster AND 1.5B K=512 attention not slower AND whole-prefill 1.5B K=3900 >= 1.3x faster AND the gemv control within +-3% in every pair.
- **Parked:** 0.5B attention in 1.05-1.5x (real but small; recorded, default only for the models that pass), or any listed condition within its last 3%.
- **Not shipped (kill switch stays, default unchanged):** 0.5B < 1.05x or any slowdown > 3% on a cell where the default would engage.
- If hd64 fails and hd128 passes, the default engages for hd128 only (recorded as such), not a blanket ship.

## Also recorded, not gating
Whole-prefill K=2048 and 3900 for the 0.5B; the ratio at K=128/256 (below the fast-prefill floor these launches are not used, listed for completeness); registers/thread for the 128 kernels from the PTX; a peer TTFT measurement is a separate item (`bench_peer_prefill.py`) and is not claimed here.
