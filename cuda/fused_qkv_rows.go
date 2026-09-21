package cuda

// fusedQKVRowsPerWarp chooses how many output rows each warp of fused_rms_qkv_rows walks (1 = the original one-row-per-warp kernel, which the caller keeps for
// rpw <= 1). Every warp of every block first redundantly recomputes the layer's rmsnorm + int8 quantisation, so the kernel is fastest when the grid is small enough that
// few blocks pay that prologue and large enough to fill the card. Measured on ten real geometries (docs/measurements/fused-rms-qkv-2026-09-21.md, ncu kernel time,
// random int4 weights): the best setting lands where the grid is about 64-100 blocks — D7 (4608 rows) rpw 8 = 72 blocks, 0.64x of the original's time; qwen2.5-3b (2560) rpw 4 =
// 80 blocks, 0.76x; 1.5B (2048) rpw 2-4, 0.72x; gemma3-1b (1536) rpw 2, 0.76x — and the small 0.5B (1152 rows) gains nothing (rpw 2 is 6% WORSE), hence the floor below.
//
// The rule: the largest power of two (<= 16) that still leaves >= 64 blocks of 8*rpw rows, for projections of at least 1536 rows; else 1. The kernel is bit-identical for any
// value (TestFusedQKVRowsBitIdentical), so a wrong pick on an unmeasured geometry can only cost speed, never change a logit.
func fusedQKVRowsPerWarp(nrows int) int {
	if nrows < 1536 {
		return 1
	}
	rpw := 1
	for cand := 2; cand <= 16; cand *= 2 {
		if (nrows+8*cand-1)/(8*cand) >= 64 {
			rpw = cand
		}
	}
	return rpw
}

// fusedGURowsPerWarp chooses the rows-per-warp of fused_rms_gu_rows for a (hidden, intermediate) geometry; 8 is the original fused_rms_gu (which the caller keeps for 8).
// Unlike the QKV kernel there is no simple rule: measured on nine real geometries (15 launches each, ncu kernel time, random int4 weights, spreads 1-10 us) the win jumps around
// at grid-size thresholds (wave quantisation on 40 SMs) — rows-per-warp 16 is -13% on qwen3-1.7b and -8% on gemma3-1b, but +33% on the 0.5B and +4% on qwen2.5-3b, which prefers
// 32 (-11%). So this is a TABLE of the geometries measured to win, and every other geometry keeps the original kernel. The kernel is bit-identical for any value
// (TestFusedGURowsBitIdentical), so a miss can only cost speed. Medians, original -> chosen: D7 (3584,18944) 214.9 -> 208.0 us (rpw 16); qwen2.5-3b (2048,11008) 76.9 -> 68.1 (32);
// qwen3-4b (2560,9728) 79.6 -> 76.2 (16); qwen3-1.7b (2048,6144) 46.3 -> 40.3 (16); llama3-8b (4096,14336) 179.6 -> 164.2 (32); gemma3-1b (1152,6912) 35.5 -> 32.7 (16);
// phi3-mini (3072,8192) 79.1 -> 75.7 (16). Not in the table (no gain): 1.5B (1536,8960), 0.5B (896,4864).
func fusedGURowsPerWarp(hidden, inter int) int {
	switch [2]int{hidden, inter} {
	case [2]int{3584, 18944}, [2]int{2560, 9728}, [2]int{2048, 6144}, [2]int{1152, 6912}, [2]int{3072, 8192}:
		return 16
	case [2]int{2048, 11008}, [2]int{4096, 14336}:
		return 32
	}
	return 8
}
