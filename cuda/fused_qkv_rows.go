//go:build cuda

package cuda

import gc "github.com/eitamring/gocudrv/cuda"

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

// waveRowsPerWarpFor is the rows-per-warp that sizes a fused-projection grid (blocks of 8 warps, one prologue per block) to ONE resident wave: blocks = (resident blocks per SM) x (SM count),
// where resident blocks per SM is limited by threads (256 per block) and by the block's dynamic shared memory, and rows-per-warp = ceil(rows / (8 x blocks)). 0 means "cannot tell".
//
// Why: every block first recomputes the layer's rmsnorm + int8 quantisation before streaming any weight, which costs ~24 us on D7's gate/up and never overlaps the block's own traffic. More blocks
// than one wave means the tail of the grid runs partly-empty waves (wave quantisation on 40 SMs: D7 gate/up measured 214.6 us at 592 blocks, 238 us at 132, and 196.9 us at 119 = one full wave of 3 per SM);
// fewer blocks means fewer prologues. The rule replaced a per-geometry table with a formula that is at least as good almost everywhere, and finds wins the table missed
// (docs/measurements/fused-rms-qkv-2026-09-21.md, "wave rule"). It assumes registers do not limit occupancy (true for these kernels on the measured card); if they did, the grid would simply be larger than one wave.
// The kernels are bit-identical for any rows-per-warp, so a bad pick costs speed, never a logit. Measured on one card (RTX 2070 SUPER, 40 SMs) only.
func waveRowsPerWarpFor(rows, smemPerBlock, sms, threadsPerSM, smemPerSM int) int {
	if rows < 1 || smemPerBlock < 1 || sms < 1 || threadsPerSM < 256 || smemPerSM < smemPerBlock {
		return 0
	}
	per := min(threadsPerSM/256, smemPerSM/smemPerBlock)
	blocks := per * sms
	return max(1, (rows+8*blocks-1)/(8*blocks))
}

func (r *cudaResident) waveRowsPerWarp(rows, smemPerBlock int) int {
	return waveRowsPerWarpFor(rows, smemPerBlock, r.smCount, r.smThreads, r.smSmem)
}

// readSMShape reads the device shape waveRowsPerWarp needs: SM count, max threads per SM, max shared memory per SM. Any failure leaves the field zero, which makes waveRowsPerWarp
// answer "cannot tell" and the caller fall back to the static rules.
func (r *cudaResident) readSMShape() {
	d := r.dev.Context().Device()
	if d == nil {
		return
	}
	if v, err := d.Attribute(gc.DeviceAttributeMultiprocessorCount); err == nil {
		r.smCount = v
	}
	if v, err := d.Attribute(gc.DeviceAttributeMaxThreadsPerMultiprocessor); err == nil {
		r.smThreads = v
	}
	if v, err := d.Attribute(gc.DeviceAttributeMaxSharedMemoryPerMultiprocessor); err == nil {
		r.smSmem = v
	}
}
