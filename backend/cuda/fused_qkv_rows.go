//go:build cuda

package cuda

import gc "github.com/eitamring/gocudrv/cuda"

// fusedQKVRowsPerWarp chooses how many output rows each warp of fused_rms_qkv_rows walks (1 = the original
// one-row-per-warp kernel, which the caller keeps for rpw <= 1). Every warp of every block first redundantly recomputes
// the layer's rmsnorm and int8 quantisation, so the kernel is fastest when the grid is small enough that few blocks pay
// that prologue and large enough to fill the card. The rule: the largest power of two (<= 16) that still leaves >= 64
// blocks of 8*rpw rows, for projections of at least 1536 rows, else 1; the smallest measured geometry (1152 rows) gains
// nothing, hence the floor. The kernel is bit-identical for any value (TestFusedQKVRowsBitIdentical), so a wrong pick on
// an unmeasured geometry can only cost speed. Measurements: docs/code-notes/cuda.md#fusedQKVRowsPerWarp,
// docs/measurements/fused-rms-qkv-2026-09-21.md.
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

// fusedGURowsPerWarp chooses the rows-per-warp of fused_rms_gu_rows for a (hidden, intermediate) geometry; 8 is the
// original fused_rms_gu, which the caller keeps for 8. There is no simple rule: the win jumps around at grid-size
// thresholds (wave quantisation), so this is a table of the geometries measured to win, and every other geometry keeps
// the original kernel. The kernel is bit-identical for any value (TestFusedGURowsBitIdentical), so a miss can only cost
// speed. The table's measurements: docs/code-notes/cuda.md#fusedGURowsPerWarp.
func fusedGURowsPerWarp(hidden, inter int) int {
	switch [2]int{hidden, inter} {
	case [2]int{3584, 18944}, [2]int{2560, 9728}, [2]int{2048, 6144}, [2]int{1152, 6912}, [2]int{3072, 8192}:
		return 16
	case [2]int{2048, 11008}, [2]int{4096, 14336}:
		return 32
	}
	return 8
}

// waveRowsPerWarpFor is the rows-per-warp that sizes a fused-projection grid (blocks of 8 warps, one prologue per block)
// to ONE resident wave: blocks = (resident blocks per SM) x (SM count), where resident blocks per SM is limited by
// threads (256 per block) and by the block's dynamic shared memory, and rows-per-warp = ceil(rows / (8 x blocks)). 0
// means "cannot tell". Every block first recomputes the layer's rmsnorm and int8 quantisation before streaming any
// weight, so more blocks than one wave leaves the tail running partly-empty waves while fewer blocks mean fewer
// prologues. It assumes registers do not limit occupancy (true for these kernels on the measured card); if they did, the
// grid would simply be larger than one wave. The kernels are bit-identical for any rows-per-warp, so a bad pick costs
// speed, never a logit. Measured on one card (RTX 2070 SUPER, 40 SMs) only: docs/code-notes/cuda.md#waveRowsPerWarpFor,
// docs/measurements/fused-rms-qkv-2026-09-21.md ("wave rule").
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
	defer r.applySMShapeOverride()
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

// smShapeOverride, when set, replaces the SM shape readSMShape read from the driver. It is nil in every build but one that links cuda/testhooks.go
// (-tags goinfer_testhooks), which sets it to make a card with more or fewer SMs than the one we own run through the wave sizing and the grids it picks.
// A grid sized for 128 SMs is correct on 40 (the kernels are bit-identical for any rows-per-warp), so this reaches the launch math with real execution.
var smShapeOverride func() (sms, threadsPerSM, smemPerSM int, ok bool)

func (r *cudaResident) applySMShapeOverride() {
	if smShapeOverride == nil {
		return
	}
	if sms, thr, smem, ok := smShapeOverride(); ok {
		r.smCount, r.smThreads, r.smSmem = sms, thr, smem
	}
}
