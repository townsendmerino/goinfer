//go:build linux

package gpudev

import (
	_ "embed"
	"fmt"
)

// cuda_gemv.go publishes the TUNED generic quantized GEMV — aikit's GPU twin of
// linalg's W4A8/W8A8 matmul, and the payload of the Phase-1b blob-split
// (docs/task-native-gpu.md). The device layer in cuda.go is the substrate; this is
// the first compute aikit ships on top of it.
//
// The kernels are lifted verbatim from goinfer's decode path (see gemv_quant.cu for
// the provenance and the bit-identity rule). Consumers get the PTX and the entry
// point names, so nothing about the launch is hidden: a caller that needs different
// geometry, a different stream, or an accumulate epilogue just builds the args
// itself. NewQuantGEMV + GEMVGrid are the ergonomic path, not a wall.

// QuantGEMVPTX is the compiled gemv_quant.cu, JIT-loaded by the driver at run time.
// Exported so a consumer can load it into a module it already owns rather than going
// through NewQuantGEMV. Regenerate with ./build_ptx.sh — never hand-edit.
//
//go:embed testdata/gemv_quant.ptx
var QuantGEMVPTX []byte

// Kernel entry points in QuantGEMVPTX.
const (
	// KernelGEMVW4A8 is the int4-weight / int8-activation GEMV with f16 per-group
	// weight scales. Parameters, in order:
	//
	//	W          [N*Kwords] u32   — 8 int4 per word, nibble-permuted at pack time
	//	a          [2*Kwords] i32   — int8 activations, 4 per word
	//	gs         [N*Kgroups] f16  — per-group weight scales
	//	aScalePtr  f32*             — activation scale, read ON DEVICE
	//	bias       f32* or null     — per-row bias; bind ArgNull() when absent
	//	N          i32              — rows
	//	Kwords     i32              — K/8
	//	Kgroups    i32              — K/32
	//	dst        [N] f32          — output
	//	accum      i32              — 1 ⇒ dst[n] += val, 0 ⇒ dst[n] = val
	KernelGEMVW4A8 = "gemv_w4a8_fwd"

	// KernelGEMVW8A8 is the int8-weight / int8-activation GEMV with f32 per-row
	// weight scales. Parameters, in order:
	//
	//	W          [N*Kdiv4] i32    — int8 weights, 4 per word, row-major
	//	a          [Kdiv4] i32      — int8 activations, 4 per word
	//	wScale     [N] f32          — per-row weight scale
	//	aScalePtr  f32*             — activation scale, read ON DEVICE
	//	bias       f32* or null     — per-row bias; bind ArgNull() when absent
	//	N          i32              — rows
	//	Kdiv4      i32              — K/4
	//	dst        [N] f32          — output
	//	accum      i32              — 1 ⇒ dst[n] += val, 0 ⇒ dst[n] = val
	KernelGEMVW8A8 = "gemv_w8a8_fwd"
)

// GEMVWarpsPerBlock is the block shape both kernels were tuned at: one output row
// per warp, 8 warps (256 threads) per block.
const GEMVWarpsPerBlock = 8

// QuantGEMV holds the compiled pipelines for both quantized GEMVs.
type QuantGEMV struct {
	W4A8 Pipeline
	W8A8 Pipeline
}

// NewQuantGEMV loads QuantGEMVPTX on this device and builds both pipelines. The
// module is tracked by the Device, so ReleaseObjects frees it.
func (d *Device) NewQuantGEMV() (QuantGEMV, error) {
	lib, err := d.CompileLibrary(QuantGEMVPTX)
	if err != nil {
		return QuantGEMV{}, fmt.Errorf("cuda: load quantized-GEMV module: %w", err)
	}
	w4, err := d.NewComputePipeline(lib, KernelGEMVW4A8)
	if err != nil {
		return QuantGEMV{}, err
	}
	w8, err := d.NewComputePipeline(lib, KernelGEMVW8A8)
	if err != nil {
		return QuantGEMV{}, err
	}
	return QuantGEMV{W4A8: w4, W8A8: w8}, nil
}

// GEMVGrid is the launch geometry both kernels expect: one row per warp, warps
// blocks of 32 threads each. Passing GEMVWarpsPerBlock reproduces the tuned shape
// (grid = ceil(rows/8), block = 256) exactly.
//
// Both kernels bounds-check their row index, so a grid that overhangs rows is safe
// — which it will whenever warps does not divide rows.
func GEMVGrid(rows, warps int) LaunchConfig {
	if rows <= 0 || warps <= 0 {
		return LaunchConfig{}
	}
	return LaunchConfig{
		GridX: uint32((rows + warps - 1) / warps), GridY: 1, GridZ: 1,
		BlockX: uint32(warps * 32), BlockY: 1, BlockZ: 1,
	}
}
