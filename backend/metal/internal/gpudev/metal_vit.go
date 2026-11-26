//go:build darwin

package gpudev

import (
	_ "embed"
	"fmt"
)

// metal_vit.go is the Metal mirror of cuda_vit.go — the transformer-encoder kernel
// set a ViT forward needs on top of the metal.go device layer (docs/task-native-gpu.md,
// Phase 3). The exported surface (ViT, NewViT, the kernel-name consts, ViTBlock) is
// identical to the CUDA side: they are build-tag mutually exclusive, so a consumer that
// only touches this surface stays platform-agnostic — the whole point of the substrate.
//
// Every kernel mirrors vision/encoder.go's arithmetic exactly, EXCEPT for the one thing
// that does not port: MSL has no `double`. The CUDA kernels accumulate LayerNorm
// mean/variance, softmax sums, and GELU in double to match the CPU tower's float64 math;
// here those reductions run in f32 with a pairwise (tree) reduction over threadgroup
// memory, which is the best f32 can do. That costs a little precision — see
// metal_vit_test.go for the achieved per-kernel tolerances and why they differ from the
// CUDA bars. The tanh-GELU formula, the max-subtract softmax, and the bidirectional
// 1/sqrt(hd) attention are otherwise identical.
//
// Metal ↔ CUDA divergences baked into these kernels (inverses of gpu/cuda.go's):
//   - dispatchThreads launches EXACTLY n threads, so no per-kernel bounds check.
//   - per-row / per-(head,query) kernels use one threadgroup per unit: the CUDA
//     blockIdx.x is threadgroup_position_in_grid, threadIdx.x is
//     thread_position_in_threadgroup, blockDim.x is threads_per_threadgroup.
//   - scalars bind as 1-element buffers (there is no by-value arg on Metal).

// vitMSL is the encoder kernel set, compiled at load by NewViT (Metal compiles MSL at
// runtime, so unlike the CUDA side there is no embedded PTX). LNBLOCK matches ViTBlock,
// and — like ViTBlock — is a BIT-IDENTITY dependency, not just an array size: it is the
// width of the cross-thread f32 sum reductions in vit_siglip.metal (layernorm
// mean/variance, softmax sum), and f32 addition is not associative, so changing it changes
// the bits. Keep it equal to the host ViTBlock and do not sweep either for performance
// without re-baselining the parity gate.
//
// It is the vit_*.metal files beside this file joined in the order below;
// TestViTSource_isPinned holds the joined bytes.
var vitMSL = "\n" + vitSigLIP + vitQwen + vitGEMMTiled + vitGEMMSG

// vitMSL's sources: the SigLIP set, the Qwen2.5-VL additions, the tiled GEMMs and the simdgroup_matrix f32 GEMMs.
var (
	//go:embed vit_siglip.metal
	vitSigLIP string
	//go:embed vit_qwen.metal
	vitQwen string
	//go:embed vit_gemm_tiled.metal
	vitGEMMTiled string
	//go:embed vit_gemm_sg.metal
	vitGEMMSG string
)

// Kernel entry points in vitMSL (identical names to the CUDA side).
const (
	KernelQuantRows = "quant_rows"
	KernelGEMMW8A8  = "gemm_w8a8"
	KernelGEMMF32   = "gemm_f32"
	KernelAddBias   = "add_bias"
	KernelAddVec    = "add_vec"
	KernelLayerNorm = "layernorm"
	KernelGELUTanh  = "gelu_tanh"
	KernelAttention = "attention"

	// KernelAttentionTiled is the query-tiled online-softmax kernel. Dispatch via
	// AttentionTiledDispatch, and only when AttentionTiledEligible(hd) — see the kernel's
	// own doc comment.
	KernelAttentionTiled = "attention_tiled"

	// --- Qwen2.5-VL additions (identical names to the CUDA side) ---
	KernelRMSNorm           = "rmsnorm"
	KernelRopeQK            = "rope_qk"
	KernelAttentionSeg      = "attention_seg"
	KernelAttentionSegTiled = "attention_seg_tiled"
	KernelSiLUMul           = "silu_mul"
	KernelGELUErf           = "gelu_erf"

	// --- tiled GEMMs (throughput; identical names to the CUDA side) ---
	KernelGEMMW8A8Tiled = "gemm_w8a8_tiled"
	KernelGEMMF32Tiled  = "gemm_f32_tiled"

	// KernelGEMMW8A8Reg is the register-blocked, packed-word int8 GEMM (identical name
	// to the CUDA side's gemm_w8a8_reg). Pick it via GEMMW8A8Plan.
	KernelGEMMW8A8Reg = "gemm_w8a8_reg"

	// KernelGEMMF32SG is the production f32 GEMM on simdgroup_matrix (Metal-only; CUDA has
	// no analogue in this kernel set). Same signature as gemm_f32_tiled. Launch with SGDims.
	KernelGEMMF32SG = "gemm_f32_sg"

	// KernelGEMMF32SGBig is the aligned (M%32==0, N%32==0, K%8==0) fast path: 32×32 tile
	// (SGBigBlock), direct device loads, no staging/barriers/bounds. Pick it via GEMMF32Plan.
	KernelGEMMF32SGBig = "gemm_f32_sg_big"

	// KernelGEMMW8A8Bias is gemm_w8a8 with fused bias epilogue: C[m,n] = (Aq·Bq)*aScale*bScale + bias[n].
	KernelGEMMW8A8Bias = "gemm_w8a8_bias"

	// KernelGEMMW8A8BiasAdd is gemm_w8a8 with fused bias and residual addition:
	// residual[m,n] += (Aq·Bq)*aScale*bScale + bias[n].
	KernelGEMMW8A8BiasAdd = "gemm_w8a8_bias_add"

	// KernelGEMMF32Bias is gemm_f32 with fused bias epilogue: C[m,n] = A·B + bias[n].
	KernelGEMMF32Bias = "gemm_f32_bias"

	// KernelGEMMF32BiasAdd is gemm_f32 with fused bias and residual addition:
	// residual[m,n] += A·B + bias[n].
	KernelGEMMF32BiasAdd = "gemm_f32_bias_add"
)

// SGBlock is gemm_f32_sg's threadgroup output-tile width (must match vitMSL's SGBM/SGBN);
// SGThreads is its threadgroup size (4 simdgroups × 32).
const (
	SGBlock   = 32
	SGThreads = 128
)

// SGDims is gemm_f32_sg's uniform-threadgroup 2-D geometry for Run2D: one SGBlock×SGBlock
// output tile per threadgroup, SGThreads threads each.
func SGDims(M, N int) (gx, gy, tgx, tgy int) {
	return (N + SGBlock - 1) / SGBlock, (M + SGBlock - 1) / SGBlock, SGThreads, 1
}

// SGBigBlock is gemm_f32_sg_big's threadgroup output-tile width (must match the kernel's 32).
const SGBigBlock = 32

// GEMMF32Plan picks the f32 GEMM kernel and its Run2D geometry for a shape M×N×K. The aligned
// no-barrier kernel (gemm_f32_sg_big — same 32×32 tile, but direct device loads and no bounds
// checks) is chosen when M and N are multiples of 32 and K of 8 — which every large encoder /
// ViT GEMM satisfies — and the general gemm_f32_sg (any shape) otherwise. Both bind at buffers
// 0..5 (A, B, C, M, N, K), so the caller just sets its scalar buffers and calls Run2D.
func (v ViT) GEMMF32Plan(M, N, K int) (p Pipeline, gx, gy, tgx, tgy int) {
	if M%SGBigBlock == 0 && N%SGBigBlock == 0 && K%8 == 0 {
		return v.GEMMF32SGBig, N / SGBigBlock, M / SGBigBlock, SGThreads, 1
	}
	gx, gy, tgx, tgy = SGDims(M, N)
	return v.GEMMF32SG, gx, gy, tgx, tgy
}

// GEMMTile is the tiled GEMMs' tile width; it must match vitMSL's TILE, which sizes the
// threadgroup As/Bs staging arrays. Mirrors cuda_vit.go's GEMMTile.
const GEMMTile = 16

// TileDims is the tiled GEMMs' uniform-threadgroup 2-D geometry for Run2D: one
// GEMMTile×GEMMTile output tile per threadgroup. (The CUDA side returns a LaunchConfig
// from TileGrid; Metal's Run2D takes the grid/threadgroup extents directly, so this
// returns them as four ints instead.)
func TileDims(M, N int) (gx, gy, tgx, tgy int) {
	return (N + GEMMTile - 1) / GEMMTile, (M + GEMMTile - 1) / GEMMTile, GEMMTile, GEMMTile
}

// IntRegBlock/IntRegK are gemm_w8a8_reg's per-threadgroup output-tile width and K-step
// in ELEMENTS (IBK words of four int8); together they form the kernel's alignment
// requirement. Named to match cuda_vit.go's identical constants.
//
// 16 rather than 64 is deliberate: SigLIP-so400m's intermediate width is 4304, a
// multiple of 16 but NOT of 64, so a K%64 kernel would have missed the MLP projections
// that are most of a ViT's work.
const (
	IntRegBlock = 64
	IntRegK     = 16
)

// GEMMW8A8Plan picks the int8 GEMM kernel and its Run2D geometry for a shape M×N×K — the
// int8 analogue of GEMMF32Plan, and the Metal twin of cuda_vit.go's GEMMW8A8Plan
// (docs/audit-metal-2026-09-12.md).
//
// It needs only K%16==0: M and N edge tiles stage zeros and skip their stores, so the
// inner loop stays unguarded while any M and N are accepted — SigLIP-so400m's
// intermediate width (4304) is a multiple of 16 but not 64, and an M%64/N%64
// requirement would have excluded the MLP projections, which are most of a ViT's work.
// A K%16!=0 shape falls back to gemm_w8a8_tiled. Both bind identically (A, aScale, B,
// bScale, C, M, N, K) and produce identical bits (int32 addition is associative), so a
// caller only swaps the pipeline and Run2D geometry.
func (v ViT) GEMMW8A8Plan(M, N, K int) (p Pipeline, gx, gy, tgx, tgy int) {
	if K%IntRegK == 0 && M > 0 && N > 0 && K > 0 {
		return v.GEMMW8A8Reg, (N + IntRegBlock - 1) / IntRegBlock, (M + IntRegBlock - 1) / IntRegBlock, IntRegBlock / 4, IntRegBlock / 4
	}
	gx, gy, tgx, tgy = TileDims(M, N)
	return v.GEMMW8A8Tiled, gx, gy, tgx, tgy
}

// GEMMW8A8BiasPlan picks the fused int8 GEMM+bias kernel and Run2D geometry for M×N×K.
// Returns an empty pipeline if K%IntRegK != 0.
// Bind signature: (Aw [[0]], aScale [[1]], Bw [[2]], bScale [[3]], bias [[4]], C [[5]], M [[6]], N [[7]], K [[8]]).
func (v ViT) GEMMW8A8BiasPlan(M, N, K int) (p Pipeline, gx, gy, tgx, tgy int) {
	if K%IntRegK == 0 && M > 0 && N > 0 && K > 0 {
		return v.GEMMW8A8Bias, (N + IntRegBlock - 1) / IntRegBlock, (M + IntRegBlock - 1) / IntRegBlock, IntRegBlock / 4, IntRegBlock / 4
	}
	return Pipeline{}, 0, 0, 0, 0
}

// GEMMW8A8BiasAddPlan picks the fused int8 GEMM+bias+residual add kernel and Run2D geometry for M×N×K.
// Returns an empty pipeline if K%IntRegK != 0.
// Bind signature: (Aw [[0]], aScale [[1]], Bw [[2]], bScale [[3]], bias [[4]], residual [[5]], M [[6]], N [[7]], K [[8]]).
func (v ViT) GEMMW8A8BiasAddPlan(M, N, K int) (p Pipeline, gx, gy, tgx, tgy int) {
	if K%IntRegK == 0 && M > 0 && N > 0 && K > 0 {
		return v.GEMMW8A8BiasAdd, (N + IntRegBlock - 1) / IntRegBlock, (M + IntRegBlock - 1) / IntRegBlock, IntRegBlock / 4, IntRegBlock / 4
	}
	return Pipeline{}, 0, 0, 0, 0
}

// GEMMF32BiasPlan picks the fused f32 GEMM+bias kernel and Run2D geometry for M×N×K.
// Bind signature: (A [[0]], B [[1]], bias [[2]], C [[3]], M [[4]], N [[5]], K [[6]]).
func (v ViT) GEMMF32BiasPlan(M, N, K int) (p Pipeline, gx, gy, tgx, tgy int) {
	if M > 0 && N > 0 && K > 0 {
		gx, gy, tgx, tgy = SGDims(M, N)
		return v.GEMMF32Bias, gx, gy, tgx, tgy
	}
	return Pipeline{}, 0, 0, 0, 0
}

// GEMMF32BiasAddPlan picks the fused f32 GEMM+bias+residual add kernel and Run2D geometry for M×N×K.
// Bind signature: (A [[0]], B [[1]], bias [[2]], residual [[3]], M [[4]], N [[5]], K [[6]]).
func (v ViT) GEMMF32BiasAddPlan(M, N, K int) (p Pipeline, gx, gy, tgx, tgy int) {
	if M > 0 && N > 0 && K > 0 {
		gx, gy, tgx, tgy = SGDims(M, N)
		return v.GEMMF32BiasAdd, gx, gy, tgx, tgy
	}
	return Pipeline{}, 0, 0, 0, 0
}

// AttnTiledMaxHD/AttnTiledKTile/AttnTiledQTile mirror attention_tiled's AT_MAXHD/AT_KTILE/
// AT_QTILE — kept in sync by hand, like GEMMTile/SGBlock above mirror their own
// kernels' #defines elsewhere in this file.
const (
	AttnTiledMaxHD = 128
	AttnTiledKTile = 16
	AttnTiledQTile = 32
)

// AttentionTiledEligible reports whether attention_tiled can serve a tower with this
// head dim: its K/V staging arrays are sized statically for hd <= AttnTiledMaxHD. A wider
// hd MUST use the untiled `attention` kernel instead — this is not a fallback the kernel
// itself performs, callers must check it themselves before dispatching.
func AttentionTiledEligible(hd int) bool { return hd > 0 && hd <= AttnTiledMaxHD }

// AttentionTiledMinNP mirrors cuda_vit.go's constant of the same name (that file is
// //go:build linux, so darwin needs its own copy) — it is CUDA's measured crossover, kept
// here only so the two files' exported surfaces match. Do NOT gate a Metal dispatch on it;
// see AttentionTiledEnabledOnMetal below.
const AttentionTiledMinNP = 3072

// AttentionTiledEnabledOnMetal is false: the CUDA crossover above does not transfer.
// BenchmarkMetalAttention on a real M1 Pro found untiled `attention` faster than
// `attention_tiled` at EVERY shape tested (np=729..4096, the full realistic ViT/Qwen
// range) — 1.46-1.85x, not close, and not narrowing at the high end the way a real
// crossover approaching from one side would. CUDA's dynamic-shared-memory occupancy
// penalty (attention_tiled's own doc comment above) that motivates the tiled kernel
// there does not describe Apple's unified-memory/threadgroup-memory architecture, so
// there is no reason to expect np=3072 — or any nearby number — to be Metal's crossover
// too. Callers MUST check this (not just AttentionTiledEligible) before dispatching
// attention_tiled. The kernel itself is correct and stays tested
// (TestMetal_vitAttentionTiled) for whenever a real Metal shape justifies flipping this.
const AttentionTiledEnabledOnMetal = false

// AttentionTiledDispatch returns the Run1D geometry (n, tg) for attention_tiled at a
// given (np, nH). Callers MUST check AttentionTiledEligible(hd) first — an ineligible hd
// dispatched anyway is an out-of-bounds write into the kernel's fixed-size Ks/Vs arrays,
// not a graceful decline.
func AttentionTiledDispatch(np, nH int) (n, tg int) {
	tilesPerHead := (np + AttnTiledQTile - 1) / AttnTiledQTile
	return nH * tilesPerHead * AttnTiledQTile, AttnTiledQTile
}

// AttentionSegTiledDispatch returns the Run1D geometry (n, tg) for attention_seg_tiled at a
// given (seq, nH). Callers MUST check AttentionTiledEligible(hd) first.
func AttentionSegTiledDispatch(seq, nH int) (n, tg int) {
	tilesPerHead := (seq + AttnTiledQTile - 1) / AttnTiledQTile
	return nH * tilesPerHead * AttnTiledQTile, AttnTiledQTile
}

// ViTBlock is the threadgroup width the per-row/attention kernels reduce at; it must
// match vitMSL's LNBLOCK (the static threadgroup reduction arrays are sized to it).
//
// It is PART OF THE BIT-IDENTITY CONTRACT, not just a performance knob. The layernorm
// and softmax kernels sum floats across `tgsz` (== this width) threads, and f32 addition
// is not associative, so the summation order — and therefore the exact bits — is fixed
// by this value. Precise-math compilation (CompileLibraryPrecise) removes the compiler's
// discretion over contraction/reassociation but NOT this: the width is chosen here, in
// host code. Do not sweep it for a speed win without re-baselining the ViT parity gate —
// a small consistent shift passes the tolerance-based maxAbsDiff check while the numbers
// have moved. Change ViTBlock and LNBLOCK together (see the cross-thread `+=` reductions
// in vitMSL, each tagged as bit-identity-coupled).
const ViTBlock = 256

// AttentionThreadgroupBytes is the threadgroup memory one `attention` or `attention_seg` dispatch needs when its
// dynamic score row holds n floats (n patches, or the longest segment): 4 bytes per element plus the kernel's two static
// ViTBlock-float reduction arrays. A dispatch over Device.MaxThreadgroupMemoryLength aborts its command buffer yet reports
// Completed, so a caller sizing n from model or image dimensions checks this first and declines.
func AttentionThreadgroupBytes(n int) int { return n*4 + 2*ViTBlock*4 }

// ViT holds the compiled encoder kernel pipelines — identical shape to the CUDA ViT.
type ViT struct {
	QuantRows Pipeline
	GEMMW8A8  Pipeline
	GEMMF32   Pipeline
	AddBias   Pipeline
	AddVec    Pipeline
	LayerNorm Pipeline
	GELUTanh  Pipeline
	Attention Pipeline

	// AttentionTiled is the query-tiled online-softmax kernel — see its own doc
	// comment in vitMSL.
	AttentionTiled Pipeline

	// Qwen2.5-VL additions.
	RMSNorm           Pipeline
	RopeQK            Pipeline
	AttentionSeg      Pipeline
	AttentionSegTiled Pipeline
	SiLUMul           Pipeline
	GELUErf           Pipeline

	// Tiled GEMMs — same math, staged through threadgroup memory.
	GEMMW8A8Tiled Pipeline
	GEMMF32Tiled  Pipeline

	// GEMMW8A8Reg is the register-blocked, packed-word-staged int8 GEMM. Pick it via
	// GEMMW8A8Plan, never directly — it requires K%16==0 and has no bounds-checked
	// fallback path of its own.
	GEMMW8A8Reg Pipeline

	// Production f32 GEMM on simdgroup_matrix (Metal-only); SGBig is the aligned fast path.
	GEMMF32SG    Pipeline
	GEMMF32SGBig Pipeline

	// Fused GEMM epilogues (bias and residual addition)
	GEMMW8A8Bias    Pipeline
	GEMMW8A8BiasAdd Pipeline
	GEMMF32Bias     Pipeline
	GEMMF32BiasAdd  Pipeline
}

// NewViT compiles vitMSL on this device and builds every encoder pipeline. The library
// is tracked by the Device, so ReleaseObjects frees it.
func (d *Device) NewViT() (ViT, error) {
	lib, err := d.CompileLibraryPrecise(vitMSL, MSL3_1)
	if err != nil {
		return ViT{}, fmt.Errorf("metal: compile ViT library: %w", err)
	}
	var v ViT
	for _, bind := range []struct {
		name string
		dst  *Pipeline
	}{
		{KernelQuantRows, &v.QuantRows},
		{KernelGEMMW8A8, &v.GEMMW8A8},
		{KernelGEMMF32, &v.GEMMF32},
		{KernelAddBias, &v.AddBias},
		{KernelAddVec, &v.AddVec},
		{KernelLayerNorm, &v.LayerNorm},
		{KernelGELUTanh, &v.GELUTanh},
		{KernelAttention, &v.Attention},
		{KernelAttentionTiled, &v.AttentionTiled},
		{KernelRMSNorm, &v.RMSNorm},
		{KernelRopeQK, &v.RopeQK},
		{KernelAttentionSeg, &v.AttentionSeg},
		{KernelAttentionSegTiled, &v.AttentionSegTiled},
		{KernelSiLUMul, &v.SiLUMul},
		{KernelGELUErf, &v.GELUErf},
		{KernelGEMMW8A8Tiled, &v.GEMMW8A8Tiled},
		{KernelGEMMF32Tiled, &v.GEMMF32Tiled},
		{KernelGEMMF32SG, &v.GEMMF32SG},
		{KernelGEMMF32SGBig, &v.GEMMF32SGBig},
		{KernelGEMMW8A8Reg, &v.GEMMW8A8Reg},
		{KernelGEMMW8A8Bias, &v.GEMMW8A8Bias},
		{KernelGEMMW8A8BiasAdd, &v.GEMMW8A8BiasAdd},
		{KernelGEMMF32Bias, &v.GEMMF32Bias},
		{KernelGEMMF32BiasAdd, &v.GEMMF32BiasAdd},
	} {
		p, err := d.NewComputePipeline(lib, bind.name)
		if err != nil {
			return ViT{}, err
		}
		*bind.dst = p
	}
	return v, nil
}
