// Package launchgeom holds the launch geometry the CUDA kernels hard-code, and the functions that derive launch shapes
// from it. Each constant mirrors a #define or template parameter in a .cu source and must change with it. It imports
// only the standard library, so kernel tests can launch the kernels the way package cuda does without reaching decoder.
// The values are pinned by cuda's TestLaunchGeometry_identity.
package launchgeom

// FaWarps is fa_partial's warps per CTA (FA_WARPS in decode_fa.cu): the partial buffer holds nSplit*FaWarps units per
// head.
const FaWarps = 8

// FaMaxRows is the most verify rows one multi-row attention call covers (the partial buffer is sized for it); a longer
// batch is processed in chunks of this many rows.
const FaMaxRows = 16

// AttnFusedBM, AttnFusedThreads and AttnFusedKPAD equal BM, WARPS*32 and KPAD in attn_fused.cu. BN equals BM there, so
// one constant serves both the query tile and the key tile. Below AttnFusedMinRows query rows (one m16n8k8 tile's M)
// the fused kernel stages 64 rows of K/V for a handful of real ones, and attn_batched, also exact, is faster and
// bit-identical.
const (
	AttnFusedBM      = 64
	AttnFusedThreads = 128
	AttnFusedKPAD    = 8
	AttnFusedMinRows = 16
)

// FaRowsPerCTA mirrors FaRows<HD, G>::R in decode_fa.cu: rows handled per CTA, chosen so the R*G*(hd/32) accumulators
// stay near 64 registers. The kernel and this function must agree (the grid's z extent is derived from it);
// TestFlashDecodeRowsBitIdentical fails if they do not, because rows would be skipped.
func FaRowsPerCTA(hd, g int) int {
	raw := 64 / ((hd / 32) * g)
	return min(max(raw, 1), 8)
}

// FaRun is a maximal run of consecutive verify rows that share a chunk partition (equal WinStart and Per), so one
// multi-row launch can serve them and each row still sees exactly the partition the M=1 kernel gives it.
type FaRun struct{ Row0, N, NKeys0, WinStart, Per int }

// FaRowRuns groups rows i=0..m-1, with nKeys_i = n0+i and the layer's sliding window, into runs. Per_i = ceil(span_i / s).
func FaRowRuns(n0, m, window, s int) []FaRun {
	var runs []FaRun
	for i := range m {
		nk := n0 + i
		ws := 0
		if window > 0 && nk > window {
			ws = nk - window
		}
		per := (nk - ws + s - 1) / s
		if k := len(runs) - 1; k >= 0 && runs[k].WinStart == ws && runs[k].Per == per {
			runs[k].N++
			continue
		}
		runs = append(runs, FaRun{Row0: i, N: 1, NKeys0: nk, WinStart: ws, Per: per})
	}
	return runs
}

// ImgBlockMaxNWin widens a layer's causal shared-memory row count (causalMaxNWin) for a bidirectional image block
// [imgStart,imgEnd). cuda's checkPrefillShmemImg and prefillCore's attn_img_batched launch both call it, so the check
// and the allocation cannot drift apart. No block (imgEnd<=imgStart) returns causalMaxNWin. A query row inside the block
// sees keys [max(pos-window+1,0), imgEnd) when windowed and [0, imgEnd) when not (derivation in the header of
// cuda/attn_img_prefill.cu): the widest row is imgEnd when the block starts at or before window-1, else
// imgLen+window-1.
func ImgBlockMaxNWin(causalMaxNWin, window, imgStart, imgEnd int) int {
	if imgEnd <= imgStart {
		return causalMaxNWin
	}
	blockMax := imgEnd
	if window > 0 {
		if imgStart > window-1 {
			blockMax = (imgEnd - imgStart) + window - 1
		}
	}
	return max(causalMaxNWin, blockMax)
}

// ImgBlocksMaxNWin is ImgBlockMaxNWin over several blocks: each row sees at most its own block, so the widest window is
// the widest over the blocks.
func ImgBlocksMaxNWin(causalMaxNWin, window int, blocks [][2]int) int {
	n := causalMaxNWin
	for _, b := range blocks {
		n = max(n, ImgBlockMaxNWin(causalMaxNWin, window, b[0], b[1]))
	}
	return n
}

// GemvRNRows and GemvRNMT equal RN and MT in gemv_w4a8_rn.cu: each warp computes GemvRNRows output rows and keeps
// GemvRNMT accumulators per row. GemvRNThreads is the block size it is launched with. Results are bit-identical for any
// RN; 2 is the profiled optimum at the register and occupancy limit. TestGemvRN_matchesCuDefines holds the two to the .cu.
const (
	GemvRNRows    = 2
	GemvRNMT      = 16
	GemvRNThreads = 256
)

// GemvRNGridX is gemv_w4a8_rn's grid for n output rows: one warp per GemvRNRows rows, GemvRNThreads/32 warps per block.
func GemvRNGridX(n int) int {
	warps := (n + GemvRNRows - 1) / GemvRNRows
	perBlock := GemvRNThreads / 32
	return (warps + perBlock - 1) / perBlock
}

// KQuantGemvRows equals KQ_ROWS in actgroup.cu: the rows a warp of the standalone native Q5_K and Q6_K GEMVs (gemv_q5k_g32,
// gemv_q6k_g32), A1-G0's pick (docs/tasks/task-kernel-survey-2026-10.md, "KS-A1 on CUDA"). TestKQuantGemv_matchesCuDefine
// holds the two together.
const KQuantGemvRows = 1

// KQuantGemvGridX is gemv_q5k_g32 / gemv_q6k_g32's grid for n output rows: 8 warps a 256-thread block, KQuantGemvRows rows a
// warp.
func KQuantGemvGridX(n int) int {
	return (n + 8*KQuantGemvRows - 1) / (8 * KQuantGemvRows)
}

// GemmMMABM, GemmMMABN, GemmMMAThreads, GemmMMAKStep and GemmMMAAPad equal GBM, GBN, GWARPS*32, GKSTEP and GAPAD in
// gemm_w4a8_mma.cu: a block computes a GemmMMABM x GemmMMABN output tile with GemmMMAThreads threads, staging GemmMMAKStep K
// elements of the activation panel per pass with a GemmMMAAPad-word row pad. TestGemmMMA_matchesCuDefines holds them to the .cu.
const (
	GemmMMABM      = 64
	GemmMMABN      = 64
	GemmMMAThreads = 128
	GemmMMAKStep   = 128
	GemmMMAAPad    = 4
)

// GemmMMAShmem is gemm_w4a8_mma's shared memory in bytes: the activation panel, [GBM][GKSTEP/4 + GAPAD] int words.
func GemmMMAShmem() uint32 {
	return uint32(GemmMMABM * (GemmMMAKStep/4 + GemmMMAAPad) * 4)
}

// GemmMMAGrid is gemm_w4a8_mma's grid for an n-column, m-row output: one block a GemmMMABN x GemmMMABM tile.
func GemmMMAGrid(n, m int) (x, y uint32) {
	return uint32((n + GemmMMABN - 1) / GemmMMABN), uint32((m + GemmMMABM - 1) / GemmMMABM)
}
