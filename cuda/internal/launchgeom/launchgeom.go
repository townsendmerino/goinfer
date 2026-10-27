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
