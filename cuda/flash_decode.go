//go:build cuda

package cuda

import (
	"fmt"
	"os"
	"strconv"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// faWarps is fa_partial's warps per CTA (FA_WARPS in decode_fa.cu): the partial buffer holds nSplit*faWarps
// units per head.
const faWarps = 8

// faMaxRows is the most verify rows one multi-row attention call covers (the partial buffer is sized for it); a longer
// batch is processed in chunks of this many rows.
const faMaxRows = 16

// flashDecodeDefaultMinKeys is the attended-span floor for the lane: below it the exact
// path runs. Set from the served forced-on ladder (docs/measurements/attn-decode-fa-served-2026-09-20.md): the 0.5B and
// gemma3-1b lose 3-9% up to 1024 keys and win from 2048, so 2048 is the lowest floor with no measured regression.
const flashDecodeDefaultMinKeys = 2048

// loadFlashDecode wires the opt-in flash-decode lane when GOINFER_CUDA_FLASH_DECODE=S (S >= 1) is set. A stock
// binary loads nothing and allocates nothing. Any failure leaves the lane OFF (faSplit 0), never half-on.
func (r *cudaResident) loadFlashDecode(m *decoder.Model, nLayers int) {
	s, err := strconv.Atoi(os.Getenv("GOINFER_CUDA_FLASH_DECODE"))
	if err != nil || s < 1 {
		return
	}
	maxHd := 0
	for l := 0; l < nLayers; l++ {
		if h := m.HeadDimAtResident(l); h > maxHd {
			maxHd = h
		}
	}
	if maxHd <= 0 {
		fmt.Fprintf(os.Stderr, "[cuda] GOINFER_CUDA_FLASH_DECODE ignored: no positive head dim\n")
		return
	}
	mod, e := r.dev.CompileLibrary(decodeFAPTX)
	if e != nil {
		fmt.Fprintf(os.Stderr, "[cuda] GOINFER_CUDA_FLASH_DECODE ignored: %v\n", e)
		return
	}
	var pipes [3]Pipeline
	for i, name := range []string{"fa_partial_64", "fa_partial_128", "fa_partial_256"} {
		if pipes[i], e = r.dev.NewComputePipeline(mod, name); e != nil {
			fmt.Fprintf(os.Stderr, "[cuda] GOINFER_CUDA_FLASH_DECODE ignored: %v\n", e)
			return
		}
	}
	if r.faCombine, e = r.dev.NewComputePipeline(mod, "fa_combine"); e != nil {
		fmt.Fprintf(os.Stderr, "[cuda] GOINFER_CUDA_FLASH_DECODE ignored: %v\n", e)
		return
	}
	r.faPartial = pipes
	// Multi-row (speculative verify) variant: 24 kernels + the row combine. A load failure leaves them zero, which only disables
	// the verify lane (speculation then falls back to the exact-attention scope), never the M=1 lane.
	rowsOK := true
	for hi, hd := range []int{64, 128, 256} {
		for g := 1; g <= 8 && rowsOK; g++ {
			p, le := r.dev.NewComputePipeline(mod, fmt.Sprintf("fa_partial_rows_%d_g%d", hd, g))
			if le != nil {
				fmt.Fprintf(os.Stderr, "[cuda] flash-decode multi-row kernel fa_partial_rows_%d_g%d not loaded (verify lane off): %v\n", hd, g, le)
				rowsOK = false
				break
			}
			r.faRows[hi][g] = p
		}
	}
	if rowsOK {
		if r.faCombineRows, e = r.dev.NewComputePipeline(mod, "fa_combine_rows"); e != nil {
			fmt.Fprintf(os.Stderr, "[cuda] flash-decode fa_combine_rows not loaded (verify lane off): %v\n", e)
			rowsOK = false
		}
	}
	if !rowsOK {
		r.faRows, r.faCombineRows = [3][9]Pipeline{}, Pipeline{}
	}
	// The partial buffer holds faMaxRows rows so a verify batch fits; row 0 of the rows layout is the M=1 layout.
	r.faBuf = r.af(faMaxRows * r.nH * s * faWarps * (maxHd + 4))
	r.faVerify = r.faCombineRows != (Pipeline{}) && os.Getenv("GOINFER_CUDA_FLASH_DECODE_VERIFY") != "0"
	r.faMinKeys = flashDecodeDefaultMinKeys
	if v, err := strconv.Atoi(os.Getenv("GOINFER_CUDA_FLASH_DECODE_MIN_KEYS")); err == nil && v >= 0 {
		r.faMinKeys = v
	}
	r.faSplit = s
}

// faEligible reports whether layer l can take the lane: a supported head dim, a GQA group of at most 8, and no
// attention sink (gpt-oss folds the sink into the softmax max and denominator, which fa_combine does not).
func (r *cudaResident) faEligible(l int) bool {
	if r.faSplit < 1 || r.faCombine == (Pipeline{}) {
		return false
	}
	Ly := &r.layers[l]
	if (Ly.hd != 64 && Ly.hd != 128 && Ly.hd != 256) || Ly.nKV < 1 || r.nH%Ly.nKV != 0 || r.nH/Ly.nKV > 8 {
		return false
	}
	return r.sinkArg(l) == ArgNull()
}

// flashDecodeAttn is the lane: fa_partial (nKV x S CTAs of 8 warps) then fa_combine (nH blocks), writing r.cctx
// exactly as the exact path would. NOT bit-identical to it; see decode_fa.cu.
func (r *cudaResident) flashDecodeAttn(l, pos int) error {
	Ly := &r.layers[l]
	nKeys := pos + 1
	winStart := 0
	if Ly.window > 0 && nKeys > int(Ly.window) {
		winStart = nKeys - int(Ly.window)
	}
	return r.flashDecodeLaunch(r.qB, r.kc[l], r.vc[l], r.cctx, Ly.hd, Ly.nKV, winStart, nKeys)
}

// flashDecodeLaunch is flashDecodeAttn's body over explicit buffers, so a test can drive it on synthetic K/V.
func (r *cudaResident) flashDecodeLaunch(q, kc, vc, ctx Buffer, hd, nKV, winStart, nKeys int) error {
	G := r.nH / nKV
	which := map[int]int{64: 0, 128: 1, 256: 2}[hd]
	shm := uint32((G*hd + faWarps*64) * 4)
	if e := r.launch(r.faPartial[which], LaunchConfig{GridX: uint32(nKV), GridY: uint32(r.faSplit), GridZ: 1, BlockX: faWarps * 32, BlockY: 1, BlockZ: 1, SharedMemBytes: shm},
		Arg(q), Arg(kc), Arg(vc), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(nKV)), gpu.ArgValue(int32(winStart)),
		gpu.ArgValue(int32(nKeys)), gpu.ArgValue(r.attnScale), gpu.ArgValue(int32(r.faSplit)), Arg(r.faBuf)); e != nil {
		return e
	}
	r.faLaunches++
	return r.launch(r.faCombine, LaunchConfig{GridX: uint32(r.nH), GridY: 1, GridZ: 1, BlockX: uint32(hd), BlockY: 1, BlockZ: 1, SharedMemBytes: uint32(3 * r.faSplit * faWarps * 4)},
		Arg(r.faBuf), gpu.ArgValue(int32(hd)), gpu.ArgValue(int32(r.faSplit*faWarps)), Arg(ctx))
}

// faRowsPerCTA mirrors FaRows<HD, G>::R in decode_fa.cu: rows handled per CTA, chosen so the R*G*(hd/32) accumulators stay near
// 64 registers. The kernel and this function must agree (the grid's z extent is derived from it); TestFlashDecodeRowsBitIdentical
// fails if they do not, because rows would be skipped.
func faRowsPerCTA(hd, g int) int {
	raw := 64 / ((hd / 32) * g)
	return min(max(raw, 1), 8)
}

// faRun is a maximal run of consecutive verify rows that share a chunk partition (equal winStart and per), so one multi-row launch
// can serve them and each row still sees exactly the partition the M=1 kernel gives it.
type faRun struct{ row0, n, nKeys0, winStart, per int }

// faRowRuns groups rows i=0..m-1, with nKeys_i = n0+i and the layer's sliding window, into runs. per_i = ceil(span_i / S).
func faRowRuns(n0, m, window, s int) []faRun {
	var runs []faRun
	for i := 0; i < m; i++ {
		nk := n0 + i
		ws := 0
		if window > 0 && nk > window {
			ws = nk - window
		}
		per := (nk - ws + s - 1) / s
		if k := len(runs) - 1; k >= 0 && runs[k].winStart == ws && runs[k].per == per {
			runs[k].n++
			continue
		}
		runs = append(runs, faRun{row0: i, n: 1, nKeys0: nk, winStart: ws, per: per})
	}
	return runs
}

// flashDecodeRowsLaunch runs the multi-row lane over ONE run of rows: fa_partial_rows then fa_combine_rows. q, kc, vc, ctx are the
// batch's buffers; q and ctx rows are [row][nH*hd]. Output row (run.row0+i) equals the M=1 lane's output at nKeys = run.nKeys0+i.
func (r *cudaResident) flashDecodeRowsLaunch(q, kc, vc, ctx Buffer, hd, nKV int, run faRun) error {
	G := r.nH / nKV
	which := map[int]int{64: 0, 128: 1, 256: 2}[hd]
	R := faRowsPerCTA(hd, G)
	shm := uint32((R*G*hd + faWarps*R*64) * 4)
	grid := uint32((run.n + R - 1) / R)
	if e := r.launch(r.faRows[which][G], LaunchConfig{GridX: uint32(nKV), GridY: uint32(r.faSplit), GridZ: grid, BlockX: faWarps * 32, BlockY: 1, BlockZ: 1, SharedMemBytes: shm},
		Arg(q), Arg(kc), Arg(vc), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(nKV)), gpu.ArgValue(int32(run.winStart)),
		gpu.ArgValue(int32(run.nKeys0)), gpu.ArgValue(int32(run.per)), gpu.ArgValue(r.attnScale), gpu.ArgValue(int32(r.faSplit)),
		gpu.ArgValue(int32(run.n)), gpu.ArgValue(int32(run.row0)), Arg(r.faBuf)); e != nil {
		return e
	}
	r.faLaunches++
	r.faRowLaunches++
	return r.launch(r.faCombineRows, LaunchConfig{GridX: uint32(r.nH), GridY: uint32(run.n), GridZ: 1, BlockX: uint32(hd), BlockY: 1, BlockZ: 1, SharedMemBytes: uint32(3 * r.faSplit * faWarps * 4)},
		Arg(r.faBuf), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(hd)), gpu.ArgValue(int32(r.faSplit*faWarps)), gpu.ArgValue(int32(run.row0)), Arg(ctx))
}

// verifyLaneFrom returns the first row of an M-row batch (rows at positions startPos..startPos+M-1) that the multi-row lane serves;
// rows below it take the exact attention, rows from it up take the lane; M means no lane rows. Per row it is EXACTLY the M=1 decode
// rule: lane on, no exact-attention scope held, the layer eligible, and the row's attended span at or over the floor. The rule is
// monotone in the row (the span never shrinks), so the lane rows are a suffix. Only all-rows verify tails qualify (a prompt prefill
// is not decode-consistent-by-construction and keeps its exact/fused attention), and an image block never does.
func (r *cudaResident) verifyLaneFrom(l, startPos, m, tail int, hasImage bool) int {
	if !r.faVerify || hasImage || (tail != tailAllLogits && tail != tailAllArgmax) || r.faExactScope.Load() != 0 || !r.faEligible(l) {
		return m
	}
	win := int(r.layers[l].window)
	for i := 0; i < m; i++ {
		nk := startPos + i + 1
		nWin := nk
		if win > 0 && nk > win {
			nWin = win
		}
		if nWin >= r.faMinKeys {
			return i
		}
	}
	return m
}

// flashVerifyAttn runs rows [from, m) of the batch through the multi-row lane, grouping them into runs that share a chunk partition
// and chunking a run longer than faMaxRows. Runs execute one after another on the one stream and reuse the partial buffer's slots.
func (r *cudaResident) flashVerifyAttn(l, startPos, from, m int, q, ctx Buffer) error {
	Ly := &r.layers[l]
	for _, run := range faRowRuns(startPos+from+1, m-from, int(Ly.window), r.faSplit) {
		for a := 0; a < run.n; a += faMaxRows {
			sub := run
			sub.row0 = from + run.row0 + a
			sub.n = min(faMaxRows, run.n-a)
			sub.nKeys0 = run.nKeys0 + a
			if e := r.flashDecodeRowsLaunch(q, r.kc[l], r.vc[l], ctx, Ly.hd, Ly.nKV, sub); e != nil {
				return e
			}
		}
	}
	return nil
}
