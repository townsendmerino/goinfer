//go:build cuda

package cuda

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	gpu "github.com/townsendmerino/goinfer/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/cuda/internal/launchgeom"
	"github.com/townsendmerino/goinfer/decoder"
)

// faWarps and faMaxRows are the flash-decode lane's launch geometry (package launchgeom has their contracts).
const (
	faWarps   = launchgeom.FaWarps
	faMaxRows = launchgeom.FaMaxRows
)

// flashDecodeDefaultMinKeys is the attended-span floor for the lane: below it the exact path runs. 2048 is the lowest
// floor with no measured regression on the served forced-on ladder
// (docs/measurements/attn-decode-fa-served-2026-09-20.md).
const flashDecodeDefaultMinKeys = 2048

// flashDecodeDefaultSplit is S, the key-split count per kv head, when GOINFER_CUDA_FLASH_DECODE is unset. 16 is the S the
// fidelity gate was pre-registered and passed at (docs/measurements/attn-decode-fa-fidelity-2026-09-20.md).
const flashDecodeDefaultSplit = 16

// flashDecodeSplit resolves GOINFER_CUDA_FLASH_DECODE to S. Default on: unset means flashDecodeDefaultSplit; "0", "off"
// and "false" turn the lane off (the exact attention path); a positive integer picks S; anything else is also off, since
// a typo must not silently enable a non-exact path at some other S.
func flashDecodeSplit(v string, set bool) int {
	if !set || v == "" {
		return flashDecodeDefaultSplit
	}
	switch strings.ToLower(v) {
	case "off", "false":
		return 0
	}
	if s, err := strconv.Atoi(v); err == nil && s >= 1 {
		return s
	}
	return 0
}

// loadFlashDecode wires the flash-decode lane (default on; GOINFER_CUDA_FLASH_DECODE=0 keeps the exact path). It acts
// only once a layer's attended span reaches faMinKeys, and any failure leaves the lane off (faSplit 0), never half-on.
//
// The default does not reach a C' expert-cache model (cacheExperts): those are sized to the byte from free VRAM
// (docs/measurements/moe-streaming-decode-overlap-ceiling-2026-09-22.md), the lane's partial buffer
// (faMaxRows*nH*S*faWarps*(hd+4) f32) would come out of that budget, and none of the lane's fidelity evidence covers a
// MoE. Setting the variable explicitly still turns it on there.
func (r *cudaResident) loadFlashDecode(m *decoder.Model, nLayers int) {
	v, explicit := r.lookupKnob("GOINFER_CUDA_FLASH_DECODE")
	s := flashDecodeSplit(v, explicit)
	if !explicit && r.cacheExperts {
		return
	}
	if s < 1 {
		return
	}
	maxHd := 0
	for l := range nLayers {
		if h := m.HeadDimAtResident(l); h > maxHd {
			maxHd = h
		}
	}
	if maxHd <= 0 {
		fmt.Fprintf(os.Stderr, "[cuda] flash-decode lane unavailable (exact decode attention used): no positive head dim\n")
		return
	}
	mod, e := r.dev.CompileLibrary(r.kvPTX(decodeFAPTX, decodeFAKV16PTX))
	if e != nil {
		fmt.Fprintf(os.Stderr, "[cuda] flash-decode lane unavailable (exact decode attention used): %v\n", e)
		return
	}
	var pipes [3]Pipeline
	for i, name := range []string{"fa_partial_64", "fa_partial_128", "fa_partial_256"} {
		if pipes[i], e = r.dev.NewComputePipeline(mod, name); e != nil {
			fmt.Fprintf(os.Stderr, "[cuda] flash-decode lane unavailable (exact decode attention used): %v\n", e)
			return
		}
	}
	if r.faCombine, e = r.dev.NewComputePipeline(mod, "fa_combine"); e != nil {
		fmt.Fprintf(os.Stderr, "[cuda] flash-decode lane unavailable (exact decode attention used): %v\n", e)
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
	r.faVerify = r.faCombineRows != (Pipeline{}) && r.knobValue("GOINFER_CUDA_FLASH_DECODE_VERIFY") != "0"
	r.faMinKeys = flashDecodeDefaultMinKeys
	if v, err := strconv.Atoi(r.knobValue("GOINFER_CUDA_FLASH_DECODE_MIN_KEYS")); err == nil && v >= 0 {
		r.faMinKeys = v
	}
	r.faSplit = s
}

// faEligible reports whether layer l can take the lane: a supported head dim, a GQA group of at most 8, and no
// attention sink (gpt-oss folds the sink into the softmax max and denominator, which fa_combine does not).
func (r *cudaResident) faEligible(l int) bool {
	if r.faSplit < 1 || r.faCombine == (Pipeline{}) || r.attnCap != 0 { // the lane has no cap: a capped layer takes the exact kernels
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
	return r.flashDecodeLaunch(r.qB, r.kvK(l), r.kvV(l), r.kvT(l), r.cctx, Ly.hd, Ly.nKV, winStart, nKeys)
}

// flashDecodeLaunch is flashDecodeAttn's body over explicit buffers, so a test can drive it on synthetic K/V.
func (r *cudaResident) flashDecodeLaunch(q, kc, vc, kvt, ctx Buffer, hd, nKV, winStart, nKeys int) error {
	G := r.nH / nKV
	which := map[int]int{64: 0, 128: 1, 256: 2}[hd]
	shm := uint32((G*hd + faWarps*64) * 4)
	if e := r.launch(r.faPartial[which], LaunchConfig{GridX: uint32(nKV), GridY: uint32(r.faSplit), GridZ: 1, BlockX: faWarps * 32, BlockY: 1, BlockZ: 1, SharedMemBytes: shm},
		Arg(q), Arg(kc), Arg(vc), Arg(kvt), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(nKV)), gpu.ArgValue(int32(winStart)),
		gpu.ArgValue(int32(nKeys)), gpu.ArgValue(r.attnScale), gpu.ArgValue(int32(r.faSplit)), Arg(r.faBuf)); e != nil {
		return e
	}
	r.faLaunches++
	return r.launch(r.faCombine, LaunchConfig{GridX: uint32(r.nH), GridY: 1, GridZ: 1, BlockX: uint32(hd), BlockY: 1, BlockZ: 1, SharedMemBytes: uint32(3 * r.faSplit * faWarps * 4)},
		Arg(r.faBuf), gpu.ArgValue(int32(hd)), gpu.ArgValue(int32(r.faSplit*faWarps)), Arg(ctx))
}

// faRowsPerCTA is launchgeom.FaRowsPerCTA.
func faRowsPerCTA(hd, g int) int { return launchgeom.FaRowsPerCTA(hd, g) }

// faRun and faRowRuns group a verify batch's rows into runs one multi-row launch can serve (package launchgeom).
type faRun = launchgeom.FaRun

func faRowRuns(n0, m, window, s int) []faRun { return launchgeom.FaRowRuns(n0, m, window, s) }

// flashDecodeRowsLaunch runs the multi-row lane over ONE run of rows: fa_partial_rows then fa_combine_rows. q, kc, vc, ctx are the
// batch's buffers; q and ctx rows are [row][nH*hd]. Output row (run.Row0+i) equals the M=1 lane's output at nKeys = run.NKeys0+i.
func (r *cudaResident) flashDecodeRowsLaunch(q, kc, vc, kvt, ctx Buffer, hd, nKV int, run faRun) error {
	G := r.nH / nKV
	which := map[int]int{64: 0, 128: 1, 256: 2}[hd]
	R := faRowsPerCTA(hd, G)
	shm := uint32((R*G*hd + faWarps*R*64) * 4)
	grid := uint32((run.N + R - 1) / R)
	if e := r.launch(r.faRows[which][G], LaunchConfig{GridX: uint32(nKV), GridY: uint32(r.faSplit), GridZ: grid, BlockX: faWarps * 32, BlockY: 1, BlockZ: 1, SharedMemBytes: shm},
		Arg(q), Arg(kc), Arg(vc), Arg(kvt), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(nKV)), gpu.ArgValue(int32(run.WinStart)),
		gpu.ArgValue(int32(run.NKeys0)), gpu.ArgValue(int32(run.Per)), gpu.ArgValue(r.attnScale), gpu.ArgValue(int32(r.faSplit)),
		gpu.ArgValue(int32(run.N)), gpu.ArgValue(int32(run.Row0)), Arg(r.faBuf)); e != nil {
		return e
	}
	r.faLaunches++
	r.faRowLaunches++
	return r.launch(r.faCombineRows, LaunchConfig{GridX: uint32(r.nH), GridY: uint32(run.N), GridZ: 1, BlockX: uint32(hd), BlockY: 1, BlockZ: 1, SharedMemBytes: uint32(3 * r.faSplit * faWarps * 4)},
		Arg(r.faBuf), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(hd)), gpu.ArgValue(int32(r.faSplit*faWarps)), gpu.ArgValue(int32(run.Row0)), Arg(ctx))
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
	for i := range m {
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
		for a := 0; a < run.N; a += faMaxRows {
			sub := run
			sub.Row0 = from + run.Row0 + a
			sub.N = min(faMaxRows, run.N-a)
			sub.NKeys0 = run.NKeys0 + a
			if e := r.flashDecodeRowsLaunch(q, r.kvK(l), r.kvV(l), r.kvT(l), ctx, Ly.hd, Ly.nKV, sub); e != nil {
				return e
			}
		}
	}
	return nil
}
