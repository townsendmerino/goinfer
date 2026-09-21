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

// flashDecodeDefaultMinKeys is the attended-span floor for the lane until a ladder sets it: below it the exact
// path runs. Deliberately the split-KV conservative default; the R6 ladder replaces it with a measured value.
const flashDecodeDefaultMinKeys = 1024

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
	comb, e := r.dev.NewComputePipeline(mod, "fa_combine")
	if e != nil {
		fmt.Fprintf(os.Stderr, "[cuda] GOINFER_CUDA_FLASH_DECODE ignored: %v\n", e)
		return
	}
	r.faPartial, r.faCombine = pipes, comb
	r.faBuf = r.af(r.nH * s * faWarps * (maxHd + 4))
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
	return r.launch(r.faCombine, LaunchConfig{GridX: uint32(r.nH), GridY: 1, GridZ: 1, BlockX: uint32(hd), BlockY: 1, BlockZ: 1},
		Arg(r.faBuf), gpu.ArgValue(int32(hd)), gpu.ArgValue(int32(r.faSplit*faWarps)), Arg(ctx))
}
