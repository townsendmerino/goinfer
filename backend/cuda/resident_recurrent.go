//go:build cuda

package cuda

import (
	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// dnetParams carries the model-level Gated-DeltaNet geometry (uniform across the linear layers).
// keyDim/valueDim/convDim are derived once rather than at every dispatch because the three are
// easy to conflate: convDim is 2*keyDim+valueDim (the conv runs over [q|k|v] together), and
// rep = nv/nk is the GVA factor mapping value heads to key heads.
type dnetParams struct {
	convK      int
	hk, hv     int
	nk, nv     int
	rep        int
	keyDim     int
	valueDim   int
	convDim    int
	stateElems int
	qScale     float32
}

// deltaNetGateSettings reads a DeltaNet layer's two settings from its operation: the write gate's scale (2 where NegEigval
// widens sigmoid's [0, 1) to [0, 2), else 1) and the output norm's own epsilon (ONormEps; 0 means the model's NormEps).
// program.CUDA claims both (DeltaNegEigval, DeltaONormEps) on the strength of this; dnSettingPlantForTest drops one for the
// gates.
func deltaNetGateSettings(ops []program.Op) (betaScale, oNormEps float32) {
	betaScale = 1
	for _, o := range ops {
		if o.Kind != program.OpDeltaNet {
			continue
		}
		if o.Delta.NegEigval && dnSettingPlantForTest != dnSettingPlantDropNegEigval {
			betaScale = 2
		}
		if o.Delta.ONormEps != 0 && dnSettingPlantForTest != dnSettingPlantDropONormEps {
			oNormEps = float32(o.Delta.ONormEps)
		}
	}
	return betaScale, oNormEps
}

// dnSettingPlantForTest drops one of deltaNetGateSettings' settings at build. Set by tests only.
var dnSettingPlantForTest int

const (
	dnSettingPlantDropNegEigval = iota + 1 // the write gate keeps sigmoid's [0, 1)
	dnSettingPlantDropONormEps             // the output norm takes the model's NormEps
)

// winFloats is a DeltaNet layer's conv window in floats, as the setup job allocates dnWin; stateElems is its recurrent state.
func (dp *dnetParams) winFloats() int { return (dp.convK - 1) * dp.convDim }

// deltaNetMixer runs one Gated-DeltaNet layer's sequence mixer and folds its output into the
// residual — the recurrent replacement for segA + rope + attention + o-proj.
//
//	norm → in_proj_qkv → conv(ring) → l2norm(q,k) → gates → delta rule(state) → gated norm × silu(z)
//	     → out_proj + residual
//
// With graphs on, captureGraphs records this together with segBFFN as the layer's single gSegA segment: the state
// buffers' contents ARE the per-token state, and replay reads current contents like any other buffer.
//
// The two small gate projections (dnB/dnA) run off the SAME quantized activation as the big ones.
// On the CPU they are f32 by deliberate choice: deltaNetWeights keeps inProjB/inProjA unquantized
// because they feed the write/decay gates, where the recurrence is most precision-sensitive. This
// is the one place the resident path is knowingly coarser than the reference, and the parity gate
// scores the gates as their own stage so the cost of that choice is visible rather than assumed.
func (r *cudaResident) deltaNetMixer(Ly *cudaLayer, l int) error {
	dp := r.dnet
	nullBias := ArgNull()
	if e := r.rms(r.x, Ly.preNorm, r.aq, r.aSc); e != nil {
		return e
	}
	if e := r.doG(Ly.dnQKV, r.aq, r.aSc, nullBias, r.dnMixed, 0); e != nil {
		return e
	}
	if e := r.doG(Ly.dnB, r.aq, r.aSc, nullBias, r.dnBt, 0); e != nil {
		return e
	}
	if e := r.doG(Ly.dnA, r.aq, r.aSc, nullBias, r.dnAt, 0); e != nil {
		return e
	}
	if e := r.doG(Ly.dnZ, r.aq, r.aSc, nullBias, r.dnZOut, 0); e != nil {
		return e
	}
	if e := r.launch(r.dnConv, g1cfg(dp.convDim, 256),
		Arg(r.dnMixed), Arg(Ly.dnConvW), Arg(Ly.dnWin), Arg(r.dnConvOut),
		gpu.ArgValue(int32(dp.convDim)), gpu.ArgValue(int32(dp.convK))); e != nil {
		return e
	}
	if e := r.launch(r.dnGates, g1cfg(dp.nv, 64),
		Arg(r.dnBt), Arg(r.dnAt), Arg(Ly.dnDtBias), Arg(Ly.dnNegExpA), Arg(r.dnHeadP),
		gpu.ArgValue(int32(dp.nv)), gpu.ArgValue(Ly.dnBetaScale)); e != nil {
		return e
	}
	if e := r.launch(r.dnNorm, LaunchConfig{GridX: uint32(dp.nk), GridY: 1, GridZ: 1,
		BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 2 * 128 * 4},
		Arg(r.dnConvOut), Arg(r.dnQn), Arg(r.dnKn),
		gpu.ArgValue(int32(dp.nk)), gpu.ArgValue(int32(dp.hk)), gpu.ArgValue(int32(dp.keyDim)),
		gpu.ArgValue(dp.qScale)); e != nil {
		return e
	}
	// v is the conv output's third slice; vBase points at it rather than copying it out.
	if e := r.launch(r.dnRule, g1cfg(dp.valueDim, 128),
		Arg(r.dnQn), Arg(r.dnKn), Arg(r.dnConvOut), Arg(r.dnHeadP), Arg(Ly.dnState), Arg(r.dnCore),
		gpu.ArgValue(int32(dp.nv)), gpu.ArgValue(int32(dp.hk)), gpu.ArgValue(int32(dp.hv)),
		gpu.ArgValue(int32(dp.rep)), gpu.ArgValue(int32(2*dp.keyDim))); e != nil {
		return e
	}
	if e := r.launch(r.dnGNorm, LaunchConfig{GridX: uint32(dp.nv), GridY: 1, GridZ: 1,
		BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: 128 * 4},
		Arg(r.dnCore), Arg(r.dnZOut), Arg(Ly.dnNormW), Arg(r.dnGated),
		gpu.ArgValue(int32(dp.nv)), gpu.ArgValue(int32(dp.hv)), gpu.ArgValue(Ly.dnGNormEps)); e != nil {
		return e
	}
	// Quantize the gated output, then out_proj straight into the residual (accum=1), exactly as
	// the attention path's o-proj does.
	if e := r.launch(r.fQ, r.actCfg(256, 256*4, dp.valueDim),
		Arg(r.dnGated), gpu.ArgValue(int32(dp.valueDim)), Arg(r.dnGq), Arg(r.dnGSc)); e != nil {
		return e
	}
	return r.doG(Ly.dnOut, r.dnGq, r.dnGSc, nullBias, r.x, 1)
}

// layerTail is everything launchToken does AFTER the FFN pre-half: the g4moe accumulator clear,
// the C′ expert DMA, segC, and the two capture seams. Split out so the Gated-DeltaNet mixer path
// rejoins it instead of duplicating it — a DeltaNet layer's FFN can be MoE (qwen3_5_moe,
// qwen3_next) and would otherwise skip the router readback entirely.
func (r *cudaResident) layerTail(Ly *cudaLayer, l int, gC bool, x Buffer) error {
	g4moe := Ly.walk.FFN.Kind == program.SublayerGemma4MoE
	routed := Ly.routedFFN()
	if g4moe {
		// segC(l-1) writes AND reads g4x2 on r.stream (CU_STREAM_NON_BLOCKING). The clear must land after that read and
		// before segC(l) accumulates into it. A memset ON r.stream is ordered by the stream itself, in every mode, and
		// drains nothing. Do not clear with gpu.Upload of a host zero slice: it runs its copy on the context's legacy null
		// stream, which has NO ordering against r.stream (aikit gpu/cuda.go), so it needs a Sync first, plus two more
		// context syncs and a pageable copy. The router readback that follows (loadRoutedExperts) still syncs r.stream
		// itself, so the wait is not lost, only moved to where it was needed anyway.
		if e := r.stream.ZeroAsync(r.g4x2, r.hidden*4); e != nil {
			return e
		}
		r.profMark(&r.profClear)
	}
	if routed && r.cacheExperts {
		if e := r.loadRoutedExperts(Ly); e != nil {
			return e
		}
		if g4moe {
			r.profMark(&r.profRT)
		}
	}
	// segC: the post-readback MoE half (expert loop + join). Dense layers have none.
	if routed {
		if gC {
			if e := Ly.gSegC.Replay(); e != nil {
				return e
			}
			if r.graphsSync {
				if err := r.stream.Sync(); err != nil {
					return err
				}
			}
		} else if e := r.segC(Ly, l, x); e != nil {
			return e
		}
		if g4moe {
			r.profMark(&r.profSegC)
		}
	}
	// hidden-state seam: this layer's OUTPUT residual, for a drafter that taps it.
	if len(r.hidCapTaps) > 0 {
		for slot, tap := range r.hidCapTaps {
			if tap == l {
				r.capVec(x, r.hidCapOut, slot, r.hidden)
				break
			}
		}
	}
	if r.layerCap { // DEBUG: snapshot the residual after this layer (divergence-localization probe)
		if err := r.stream.Sync(); err != nil {
			return err
		}
		h := make([]float32, r.hidden)
		if err := gpu.Download(x, h); err != nil {
			return err
		}
		r.layerCapBuf = append(r.layerCapBuf, h)
	}
	return nil
}
