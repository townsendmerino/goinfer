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

// captureState holds cudaResident's hidden-state and diagnostic captures: the per-layer residual probe, the drafter taps (per
// token and batched), Gemma 4's MoE-branch readback and the per-sublayer contributions. Each is off (nil or false) in production.
type captureState struct {
	layerCap    bool // DEBUG probe: snapshot the residual r.x after every layer (localizes where a full-forward divergence first appears)
	layerCapBuf [][]float32
	// hidCap is the hidden-state seam a resident CUDA target would need to feed a hidden-state drafter (DFlash,
	// DSpark): the resident analogue of decoder.Model.ForwardCapture, which exists only on the CPU forward. It is NOT
	// wired into production: SetHiddenCapture/HiddenCapture have no non-test caller (the block drafter arms the batched
	// counterpart below, via SetBatchedCapture), so nothing reaches this per-token seam yet
	// (docs/code-notes/cuda.md#cudaResident.hidCap).
	//
	// It is distinct from layerCap, deliberately: layerCap is a divergence-localization probe (EVERY layer, a
	// stream.Sync() and a download each, into an unbounded buffer: 36 syncs per token at 36 layers), far too
	// expensive to decode against. hidCap copies only the TAPPED layers into fixed slots, so a 5-tap drafter costs 5.
	hidCapTaps []int       // layer indices to capture, ascending; nil ⇒ seam off
	hidCapOut  [][]float32 // [len(hidCapTaps)][hidden], overwritten per token
	// The BATCHED counterpart, for the block-drafting verify: one download per tap covering
	// all M rows, instead of the per-token seam's one per tap PER TOKEN.
	capBTaps []int       // layer indices, ascending; nil ⇒ off
	capBOut  [][]float32 // [len(capBTaps)][M*hidden], overwritten per batched call
	// g4cap (GOINFER_G4_CAPTURE) reads back the four MoE-layer buffers (rn / wgt / x1 / x2), so a whole-forward miss localizes to
	// router vs dense vs expert vs join in one run.
	g4cap                               bool
	g4capRn, g4capWgt, g4capX1, g4capX2 [][]float32
	g4capIdx                            [][]uint32 // APPEND order (token-outer, layer-inner), matching the CPU routerCaptureBuf — for the per-POSITION routing-agreement check (a top-k flip at pos N reads like accumulation in a cosine)
	// g4capLayer is g4capIdx's layer index, recorded in parallel so a consumer replaying the trace through a
	// per-layer cache need not infer the (token, layer) shape from the append order and a divisor.
	g4capLayer []int
	// Per-sublayer contribution capture (diagnostic; off in production, zero cost). When subCap
	// is set, launchToken copies the sandwich-normed o-proj output (attention contribution) and
	// down output (MLP contribution) per layer — the exact dp4a-path analogue of the decoder's
	// ForwardSubCapture, so a cross-backend per-sublayer diff is possible.
	subCap                                 bool
	subAttnC, subMLPC, subCtxC, subMLPpreC [][]float32
}

// dnetState is the Gated-DeltaNet mixer: its kernels (per token, row-batched and the speculative verify's), the family's
// parameters, the verify arena and the per-token scratch. Zero for every other family.
type dnetState struct {
	// Gated-DeltaNet mixer (deltanet.ptx, its own module). Loaded only for that family; every other model leaves
	// these zero and never dispatches them.
	dnConv, dnGates, dnNorm, dnRule, dnGNorm Pipeline
	// Row-batched twins of the five, for prompt prefill (prefillDeltaNetRows): one launch per layer covers M rows.
	dnConvRows, dnGatesRows, dnNormRows, dnRuleRows, dnGNormRows Pipeline
	dnRuleRows128                                                Pipeline // the scan with its state row in registers (hk == hv == 128)
	// The speculative verify's twins of dnConvRows, dnRuleRows and dnRuleRows128, which never store the window or the state, and the arena
	// a verify pass keeps its rows in (recurrent_spec.go). dnSpecOn is set for the duration of one VerifyLastNArgmax.
	dnConvRowsSpec, dnRuleRowsSpec, dnRuleRows128Spec Pipeline
	dnSpec                                            *dnSpecArena
	dnSpecOn                                          bool
	dnQSplit, dnAttnGate                              Pipeline // the family's fused double-width q_proj + output gate
	dnet                                              *dnetParams
	// Gated-DeltaNet per-token scratch (allocated only when any layer isDeltaNet). Sized from
	// dnetParams, which is NOT the attention geometry: convDim is 2*nk*hk + nv*hv and has no
	// relation to qDim/kvDim, so none of the buffers above can be reused for it.
	dnMixed, dnConvOut, dnQn, dnKn, dnHeadP, dnBt, dnAt, dnZOut, dnCore, dnGated Buffer
	dnGq, dnGSc                                                                  Buffer // int8 activation + scale for the gated output's out_proj GEMV
	dnQg                                                                         Buffer // [2*qDim] the fused [query ‖ gate] q_proj output, before the split
	dnAGate                                                                      Buffer // [qDim] attention output gate (qGate layers)
}
