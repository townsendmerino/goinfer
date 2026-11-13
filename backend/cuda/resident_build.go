//go:build cuda

package cuda

import (
	"fmt"
	"math"
	"os"
	"runtime"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/decoder/fit"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// buildDecline marks an error from a build phase as a decline: buildResidentMode's declined closure turns it into the typed
// decoder.DeclineResident, where any other error from a phase is returned as it is.
type buildDecline struct{ error }

func decline(err error) error { return buildDecline{err} }

// buildState is what a resident build resolves before it allocates anything, handed from one phase to the next. A phase fills the
// fields its successors read.
type buildState struct {
	m     *decoder.Model
	w     *decoder.Weights
	plans bool // plan-only or memory plan: the build stops before it allocates
	// kq reads the file's K-quant blocks for planProj; the build closes it when it returns.
	kq *decoder.KQuantReader

	shape program.Shape

	H, nLayers, nH, I, vocab int
	pleP                     int

	nE, topK, moeInter, sharedInter, nGroup, topkGroup int
	moeSig, moeNorm, sharedUngated, isMoE              bool
	moeScale                                           float64
	isG4MoE, isGemma4                                  bool

	qLoRA, kvLoRA, qkNope, qkRope, vHead int
	interleave                           bool
	mlaAttnScale, mlaRopeScale           float64
	mlaOK                                bool

	dnConvK, dnHK, dnHV, dnNK, dnNV int
	dnAttnGate, dnetOK              bool
	dnetP                           *dnetParams
	mambaP                          *mambaParams

	sandwich, postOnly, qkNormWhole, layerNorm, parallelBlock bool
	logitScale                                                float32
	actG32                                                    bool

	hls       []hlayer // each layer's planned weights, filled by planLayers
	hlm       hostW    // the head's
	loadAbort <-chan struct{}
}

// planProj plans a projection or the head: the file's own Q5_K or Q6_K blocks where the q4k lane uploads them natively
// (decoder.Model.KQuantNative, which the fit pricing also reads), else planWeight. s.kq reads the file until the build returns.
func (s *buildState) planProj(src *linalg.WeightMat) (hostW, error) {
	m := s.m
	kind := m.KQuantNative("cuda", src)
	if kind == "" {
		return planWeight(src)
	}
	if s.kq == nil {
		var e error
		if s.kq, e = m.OpenKQuant(); e != nil || s.kq == nil {
			return hostW{}, fmt.Errorf("cuda: the file's %s blocks are unreadable: %v", kind, e)
		}
	}
	row, e := s.kq.RowSource(src)
	if e != nil {
		return hostW{}, e
	}
	h, e := planKQuant(src, kind, row)
	if e == nil && kquantKeepInt8ForTest {
		alt, e2 := planWeight(src)
		if e2 != nil {
			return hostW{}, e2
		}
		h.alt = &alt
	}
	return h, e
}

// admit resolves the geometry the build dispatches on and declines what this backend does not implement: the features outside
// the program, a layer shape or operation CUDA's table lacks, and the per-family and per-32 activation sub-shapes. It reads the
// model and allocates nothing.
func (s *buildState) admit() error {
	m := s.m
	w := m.Weights()
	if w == nil || len(w.Layers) == 0 {
		return decline(fmt.Errorf("the model has no loaded layer weights"))
	}
	// Admission: anything this backend does not implement must decline, or it is silently dropped and the logits are wrong. The
	// operations and parameters are checked against CUDA's table (program.CUDA) below; the features the program does not express
	// (decoder.Model.FeaturesOutsideProgram) are checked here.
	if missing := m.FeaturesOutsideProgram(); len(missing) > 0 {
		return decline(fmt.Errorf("cuda does not implement %v, which this model needs", missing))
	}
	// The layer shape the build dispatches on comes from the program (OP4cu-b1); a program whose layers disagree on it, or ask
	// for two placements at once, declines here, naming the layer.
	shape, shapeErr := program.ShapeOf(m.Program())
	if shapeErr != nil {
		return decline(shapeErr)
	}
	// Every operation and parameter must be one CUDA's walker implements (program.CUDA, OP4cu-c1); the decline names the first
	// layer, operation and parameter outside the table.
	if tableErr := program.CUDA.Check(m.Program()); tableErr != nil && legacyLayerForTest == nil {
		return decline(tableErr)
	}

	H, nLayers, nH, _, _, I, vocab := m.Dims() // nKV, hd are per-layer now (KVHeadsAtResident/HeadDimAtResident); model-level unused
	// Gemma 4 E-model PLE width P (0 for every other model). The resident embedding row is then [hidden ‖ nLayers*P] (ResidentEmbedLen):
	// the host computes the per-layer inputs, and each layer's PLE branch reads its slice of the tail (S1 on CUDA).
	pleP := m.Gemma4PLEDimResident()
	if want := H + nLayers*pleP; pleP > 0 && m.ResidentEmbedLen() != want {
		return decline(fmt.Errorf("gemma4 E-model: the decoder's resident embedding row is %d floats, but hidden+layers*P is %d", m.ResidentEmbedLen(), want))
	}

	// ---- MoE knobs (ok=false for a dense model; every field then stays zero) ----
	// sharedUngated picks which shared-expert combine runs: GLM/DeepSeek add the shared output
	// straight into the residual, Qwen-MoE scales it by sigmoid(SharedGate·h) first. Both are
	// the same kernel (shared_gate_combine, `ungated` flag); only the gated one needs the extra
	// [1,hidden] weight.
	nE, topK, moeInter, sharedInter, moeSig, moeNorm, sharedUngated, moeScale, nGroup, topkGroup, isMoE := m.MoEResidentParams()
	// Gemma-4 enable_moe_block sets arch.MoE (so isMoE is true) but is NOT the generic MoE: it runs
	// gemma4MoeMLP (parallel dense‖MoE + join), so it bypasses the generic-MoE admission checks below
	// (gelu-tanh act, the attention sandwich norm) and gets its own int4-shape checks. Its nE/topK/
	// moeInter still come from arch.MoE (MoEResidentParams), matching the bundle.
	isG4MoE := m.HasGemma4MoEResident()
	isGemma4 := m.IsGemma4Resident()
	if isMoE && !isG4MoE {
		// Decline anything the dispatch does not implement loudly rather than dropping it; FeatMoE is one flag and cannot
		// express these sub-shapes. The gated shared expert (Qwen-MoE) is a derived feature (FeatMoEGatedShared,
		// decoder/features.go) that CUDA does not declare, so the admission check above declines it before this switch; a copy
		// of that decline here could drift from the taxonomy. moe_route's MOE_MAX_E / MOE_MAX_G come from decoder's declaration
		// (ResidentBackendMoECap), which the load path's admission gate already applied; do not restate the numbers here.
		capE, capG, _ := decoder.ResidentBackendMoECap("cuda")
		switch {
		case nE > capE:
			return decline(fmt.Errorf("MoE nE=%d exceeds moe_route's MOE_MAX_E=%d", nE, capE))
		case nGroup > capG:
			return decline(fmt.Errorf("MoE nGroup=%d exceeds moe_route's MOE_MAX_G=%d", nGroup, capG))
		case m.GatedActResident() != 1: // decoder.ActSiLU — decoder/mlp.go errors on any other
			return decline(fmt.Errorf("MoE experts are SwiGLU-only, arch act=%d", m.GatedActResident()))
		case shape.Sandwich:
			return decline(fmt.Errorf("MoE + sandwich norms: the combine accumulates straight into " +
				"the residual, so there is no seam to normalize the block output at"))
		case moeInter%32 != 0 || H%32 != 0:
			return decline(fmt.Errorf("MoE int4 needs moeInter(%d) and hidden(%d) both multiples of 32", moeInter, H))
		case sharedInter > 0 && sharedInter%32 != 0:
			return decline(fmt.Errorf("MoE int4 shared expert needs sharedInter(%d) a multiple of 32", sharedInter))
		}
	}
	if isG4MoE {
		// gemma4MoeMLP's indexed-expert + dense GEMVs are int4/W4A8: the widths the kernels stride by
		// must be multiples of the group size (hidden) / 8-nibble word (moeInter, denseInter).
		if moeInter%32 != 0 || H%32 != 0 {
			return decline(fmt.Errorf("gemma4 MoE int4 needs moeInter(%d) and hidden(%d) both multiples of 32", moeInter, H))
		}
		if capE, _, _ := decoder.ResidentBackendMoECap("cuda"); nE > capE {
			return decline(fmt.Errorf("gemma4 MoE nE=%d exceeds moe_route's MOE_MAX_E=%d", nE, capE))
		}
	}

	// ---- host pack all weights (CPU; any incompatible shape → decline) ----
	// MLA geometry (DeepSeek / Kimi).
	qLoRA, kvLoRA, qkNope, qkRope, vHead, interleave, mlaAttnScale, mlaRopeScale, mlaOK := m.MLAResidentParams()

	// Gated-DeltaNet hybrid geometry. Resolved before the per-layer pack because BOTH layer kinds
	// branch on it: the linear layers take the recurrence, and the softmax ones carry the fused
	// double-width q_proj that no other family has.
	dnConvK, dnHK, dnHV, dnNK, dnNV, dnAttnGate, dnetOK := m.Qwen35ResidentParams()
	var dnetP *dnetParams
	if dnetOK {
		keyDim, valueDim := dnNK*dnHK, dnNV*dnHV
		dnetP = &dnetParams{
			convK: dnConvK, hk: dnHK, hv: dnHV, nk: dnNK, nv: dnNV, rep: dnNV / dnNK,
			keyDim: keyDim, valueDim: valueDim, convDim: 2*keyDim + valueDim,
			stateElems: dnNV * dnHV * dnHK,
			qScale:     float32(1 / math.Sqrt(float64(dnHK))),
		}
	}

	// Mamba-2 hybrid geometry (Nemotron-H). A block of that family carries one operation (cudaMixerOf), and its non-gated MoE blocks are not built
	// here: they decline by name, as does Granite-4.0-H, and an int8 load (buildMambaParams).
	mambaP, mambaErr := buildMambaParams(m)
	if mambaErr != nil {
		return decline(mambaErr)
	}
	if mambaP != nil && isMoE {
		return decline(fmt.Errorf("Nemotron-H's non-gated relu² MoE blocks are not built on CUDA"))
	}

	sandwich := shape.Sandwich
	// Olmo 3 / Olmo Hybrid: post-only norms.
	postOnly := shape.PostOnly
	qkNormWhole := shape.QKNormWhole
	// Cohere/Command-R and Cohere2/Command-R7B: LayerNorm and parallel blocks.
	layerNorm := shape.LayerNorm
	parallelBlock := shape.Parallel
	logitScale, _ := m.LogitScaleResident() // ok=false ⇒ 1, already the applyLogitScale no-op value
	// Per-32 ACTIVATION quantization (decoder.Options.ActQuantGroup, actgroup.cu): implemented for the
	// plain dense RMSNorm decode path. Every path whose kernels read ONE activation scale per vector
	// declines by name rather than reading a per-group buffer as a per-vector one.
	actG32 := m.ActQuantGroup() == 32
	if actG32 {
		switch {
		case isMoE && !moeActG32DeclineOffForTest:
			return decline(fmt.Errorf("per-32 activations: the MoE expert/router kernels read per-vector activation scales"))
		case mlaOK:
			return decline(fmt.Errorf("per-32 activations: the MLA path is not implemented"))
		case isGemma4:
			return decline(fmt.Errorf("per-32 activations: the Gemma 4 path is not implemented"))
		case dnetOK:
			return decline(fmt.Errorf("per-32 activations: the Gated DeltaNet path is not implemented"))
		case layerNorm, postOnly, parallelBlock:
			return decline(fmt.Errorf("per-32 activations: LayerNorm / post-norm-only / parallel-block families are not implemented"))
		case H%32 != 0 || I%32 != 0:
			return decline(fmt.Errorf("per-32 activations need hidden (%d) and FFN width (%d) to be multiples of 32", H, I))
		}
	}

	s.w, s.shape = w, shape
	s.H, s.nLayers, s.nH, s.I, s.vocab, s.pleP = H, nLayers, nH, I, vocab, pleP
	s.nE, s.topK, s.moeInter, s.sharedInter, s.nGroup, s.topkGroup = nE, topK, moeInter, sharedInter, nGroup, topkGroup
	s.moeSig, s.moeNorm, s.sharedUngated, s.isMoE, s.moeScale = moeSig, moeNorm, sharedUngated, isMoE, moeScale
	s.isG4MoE, s.isGemma4 = isG4MoE, isGemma4
	s.qLoRA, s.kvLoRA, s.qkNope, s.qkRope, s.vHead, s.interleave = qLoRA, kvLoRA, qkNope, qkRope, vHead, interleave
	s.mlaAttnScale, s.mlaRopeScale, s.mlaOK = mlaAttnScale, mlaRopeScale, mlaOK
	s.dnConvK, s.dnHK, s.dnHV, s.dnNK, s.dnNV, s.dnAttnGate, s.dnetOK, s.dnetP = dnConvK, dnHK, dnHV, dnNK, dnNV, dnAttnGate, dnetOK, dnetP
	s.mambaP = mambaP
	s.sandwich, s.postOnly, s.qkNormWhole, s.layerNorm, s.parallelBlock = sandwich, postOnly, qkNormWhole, layerNorm, parallelBlock
	s.logitScale, s.actG32 = logitScale, actG32
	return nil
}

// planLayers validates and host-plans every layer's weights and the head, declining what CUDA cannot hold, and leaves s.hls and
// s.hlm for the upload. With s.plans set the experts are validated instead of packed. It allocates nothing on the device.
func (s *buildState) planLayers() error {
	m, w, shape, planProj, plans := s.m, s.w, s.shape, s.planProj, s.plans
	H, nLayers, nH, pleP := s.H, s.nLayers, s.nH, s.pleP
	nE, moeInter, sharedInter := s.nE, s.moeInter, s.sharedInter
	sharedUngated, isMoE := s.sharedUngated, s.isMoE
	qLoRA, kvLoRA, qkNope, qkRope, vHead := s.qLoRA, s.kvLoRA, s.qkNope, s.qkRope, s.vHead
	dnAttnGate, dnetOK, mambaP := s.dnAttnGate, s.dnetOK, s.mambaP
	sandwich, postOnly, qkNormWhole, parallelBlock, actG32 := s.sandwich, s.postOnly, s.qkNormWhole, s.parallelBlock, s.actG32
	if planStartForTest != nil && !plans {
		planStartForTest()
	}
	hls := make([]hlayer, nLayers)
	// Under the C' expert cache the routed experts are packed straight into their pinned host copy when the build uploads, so planning only validates them.
	expertStack, sharedStack, sharedOne := packWeightStack, packWeightStack, packWeight
	if m.MoECacheExperts() {
		expertStack = planWeightStack
	}
	if plans {
		expertStack, sharedStack, sharedOne = planWeightStack, planWeightStack, planWeight
	}
	loadAbort := m.LoadAbort() // the swap tripwire's channel (Options.LoadAbort); nil never fires
	for l := range nLayers {
		select {
		case <-loadAbort:
			return fmt.Errorf("cuda: resident build aborted while planning layer %d of %d: %w", l, nLayers, decoder.ErrLoadAborted)
		default:
		}
		lw := &w.Layers[l]
		var hl hlayer
		hl.isMoE = isMoE && lw.Experts != nil // same key as decoder/mlp.go; false on dense prefix layers
		var mixerOK bool
		hl.mixer, hl.ffn, mixerOK = cudaMixerOf(m.Program().Layers[l])
		if !mixerOK {
			x, _ := program.MixerOf(m.Program().Layers[l])
			return decline(fmt.Errorf("layer %d: the %v mixer is not built on CUDA", l, x))
		}
		switch hl.mixer {
		case decoder.MixerAttention, decoder.MixerDeltaNet, decoder.MixerMLA:
		case decoder.MixerMamba, decoder.MixerNone:
			if mambaP == nil {
				return decline(fmt.Errorf("layer %d: the %v mixer is built on CUDA for Nemotron-H only", l, hl.mixer))
			}
		default:
			return decline(fmt.Errorf("layer %d: the %v mixer is not built on CUDA", l, hl.mixer))
		}
		// The layer's steps come from its operations through CUDA's fusion table (OP4cu-b2); after the upload, the sublayers they
		// run are checked against what each layer's upload holds.
		progSteps, stepErr := cudaStepsOf(m.Program().Layers[l])
		if stepErr != nil && legacyLayerForTest == nil {
			return decline(fmt.Errorf("layer %d: %w", l, stepErr))
		}
		hl.steps = progSteps
		g4b, isG4 := m.Gemma4MoEResidentLayer(l)
		hl.g4moe = isG4
		var proj []struct {
			dst *hostW
			src *linalg.WeightMat
		}
		type projEnt = struct {
			dst *hostW
			src *linalg.WeightMat
		}
		switch {
		case hl.mixer == decoder.MixerDeltaNet:
			// Gated-DeltaNet mixer layer: no attention projections at all. The three dominant
			// projections carry the model's quant; inB/inA are f32 on the CPU by deliberate
			// choice (they feed the write/decay gates) and packWeight quantizes them to int8
			// here — the one place this port is knowingly coarser than the reference.
			qkv, z, outP, inB, inA, cW, dtB, negA, nW := m.Qwen35DeltaWeights(l)
			hl.dnConvW, hl.dnDtBias, hl.dnNegExpA, hl.dnNormW = cW, dtB, negA, nW
			hl.dnBetaScale, hl.dnONormEps = deltaNetGateSettings(m.Program().Layers[l])
			// Name the missing tensor. Every one of these becomes a device up32, and an empty
			// slice there is a 0-byte allocation — which this driver reports as "invalid length"
			// with no hint at WHICH of the four (or which loader) produced it. Two different
			// checkpoints have already failed exactly that way during this bring-up.
			for _, chk := range []struct {
				name string
				n    int
			}{{"conv_w", len(cW)}, {"dt_bias", len(dtB)}, {"neg_exp_a", len(negA)}, {"norm_w", len(nW)}} {
				if chk.n == 0 {
					return decline(fmt.Errorf("layer %d: DeltaNet %s is empty — the loader did not "+
						"populate it for this container", l, chk.name))
				}
			}
			bWM := linalg.WrapF32(inB, len(dtB), H)
			aWM := linalg.WrapF32(inA, len(dtB), H)
			proj = []projEnt{{&hl.dnQKV, qkv}, {&hl.dnZ, z}, {&hl.dnOut, outP},
				{&hl.dnB, &bWM}, {&hl.dnA, &aWM}}
		case hl.mixer == decoder.MixerMamba:
			hm, e := packMambaLayer(m, l, H, mambaP)
			if e != nil {
				return decline(e)
			}
			hl.mamba = hm // packed above; nothing to pack through proj
		case hl.mixer == decoder.MixerNone:
			// A feed-forward-only block: the non-gated up and down projections (relu² between them). lw.GateProj is empty by design.
			proj = []projEnt{{&hl.u, &lw.UpProj}, {&hl.d, &lw.DownProj}}
		case dnetOK && dnAttnGate:
			// The same family's gated softmax layer (qwen3_5/qwen3_5_moe/qwen3_next; not every dnetOK family: Olmo Hybrid's
			// full-attention layer is olmo3's plain scheme, dnAttnGate=false, and falls through to default below). Its weights live
			// off lw.QProj, and q_proj is double width, [query ‖ gate] per head.
			hl.qGate = dnAttnGate
			qP, kP, vP, oP, qN, kN := m.Qwen35AttnWeights(l)
			hl.qNorm, hl.kNorm = qN, kN
			// Same reason as the DeltaNet check below: these become device uploads, and an empty
			// one is an "invalid length" with no indication of which.
			if len(qN) == 0 || len(kN) == 0 {
				return decline(fmt.Errorf("layer %d: qwen35 softmax layer has empty q_norm/k_norm "+
					"(%d/%d) — the loader did not populate them for this container", l, len(qN), len(kN)))
			}
			proj = []projEnt{{&hl.q, qP}, {&hl.k, kP}, {&hl.v, vP}, {&hl.o, oP}}
		case hl.mixer == decoder.MixerMLA:
			qA, qANorm, qB, qProj, kvA, kvANorm, kvB, oProj := m.MLALayerWeights(l)
			qAQ, qBQ, qQ, kvAQ, oQ := m.MLALayerInt8(l)
			hl.mlaQANorm, hl.mlaKVANorm = qANorm, kvANorm
			latDim := kvLoRA + qkRope
			// wm is a projection's int8 form when the load holds one (the codes packWeight makes from the f32), else its f32.
			wm := func(q *linalg.WeightMat, f []float32, rows, cols int) *linalg.WeightMat {
				if q.Rows() > 0 {
					return q
				}
				w := linalg.WrapF32(f, rows, cols)
				return &w
			}
			if qLoRA > 0 {
				proj = append(proj,
					projEnt{&hl.mlaQA, wm(qAQ, qA, qLoRA, H)},
					projEnt{&hl.mlaQB, wm(qBQ, qB, nH*(qkNope+qkRope), qLoRA)},
				)
			} else {
				proj = append(proj, projEnt{&hl.mlaQ, wm(qQ, qProj, nH*(qkNope+qkRope), H)})
			}
			kvaWM := wm(kvAQ, kvA, latDim, H)
			oWM := wm(oQ, oProj, H, nH*vHead)
			proj = append(proj,
				projEnt{&hl.mlaKVA, kvaWM},
				projEnt{&hl.o, oWM},
			)
			// kvB is [nH*(qkNope+vHead), kvLoRA] row-major (per head: k_nope rows ‖ v rows). Slice it into W_UKᵀ [nH, kvLoRA,
			// qkNope] (transposed for the absorb GEMV) and W_UV [nH, vHead, kvLoRA] (the lift, as-is), as
			// webgpuBackend.BuildResident does (gpu/residency.go).
			hRow := qkNope + vHead
			wuk := make([]float32, nH*kvLoRA*qkNope)
			wuv := make([]float32, nH*vHead*kvLoRA)
			for h := range nH {
				for d := range qkNope {
					src := kvB[(h*hRow+d)*kvLoRA : (h*hRow+d)*kvLoRA+kvLoRA]
					for cc := range kvLoRA {
						wuk[(h*kvLoRA+cc)*qkNope+d] = src[cc]
					}
				}
				for ev := range vHead {
					copy(wuv[(h*vHead+ev)*kvLoRA:(h*vHead+ev)*kvLoRA+kvLoRA], kvB[(h*hRow+qkNope+ev)*kvLoRA:(h*hRow+qkNope+ev)*kvLoRA+kvLoRA])
				}
			}
			hl.mlaWUK, hl.mlaWUV = wuk, wuv
		default:
			if m.KVSrcAtResident(l) != l {
				// A KV-shared layer (Gemma 4 E-model) carries only Q and O: its K/V are its source layer's.
				proj = []projEnt{{&hl.q, &lw.QProj}, {&hl.o, &lw.OProj}}
			} else {
				proj = []projEnt{{&hl.q, &lw.QProj}, {&hl.k, &lw.KProj}, {&hl.o, &lw.OProj}}
				if !m.VFromKResident(l) { // K=V (attention_k_eq_v) global layers carry NO v_proj — V=v_norm(k)
					proj = append(proj, projEnt{&hl.v, &lw.VProj})
				}
			}
		}
		if hl.g4moe {
			// Gemma-4 MoE dense branch: lw.GateProj/UpProj/DownProj are EMPTY (the dense MLP lives in
			// the gemma4moe sub-block), so pack from the bundle instead.
			proj = append(proj,
				struct {
					dst *hostW
					src *linalg.WeightMat
				}{&hl.g, g4b.MlpGate},
				struct {
					dst *hostW
					src *linalg.WeightMat
				}{&hl.u, g4b.MlpUp},
				struct {
					dst *hostW
					src *linalg.WeightMat
				}{&hl.d, g4b.MlpDown})
		} else if !hl.isMoE && hl.ffn && hl.mixer != decoder.MixerNone {
			// A routed layer carries no dense FFN — GateProj/UpProj/DownProj are empty, and
			// packing them would fail on a zero shape rather than mean anything.
			proj = append(proj,
				struct {
					dst *hostW
					src *linalg.WeightMat
				}{&hl.g, &lw.GateProj},
				struct {
					dst *hostW
					src *linalg.WeightMat
				}{&hl.u, &lw.UpProj},
				struct {
					dst *hostW
					src *linalg.WeightMat
				}{&hl.d, &lw.DownProj})
		}
		if pleP > 0 {
			if lw.PLEGate.Rows() != pleP || lw.PLEProj.Rows() != H || len(lw.PostPLENorm) != H {
				return decline(fmt.Errorf("layer %d: PLE weights are not the expected shape (gate rows %d want %d, proj rows %d want %d, post-norm %d want %d)",
					l, lw.PLEGate.Rows(), pleP, lw.PLEProj.Rows(), H, len(lw.PostPLENorm), H))
			}
			proj = append(proj, projEnt{&hl.pleGate, &lw.PLEGate}, projEnt{&hl.pleProj, &lw.PLEProj})
			hl.postPLENorm = lw.PostPLENorm
		}
		for _, p := range proj {
			hw, e := planProj(p.src) // validated now, packed into the device buffer at upload
			if e != nil {
				return decline(e)
			}
			if hw.kind == "q4k" && !actG32 {
				// gemv_q4k_g32's minimum term reads the per-group activation sums only the per-32
				// quantizers write; --quant q4k always stamps per-32, so this is defence in depth.
				return decline(fmt.Errorf("layer %d: a Q4_K weight needs per-32 activations (ActQuantGroup 32)", l))
			}
			*p.dst = hw
		}
		if hl.g4moe {
			// Gemma-4 MoE: stack the experts (each ExpertsGateUp is the fused [2*moeInter,hidden]
			// gate‖up), the f32 router (RouterProjScaled, scale folded), the 5 norms + per-expert
			// scale + layerScalar. The dense g/u/d were packed above via the proj list.
			var e error
			if hl.expGU, e = expertStack(g4b.ExpertsGateUp...); e != nil {
				return decline(fmt.Errorf("layer %d gemma4 expert gate‖up stack: %w", l, e))
			}
			if hl.expDown, e = expertStack(g4b.ExpertsDown...); e != nil {
				return decline(fmt.Errorf("layer %d gemma4 expert down stack: %w", l, e))
			}
			if hl.expGU.kind != "int4" || hl.expDown.kind != "int4" {
				return decline(fmt.Errorf("layer %d: gemma4 experts are %q/%q — the resident MoE GEMVs are int4-only (load with Quant: \"int4\")", l, hl.expGU.kind, hl.expDown.kind))
			}
			if hl.expGU.N != nE*2*moeInter || hl.expDown.N != nE*H {
				return decline(fmt.Errorf("layer %d: gemma4 stacked rows %d/%d, want %d/%d", l, hl.expGU.N, hl.expDown.N, nE*2*moeInter, nE*H))
			}
			hl.router = g4b.RouterProjScaled  // [nE*hidden] f32, routerScale·hidden^-0.5 folded in
			hl.routerBs = make([]float32, nE) // gemma4 has no router bias (moe_route reads it unconditionally)
			hl.perExpertScale = g4b.PerExpertScale
			hl.g4preFFN, hl.g4postFFN1, hl.g4preFFN2, hl.g4postFFN2, hl.g4post =
				g4b.PreFFNNorm, g4b.PostFFNNorm1, g4b.PreFFNNorm2, g4b.PostFFNNorm2, g4b.PostFFNNorm
			hl.layerScalar = g4b.LayerScalar
		}
		if hl.isMoE {
			if len(lw.Experts) != nE {
				return decline(fmt.Errorf("layer %d: %d experts, arch says %d", l, len(lw.Experts), nE))
			}
			// The router stays f32 — see cudaResident.moe. It is loaded unquantized (weights.go
			// uses loadMat, not loadMatQ) precisely so this holds; if that ever changes, decline
			// rather than quietly quantize the one matrix that must not be.
			rf, ok := lw.Router.F32()
			if !ok {
				return decline(fmt.Errorf("layer %d: router is %q, not f32 — quantizing it would flip "+
					"experts near a tie, which is a cliff and not a small error", l, lw.Router.Kind()))
			}
			if lw.Router.Rows() != nE || lw.Router.Cols() != H {
				return decline(fmt.Errorf("layer %d: router is %dx%d, want %dx%d", l, lw.Router.Rows(), lw.Router.Cols(), nE, H))
			}
			hl.router = rf
			// bias is read unconditionally by moe_route; zeros when the arch has none.
			hl.routerBs = lw.RouterBias
			if hl.routerBs == nil {
				hl.routerBs = make([]float32, nE)
			} else if len(hl.routerBs) != nE {
				return decline(fmt.Errorf("layer %d: router bias len %d != nE %d", l, len(hl.routerBs), nE))
			}
			// gate‖up INTERLEAVED per expert (g0,u0,g1,u1,...): one row range of width
			// 2*moeInter is then exactly expert e's pair, which is what the indexed GEMV +
			// glu_quant(gOff=0, uOff=moeInter) pair expects.
			gu := make([]*linalg.WeightMat, 0, 2*nE)
			dn := make([]*linalg.WeightMat, 0, nE)
			for e := range lw.Experts {
				ex := &lw.Experts[e]
				gu = append(gu, &ex.Gate, &ex.Up)
				dn = append(dn, &ex.Down)
			}
			var e error
			if hl.expGU, e = expertStack(gu...); e != nil {
				return decline(fmt.Errorf("layer %d gate/up stack: %w", l, e))
			}
			if hl.expDown, e = expertStack(dn...); e != nil {
				return decline(fmt.Errorf("layer %d down stack: %w", l, e))
			}
			// The MoE GEMVs are w4a8 only: they read f16 group scales and unpack nibbles, so an
			// int8 stack would be read as int4 and return garbage rather than error.
			if hl.expGU.kind != "int4" || hl.expDown.kind != "int4" {
				return decline(fmt.Errorf("layer %d: MoE experts are %q/%q — the resident MoE GEMVs are "+
					"int4-only (load with Quant: \"int4\")", l, hl.expGU.kind, hl.expDown.kind))
			}
			if hl.expGU.N != nE*2*moeInter || hl.expDown.N != nE*H {
				return decline(fmt.Errorf("layer %d: stacked rows %d/%d, want %d/%d — the kernel's "+
					"rowsPerExpert stride would land on the wrong expert",
					l, hl.expGU.N, hl.expDown.N, nE*2*moeInter, nE*H))
			}
			// Always-on shared expert (ungated). gate‖up concatenated the same way as a routed
			// expert, so one dense GEMV + the glu_quant offset split covers it.
			if sharedInter > 0 {
				se := &lw.SharedExpert
				if se.Gate.Rows() != sharedInter || se.Down.Cols() != sharedInter {
					return decline(fmt.Errorf("layer %d: shared expert shape gate %dx%d down %dx%d disagrees with sharedInter=%d",
						l, se.Gate.Rows(), se.Gate.Cols(), se.Down.Rows(), se.Down.Cols(), sharedInter))
				}
				var e error
				if hl.shGU, e = sharedStack(&se.Gate, &se.Up); e != nil {
					return decline(fmt.Errorf("layer %d shared gate/up: %w", l, e))
				}
				if hl.shDown, e = sharedOne(&se.Down); e != nil {
					return decline(fmt.Errorf("layer %d shared down: %w", l, e))
				}
				if hl.shGU.kind != "int4" || hl.shDown.kind != "int4" {
					return decline(fmt.Errorf("layer %d: shared expert is %q/%q — int4-only", l, hl.shGU.kind, hl.shDown.kind))
				}
				hl.hasShared = true
				if !sharedUngated {
					// Qwen-MoE's sigmoid-gated shared expert: out += sigmoid(SharedGate·h)·shared(h), shared_gate_combine's ungated=0
					// branch; it needs the [1,hidden] SharedGate weight uploaded.
					if lw.SharedGate.Rows() == 0 {
						return decline(fmt.Errorf("layer %d: arch declares a GATED shared expert "+
							"but SharedGate is empty — this loader did not populate it", l))
					}
					if hl.shGate, e = sharedOne(&lw.SharedGate); e != nil {
						return decline(fmt.Errorf("layer %d shared gate: %w", l, e))
					}
				}
			}
		}
		hl.preNorm, hl.postNorm = lw.PreAttnNorm, lw.PreMLPNorm
		if hl.mixer == decoder.MixerMamba || hl.mixer == decoder.MixerNone {
			// A single-operation block: its one pre-norm and the weights packed above, no attention geometry, QK-norm or rope table.
			if len(hl.preNorm) != H {
				return decline(fmt.Errorf("layer %d: pre-norm has %d values, want hidden %d", l, len(hl.preNorm), H))
			}
			hls[l] = hl
			continue
		}
		if !dnetOK || !dnAttnGate {
			// A gated dnetOK family's softmax layer keeps its QK-norm weights off lw (packed with its projections above). Everyone
			// else, including a dnetOK family whose softmax layer is not gated (Olmo Hybrid), reads them like an ordinary GQA/MHA
			// layer's lw.QNorm/KNorm.
			hl.qNorm, hl.kNorm = lw.QNorm, lw.KNorm
		}
		// Per-layer RoPE table (Gemma's local 10k vs global 1M base; Mellum's YaRN-on-global).
		// Uniform-rope families hand back the same slice for every layer.
		if mambaP != nil {
			// Nemotron-H's attention is NoPE: an all-zero table makes rope_kv the identity rotation (cos 0 = 1, sin 0 = 0), as WebGPU's zeroed invFreq does.
			hl.invFreq = make([]float32, m.RotaryDimAtResident(l)/2)
		} else if hl.mixer != decoder.MixerDeltaNet {
			hl.invFreq = m.RopeInvFreqLayerResident(l) // Gemma 4: real per-layer table, not the generic one
		}
		// Per-layer window: only LOCAL layers are windowed; global layers stay full causal.
		if m.LayerIsLocalResident(l) {
			hl.window = int32(m.SlidingWindowResident())
		}
		if hl.mixer == decoder.MixerDeltaNet {
			// A DeltaNet layer has no attention geometry, QK-norm or rope table, which the checks below read; its own shapes are
			// validated at upload, where the state buffers are sized from dnetParams. postOnly is a model-level flag, but a DeltaNet
			// layer reaches NormPre2 through NormPlacementLinear: it needs real preNorm/postNorm (checked here) and has no
			// PostAttnNorm/PostMLPNorm tensors, so that pair is never populated for it.
			if len(hl.preNorm) == 0 || len(hl.postNorm) == 0 {
				return decline(fmt.Errorf("layer %d missing pre/pre-MLP norm", l))
			}
			hls[l] = hl
			continue
		}
		if hl.mixer == decoder.MixerMLA {
			if len(hl.preNorm) == 0 || len(hl.postNorm) == 0 {
				return decline(fmt.Errorf("layer %d missing pre/pre-MLP norm", l))
			}
			hls[l] = hl
			continue
		}
		// Gemma sandwich / Olmo 3 postOnly: both dispatch PostAttnNorm/PostMLPNorm on the sublayer output, so they must be
		// present when the arch declares either (a missing one would silently drop the norm). Non-DeltaNet layers only (see the
		// continue above).
		if sandwich || postOnly {
			hl.postAttnNorm, hl.postMLPNorm = lw.PostAttnNorm, lw.PostMLPNorm
			if len(hl.postAttnNorm) != H || len(hl.postMLPNorm) != H {
				return decline(fmt.Errorf("layer %d: arch declares sandwich/postOnly norms but PostAttnNorm/PostMLPNorm are not len==hidden(%d) (got %d/%d)",
					l, H, len(hl.postAttnNorm), len(hl.postMLPNorm)))
			}
		}
		if hdL := m.HeadDimAtResident(l); shape.QKNorm {
			// Per-layer head_dim: Gemma 4's global layers have q_norm/k_norm of GlobalHeadDim, not the model HeadDim, so validate
			// against the layer's own width. QKNormWhole (Olmo 3/Olmo Hybrid): the norm reduces over the whole q/k vector, so the
			// weight is nH*hdL / nKV*hdL wide, and the resident launch (one Q block plus one K block, see segA) sizes both blocks
			// from a single hd, which is correct only when nH==nKV (MHA). Decline rather than mis-normalize a GQA family.
			wantQ, wantK := hdL, hdL
			nKVL := m.KVHeadsAtResident(l)
			if qkNormWhole {
				if nH != nKVL {
					return decline(fmt.Errorf("layer %d: arch claims QKNormWhole but nH=%d != nKV=%d — the resident whole-vector kernel launch requires MHA", l, nH, nKVL))
				}
				wantQ, wantK = nH*hdL, nKVL*hdL
			}
			if m.KVSrcAtResident(l) != l {
				wantK = 0 // a KV-shared layer has no k_norm either
			}
			if len(hl.qNorm) != wantQ || len(hl.kNorm) != wantK {
				return decline(fmt.Errorf("layer %d: arch claims QK-norm but QNorm/KNorm are not the expected width (want %d/%d, got %d/%d)",
					l, wantQ, wantK, len(hl.qNorm), len(hl.kNorm)))
			}
		}
		if !postOnly && len(hl.preNorm) == 0 {
			return decline(fmt.Errorf("layer %d missing pre-norm", l))
		}
		// parallelBlock (Cohere/Command-R) has no pre-MLP norm tensor at all — the MLP reuses the
		// SAME shared input norm the attention branch already computed (segBFFN's r.aq/r.aSc
		// reuse), so hl.postNorm is legitimately empty for it and must not be required.
		if hl.ffn && !postOnly && !parallelBlock && len(hl.postNorm) == 0 {
			return decline(fmt.Errorf("layer %d missing pre-MLP norm", l))
		}
		if lw.QBias != nil {
			hl.qb, hl.kb, hl.vb, hl.hasBias = lw.QBias, lw.KBias, lw.VBias, true
		}
		// Captured independently of QBias: the two travel together in Qwen2 but not in general (GPT-2 carries an o_proj bias
		// with no q/k/v bias), so folding this into the branch above would silently drop it for the families FeatOutBias exists
		// for.
		if lw.OBias != nil {
			hl.ob, hl.hasOBias = lw.OBias, true
		}
		hls[l] = hl
	}
	lmSrc := &w.LMHead
	if w.LMHead.Rows() == 0 {
		lmSrc = &w.Embed // tied embeddings
	}
	hlm, e := planProj(lmSrc)
	if e != nil {
		return decline(e)
	}
	if planCompleteForTest != nil && !plans {
		planCompleteForTest()
	}
	s.hls, s.hlm, s.loadAbort = hls, hlm, loadAbort
	return nil
}

// newResident builds the cudaResident the setup job fills: its fields from the planned geometry, the executor goroutine, the layer
// geometry and the KV context plan (r.ctxCap), declining what it cannot hold. Plan-only and the memory plan end here, returning
// planOnlySlack and errMemPlanned. A MoE build whose context is planned opens its device here (openForPlan), and every error
// return after that closes it.
func (s *buildState) newResident(planOnly bool, mp *memPlanRun) (*cudaResident, error) {
	m, shape := s.m, s.shape
	H, nLayers, nH, I, vocab, pleP := s.H, s.nLayers, s.nH, s.I, s.vocab, s.pleP
	nE, topK, moeInter, sharedInter, nGroup, topkGroup := s.nE, s.topK, s.moeInter, s.sharedInter, s.nGroup, s.topkGroup
	moeSig, moeNorm, isMoE, moeScale := s.moeSig, s.moeNorm, s.isMoE, s.moeScale
	isG4MoE, isGemma4 := s.isG4MoE, s.isGemma4
	qLoRA, kvLoRA, qkNope, qkRope, vHead, interleave := s.qLoRA, s.kvLoRA, s.qkNope, s.qkRope, s.vHead, s.interleave
	mlaAttnScale, mlaRopeScale, mlaOK := s.mlaAttnScale, s.mlaRopeScale, s.mlaOK
	dnetP, mambaP := s.dnetP, s.mambaP
	sandwich, postOnly, qkNormWhole, layerNorm, parallelBlock := s.sandwich, s.postOnly, s.qkNormWhole, s.layerNorm, s.parallelBlock
	logitScale, actG32 := s.logitScale, s.actG32
	hls, hlm, loadAbort := s.hls, s.hlm, s.loadAbort
	attnTempBeta, attnTempOrigMaxPos := m.AttnTempParams() // Ministral 3, FeatAttnTemp; 0 for every other family
	// Computed once: both ctxCap and extraBytes below need it, and a struct literal cannot read a sibling field set in the
	// same literal.
	kvSlotsReq := cudaKVSlotsRequest(m)
	// The driver's 2 MiB rounding over the weight buffers is the plan's figure (decoder.Model.PlanSlack, this package's
	// report): the plan prices it as a fixed term, so the context it picks already leaves room for it, and the build adds no
	// copy of its own. Plan-only, it is computed here, over the buffers just planned.
	if planOnly {
		if allocSlackOffForTest {
			return nil, planOnlySlack(0) // the planted defect of TestBuildScratchAccounting_plantedDefect: a plan that prices no slack
		}
		return nil, planOnlySlack(packedAllocSlack([]any{hls, hlm}))
	}
	allocSlack := m.PlanSlack("cuda")
	if allocSlackSeenForTest != nil && mp == nil {
		allocSlackSeenForTest(allocSlack, packedAllocSlack([]any{hls, hlm}))
	}
	attnCap := m.AttnLogitSoftcapResident()
	if attnSoftcapDropForTest {
		attnCap = 0
	}
	if attnCap != 0 && mlaOK {
		return nil, decline(fmt.Errorf("an attention softcap on a latent-attention model: mla_attn has no cap"))
	}
	r := &cudaResident{
		prog:      m.Program(),
		loadAbort: loadAbort,
		knob:      m.Knob,
		kvF16:     m.KVCacheF16(),
		kvI8:      m.KVDeliveredFor("cuda") == mem.KVI8,
		hidden:    H, nLayers: nLayers, nH: nH, inter: I, vocab: vocab,
		eps: m.NormEps(), attnScale: m.AttnScale(), finalSoftcap: m.FinalLogitSoftcapResident(), attnCap: attnCap, faSoftcap: m.FlashDecodeSoftcap(),
		attnTempBeta: attnTempBeta, attnTempOrigMaxPos: attnTempOrigMaxPos,
		qkNorm: shape.QKNorm, qkNormWhole: qkNormWhole, rmsAddOne: m.RMSAddOne(),
		act: int32(m.GatedActResident()), sandwich: sandwich, postOnly: postOnly,
		layerNorm: layerNorm, parallelBlock: parallelBlock, logitScale: logitScale, actG32: actG32,
		moe: isMoE, nE: nE, topK: topK, moeInter: moeInter,
		moeScale: float32(moeScale), nGroup: nGroup, topkGroup: topkGroup,
		sharedInter: sharedInter,
		gemma4Moe:   isG4MoE, gemma4Dense: isGemma4, g4cap: os.Getenv("GOINFER_G4_CAPTURE") != "",
		// C′: DMA the routed int4 experts host→VRAM slots per token (device read, correct). The
		// path to running a model whose experts exceed VRAM. Off by default; byte-identical when off.
		cacheExperts: m.MoECacheExperts(),
		// MoEExpertsUnpinned: the cache's slots upload from pageable pages of a packed-experts file beside the model, not a pinned copy.
		unpinnedExperts: m.MoEExpertsUnpinned(), unpinnedDir: modelDir(m.SourcePath()),
		cacheProf: os.Getenv("GOINFER_MOE_CACHE_PROF") != "",
		dnet:      dnetP,
		mamba:     mambaP,
		isMLA:     mlaOK, mlaRank: kvLoRA, mlaLatDim: kvLoRA + qkRope,
		mlaQKHead: qkNope + qkRope, mlaQKNope: qkNope, mlaQKRope: qkRope,
		mlaVHead: vHead, mlaQLoRA: qLoRA, mlaInterleave: interleave,
		mlaRopeScale:    float32(mlaRopeScale),
		allocSlackBytes: allocSlack,
		ctxExplicit:     ctxExplicitFor(m),
		// One conversation may use the whole KV pool (KV4a-3) unless a context was asked for or pinned by the fit guard, or a companion's
		// device K/V scales with the context (a --drafter's, priced at the planned one).
		kvRaise:    m.ResidentContextRequest() <= 0 && m.ExtraResidentKVPerPosition() == 0,
		modelCtx:   m.Config().MaxPositions,
		kvSlotsReq: kvSlotsReq,
	}
	if moeSig {
		r.moeSigmoid = 1
	}
	if moeNorm {
		r.moeNormTopK = 1
	}
	if mlaOK && mlaAttnScale != 0 {
		r.attnScale = float32(mlaAttnScale)
	}
	// C' device expert slots per layer: an LRU cache of nSlots experts (clamped [topK, nE]), costing
	// nLayers*nSlots*perExpert of VRAM; more slots trade VRAM for fewer per-token DMAs. The request comes from
	// --moe-cache-slots (GOINFER_MOE_CACHE_SLOTS is still honoured).
	//
	// With caching on and no explicit request the default is 8*topK (capped at nE): a bounded multiple, neither topK nor the
	// "ask for all, let allocSlots cap to free VRAM" that MoECacheSlotsRequest documents for 0. The floor is topK (one
	// token's routed set must be resident at once) and allocSlots still caps to measured free VRAM. Do not default to "ask
	// for all": allocSlots's headroom (marginBytes) covers per-token costs, not what the forward allocates after it, so on
	// the real 26B it capped 128 slots to 34 and the warm forward then died with CUDA_ERROR_OUT_OF_MEMORY. 8*topK sits just
	// above the knee of one measured sweep; it is a heuristic, not a derived constant, because where the knee falls depends
	// on the model's routing entropy. The sweep and the history:
	// docs/code-notes/cuda.md#BuildResident: MoE cache slots default.
	r.cacheSlots = topK
	r.cacheSlotsReq = topK
	if r.cacheExperts && nE > 0 {
		if d := 8 * topK; d > topK {
			r.cacheSlots = min(d, nE)
			r.cacheSlotsReq = r.cacheSlots
		}
	}
	if r.cacheExperts {
		// The request comes from Options (--moe-cache-slots); unset keeps the 8*topK default above, not the accessor's "ask for
		// all". A request BELOW topK cannot be honoured, since one token's own top-k must fit, so it is refused rather than
		// floored: flooring left the 8*topK default in place, and a gate asking for fewer slots than experts got the identity
		// mapping and discriminated by routing luck.
		if req := m.MoECacheSlotsRequest(); req > 0 && req < topK {
			// A hard error, not declined(): declined() swallows the error and falls back to the staged path, which suits a shape
			// this backend does not implement but not an operator flag that cannot mean what it says.
			return nil, fmt.Errorf("cuda: --moe-cache-slots %d is below top-k %d — one "+
				"token's own routed experts must all be resident at once, so this cannot be "+
				"honoured (G-07)", req, topK)
		} else if req >= topK {
			r.cacheSlots = req
			r.cacheSlotsReq = req
			if nE > 0 && r.cacheSlots > nE {
				r.cacheSlots = nE
			}
		}
	}
	if mp == nil { // the memory plan runs no job: it opens no device
		r.reqCh = make(chan func() error)
		r.ackCh = make(chan error)
		go func() {
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			for j := range r.reqCh {
				r.ackCh <- runJob(j)
			}
		}()
	}

	// (runJob — the executor's panic boundary — is defined below.)

	// The context margin, on the planned geometry, before the context is planned (ctxMarginFor): the setup job replaces r.layers,
	// and the fused-attention flags are set again where the modules load.
	r.layers = planLayerGeometry(m, hls, nH, r)
	if e := r.readRecurrentFloats(m); e != nil {
		return nil, decline(e)
	}
	r.pleP, r.eModel = pleP, pleP > 0
	for l := range r.layers {
		r.eModel = r.eModel || r.layers[l].kvShared
	}
	r.fastAttn, r.fastGemm = fastPrefillEnabled(r.knobValue("GOINFER_CUDA_FAST_PREFILL"))
	if m.ExactPrefill() {
		r.fastAttn, r.fastGemm = false, false
	}
	r.planning = true
	// A MoE build whose context is planned opens its device and pays moe_route's local-memory pool (payRoutePool) before the plan
	// reads free VRAM, on the context it keeps: the startup self-test's pool is released with its own context before the build, so
	// nothing else has paid it (docs/tasks/parked/task-kernel-survey-2026-10.md, "KS-H6 build, review amendment"). It checks the weights
	// first, against the margin at the floor, so a model that does not fit gets checkWeightsFit's decline rather than a warm-up's
	// failure. The setup job keeps this device and stream, and checks the weights again against the margin it plans.
	if _, floor, plans := ctxPlanRange(m, m.ResidentContextRequest(), m.Config().MaxPositions); r.moe && plans {
		r.ctxMarginBytes = r.ctxMarginFor(floor)
		if mp != nil {
			mp.out.Margin, mp.out.ExtraBytes, mp.out.KVSlotsReq = r.ctxMargin(), m.ExtraResidentBytes(), r.kvSlotsReq
			if e := mp.weightsCheck(m, "weightsOpen", r.ctxMargin()); e != nil {
				return nil, e
			}
			mp.out.PoolPaid = true // openForPlan pays the pool only after this check passes
			if mp.derive {
				mp.in.Plan, mp.in.Weights = mp.in.WeightsOpen-mp.pool, mp.in.WeightsOpen-mp.pool
			}
		} else if e := r.do(func() error { return r.openForPlan(m) }); e != nil {
			r.Close()
			return nil, decline(e)
		}
	}
	owners, mlaPools := kvLayerCounts(r.layers)
	// Resolve the resident KV capacity HERE, at construction, not at the KV allocation site: several buffers are sized from it
	// earlier (the split-KV score scratch among them), and a zero-value ctxCap makes those 0-byte allocations that fail the whole
	// resident build. cap = min(model context window, request); request 0 ⇒ the 4096 default, so a caller who did not ask
	// allocates exactly what they always did.
	// The plan reads free VRAM: a MoE build has paid its local-memory pool above, on the context it keeps.
	readFree := liveCtxFree
	if mp != nil {
		readFree = func() (int64, bool) {
			if mp.derive {
				mp.in.PlanOK = true // called only when the context is planned
			}
			return mp.in.Plan, mp.in.PlanOK
		}
	}
	if recurrentStatePlannedForTest != nil && mp == nil {
		recurrentStatePlannedForTest(m.Plan("cuda", 1<<40, fit.PlanRequest{Ctx: 1}).StateBytes) // the state does not grow with the context
	}
	r.ctxCap, r.ctxMarginBytes = resolveCtxCapFitFree(m, m.ResidentContextRequest(), m.Config().MaxPositions, kvSlotsReq,
		func(ctx int) int64 { r.ctxMarginCtx = ctx; return r.ctxMarginFor(ctx) },
		kvRoundingBound(owners, mlaPools), readFree)
	planned := r.layers
	r.planning, r.layers = false, nil
	r.ctxPlanned = r.ctxCap
	// A drafter's device K/V is priced at the FINAL ctxCap. resolveCtxCapFit's ExtraBytes consult priced it at candidate, the
	// widest ctx that call considered, and the real allocation can only be <= candidate, so this is never a smaller price than the
	// one planned against; checkKVFits enforces it.
	r.extraBytes = m.ExtraResidentBytes() + m.ExtraResidentKVPerPosition()*int64(r.ctxCap)
	if mp != nil {
		return nil, r.finishMemPlan(m, hls, planned, mp)
	}
	return r, nil
}
