//go:build cuda

package cuda

import (
	"fmt"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// projEnt is one projection planLayerProjs plans: the host weight it fills and its source.
type projEnt = struct {
	dst *hostW
	src *linalg.WeightMat
}

// planLayer plans layer l's weights into hls[l]: it classifies the layer's mixer and steps, then plans and checks its
// projections, experts and norms, declining what CUDA cannot hold.
func (s *buildState) planLayer(l int, hls []hlayer, loadAbort <-chan struct{}, expertStack, sharedStack func(ws ...*linalg.WeightMat) (hostW, error), sharedOne func(w *linalg.WeightMat) (hostW, error)) error {
	m, w := s.m, s.w
	bv := m.BackendView()
	nLayers := s.nLayers
	isMoE := s.isMoE
	mambaP := s.mambaP
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
	g4b, isG4 := bv.Layer(l).Gemma4MoEBundle()
	hl.g4moe = isG4
	if e := s.planLayerProjs(l, &hl, lw, g4b); e != nil {
		return e
	}
	if e := s.planLayerExperts(l, &hl, lw, g4b, expertStack, sharedStack, sharedOne); e != nil {
		return e
	}
	if e := s.planLayerNorms(l, &hl, lw); e != nil {
		return e
	}
	hls[l] = hl
	return nil
}

// planLayerProjs lists layer l's projections for its mixer kind, with the dense feed-forward and per-layer-embedding ones, and
// plans each into hl, filling the per-family fields that come with them and declining a shape or a quant CUDA cannot run.
func (s *buildState) planLayerProjs(l int, hl *hlayer, lw *decoder.LayerWeights, g4b decoder.Gemma4MoEResidentBundle) error {
	m, planProj := s.m, s.planProj
	bv := m.BackendView()
	H, nH, pleP := s.H, s.nH, s.pleP
	qLoRA, kvLoRA, qkNope, qkRope, vHead := s.qLoRA, s.kvLoRA, s.qkNope, s.qkRope, s.vHead
	dnAttnGate, dnetOK, mambaP := s.dnAttnGate, s.dnetOK, s.mambaP
	actG32 := s.actG32
	var proj []struct {
		dst *hostW
		src *linalg.WeightMat
	}
	switch {
	case hl.mixer == decoder.MixerDeltaNet:
		// Gated-DeltaNet mixer layer: no attention projections at all. The three dominant
		// projections carry the model's quant; inB/inA are f32 on the CPU by deliberate
		// choice (they feed the write/decay gates) and packWeight quantizes them to int8
		// here — the one place this port is knowingly coarser than the reference.
		qkv, z, outP, inB, inA, cW, dtB, negA, nW := bv.Layer(l).Qwen35DeltaWeights()
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
		qP, kP, vP, oP, qN, kN := bv.Layer(l).Qwen35AttnWeights()
		hl.qNorm, hl.kNorm = qN, kN
		// Same reason as the DeltaNet check below: these become device uploads, and an empty
		// one is an "invalid length" with no indication of which.
		if len(qN) == 0 || len(kN) == 0 {
			return decline(fmt.Errorf("layer %d: qwen35 softmax layer has empty q_norm/k_norm "+
				"(%d/%d) — the loader did not populate them for this container", l, len(qN), len(kN)))
		}
		proj = []projEnt{{&hl.q, qP}, {&hl.k, kP}, {&hl.v, vP}, {&hl.o, oP}}
	case hl.mixer == decoder.MixerMLA:
		qA, qANorm, qB, qProj, kvA, kvANorm, kvB, oProj := bv.Layer(l).MLAWeights()
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
		if bv.Layer(l).KVSrc() != l {
			// A KV-shared layer (Gemma 4 E-model) carries only Q and O: its K/V are its source layer's.
			proj = []projEnt{{&hl.q, &lw.QProj}, {&hl.o, &lw.OProj}}
		} else {
			proj = []projEnt{{&hl.q, &lw.QProj}, {&hl.k, &lw.KProj}, {&hl.o, &lw.OProj}}
			if !bv.Layer(l).VFromK() { // K=V (attention_k_eq_v) global layers carry NO v_proj — V=v_norm(k)
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
	return nil
}

// planLayerExperts plans a routed layer's expert stacks, router and shared expert, or Gemma 4's parallel MoE block, checking each
// against the arch's geometry.
func (s *buildState) planLayerExperts(l int, hl *hlayer, lw *decoder.LayerWeights, g4b decoder.Gemma4MoEResidentBundle, expertStack, sharedStack func(ws ...*linalg.WeightMat) (hostW, error), sharedOne func(w *linalg.WeightMat) (hostW, error)) error {
	H := s.H
	nE, moeInter, sharedInter := s.nE, s.moeInter, s.sharedInter
	sharedUngated := s.sharedUngated
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
	return nil
}

// planLayerNorms takes layer l's norms, rope table and window, and checks the norms and geometry its mixer needs.
func (s *buildState) planLayerNorms(l int, hl *hlayer, lw *decoder.LayerWeights) error {
	m, shape := s.m, s.shape
	bv := m.BackendView()
	H, nH := s.H, s.nH
	dnAttnGate, dnetOK, mambaP := s.dnAttnGate, s.dnetOK, s.mambaP
	sandwich, postOnly, qkNormWhole, parallelBlock := s.sandwich, s.postOnly, s.qkNormWhole, s.parallelBlock
	hl.preNorm, hl.postNorm = lw.PreAttnNorm, lw.PreMLPNorm
	if hl.mixer == decoder.MixerMamba || hl.mixer == decoder.MixerNone {
		// A single-operation block: its one pre-norm and the weights packed above, no attention geometry, QK-norm or rope table.
		if len(hl.preNorm) != H {
			return decline(fmt.Errorf("layer %d: pre-norm has %d values, want hidden %d", l, len(hl.preNorm), H))
		}
		return nil
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
		hl.invFreq = make([]float32, bv.Layer(l).RotaryDim()/2)
	} else if hl.mixer != decoder.MixerDeltaNet {
		hl.invFreq = bv.Layer(l).RopeInvFreq() // Gemma 4: real per-layer table, not the generic one
	}
	// Per-layer window: only LOCAL layers are windowed; global layers stay full causal.
	if bv.Layer(l).IsLocal() {
		hl.window = int32(bv.SlidingWindow())
	}
	if hl.mixer == decoder.MixerDeltaNet {
		// A DeltaNet layer has no attention geometry, QK-norm or rope table, which the checks below read; its own shapes are
		// validated at upload, where the state buffers are sized from dnetParams. postOnly is a model-level flag, but a DeltaNet
		// layer reaches NormPre2 through NormPlacementLinear: it needs real preNorm/postNorm (checked here) and has no
		// PostAttnNorm/PostMLPNorm tensors, so that pair is never populated for it.
		if len(hl.preNorm) == 0 || len(hl.postNorm) == 0 {
			return decline(fmt.Errorf("layer %d missing pre/pre-MLP norm", l))
		}
		return nil
	}
	if hl.mixer == decoder.MixerMLA {
		if len(hl.preNorm) == 0 || len(hl.postNorm) == 0 {
			return decline(fmt.Errorf("layer %d missing pre/pre-MLP norm", l))
		}
		return nil
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
	if hdL := bv.Layer(l).HeadDim(); shape.QKNorm {
		// Per-layer head_dim: Gemma 4's global layers have q_norm/k_norm of GlobalHeadDim, not the model HeadDim, so validate
		// against the layer's own width. QKNormWhole (Olmo 3/Olmo Hybrid): the norm reduces over the whole q/k vector, so the
		// weight is nH*hdL / nKV*hdL wide, and the resident launch (one Q block plus one K block, see segA) sizes both blocks
		// from a single hd, which is correct only when nH==nKV (MHA). Decline rather than mis-normalize a GQA family.
		wantQ, wantK := hdL, hdL
		nKVL := bv.Layer(l).KVHeads()
		if qkNormWhole {
			if nH != nKVL {
				return decline(fmt.Errorf("layer %d: arch claims QKNormWhole but nH=%d != nKV=%d — the resident whole-vector kernel launch requires MHA", l, nH, nKVL))
			}
			wantQ, wantK = nH*hdL, nKVL*hdL
		}
		if bv.Layer(l).KVSrc() != l {
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
	return nil
}
