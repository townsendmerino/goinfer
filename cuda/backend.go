//go:build cuda

package cuda

import (
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"strconv"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

func init() {
	decoder.RegisterBackend("cuda", func() (decoder.Backend, error) {
		return &cudaBackend{}, nil
	})
	// A live query, unlike Metal's registered probe: CUDA VRAM is a separate pool from host RAM, so the driver's free figure
	// is reliable. It creates a throwaway device with no kernels loaded and releases it right after the query (the bare
	// create-query-release shape of cuda/alloc_floor_test.go), not the persistent executor cudaResident needs.
	decoder.RegisterMemoryProbe("cuda", func() (int64, bool) {
		dev, err := CreateSystemDefaultDevice()
		if err != nil {
			return 0, false
		}
		defer dev.ReleaseObjects()
		free, _, err := dev.Context().MemInfo()
		if err != nil {
			return 0, false
		}
		return int64(free), true
	})
}

// Compile-time seam checks: cudaBackend must satisfy the decoder's backend +
// residency interfaces (catches signature drift against decoder/residency.go).
var (
	_ decoder.Backend          = (*cudaBackend)(nil)
	_ decoder.ResidencyBackend = (*cudaBackend)(nil)
	_ decoder.ResidentForward  = (*cudaResident)(nil)
	_ decoder.ResidentAdapter  = (*cudaResident)(nil)
)

// cudaBackend implements decoder.Backend + decoder.ResidencyBackend.
type cudaBackend struct {
	resident *cudaResident // set by BuildResident; shut down in Close
}

func (b *cudaBackend) Name() string { return "cuda" }

// MatmulBT is the CPU fallback (the shared SIMD linalg kernels): CUDA has no partial GPU path, so a model the resident
// runner does not admit, or a box with no driver, runs entirely on the CPU, and DecodePath says so.
func (b *cudaBackend) MatmulBT(a, bmat, dst []float32, M, K, N int) {
	linalg.MatmulBT(a, bmat, dst, M, K, N)
}

// layerFusable reports whether one layer permits the fused super-kernels. fQKV always reads Q/K/V, so those must be
// int4. fGU also reads gate/up/down as int4+f16-scales but runs only on dense-FFN layers: MoE layers take moeMLP and
// leave g/u/d unpacked, so they are exempt (requiring g/u/d there would strip fQKV from every MoE model), while skipping
// the check on a dense int8 gate/up layer would pass fGU a nil ws16 and crash the executor.
func layerFusable(qkvInt4, moe, guInt4 bool) bool {
	return qkvInt4 && (moe || guInt4)
}

// BuildResident builds a resident CUDA decoder from a loaded Model: host-packs the int4/int8/f32 projections, spawns a
// LockOSThread-pinned CUDA executor, and uploads weights and KV scratch once, returning a *cudaResident. It declines
// (ok=false, no crash) when the driver is absent or a projection shape is not residency-compatible; the decoder then
// uses the staged/CPU path.
func (b *cudaBackend) BuildResident(m *decoder.Model) (rf decoder.ResidentForward, ok bool, err error) {
	// Never crash the process on a missing/broken driver: recover → decline → fallback.
	defer func() {
		if p := recover(); p != nil {
			rf, ok, err = nil, false, fmt.Errorf("cuda: BuildResident recovered from panic: %v", p)
		}
	}()

	// The decline reason is printed unconditionally, not behind a debug flag: declining moves the whole forward to CPU, and
	// the runtime must name why it is off the fast path. It travels as a typed decline (decoder.DeclineResident): decoder's
	// load path prints it once and records it as the model's ResidentDecline, which DecodePath and `serve check` report.
	declined := func(e error) (decoder.ResidentForward, bool, error) {
		// The reason names the cause and what to try; a recovered panic's stack goes to stderr beside it.
		reason, detail := declineAdvice(e.Error(), m.MoECacheExperts())
		if detail != "" {
			fmt.Fprintf(os.Stderr, "[cuda] resident build failure, detail:\n%s\n", detail)
		}
		return nil, false, decoder.DeclineResident("cuda: %s", reason)
	}

	w := m.Weights()
	if w == nil || len(w.Layers) == 0 {
		return declined(fmt.Errorf("the model has no loaded layer weights"))
	}

	// Admission: DecodeRunnerEligible was scoped to the richer WebGPU runner, so it is too permissive for this backend. A
	// feature this backend does not implement must decline, or it is silently dropped and the logits are wrong (Qwen3's
	// QK-norm ignored; a partial-rotary model reading invFreq out of bounds because the rope kernel hardcodes half = hd/2).
	// The taxonomy is decoder/features.go, shared by all backends.
	if missing := m.MissingResidentFeatures(decoder.ResidentBackendFeatures("cuda")); len(missing) > 0 {
		return declined(fmt.Errorf("arch needs unimplemented feature(s) %v", missing))
	}

	H, nLayers, nH, _, _, I, vocab := m.Dims() // nKV, hd are per-layer now (KVHeadsAtResident/HeadDimAtResident); model-level unused
	// Gemma 4 E-model PLE width P (0 for every other model). The resident embedding row is then [hidden ‖ nLayers*P] (ResidentEmbedLen):
	// the host computes the per-layer inputs, and each layer's PLE branch reads its slice of the tail (S1 on CUDA).
	pleP := m.Gemma4PLEDimResident()
	if want := H + nLayers*pleP; pleP > 0 && m.ResidentEmbedLen() != want {
		return declined(fmt.Errorf("gemma4 E-model: the decoder's resident embedding row is %d floats, but hidden+layers*P is %d", m.ResidentEmbedLen(), want))
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
			return declined(fmt.Errorf("MoE nE=%d exceeds moe_route's MOE_MAX_E=%d", nE, capE))
		case nGroup > capG:
			return declined(fmt.Errorf("MoE nGroup=%d exceeds moe_route's MOE_MAX_G=%d", nGroup, capG))
		case m.GatedActResident() != 1: // decoder.ActSiLU — decoder/mlp.go errors on any other
			return declined(fmt.Errorf("MoE experts are SwiGLU-only, arch act=%d", m.GatedActResident()))
		case m.SandwichNormResident():
			return declined(fmt.Errorf("MoE + sandwich norms: the combine accumulates straight into " +
				"the residual, so there is no seam to normalize the block output at"))
		case moeInter%32 != 0 || H%32 != 0:
			return declined(fmt.Errorf("MoE int4 needs moeInter(%d) and hidden(%d) both multiples of 32", moeInter, H))
		case sharedInter > 0 && sharedInter%32 != 0:
			return declined(fmt.Errorf("MoE int4 shared expert needs sharedInter(%d) a multiple of 32", sharedInter))
		}
	}
	if isG4MoE {
		// gemma4MoeMLP's indexed-expert + dense GEMVs are int4/W4A8: the widths the kernels stride by
		// must be multiples of the group size (hidden) / 8-nibble word (moeInter, denseInter).
		if moeInter%32 != 0 || H%32 != 0 {
			return declined(fmt.Errorf("gemma4 MoE int4 needs moeInter(%d) and hidden(%d) both multiples of 32", moeInter, H))
		}
		if capE, _, _ := decoder.ResidentBackendMoECap("cuda"); nE > capE {
			return declined(fmt.Errorf("gemma4 MoE nE=%d exceeds moe_route's MOE_MAX_E=%d", nE, capE))
		}
	}

	// ---- host pack all weights (CPU; any incompatible shape → decline) ----
	type hlayer struct {
		q, k, v, o, g, u, d       hostW
		pleGate, pleProj          hostW     // Gemma 4 E-model per-layer embedding branch (S1 on CUDA); zero unless pleP > 0
		postPLENorm               []float32 // the PLE branch's post-norm weight [hidden]
		qb, kb, vb                []float32
		ob                        []float32 // attention output-projection bias (GPT-2 / gpt-oss)
		preNorm, postNorm         []float32
		qNorm, kNorm              []float32
		postAttnNorm, postMLPNorm []float32 // Gemma sandwich (nil unless the arch declares it)
		invFreq                   []float32 // per-layer RoPE table
		window                    int32
		hasBias                   bool
		hasOBias                  bool

		// MoE, per layer: GLM/DeepSeek run dense for the first FirstKDense layers and route
		// after, so this is keyed off the layer's own Experts (as decoder/mlp.go does), not
		// off the arch.
		isMoE            bool
		expGU, expDown   hostW
		router, routerBs []float32
		hasShared        bool
		shGU, shDown     hostW
		shGate           hostW // [1, hidden] sigmoid gate (Qwen-MoE); empty ⇒ ungated (GLM/DeepSeek)

		// Gemma-4 enable_moe_block (parallel dense‖MoE). g4moe routes the upload to gemma4MoeMLP's
		// fields: the dense branch reuses g/u/d (packed from mlpGate/up/down), the router reuses
		// router (RouterProjScaled, f32) + routerBs (zeros), the experts reuse expGU/expDown.
		g4moe                                               bool
		g4preFFN, g4postFFN1, g4preFFN2, g4postFFN2, g4post []float32
		perExpertScale                                      []float32
		layerScalar                                         float32

		// Gated-DeltaNet (qwen3_5_moe / qwen3_next / qwen3_5). isDeltaNet layers carry NO
		// q/k/v/o and no KV cache — the recurrence's fixed-size state is the whole history.
		// qGate marks the family's SOFTMAX layers, whose q_proj is double width.
		isDeltaNet                            bool
		qGate                                 bool
		dnQKV, dnZ, dnOut, dnB, dnA           hostW
		dnConvW, dnDtBias, dnNegExpA, dnNormW []float32

		// MLA (DeepSeek / Kimi, FeatMLA)
		isMLA                      bool
		mlaQA, mlaQB, mlaQ, mlaKVA hostW
		mlaQANorm, mlaKVANorm      []float32
		mlaWUK, mlaWUV             []float32
	}
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

	sandwich := m.SandwichNormResident()
	// Olmo 3 / Olmo Hybrid: post-only norms.
	postOnly := m.PostOnlyNormResident()
	qkNormWhole := m.QKNormWholeResident()
	// Cohere/Command-R and Cohere2/Command-R7B: LayerNorm and parallel blocks.
	layerNorm := m.LayerNormResident()
	parallelBlock := m.ParallelBlockResident()
	logitScale, _ := m.LogitScaleResident() // ok=false ⇒ 1, already the applyLogitScale no-op value
	// Per-32 ACTIVATION quantization (decoder.Options.ActQuantGroup, actgroup.cu): implemented for the
	// plain dense RMSNorm decode path. Every path whose kernels read ONE activation scale per vector
	// declines by name rather than reading a per-group buffer as a per-vector one.
	actG32 := m.ActQuantGroup() == 32
	if actG32 {
		switch {
		case isMoE:
			return declined(fmt.Errorf("per-32 activations: the MoE expert/router kernels read per-vector activation scales"))
		case mlaOK:
			return declined(fmt.Errorf("per-32 activations: the MLA path is not implemented"))
		case isGemma4:
			return declined(fmt.Errorf("per-32 activations: the Gemma 4 path is not implemented"))
		case dnetOK:
			return declined(fmt.Errorf("per-32 activations: the Gated DeltaNet path is not implemented"))
		case layerNorm, postOnly, parallelBlock:
			return declined(fmt.Errorf("per-32 activations: LayerNorm / post-norm-only / parallel-block families are not implemented"))
		case H%32 != 0 || I%32 != 0:
			return declined(fmt.Errorf("per-32 activations need hidden (%d) and FFN width (%d) to be multiples of 32", H, I))
		}
	}
	hls := make([]hlayer, nLayers)
	for l := range nLayers {
		lw := &w.Layers[l]
		var hl hlayer
		hl.isMoE = isMoE && lw.Experts != nil // same key as decoder/mlp.go; false on dense prefix layers
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
		case dnetOK && m.Qwen35LinearLayer(l):
			// Gated-DeltaNet mixer layer: no attention projections at all. The three dominant
			// projections carry the model's quant; inB/inA are f32 on the CPU by deliberate
			// choice (they feed the write/decay gates) and packWeight quantizes them to int8
			// here — the one place this port is knowingly coarser than the reference.
			hl.isDeltaNet = true
			qkv, z, outP, inB, inA, cW, dtB, negA, nW := m.Qwen35DeltaWeights(l)
			hl.dnConvW, hl.dnDtBias, hl.dnNegExpA, hl.dnNormW = cW, dtB, negA, nW
			// Name the missing tensor. Every one of these becomes a device up32, and an empty
			// slice there is a 0-byte allocation — which this driver reports as "invalid length"
			// with no hint at WHICH of the four (or which loader) produced it. Two different
			// checkpoints have already failed exactly that way during this bring-up.
			for _, chk := range []struct {
				name string
				n    int
			}{{"conv_w", len(cW)}, {"dt_bias", len(dtB)}, {"neg_exp_a", len(negA)}, {"norm_w", len(nW)}} {
				if chk.n == 0 {
					return declined(fmt.Errorf("layer %d: DeltaNet %s is empty — the loader did not "+
						"populate it for this container", l, chk.name))
				}
			}
			bWM := linalg.WrapF32(inB, len(dtB), H)
			aWM := linalg.WrapF32(inA, len(dtB), H)
			proj = []projEnt{{&hl.dnQKV, qkv}, {&hl.dnZ, z}, {&hl.dnOut, outP},
				{&hl.dnB, &bWM}, {&hl.dnA, &aWM}}
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
				return declined(fmt.Errorf("layer %d: qwen35 softmax layer has empty q_norm/k_norm "+
					"(%d/%d) — the loader did not populate them for this container", l, len(qN), len(kN)))
			}
			proj = []projEnt{{&hl.q, qP}, {&hl.k, kP}, {&hl.v, vP}, {&hl.o, oP}}
		case mlaOK:
			hl.isMLA = true
			qA, qANorm, qB, qProj, kvA, kvANorm, kvB, oProj := m.MLALayerWeights(l)
			hl.mlaQANorm, hl.mlaKVANorm = qANorm, kvANorm
			latDim := kvLoRA + qkRope
			if qLoRA > 0 {
				qaWM := linalg.WrapF32(qA, qLoRA, H)
				qbWM := linalg.WrapF32(qB, nH*(qkNope+qkRope), qLoRA)
				proj = append(proj,
					projEnt{&hl.mlaQA, &qaWM},
					projEnt{&hl.mlaQB, &qbWM},
				)
			} else {
				qWM := linalg.WrapF32(qProj, nH*(qkNope+qkRope), H)
				proj = append(proj, projEnt{&hl.mlaQ, &qWM})
			}
			kvaWM := linalg.WrapF32(kvA, latDim, H)
			oWM := linalg.WrapF32(oProj, H, nH*vHead)
			proj = append(proj,
				projEnt{&hl.mlaKVA, &kvaWM},
				projEnt{&hl.o, &oWM},
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
		} else if !hl.isMoE {
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
				return declined(fmt.Errorf("layer %d: PLE weights are not the expected shape (gate rows %d want %d, proj rows %d want %d, post-norm %d want %d)",
					l, lw.PLEGate.Rows(), pleP, lw.PLEProj.Rows(), H, len(lw.PostPLENorm), H))
			}
			proj = append(proj, projEnt{&hl.pleGate, &lw.PLEGate}, projEnt{&hl.pleProj, &lw.PLEProj})
			hl.postPLENorm = lw.PostPLENorm
		}
		for _, p := range proj {
			hw, e := packWeight(p.src)
			if e != nil {
				return declined(e)
			}
			if hw.kind == "q4k" && !actG32 {
				// gemv_q4k_g32's minimum term reads the per-group activation sums only the per-32
				// quantizers write; --quant q4k always stamps per-32, so this is defence in depth.
				return declined(fmt.Errorf("layer %d: a Q4_K weight needs per-32 activations (ActQuantGroup 32)", l))
			}
			*p.dst = hw
		}
		if hl.g4moe {
			// Gemma-4 MoE: stack the experts (each ExpertsGateUp is the fused [2*moeInter,hidden]
			// gate‖up), the f32 router (RouterProjScaled, scale folded), the 5 norms + per-expert
			// scale + layerScalar. The dense g/u/d were packed above via the proj list.
			var e error
			if hl.expGU, e = packWeightStack(g4b.ExpertsGateUp...); e != nil {
				return declined(fmt.Errorf("layer %d gemma4 expert gate‖up stack: %w", l, e))
			}
			if hl.expDown, e = packWeightStack(g4b.ExpertsDown...); e != nil {
				return declined(fmt.Errorf("layer %d gemma4 expert down stack: %w", l, e))
			}
			if hl.expGU.kind != "int4" || hl.expDown.kind != "int4" {
				return declined(fmt.Errorf("layer %d: gemma4 experts are %q/%q — the resident MoE GEMVs are int4-only (load with Quant: \"int4\")", l, hl.expGU.kind, hl.expDown.kind))
			}
			if hl.expGU.N != nE*2*moeInter || hl.expDown.N != nE*H {
				return declined(fmt.Errorf("layer %d: gemma4 stacked rows %d/%d, want %d/%d", l, hl.expGU.N, hl.expDown.N, nE*2*moeInter, nE*H))
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
				return declined(fmt.Errorf("layer %d: %d experts, arch says %d", l, len(lw.Experts), nE))
			}
			// The router stays f32 — see cudaResident.moe. It is loaded unquantized (weights.go
			// uses loadMat, not loadMatQ) precisely so this holds; if that ever changes, decline
			// rather than quietly quantize the one matrix that must not be.
			rf, ok := lw.Router.F32()
			if !ok {
				return declined(fmt.Errorf("layer %d: router is %q, not f32 — quantizing it would flip "+
					"experts near a tie, which is a cliff and not a small error", l, lw.Router.Kind()))
			}
			if lw.Router.Rows() != nE || lw.Router.Cols() != H {
				return declined(fmt.Errorf("layer %d: router is %dx%d, want %dx%d", l, lw.Router.Rows(), lw.Router.Cols(), nE, H))
			}
			hl.router = rf
			// bias is read unconditionally by moe_route; zeros when the arch has none.
			hl.routerBs = lw.RouterBias
			if hl.routerBs == nil {
				hl.routerBs = make([]float32, nE)
			} else if len(hl.routerBs) != nE {
				return declined(fmt.Errorf("layer %d: router bias len %d != nE %d", l, len(hl.routerBs), nE))
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
			if hl.expGU, e = packWeightStack(gu...); e != nil {
				return declined(fmt.Errorf("layer %d gate/up stack: %w", l, e))
			}
			if hl.expDown, e = packWeightStack(dn...); e != nil {
				return declined(fmt.Errorf("layer %d down stack: %w", l, e))
			}
			// The MoE GEMVs are w4a8 only: they read f16 group scales and unpack nibbles, so an
			// int8 stack would be read as int4 and return garbage rather than error.
			if hl.expGU.kind != "int4" || hl.expDown.kind != "int4" {
				return declined(fmt.Errorf("layer %d: MoE experts are %q/%q — the resident MoE GEMVs are "+
					"int4-only (load with Quant: \"int4\")", l, hl.expGU.kind, hl.expDown.kind))
			}
			if hl.expGU.N != nE*2*moeInter || hl.expDown.N != nE*H {
				return declined(fmt.Errorf("layer %d: stacked rows %d/%d, want %d/%d — the kernel's "+
					"rowsPerExpert stride would land on the wrong expert",
					l, hl.expGU.N, hl.expDown.N, nE*2*moeInter, nE*H))
			}
			// Always-on shared expert (ungated). gate‖up concatenated the same way as a routed
			// expert, so one dense GEMV + the glu_quant offset split covers it.
			if sharedInter > 0 {
				se := &lw.SharedExpert
				if se.Gate.Rows() != sharedInter || se.Down.Cols() != sharedInter {
					return declined(fmt.Errorf("layer %d: shared expert shape gate %dx%d down %dx%d disagrees with sharedInter=%d",
						l, se.Gate.Rows(), se.Gate.Cols(), se.Down.Rows(), se.Down.Cols(), sharedInter))
				}
				var e error
				if hl.shGU, e = packWeightStack(&se.Gate, &se.Up); e != nil {
					return declined(fmt.Errorf("layer %d shared gate/up: %w", l, e))
				}
				if hl.shDown, e = packWeight(&se.Down); e != nil {
					return declined(fmt.Errorf("layer %d shared down: %w", l, e))
				}
				if hl.shGU.kind != "int4" || hl.shDown.kind != "int4" {
					return declined(fmt.Errorf("layer %d: shared expert is %q/%q — int4-only", l, hl.shGU.kind, hl.shDown.kind))
				}
				hl.hasShared = true
				if !sharedUngated {
					// Qwen-MoE's sigmoid-gated shared expert: out += sigmoid(SharedGate·h)·shared(h), shared_gate_combine's ungated=0
					// branch; it needs the [1,hidden] SharedGate weight uploaded.
					if lw.SharedGate.Rows() == 0 {
						return declined(fmt.Errorf("layer %d: arch declares a GATED shared expert "+
							"but SharedGate is empty — this loader did not populate it", l))
					}
					if hl.shGate, e = packWeight(&lw.SharedGate); e != nil {
						return declined(fmt.Errorf("layer %d shared gate: %w", l, e))
					}
				}
			}
		}
		hl.preNorm, hl.postNorm = lw.PreAttnNorm, lw.PreMLPNorm
		if !dnetOK || !dnAttnGate {
			// A gated dnetOK family's softmax layer keeps its QK-norm weights off lw (packed with its projections above). Everyone
			// else, including a dnetOK family whose softmax layer is not gated (Olmo Hybrid), reads them like an ordinary GQA/MHA
			// layer's lw.QNorm/KNorm.
			hl.qNorm, hl.kNorm = lw.QNorm, lw.KNorm
		}
		// Per-layer RoPE table (Gemma's local 10k vs global 1M base; Mellum's YaRN-on-global).
		// Uniform-rope families hand back the same slice for every layer.
		if !hl.isDeltaNet {
			hl.invFreq = m.RopeInvFreqLayerResident(l) // Gemma 4: real per-layer table, not the generic one
		}
		// Per-layer window: only LOCAL layers are windowed; global layers stay full causal.
		if m.LayerIsLocalResident(l) {
			hl.window = int32(m.SlidingWindowResident())
		}
		if hl.isDeltaNet {
			// A DeltaNet layer has no attention geometry, QK-norm or rope table, which the checks below read; its own shapes are
			// validated at upload, where the state buffers are sized from dnetParams. postOnly is a model-level flag, but a DeltaNet
			// layer reaches NormPre2 through NormPlacementLinear: it needs real preNorm/postNorm (checked here) and has no
			// PostAttnNorm/PostMLPNorm tensors, so that pair is never populated for it.
			if len(hl.preNorm) == 0 || len(hl.postNorm) == 0 {
				return declined(fmt.Errorf("layer %d missing pre/pre-MLP norm", l))
			}
			hls[l] = hl
			continue
		}
		if hl.isMLA {
			if len(hl.preNorm) == 0 || len(hl.postNorm) == 0 {
				return declined(fmt.Errorf("layer %d missing pre/pre-MLP norm", l))
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
				return declined(fmt.Errorf("layer %d: arch declares sandwich/postOnly norms but PostAttnNorm/PostMLPNorm are not len==hidden(%d) (got %d/%d)",
					l, H, len(hl.postAttnNorm), len(hl.postMLPNorm)))
			}
		}
		if hdL := m.HeadDimAtResident(l); m.HasQKNorm() {
			// Per-layer head_dim: Gemma 4's global layers have q_norm/k_norm of GlobalHeadDim, not the model HeadDim, so validate
			// against the layer's own width. QKNormWhole (Olmo 3/Olmo Hybrid): the norm reduces over the whole q/k vector, so the
			// weight is nH*hdL / nKV*hdL wide, and the resident launch (one Q block plus one K block, see segA) sizes both blocks
			// from a single hd, which is correct only when nH==nKV (MHA). Decline rather than mis-normalize a GQA family.
			wantQ, wantK := hdL, hdL
			nKVL := m.KVHeadsAtResident(l)
			if qkNormWhole {
				if nH != nKVL {
					return declined(fmt.Errorf("layer %d: arch claims QKNormWhole but nH=%d != nKV=%d — the resident whole-vector kernel launch requires MHA", l, nH, nKVL))
				}
				wantQ, wantK = nH*hdL, nKVL*hdL
			}
			if m.KVSrcAtResident(l) != l {
				wantK = 0 // a KV-shared layer has no k_norm either
			}
			if len(hl.qNorm) != wantQ || len(hl.kNorm) != wantK {
				return declined(fmt.Errorf("layer %d: arch claims QK-norm but QNorm/KNorm are not the expected width (want %d/%d, got %d/%d)",
					l, wantQ, wantK, len(hl.qNorm), len(hl.kNorm)))
			}
		}
		if !postOnly && len(hl.preNorm) == 0 {
			return declined(fmt.Errorf("layer %d missing pre-norm", l))
		}
		// parallelBlock (Cohere/Command-R) has no pre-MLP norm tensor at all — the MLP reuses the
		// SAME shared input norm the attention branch already computed (segBFFN's r.aq/r.aSc
		// reuse), so hl.postNorm is legitimately empty for it and must not be required.
		if !postOnly && !parallelBlock && len(hl.postNorm) == 0 {
			return declined(fmt.Errorf("layer %d missing pre-MLP norm", l))
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
	hlm, e := packWeight(lmSrc)
	if e != nil {
		return declined(e)
	}

	// ---- resident + pinned executor ----
	attnTempBeta, attnTempOrigMaxPos := m.AttnTempParams() // Ministral 3, FeatAttnTemp; 0 for every other family
	// Computed once: both ctxCap and extraBytes below need it, and a struct literal cannot read a sibling field set in the
	// same literal.
	kvSlotsReq := cudaKVSlotsRequest(m, dnetP != nil)
	allocSlack := packedAllocSlack([]any{hls, hlm})
	if allocSlackOffForTest {
		allocSlack = 0 // the planted defect of TestBuildScratchAccounting_plantedDefect: the plan as it was before the slack was priced
	}
	residentCtxCap := resolveCtxCapFitSlack(m, m.ResidentContextRequest(), m.Config().MaxPositions, kvSlotsReq, allocSlack)
	r := &cudaResident{
		knob:   m.Knob,
		hidden: H, nLayers: nLayers, nH: nH, inter: I, vocab: vocab,
		eps: m.NormEps(), attnScale: m.AttnScale(), finalSoftcap: m.FinalLogitSoftcapResident(),
		attnTempBeta: attnTempBeta, attnTempOrigMaxPos: attnTempOrigMaxPos,
		qkNorm: m.HasQKNorm(), qkNormWhole: qkNormWhole, rmsAddOne: m.RMSAddOne(),
		act: int32(m.GatedActResident()), sandwich: m.SandwichNormResident(), postOnly: postOnly,
		layerNorm: layerNorm, parallelBlock: parallelBlock, logitScale: logitScale, actG32: actG32,
		moe: isMoE, nE: nE, topK: topK, moeInter: moeInter,
		moeScale: float32(moeScale), nGroup: nGroup, topkGroup: topkGroup,
		sharedInter: sharedInter,
		gemma4Moe:   isG4MoE, gemma4Dense: isGemma4, g4cap: os.Getenv("GOINFER_G4_CAPTURE") != "",
		// C′: DMA the routed int4 experts host→VRAM slots per token (device read, correct). The
		// path to running a model whose experts exceed VRAM. Off by default; byte-identical when off.
		cacheExperts: m.MoECacheExperts(),
		cacheProf:    os.Getenv("GOINFER_MOE_CACHE_PROF") != "",
		dnet:         dnetP,
		isMLA:        mlaOK, mlaRank: kvLoRA, mlaLatDim: kvLoRA + qkRope,
		mlaQKHead: qkNope + qkRope, mlaQKNope: qkNope, mlaQKRope: qkRope,
		mlaVHead: vHead, mlaQLoRA: qLoRA, mlaInterleave: interleave,
		mlaRopeScale: float32(mlaRopeScale),
		// Resolve the resident KV capacity HERE, at construction, not at the KV allocation site:
		// several buffers are sized from it earlier (the split-KV score scratch among them), and a
		// zero-value ctxCap makes those 0-byte allocations that fail the whole resident build.
		// cap = min(model context window, request); request 0 ⇒ the 4096 default, so a caller who
		// did not ask allocates exactly what they always did.
		ctxCap:          residentCtxCap,
		ctxPlanned:      residentCtxCap,
		allocSlackBytes: allocSlack,
		ctxExplicit:     m.ResidentContextRequest() > 0,
		kvSlotsReq:      kvSlotsReq,
		// A drafter's device K/V is priced at the FINAL ctxCap. resolveCtxCapFit's ExtraBytes consult priced it at candidate,
		// the widest ctx that call considered, and the real allocation can only be <= candidate, so this is never a smaller
		// price than the one planned against; checkKVFits enforces it.
		extraBytes: m.ExtraResidentBytes() + m.ExtraResidentKVPerPosition()*int64(residentCtxCap),
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
	// on the model's routing entropy. The sweep and the history: docs/code-notes/cuda.md#BuildResident: MoE cache slots
	// default.
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
			return nil, false, fmt.Errorf("cuda: --moe-cache-slots %d is below top-k %d — one "+
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
	hFinal := w.FinalNorm
	r.reqCh = make(chan func() error)
	r.ackCh = make(chan error)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		for j := range r.reqCh {
			r.ackCh <- runJob(j)
		}
	}()

	// (runJob — the executor's panic boundary — is defined below.)

	// setup job: create the context on the pinned thread, JIT kernels, upload everything.
	setupErr := r.do(func() error {
		var e error
		if r.dev, e = CreateSystemDefaultDevice(); e != nil {
			return e
		}
		// checkWeightsFit runs first, right after the device exists and before the kernel compiles and uploads, and before this
		// backend's other declines, so a model whose fixed (non-expert) weights exceed free VRAM gets a clean decline naming
		// that rather than whichever raw driver error the first oversized allocation produced.
		if e := r.checkWeightsFit(m); e != nil {
			return e
		}
		// The module and pipeline handles cached here do not survive device exhaustion. Once the device has been drained to
		// refusal, a later launch through a cached handle returns SUCCESS and executes nothing (an all-zero output, no error,
		// and cuFuncGetAttribute answers normally); only re-loading the module and re-resolving the function immediately before
		// the launch restores it. goinfer avoids this because BuildResident declines on exhaustion and issues no more CUDA, and
		// that decline is a safety property, not an incidental fallback.
		//
		// Any change that drives the device to refusal and then keeps using the same context breaks this, silently, as zeros. A
		// retry loop, an eviction-and-rebuild or a "just try a smaller cache" must re-load rather than reuse these handles.
		// Evidence, path by path: docs/QUEUE.md A13; the full record: docs/code-notes/cuda.md#BuildResident: device exhaustion.
		gmod, e := r.dev.CompileLibrary(gemvFwdPTX)
		if e != nil {
			return e
		}
		glmod, e := r.dev.CompileLibrary(gluePTX)
		if e != nil {
			return e
		}
		// The generic quantized GEMVs come from aikit (gpu.QuantGEMVPTX), not gemv_fwd.ptx; gemv_fwd.cu keeps only the
		// LLM-specific kv_store / rope_kv, and decode stays bit-identical.
		qgemv, e := r.dev.NewQuantGEMV()
		if e != nil {
			return e
		}
		r.gemvW4, r.gemvW8 = qgemv.W4A8, qgemv.W8A8
		// kv_store and rope are not bound: the fused rope_kv below subsumes both (TestPipelineLint_boundKernelsAreLaunched flags
		// a pipeline bound and never launched). GPT-J pairwise rotation (Cohere/Cohere2/Aya, GLM-OCR) binds the pairwise twin of
		// rope_kv (cuda/rope_pairwise.cu, same argument list and launch geometry) into the same field, so no launch site
		// changes. A load failure declines: the NeoX kernel on a pairwise model is silently wrong from position 1, so there is
		// no fallback.
		r.pairwiseRoPE = m.PairwiseRoPEResident()
		if r.pairwiseRoPE {
			pwmod, pe := r.dev.CompileLibrary(ropePairwisePTX)
			if pe != nil {
				return fmt.Errorf("pairwise rope: compile rope_pairwise.ptx: %w", pe)
			}
			if r.ropeKV, e = r.dev.NewComputePipeline(pwmod, "rope_kv_pw"); e != nil {
				return e
			}
		} else if r.ropeKV, e = r.dev.NewComputePipeline(gmod, "rope_kv"); e != nil {
			return e
		}
		// kv_store at pos 0 is an exact device copy (copyF32), used by K=V layers to fill vB from kB. It is the one use of kv_store, which the fused rope_kv still subsumes for the KV cache.
		if r.kvCopy, e = r.dev.NewComputePipeline(gmod, "kv_store"); e != nil {
			return e
		}
		qmod, e2 := r.dev.CompileLibrary(fusedQKVPTX)
		if e2 != nil {
			return e2
		}
		if r.fQKV, e = r.dev.NewComputePipeline(qmod, "fused_rms_qkv"); e != nil {
			return e
		}
		if r.fGU, e = r.dev.NewComputePipeline(qmod, "fused_rms_gu"); e != nil {
			return e
		}
		if r.fQKN, e = r.dev.NewComputePipeline(qmod, "qk_norm"); e != nil {
			return e
		}
		// rows-per-warp variant of fused_rms_gu (own module; see fused_gu_rows.cu). A load failure only leaves the original kernel in use.
		if gmod2, ge2 := r.dev.CompileLibrary(fusedGURowsPTX); ge2 == nil {
			var le error
			if r.fGURows, le = r.dev.NewComputePipeline(gmod2, "fused_rms_gu_rows"); le != nil {
				r.fGURows = Pipeline{}
			}
		}
		r.readSMShape() // device shape for wave-sized fused-projection grids (waveRowsPerWarp); unreadable -> zero -> the static rules
		// rows-per-warp variant of fused_rms_qkv (own module; see fused_qkv_rows.cu). A load failure only leaves the original kernel in use.
		if rmod, re2 := r.dev.CompileLibrary(fusedQKVRowsPTX); re2 == nil {
			var le error
			if r.fQKVRows, le = r.dev.NewComputePipeline(rmod, "fused_rms_qkv_rows"); le != nil {
				r.fQKVRows = Pipeline{}
			}
		}
		fns := []struct {
			dst  *Pipeline
			name string
		}{
			{&r.fRms, "rmsnorm_quant"}, {&r.fRmsF32, "rmsnorm_f32"}, {&r.fQ, "quant_vec"},
			{&r.fAttn, "attention"}, {&r.fSw, "glu_quant"}, {&r.fRes, "residual"},
			{&r.fLN, "layernorm_quant"},
		}
		for _, f := range fns {
			if *f.dst, e = r.dev.NewComputePipeline(glmod, f.name); e != nil {
				return e
			}
		}
		// Per-32 activations: bind actgroup.cu's kernels INTO the per-vector pipeline fields. Each has
		// its per-vector sibling's exact argument list and launch shape, so every launch site runs the
		// per-32 kernel unedited; the paths that would still read a per-vector scale declined above,
		// fusion is off, and batched prefill declines (prefillStaticDecline).
		if r.actG32 {
			agmod, e3 := r.dev.CompileLibrary(actGroupPTX)
			if e3 != nil {
				return fmt.Errorf("actgroup module: %w", e3)
			}
			for _, f := range []struct {
				dst  *Pipeline
				name string
			}{
				{&r.fRms, "rmsnorm_quant_g32"}, {&r.fQ, "quant_vec_g32"}, {&r.fSw, "glu_quant_g32"},
				{&r.gemvW4, "gemv_w4a8_g32"}, {&r.gemvW8, "gemv_w8a8_g32"}, {&r.gemvQ4K, "gemv_q4k_g32"},
				{&r.fQKVg32, "fused_rms_qkv_g32"}, {&r.fGUg32, "fused_rms_gu_g32"},
			} {
				if *f.dst, e = r.dev.NewComputePipeline(agmod, f.name); e != nil {
					return fmt.Errorf("actgroup %s: %w", f.name, e)
				}
			}
		}
		// argmax_reduce lives in its own module (argmax.ptx), off glue.ptx, so a kernel fix does not force a glue.ptx
		// regeneration. moe.ptx, glue.ptx and gemv_fwd.ptx are the audited artifacts, pinned at NVRTC 12.6.85
		// (cuda/testdata/REGEN.md); every other production PTX, including argmax.ptx, is built at whatever NVRTC was on hand
		// when it was added. See cuda/argmax.cu.
		amod, e2 := r.dev.CompileLibrary(argmaxPTX)
		if e2 != nil {
			return e2
		}
		if r.fArg, e = r.dev.NewComputePipeline(amod, "argmax_reduce"); e != nil {
			return e
		}
		if r.fArgRows, e = r.dev.NewComputePipeline(amod, "argmax_rows"); e != nil {
			return e
		}
		// topk_select: its own module, same isolation as argmax_reduce above.
		tmod, e2 := r.dev.CompileLibrary(topkPTX)
		if e2 != nil {
			return e2
		}
		if r.fTopK, e = r.dev.NewComputePipeline(tmod, "topk_select"); e != nil {
			return e
		}
		// gumbel_stage1/2: own module, same isolation.
		gumod, e2 := r.dev.CompileLibrary(gumbelPTX)
		if e2 != nil {
			return e2
		}
		if r.fGumbel1, e = r.dev.NewComputePipeline(gumod, "gumbel_stage1"); e != nil {
			return e
		}
		if r.fGumbel2, e = r.dev.NewComputePipeline(gumod, "gumbel_stage2"); e != nil {
			return e
		}
		// Compute-time LoRA (docs/tasks/task-gpu-paths-2026-09.md): own module, loaded unconditionally (cheap, and whether an
		// adapter will arrive is not known here).
		lmod, e2 := r.dev.CompileLibrary(loraPTX)
		if e2 != nil {
			return e2
		}
		if r.fLoraDown, e = r.dev.NewComputePipeline(lmod, "lora_delta_down"); e != nil {
			return e
		}
		if r.fLoraUp, e = r.dev.NewComputePipeline(lmod, "lora_delta_up"); e != nil {
			return e
		}
		r.loraT = r.af(loraRMax)
		// Batched prefill kernels (weight-stationary M=len path): own module, prefill_batched.ptx, with the audited PTX
		// untouched. bGemvB dispatches int4 to bRN (gemv_w4a8_rn) unconditionally; there is no gemv_w4a8_batched module.
		{
			if pbmod, e3 := r.dev.CompileLibrary(prefillBatchedPTX); e3 == nil {
				ok := true
				load := func(dst *Pipeline, mod gpu.Library, name string) {
					if p, le := r.dev.NewComputePipeline(mod, name); le == nil {
						*dst = p
					} else {
						ok = false
					}
				}
				if rnmod, e4 := r.dev.CompileLibrary(gemvRNPTX); e4 == nil {
					load(&r.bRN, rnmod, "gemv_w4a8_rn")
				} else {
					ok = false
				}
				// int8 batched GEMV: own module, own file; the audited PTX is untouched.
				if w8mod, e5 := r.dev.CompileLibrary(gemvW8BatchedPTX); e5 == nil {
					load(&r.bW8, w8mod, "gemv_w8a8_batched")
				} else {
					ok = false
				}
				load(&r.bRms, pbmod, "rmsnorm_quant_batched")
				load(&r.bQKN, pbmod, "qk_norm_batched")
				load(&r.bNormF32, pbmod, "rmsnorm_f32_batched")
				if r.pairwiseRoPE {
					// GPT-J pairwise twin of rope_kv_batched (same signature); see the decode bind above.
					if pwmod, pe := r.dev.CompileLibrary(ropePairwisePTX); pe == nil {
						load(&r.bRopeKV, pwmod, "rope_kv_batched_pw")
					} else {
						ok = false
					}
				} else {
					load(&r.bRopeKV, pbmod, "rope_kv_batched")
				}
				load(&r.bAttn, pbmod, "attn_batched")
				load(&r.bQuant, pbmod, "quant_vec_batched")
				load(&r.bSw, pbmod, "glu_quant_batched")
				load(&r.bRes, pbmod, "residual_batched")
				if lnmod, e6 := r.dev.CompileLibrary(layernormQuantPTX); e6 == nil {
					load(&r.bLN, lnmod, "layernorm_quant_batched")
				}
				r.prefillReady = ok
			}
		}
		// Gemma 3 image-block prefill attention (decoder.ResidentImagePrefill): own module, prefill_batched.ptx untouched. A
		// load failure is not fatal: bAttnImg stays zero and PrefillImageLast declines to the CPU-prefill-then-UploadKV bridge.
		// It depends on r.prefillReady, as L2 does.
		if r.prefillReady {
			if imod, e8 := r.dev.CompileLibrary(attnImgPrefillPTX); e8 == nil {
				// &r.bAttnImg (not a bare tuple assign): TestPipelineLint_boundKernelsAreLaunched's
				// static scan recognizes `&r.<field>` or a direct NewComputePipeline RHS as a bind
				// site, matching the loadF/loadG idiom L2/L3 already use below.
				loadImg := func(dst *Pipeline, name string) {
					if pl, pe := r.dev.NewComputePipeline(imod, name); pe == nil {
						*dst = pl
						r.imgPrefillReady = true
					}
				}
				loadImg(&r.bAttnImg, "attn_img_batched")
			}
		}
		// Qwen2.5-VL m-RoPE batched prefill (decoder.ResidentMRoPEPrefill): own module, prefill_batched.ptx untouched. Loaded
		// only for an m-RoPE model, since it needs the model's MRopeSection for sec0/sec1 (unlike bAttnImg above, which loads
		// unconditionally). A load failure or a non-m-RoPE model is not fatal: PrefillMRoPELast declines to the
		// CPU-prefill-then-UploadKV bridge.
		if r.prefillReady {
			if sec := m.MRopeSectionResident(); len(sec) == 3 {
				// Pairwise m-RoPE (GLM-OCR) rotates (2d, 2d+1): its own kernel in rope_pairwise.ptx. A
				// model that needs it and cannot load it must NOT fall back to the NeoX m-RoPE kernel.
				mropePTX, mropeName := ropeMRopePrefillPTX, "rope_kv_mrope_batched"
				if r.pairwiseRoPE {
					mropePTX, mropeName = ropePairwisePTX, "rope_kv_mrope_batched_pw"
				}
				if mmod, e9 := r.dev.CompileLibrary(mropePTX); e9 == nil {
					loadMRope := func(dst *Pipeline, name string) {
						if pl, pe := r.dev.NewComputePipeline(mmod, name); pe == nil {
							*dst = pl
							r.mropePrefillReady = true
						}
					}
					loadMRope(&r.bRopeKVMRoPE, mropeName)
					if r.mropePrefillReady {
						r.mropeTakesMode = !r.pairwiseRoPE
						r.mropeSec0, r.mropeSec1 = int32(sec[0]), int32(sec[0]+sec[1])
						if m.MRopeInterleavedResident() {
							if r.pairwiseRoPE {
								// Pairwise rotation over the interleaved layout is a combination no checkpoint registers (decoder.ropeAt),
								// and rope_kv_mrope_batched_pw has no such mode: decline rather than rotate by the wrong rule.
								r.mropePrefillReady = false
							}
							r.mropeMode, r.mropeSec0, r.mropeSec1 = 1, int32(3*sec[1]), int32(3*sec[2])
						}
					}
				}
			}
		}
		// L2 fused prefill attention (docs/completed/task-prefill-gap.md §4 L2): default on above fastPrefillFloor;
		// GOINFER_CUDA_FAST_PREFILL=0 or Options.ExactPrefill keeps the exact path. Own module, prefill_batched.ptx untouched. A
		// load failure is not fatal: bAttnFused* stay zero and every selection site falls back to attn_batched, the exact path.
		r.fastAttn, r.fastGemm = fastPrefillEnabled(r.knobValue("GOINFER_CUDA_FAST_PREFILL"))
		if m.ExactPrefill() { // Options.ExactPrefill: this model's prefill stays on the exact path
			r.fastAttn, r.fastGemm = false, false
		}
		if (r.fastAttn || r.fastGemm) && r.prefillReady {
			if fmod, e6 := r.dev.CompileLibrary(attnFusedPTX); r.fastAttn && e6 == nil {
				loadF := func(dst *Pipeline, name string) {
					if pl, pe := r.dev.NewComputePipeline(fmod, name); pe == nil {
						*dst = pl
					}
				}
				loadF(&r.bAttnFused64, "attn_fused_hd64")
				loadF(&r.bAttnFused128, "attn_fused_hd128")
				if bmod, eb := r.dev.CompileLibrary(attnFusedBMPTX); eb == nil {
					loadF2 := func(dst *Pipeline, name string) {
						if pl, pe := r.dev.NewComputePipeline(bmod, name); pe == nil {
							*dst = pl
						}
					}
					loadF2(&r.bAttnBM32x64hd64, "attn_fused_bm32_hd64")
					loadF2(&r.bAttnBM32x64hd128, "attn_fused_bm32_hd128")
					loadF2(&r.bAttnBM32x32hd64, "attn_fused_bm32n32_hd64")
					loadF2(&r.bAttnBM32x32hd128, "attn_fused_bm32n32_hd128")
					loadF2(&r.bAttnBM128hd64, "attn_fused_bm128_hd64")
					loadF2(&r.bAttnBM128hd128, "attn_fused_bm128_hd128")
					if v := r.knobValue("GOINFER_CUDA_ATTN_FUSED_TILE"); v == "64x64" {
						r.attnTile = -1
					} else if v == "128x64" {
						r.attnTile = 3
					} else if v == "32x64" {
						r.attnTile = 1
					} else if v == "32x32" {
						r.attnTile = 2
					}
				}
			}
			// L3 tensor-core GEMM: own module, own half of the gate. Bound through an &r.<field> loader closure, like every pipeline
			// here: TestPipelineLint_boundKernelsAreLaunched reads a plain `r.F = pl` as a launch with no binding, i.e. a
			// nil-pipeline dispatch.
			if gmod, e7 := r.dev.CompileLibrary(gemmMMAPTX); r.fastGemm && e7 == nil {
				loadG := func(dst *Pipeline, name string) {
					if pl, pe := r.dev.NewComputePipeline(gmod, name); pe == nil {
						*dst = pl
					}
				}
				loadG(&r.bGemmMMA, "gemm_w4a8_mma")
			}
		}
		// Split-KV decode attention: a high-occupancy, bit-identical alternative to the attn_batched(M=1) decode launch (it
		// replaces exactly that launch, so it needs prefillReady). Own module. skMinKeys -1 means use the per-geometry table;
		// set it before the load so a partial split-KV load cannot leave the zero value, which would mean "always split".
		r.skMinKeys = -1
		// gpt-oss's clamped interleaved-SwiGLU expert epilogue: own module, so the audited glue.ptx/moe.ptx stay untouched.
		// Loaded only for that family; launchGluSplitExpert branches on this pipeline being populated. A load failure is not
		// fatal.
		if alpha, limit, isGptOss := m.GptOssActResident(); isGptOss {
			if gmod, ge := r.dev.CompileLibrary(gptOssActPTX); ge == nil {
				// Loader closure, not a plain assignment: see the pipeline-lint note at the L3 load above.
				loadG := func(dst *Pipeline, name string) {
					if pl, pe := r.dev.NewComputePipeline(gmod, name); pe == nil {
						*dst = pl
					}
				}
				loadG(&r.gptOssSw, "glu_quant_gptoss")
				// gpt-oss's router (route_gptoss): moe_route's "bias" steers expert selection only and takes the weight from the
				// unbiased score, which is right for DeepSeek/GLM and wrong here, where softmax runs over the selected biased logits.
				// Running gpt-oss through moe_route gives plausible mixing weights that are not this model's, a silent quality loss.
				loadG(&r.gptOssRoute, "route_gptoss")
				r.gptOssAlpha, r.gptOssLimit = alpha, limit
			}
		}
		if r.prefillReady {
			if skmod, e2 := r.dev.CompileLibrary(decodeSplitKVPTX); e2 == nil {
				skOK := true
				loadSK := func(dst *Pipeline, name string) {
					if p, le := r.dev.NewComputePipeline(skmod, name); le == nil {
						*dst = p
					} else {
						skOK = false
					}
				}
				loadSK(&r.skScores, "splitkv_scores")
				loadSK(&r.skSoftmax, "splitkv_softmax")
				loadSK(&r.skVsum, "splitkv_vsum")
				if skOK {
					r.skScoreBuf = r.af(r.nH * r.ctxCap)
					r.skInvBuf = r.af(r.nH)
					// Flash-decode V-sum SPIKE, opt-in and NOT bit-identical. Loaded and allocated
					// only when asked for, so a stock binary carries neither the pipelines nor the
					// scratch. See docs/scoping-decode-tree-recanon.md §6.
					if v, err := strconv.Atoi(os.Getenv("GOINFER_SPLITKV_VSUM_SPLIT")); err == nil && v > 1 {
						// maxHd comes from the MODEL, not r.layers: r.layers is not populated until
						// later in BuildResident, so reading it here silently yielded 0 and the
						// resulting 0-byte allocation panicked the executor and declined the whole
						// resident path to CPU. And hidden/nH is not a substitute — qwen3_5's
						// head_dim is 256 where hidden/heads is not.
						maxHd := 0
						for l := range nLayers {
							if h := m.HeadDimAtResident(l); h > maxHd {
								maxHd = h
							}
						}
						if maxHd <= 0 {
							// Never attempt the allocation on a degenerate size: that is the exact
							// failure above, and it takes the entire resident path down with it.
							fmt.Fprintf(os.Stderr, "[cuda] GOINFER_SPLITKV_VSUM_SPLIT ignored: no positive head dim\n")
						} else {
							loadSK(&r.skVsumPartial, "splitkv_vsum_partial")
							loadSK(&r.skVsumCombine, "splitkv_vsum_combine")
							if r.skVsumPartial != (Pipeline{}) && r.skVsumCombine != (Pipeline{}) {
								r.skPartialBuf = r.af(r.nH * maxHd * v)
								r.skVsumSplit = v
							}
						}
					}
					// Default ON (bit-identical; gated per layer at runtime on the effective attended span
					// nWin ≥ splitkvThreshold(nH, hd), so geometries and depths it loses on are unaffected).
					// GOINFER_SPLITKV_ATTN=0 force-disables it (A/B / rollback); GOINFER_SPLITKV_MIN_KEYS
					// overrides the per-geometry threshold so the crossover is re-measurable without a
					// rebuild (0 ⇒ always split — the force-on arm).
					r.splitkvAttn = r.knobValue("GOINFER_SPLITKV_ATTN") != "0"
					if v, err := strconv.Atoi(r.knobValue("GOINFER_SPLITKV_MIN_KEYS")); err == nil && v >= 0 {
						r.skMinKeys = v
					}
				}
			}
		}
		if r.prefillReady {
			r.loadFlashDecode(m, nLayers)
		}
		// MoE module: loaded only for a routed model, so a dense one JITs nothing extra.
		if r.moe {
			mmod, e2 := r.dev.CompileLibrary(moePTX)
			if e2 != nil {
				return e2
			}
			for _, f := range []struct {
				dst  *Pipeline
				name string
			}{
				{&r.fRoute, "moe_route"}, {&r.fRouterGemv, "gemv_f32_a8"},
				{&r.fMoEGemv, "gemv_w4a8_moe"}, {&r.fMoEWacc, "gemv_w4a8_moe_wacc"},
				{&r.fMoEWaccBias, "gemv_w4a8_moe_wacc_bias"},
				{&r.fSharedCombine, "shared_gate_combine"},
			} {
				if *f.dst, e = r.dev.NewComputePipeline(mmod, f.name); e != nil {
					return e
				}
			}
		}
		// router_f32 module: Gemma 4's own kernels, kept off the audited moe.ptx. gemma4Dense (any gemma4 checkpoint, not only
		// enable_moe_block ones) compiles it too: scale_vec (fScaleVec) is segB's dense-tail per-layer-output-scalar kernel,
		// which every dense gemma4 layer needs. The three router-specific kernels stay gemma4Moe-only.
		if r.gemma4Moe || r.gemma4Dense {
			rmod, e2 := r.dev.CompileLibrary(routerF32PTX)
			if e2 != nil {
				return e2
			}
			if r.fScaleVec, e = r.dev.NewComputePipeline(rmod, "scale_vec"); e != nil {
				return e
			}
			if r.gemma4Moe {
				for _, f := range []struct {
					dst  *Pipeline
					name string
				}{
					{&r.fRouterF32, "gemv_f32_f32"}, {&r.fScaleWgt, "scale_wgt_by_expert"},
					{&r.fRmsNW, "rmsnorm_nw"},
				} {
					if *f.dst, e = r.dev.NewComputePipeline(rmod, f.name); e != nil {
						return e
					}
				}
				if r.g4cap {
					r.g4capRn, r.g4capWgt = make([][]float32, nLayers), make([][]float32, nLayers)
					r.g4capX1, r.g4capX2 = make([][]float32, nLayers), make([][]float32, nLayers)
				}
			}
		}
		r.stream = r.dev.NewCommandQueue()

		// gpt-oss per-layer device state: the attention sinks and the per-expert gate‖up
		// bias table. Uploaded once here rather than per token — they are weights, not
		// activations. Both are nil for every other family, and r.af/r.up32 are not called
		// at all in that case (Alloc(0) is an error, not a no-op).
		if _, _, isGptOss := m.GptOssActResident(); isGptOss {
			// The glue `attention` fallback (taken when !prefillReady) has no sink parameter, so a gpt-oss model whose
			// prefill_batched.ptx fails to JIT would decode through a kernel that silently omits the learned per-head sink. Refuse
			// that case: decoder/features.go says the sink reaches both paths, and this is what makes that true.
			if !r.prefillReady {
				return fmt.Errorf("cuda: gpt-oss needs the batched attention kernel " +
					"for its learned attention sink, and prefill_batched.ptx did not load — the " +
					"glue attention fallback has no sink parameter, so decoding here would " +
					"silently drop it (N-10)")
			}
			r.gptOssSinks = make([]Buffer, nLayers)
			r.gptOssExpBias = make([]Buffer, nLayers)
			r.gptOssDownBias = make([]Buffer, nLayers)
			for l := range nLayers {
				if sk := m.GptOssSinksResident(l); len(sk) > 0 {
					r.gptOssSinks[l] = r.up32(sk)
				}
				if gb := m.GptOssExpertBiasResident(l); len(gb) > 0 {
					r.gptOssExpBias[l] = r.up32(gb)
				}
				// The per-expert DOWN bias, consumed by gemv_w4a8_moe_wacc_bias inside the router-weight product; omitting it costs min
				// cosine 0.75 against 0.997.
				if db := m.GptOssExpertDownBiasResident(l); len(db) > 0 {
					r.gptOssDownBias[l] = r.up32(db)
				}
			}
		}
		if dnetP != nil {
			// Gated-DeltaNet: its own module (nothing else here is recurrent) and its own
			// per-token scratch. convDim is 2*nk*hk + nv*hv and bears no relation to
			// qDim/kvDim, so none of the attention scratch can be reused for it.
			dmod, de := r.dev.CompileLibrary(deltaNetPTX)
			if de != nil {
				return fmt.Errorf("cuda: JIT deltanet.ptx: %w", de)
			}
			var lerr error
			loadD := func(dst *Pipeline, name string) {
				// Loader closure, not a plain assignment: see the pipeline-lint note at the L3 load above.
				pl, pe := r.dev.NewComputePipeline(dmod, name)
				if pe != nil {
					lerr = fmt.Errorf("cuda: deltanet kernel %q: %w", name, pe)
					return
				}
				*dst = pl
			}
			loadD(&r.dnConv, "delta_conv")
			loadD(&r.dnGates, "delta_gates")
			loadD(&r.dnNorm, "delta_norm")
			loadD(&r.dnRule, "delta_rule")
			loadD(&r.dnGNorm, "delta_gnorm")
			// Row-batched twins for prompt prefill (docs/tasks/task-cuda-deltanet-prefill-2026-09.md).
			loadD(&r.dnConvRows, "delta_conv_rows")
			loadD(&r.dnGatesRows, "delta_gates_rows")
			loadD(&r.dnNormRows, "delta_norm_rows")
			loadD(&r.dnRuleRows, "delta_rule_rows")
			loadD(&r.dnRuleRows128, "delta_rule_rows_128")
			loadD(&r.dnGNormRows, "delta_gnorm_rows")
			loadD(&r.dnQSplit, "delta_qsplit")
			loadD(&r.dnAttnGate, "delta_attn_gate")
			if lerr != nil {
				// Hard error, not a silent degrade: unlike gpt-oss's optional epilogue, every
				// linear layer of this family NEEDS these. A missing pipeline would dispatch
				// null and produce garbage rather than fall back to anything.
				return lerr
			}
			dp := dnetP
			r.dnMixed, r.dnConvOut = r.af(dp.convDim), r.af(dp.convDim)
			r.dnQn, r.dnKn = r.af(dp.keyDim), r.af(dp.keyDim)
			r.dnHeadP, r.dnBt, r.dnAt = r.af(dp.nv*2), r.af(dp.nv), r.af(dp.nv)
			r.dnZOut, r.dnCore, r.dnGated = r.af(dp.valueDim), r.af(dp.valueDim), r.af(dp.valueDim)
			r.dnGq, r.dnGSc = r.ai(dp.valueDim/4), r.af(1)
		}
		if r.isMLA {
			mlamod, me := r.dev.CompileLibrary(mlaPTX)
			if me != nil {
				return fmt.Errorf("cuda: JIT mla.ptx: %w", me)
			}
			var lerr error
			loadMLA := func(dst *Pipeline, name string) {
				pl, pe := r.dev.NewComputePipeline(mlamod, name)
				if pe != nil {
					lerr = fmt.Errorf("cuda: mla kernel %q: %w", name, pe)
					return
				}
				*dst = pl
			}
			loadMLA(&r.fMlaStore, "mla_latent_store")
			loadMLA(&r.fMlaHeadMV, "mla_head_matvec")
			loadMLA(&r.fMlaQRope, "mla_q_rope")
			loadMLA(&r.fMlaAttn, "mla_attn")
			if lerr != nil {
				return lerr
			}
			r.mlaKVDown = r.af(r.mlaLatDim)
			r.qAbs = r.af(r.nH * r.mlaLatDim)
			r.wsum = r.af(r.nH * r.mlaRank)
			if r.mlaQLoRA > 0 {
				r.mlaQAOut = r.af(r.mlaQLoRA)
				r.mlaQAQ = r.ai((r.mlaQLoRA + 3) / 4)
				r.mlaQASc = r.af(1)
			}
		}
		r.layers = make([]cudaLayer, nLayers)
		for l := range nLayers {
			h := &hls[l]
			L := cudaLayer{idx: l}
			if !r.postOnly || h.isDeltaNet {
				// postOnly (Olmo 3/Olmo Hybrid) has no pre-norm weight for its non-DeltaNet layers (0-length, and Alloc(0) is a hard
				// error), but Olmo Hybrid's DeltaNet layers reach NormPre2 through NormPlacementLinear and carry a real one, required
				// non-empty above.
				L.preNorm = r.up32(h.preNorm)
			}
			if (!r.postOnly && !r.parallelBlock) || h.isDeltaNet {
				// parallelBlock (Cohere/Command-R) has no pre-MLP norm tensor (h.postNorm is 0-length, the same Alloc(0) hazard);
				// segBFFN reuses segA's r.aq/r.aSc as the MLP input (see Model.ParallelBlockResident).
				L.postNorm = r.up32(h.postNorm)
			}
			if h.isDeltaNet {
				// Gated-DeltaNet mixer layer: no q/k/v/o, rope table or KV cache. It has two persistent state buffers instead, zeroed at
				// build and re-zeroed per generation: the recurrence compounds, so unlike a KV cache the next sequence cannot overwrite
				// it.
				dp := r.dnet
				L.isDeltaNet = true
				L.dnQKV, L.dnZ, L.dnOut = r.upW(h.dnQKV), r.upW(h.dnZ), r.upW(h.dnOut)
				L.dnB, L.dnA = r.upW(h.dnB), r.upW(h.dnA)
				L.dnConvW, L.dnDtBias = r.up32(h.dnConvW), r.up32(h.dnDtBias)
				L.dnNegExpA, L.dnNormW = r.up32(h.dnNegExpA), r.up32(h.dnNormW)
				L.dnWin = r.up32(make([]float32, (dp.convK-1)*dp.convDim))
				L.dnState = r.up32(make([]float32, dp.stateElems))
			} else if h.isMLA {
				L.isMLA = true
				if r.mlaQLoRA > 0 {
					L.mlaQA = r.upW(h.mlaQA)
					L.mlaQB = r.upW(h.mlaQB)
					L.mlaQANorm = r.up32(h.mlaQANorm)
				} else {
					L.mlaQ = r.upW(h.mlaQ)
				}
				L.mlaKVA = r.upW(h.mlaKVA)
				L.mlaKVANorm = r.up32(h.mlaKVANorm)
				L.mlaWUK = r.up32(h.mlaWUK)
				L.mlaWUV = r.up32(h.mlaWUV)
				L.o = r.upW(h.o)
				L.invF = r.up32(h.invFreq)
				L.mscale = float32(m.RopeMscaleLayer(l))
			} else {
				L.q, L.o = r.upW(h.q), r.upW(h.o)
				if m.KVSrcAtResident(l) == l {
					L.k = r.upW(h.k) // a KV-shared layer has no k_proj; v below (K=V layers have none)
				}
				L.invF = r.up32(h.invFreq)
				L.mscale = float32(m.RopeMscaleLayer(l)) // YaRN attention_factor; 1.0 for every family without it
				L.qGate = h.qGate
			}
			if !h.isMoE {
				// A routed layer has no dense FFN to upload: its hostW's are empty, and
				// Alloc(0) is an error rather than a harmless no-op.
				L.g, L.u, L.d = r.upW(h.g), r.upW(h.u), r.upW(h.d)
				L.ffnI = L.g.N
				if g4OneFFNWidthForTest && l > 0 {
					L.ffnI = r.layers[0].ffnI
				}
				if pleP > 0 {
					L.pleGate, L.pleProj, L.postPLENorm = r.upW(h.pleGate), r.upW(h.pleProj), r.up32(h.postPLENorm)
				}
				// Gemma 4's per-layer output scalar (out = h*layerScalar after the dense MLP residual add): present on every dense
				// gemma4 layer (1 when the checkpoint's tensor is absent, decoder/weights.go); g4moe layers override it below with their
				// bundle's value. Zero, the skip sentinel segB's dense tail checks, for every non-gemma4 family.
				L.layerScalar = m.Gemma4DenseLayerScalarAtResident(l)
			}
			if (r.sandwich || r.postOnly) && !h.isDeltaNet {
				// h.isDeltaNet: postOnly is model-level, but Olmo Hybrid's DeltaNet layers use NormPre2 and were never given
				// postAttnNorm/postMLPNorm (see the validation loop's DeltaNet continue).
				L.postAttnNorm, L.postMLPNorm = r.up32(h.postAttnNorm), r.up32(h.postMLPNorm)
			}
			// Both of these are ATTENTION side tables: a DeltaNet mixer layer has neither, and
			// its nil slices would become 0-byte allocations (a hard error, not a no-op).
			if h.hasBias && !h.isDeltaNet && !h.isMLA {
				L.qb, L.kb, L.vb, L.hasBias = r.up32(h.qb), r.up32(h.kb), r.up32(h.vb), true
			}
			if h.hasOBias && !h.isDeltaNet {
				L.ob, L.hasOBias = r.up32(h.ob), true
			}
			if r.qkNorm && !h.isDeltaNet && !h.isMLA {
				L.qNorm = r.up32(h.qNorm)
				if len(h.kNorm) > 0 { // empty on a KV-shared layer, and Alloc(0) is an error
					L.kNorm = r.up32(h.kNorm)
				}
			}
			if h.isMoE {
				L.isMoE = true
				L.routerW, L.routerB = r.up32(h.router), r.up32(h.routerBs)
				L.expGU, L.expDown = r.upExperts(h.expGU), r.upExperts(h.expDown)
				if h.hasShared {
					L.hasShared = true
					L.shGU, L.shDown = r.upW(h.shGU), r.upW(h.shDown)
					if h.shGate.N > 0 {
						L.shGateW = r.upW(h.shGate)
					}
				}
			}
			if h.g4moe {
				// Gemma-4 parallel dense‖MoE: router (folded f32 proj) + experts + the 5 norms +
				// per-expert scale + layerScalar. Dense g/u/d were uploaded via the !h.isMoE branch.
				L.g4moe = true
				L.routerW, L.routerB = r.up32(h.router), r.up32(h.routerBs)
				L.expGU, L.expDown = r.upExperts(h.expGU), r.upExperts(h.expDown)
				L.g4preFFN, L.g4postFFN1 = r.up32(h.g4preFFN), r.up32(h.g4postFFN1)
				L.g4preFFN2, L.g4postFFN2, L.g4postFFN = r.up32(h.g4preFFN2), r.up32(h.g4postFFN2), r.up32(h.g4post)
				L.perExpertScaleB = r.up32(h.perExpertScale)
				L.layerScalar = h.layerScalar
			}
			L.window = h.window
			// Per-layer attention geometry, from the same accessors the CPU forward uses (headDimAt/kvHeadsAt/isGlobalLayer through
			// the *Resident wrappers), never a recomputed interleave, so the runner cannot drift from runLayersGemma4. Uniform
			// families collapse to the model-level fields; Gemma 4's global layers report the wide head, fewer KV heads and partial
			// rotary.
			L.hd, L.nKV, L.rhalf = m.HeadDimAtResident(l), m.KVHeadsAtResident(l), m.RotaryDimAtResident(l)/2
			L.qDim, L.kvDim = nH*L.hd, L.nKV*L.hd
			if h.isDeltaNet {
				// A DeltaNet layer has no attention geometry to validate or rope table to bind. A non-zero kvDim would make the KV
				// allocator below size a cache this layer never reads, wasting VRAM on the family that most needs it.
				L.hd, L.nKV, L.rhalf, L.qDim, L.kvDim = 0, 0, 0, 0, 0
				r.layers[l] = L
				continue
			}
			if h.isMLA {
				L.hd = r.mlaQKHead
				L.nKV = 1
				L.rhalf = r.mlaQKRope / 2
				L.qDim = nH * r.mlaVHead
				L.kvDim = r.mlaLatDim
				r.layers[l] = L
				continue
			}
			if src := m.KVSrcAtResident(l); src != l {
				if g4KVSrcOffForTest {
					// Test defect: the source one OWNING layer of the same attention type earlier (when there is one).
					for j := src - 1; j >= 0; j-- {
						if m.KVSrcAtResident(j) == j && m.HeadDimAtResident(j) == m.HeadDimAtResident(src) {
							src = j
							break
						}
					}
				}
				// Gemma 4 E-model KV sharing: this layer reads layer src's cache and owns no K/V. The geometry must be the
				// source's, or its attention would index the aliased buffers with the wrong stride.
				so := &r.layers[src]
				if src > l || so.kvShared || so.hd != L.hd || so.nKV != L.nKV || so.kvDim != L.kvDim {
					return fmt.Errorf("cuda: layer %d shares layer %d's KV cache but its geometry (hd %d, nKV %d, kvDim %d) is not an owning layer's (hd %d, nKV %d, kvDim %d)",
						l, src, L.hd, L.nKV, L.kvDim, so.hd, so.nKV, so.kvDim)
				}
				L.kvShared, L.kvSrc = true, src
			}
			L.kEqV = m.VFromKResident(l) && !L.kvShared
			// Gemma 4 applies v_norm on every layer that owns its K/V; K=V layers always did. g4DropVNormForTest re-drops it on the
			// non-K=V layers so a test can reproduce the pre-fix numbers.
			L.vNorm = !L.kvShared && (L.kEqV || (isGemma4 && !g4DropVNormForTest))
			if !L.kEqV && !L.kvShared {
				L.v = r.upW(h.v) // non-K=V layers have a real v_proj weight; K=V derives V from k; a shared layer has neither
			}
			// Per-layer rope-table invariant: rope_kv rotates L.rhalf pairs per head reading invFreq[0..rhalf), so the bound
			// per-layer table must have exactly rhalf entries. Gemma 4's global and local tables differ in length and the generic
			// finalizeRoPE check cannot see that, so assert it per layer at build.
			if len(h.invFreq) != L.rhalf {
				return fmt.Errorf("cuda: layer %d rope table len=%d != rhalf=%d — per-layer invFreq must match the rotated-pair count the kernel indexes", l, len(h.invFreq), L.rhalf)
			}
			r.layers[l] = L
		}
		r.lmW = r.upW(hlm)
		r.finalNorm = r.up32(hFinal)

		// Scratch (Q/K/V projections, attention context) is allocated ONCE and shared across
		// layers, so it must fit the WIDEST per-layer geometry — Gemma 4's 512-global head, not
		// a model-level value. maxQDim/maxKVDim reduce to nH*hd / nKV*hd for uniform families.
		maxQDim, maxKVDim, maxHd := 0, 0, 0
		for l := range r.layers {
			if q := r.layers[l].qDim; q > maxQDim {
				maxQDim = q
			}
			if k := r.layers[l].kvDim; k > maxKVDim {
				maxKVDim = k
			}
			if d := m.HeadDimAtResident(l); d > maxHd {
				maxHd = d
			}
		}
		// The scratch MUST size to the widest head over LAYERS, not m.Dims() — which reports the
		// LOCAL head_dim (256 for the real 26B), not the 512 the global layers need. Cross-check
		// maxQDim (from per-layer qDim) against nH*maxHd (from the accessors independently): a
		// mismatch means the per-layer geometry drifted from its source. GPU OOB writes don't
		// reliably fault, so assert at plan time rather than wait for a parity red.
		if dnetP != nil {
			// A DeltaNet layer reports zero attention geometry, so it must not drag maxQDim/maxHd
			// to zero for the SOFTMAX layers that share this scratch. Recompute over the
			// attention layers only, and check the invariant against those.
			maxQDim, maxKVDim, maxHd = 0, 0, 0
			for l := range r.layers {
				if r.layers[l].isDeltaNet {
					continue
				}
				if q := r.layers[l].qDim; q > maxQDim {
					maxQDim = q
				}
				if k := r.layers[l].kvDim; k > maxKVDim {
					maxKVDim = k
				}
				if d := m.HeadDimAtResident(l); d > maxHd {
					maxHd = d
				}
			}
			if maxQDim == 0 {
				return fmt.Errorf("cuda: every layer is a DeltaNet mixer — no softmax layer to size attention scratch from; this family is 3:1, so a model with none is a loader bug, not a valid shape")
			}
			if dnAttnGate {
				// The softmax layers' fused q_proj is double width; both halves live in dnQg
				// until delta_qsplit separates them into qB and the gate.
				r.dnQg, r.dnAGate = r.af(2*maxQDim), r.af(maxQDim)
			}
		}
		if !r.isMLA && maxQDim != nH*maxHd {
			return fmt.Errorf("cuda: scratch maxQDim=%d != nH*maxHd=%d*%d=%d — per-layer geometry inconsistent with the accessors", maxQDim, nH, maxHd, nH*maxHd)
		}
		// r.x carries the embedding row: for an E-model that is [hidden ‖ nLayers*P PLE inputs], uploaded in ONE copy, and each layer's
		// PLE branch reads its slice with an element offset (glu_quant's uOff), so no buffer view is needed. Every other model: hidden floats.
		r.pleP, r.embLen = pleP, H+nLayers*pleP
		r.eModel = pleP > 0
		for l := range r.layers {
			r.eModel = r.eModel || r.layers[l].kvShared
		}
		r.x, r.aSc, r.aq = r.af(r.embLen), r.af(r.actScaleLen(H)), r.ai(H/4)
		if pleP > 0 {
			r.pleG = r.af(pleP)
		}
		qBufDim := maxQDim
		if r.isMLA {
			qBufDim = nH * r.mlaQKHead
		}
		r.qB, r.kB, r.vB = r.af(qBufDim), r.af(maxKVDim), r.af(maxKVDim)
		r.kc, r.vc = make([]Buffer, nLayers), make([]Buffer, nLayers)
		// r.ctxCap was resolved at construction (several earlier buffers size from it). The fit check
		// belongs HERE, though: it needs the per-layer kvDims, and running it immediately before the
		// caches are allocated is what makes `free` mean "what is actually left for KV".
		if e := r.checkKVFits(); e != nil {
			return e
		}
		for l := range r.kc {
			// Each layer's KV cache is sized by its own kvDim (Gemma 4's local 2048 vs global 1024), matching the pos*Ly.kvDim
			// stride launchToken indexes it with. The kvDim the cache is sized with must equal the one the accessors derive: a stale
			// model-level kvDim would index off the end into garbage output rather than panic, so fail loudly at plan time.
			if r.layers[l].isDeltaNet {
				// No KV cache for a recurrent mixer. Allocating one would be pure waste on the
				// family least able to afford it: 3 of every 4 layers are this kind.
				continue
			}
			if r.layers[l].isMLA {
				if r.layers[l].kvDim != r.mlaLatDim {
					return fmt.Errorf("cuda: layer %d MLA latent cache kvDim=%d != latDim=%d", l, r.layers[l].kvDim, r.mlaLatDim)
				}
				r.kc[l] = r.af(r.ctxCap * r.layers[l].kvDim)
				continue
			}
			if want := m.KVHeadsAtResident(l) * m.HeadDimAtResident(l); r.layers[l].kvDim != want {
				return fmt.Errorf("cuda: layer %d KV cache kvDim=%d != nKV*hd=%d (accessor-derived) — geometry/cache-size mismatch", l, r.layers[l].kvDim, want)
			}
			if r.layers[l].kvShared {
				continue // aliased below, once every owning layer has its buffers
			}
			r.kc[l], r.vc[l] = r.af(r.ctxCap*r.layers[l].kvDim), r.af(r.ctxCap*r.layers[l].kvDim)
		}
		// A KV-shared layer's cache IS its source's: every launch indexes r.kc[l]/r.vc[l], so aliasing the buffers needs no launch-site
		// change (the device teardown is ledger-based, so a buffer held twice is released once). UploadKV refuses these layers by name.
		for l := range r.layers {
			if r.layers[l].kvShared {
				r.kc[l], r.vc[l] = r.kc[r.layers[l].kvSrc], r.vc[r.layers[l].kvSrc]
			}
		}
		// Further resident KV slots, each a copy of the per-layer buffers just allocated (checkKVFits granted r.kvSlotsN of the
		// request). Slot 0 is the set above; UseKVSlot rebinds r.kc/r.vc.
		if n := r.kvSlotsN; n > 1 {
			r.kvSlotBufs = make([]cudaKVSlot, n)
			r.kvSlotBufs[0] = cudaKVSlot{r.kc, r.vc}
			for s := 1; s < n; s++ {
				b, ok := r.allocKVSlot(s, nLayers)
				if !ok {
					// checkKVFits sized the slot count against the free VRAM read before the build's own scratch, with the margin as the
					// only slack, so the last slot can miss by the scratch. Keep the slots that fit instead of dropping the whole resident
					// to the CPU; the first slot is the build's own and still declines.
					fmt.Fprintf(os.Stderr, "cuda: %d of %d resident KV slots granted: the device ran out of memory allocating slot %d beside the build's own scratch\n", s, n, s+1)
					r.kvSlotsN, r.kvSlotBufs = s, r.kvSlotBufs[:s]
					break
				}
				r.kvSlotBufs[s] = b
			}
		}
		if r.actG32 && maxQDim%32 != 0 {
			return fmt.Errorf("per-32 activations need the attention width (%d) to be a multiple of 32", maxQDim)
		}
		r.cctx, r.cSc, r.cq = r.af(maxQDim), r.af(r.actScaleLen(maxQDim)), r.ai(maxQDim/4)
		// v_norm (Gemma 4: every K/V-owning layer; K=V layers also derive V = v_norm(k)) reuses qk_norm with a UNIT
		// weight [maxHd] (so x*inv*w = x*inv, scale-less; addOne=0). Allocate it only when needed.
		for l := range r.layers {
			if r.layers[l].vNorm {
				ones := make([]float32, maxHd)
				for i := range ones {
					ones[i] = 1.0
				}
				r.vNormUnit = r.up32(ones)
				break
			}
		}
		r.oO = r.af(H)
		r.mSc, r.mq = r.af(r.actScaleLen(H)), r.ai(H/4)
		if r.layerNorm {
			r.zeroBias = r.af(H)
			if err := gpu.Upload(r.zeroBias, make([]float32, H)); err != nil {
				return err
			}
		}
		// Dense-FFN scratch, only if the model has a dense FFN: a model whose every layer is routed reports intermediate_size 0
		// (Qwen3.6-35B-A3B's config omits the key), and these would be 0-byte allocations, which this driver rejects as "invalid
		// length". The dense branch of segBFFN is unreachable for such a model. Scratch is sized to the widest dense layer: an
		// E-model's layers differ, and the launches take each layer's own width (cudaLayer.ffnI).
		maxI := I
		for l := range r.layers {
			maxI = max(maxI, r.layers[l].ffnI)
		}
		if maxI > 0 {
			r.gO, r.uO = r.af(maxI), r.af(maxI)
			r.dSc, r.dScr, r.dq = r.af(r.actScaleLen(maxI)), r.af(maxI), r.ai(maxI/4)
		} else if !isMoE {
			return fmt.Errorf("cuda: intermediate_size is 0 on a DENSE model — no FFN to run")
		}
		if r.moe {
			// Sized to the MoE expert width, not the dense one (Mellum's moe_intermediate_size
			// differs from intermediate_size).
			r.rLogits, r.rIdx, r.rWgt = r.af(nE), r.au32(topK), r.af(topK)
			if r.cacheExperts { // C′: per-token slot-id buffer (uploaded per layer) + readback scratch
				r.slotIdx = r.au32(topK)
				r.hostIdx = make([]uint32, topK)
				r.hostSlot = make([]uint32, topK)
				if hb, e := gpu.NewHostBuffer[uint8](r.dev, topK*4); e != nil {
					return e
				} else {
					r.slotIdxHost = hb
				}
			}
			r.moeGU = r.af(2 * moeInter)
			r.moeSc, r.moeScr, r.moeQ = r.af(1), r.af(moeInter), r.ai(moeInter/4)
			if sharedInter > 0 {
				r.shGUout = r.af(2 * sharedInter)
				r.shSc, r.shScr, r.shQ = r.af(1), r.af(sharedInter), r.ai(sharedInter/4)
				r.shDownOut = r.af(H)
				r.shGl = r.af(1) // the sigmoid gate logit; allocated unconditionally (one float)
			}
		}
		if r.gemma4Moe { // parallel dense‖MoE branch scratch
			r.g4x1, r.g4x2, r.g4rn = r.af(H), r.af(H), r.af(H)
		}
		r.dO, r.logits = r.af(H), r.af(vocab)
		r.argIdx, r.argVal = r.ai(1), r.af(1) // greedy fast-path readback (4 B vs 594 KB)
		r.topkOut = r.ai(2*topkMaxK + 2)      // sampled fast-path readback (~2 KB vs 594 KB)
		nGb := gumbelBlocks(vocab)
		r.gbKey, r.gbIdx, r.gbOut = r.af(nGb), r.ai(nGb), r.ai(1) // Gumbel-max sampling scratch (R7b)
		if hb, e := gpu.NewHostBuffer[float32](r.dev, vocab); e != nil {
			return e
		} else {
			r.logitsPinned, r.logitsHost = hb, hb.Slice()
		}
		// Pay moe_route's deferred first-launch local-memory reservation BEFORE the free reading that sizes the cache, so the
		// cap is correct by construction rather than covered by a margin. moe_route declares per-thread local scratch, and the
		// driver backs it for the device's occupancy the first time the kernel runs, not at module load; the launch demands
		// about 289 MB and retains about 138 MB on the RTX 2070 SUPER, invisible to allocSlots, which reads free VRAM before any
		// kernel has run. Forcing it here beats enlarging slotMarginBytes: the peak is transient, so paying it now means the
		// reading below sees only what is retained, and a margin would bury a named consumer in an unnamed constant.
		//
		// Naming moe_route is safe because the backing store is shared and sized by the largest kernel, and
		// TestKernelLocalMemoryCensus enumerates every entry point in every embedded module and fails, naming this site, if
		// another kernel declares more. route_gptoss does (4608 B/thread against 4416), so it is forced below wherever it is
		// bound (TestRouteGptOssGrowsPoolPastMoERoute). This was measured with sequential single-stream launches, which is what
		// goinfer does; concurrent streams would reopen whether the bound is the max or a sum.
		if r.moe && r.cacheExperts {
			// nE=1, k=1, nGroup=1 does the least work the kernel can do. Its outputs are discarded (rIdx/rWgt are overwritten by
			// every real token before being read, and rLogits is read uninitialised as logits and bias): only the allocation side
			// effect is wanted.
			if e := r.stream.Launch(r.fRoute, onecfg(1, 0),
				Arg(r.rLogits), Arg(r.rLogits), Arg(r.rIdx), Arg(r.rWgt),
				gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1)),
				gpu.ArgValue(int32(0)), gpu.ArgValue(float32(1)),
				gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1))); e != nil {
				return fmt.Errorf("pre-sizing warm-up of moe_route failed: %w", e)
			}
			if r.gptOssRoute != (Pipeline{}) {
				// nE=1, k=1, with the same discarded outputs and rLogits as the bias.
				if e := r.stream.Launch(r.gptOssRoute, onecfg(1, 0),
					Arg(r.rLogits), Arg(r.rLogits), Arg(r.rIdx), Arg(r.rWgt),
					gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1))); e != nil {
					return fmt.Errorf("pre-sizing warm-up of route_gptoss failed: %w", e)
				}
			}
			// Synchronise: the reservation must be a fact before free VRAM is read, and an async
			// launch would let allocSlots read the pre-launch figure and reproduce the whole defect.
			if e := r.stream.Sync(); e != nil {
				return fmt.Errorf("pre-sizing warm-up of moe_route failed to complete: %w", e)
			}
		}
		// C′: the core + KV + scratch are now up, so free VRAM reflects them — size the expert slot
		// cache to what actually fits (cap-and-log, never OOM) and allocate it.
		if e := r.allocSlots(); e != nil {
			return e
		}
		return r.setupErr
	})
	if setupErr != nil {
		r.Close()
		// A silent decline suits a shape this backend does not implement (the staged path serves it, slower). It is wrong when
		// the operator explicitly asked for a resident context that does not fit: degrading quietly turns that into a latency
		// mystery under load, so errKVWontFit is a hard startup error naming the GB. Everything else keeps the decline.
		if errors.Is(setupErr, errKVWontFit) {
			return nil, false, setupErr
		}
		return declined(setupErr)
	}
	// The fused path needs int4 projections on the layers it runs on (layerFusable). Anything else falls back to the unfused
	// chain, which handles mixed quant correctly.
	r.fuseQKV = true
	for l := range hls {
		h := &hls[l]
		if !layerFusable(
			h.q.kind == "int4" && h.k.kind == "int4" && h.v.kind == "int4",
			h.isMoE,
			h.g.kind == "int4" && h.u.kind == "int4" && h.d.kind == "int4") {
			r.fuseQKV = false
			break
		}
	}
	if r.knobValue("GOINFER_CUDA_NO_FUSE") != "" {
		r.fuseQKV = false
	}
	if postOnly {
		// fQKV bakes a real pre-norm weight into its rmsnorm+quant dispatch, and a postOnly arch has none (Ly.preNorm is never
		// uploaded), so it takes the unfused segA chain, which quantizes the raw residual.
		r.fuseQKV = false
	}
	if parallelBlock {
		// fQKV goes straight from x to Q/K/V and never materializes the normed, quantized activation that segBFFN's MLP branch
		// reuses as its own input (FeatParallelBlock: segA's r.aq/r.aSc); the fused path has nowhere to hand it back.
		r.fuseQKV = false
	}
	if r.isMLA {
		r.fuseQKV = false
	}
	if r.actG32 {
		// fused_rms_qkv / fused_rms_gu quantize per vector internally.
		r.fuseQKV = false
		// Their per-32 twins (fused_rms_qkv_g32 / fused_rms_gu_g32) take any per-32 weight kind, per projection, with fuseQKV's
		// structural exclusions (no pre-norm, parallel block, MLA, gated q, MoE layers). Size-gated (fusedG32MaxHidden): every
		// block redoes the rmsnorm and per-32 quant, which grows with H while the launches saved matter less as the GEMVs grow.
		// Fusion paid off at H=1536 and lost at 3072 and 3584.
		r.fuseG32 = r.knobValue("GOINFER_CUDA_NO_FUSE") == "" && !postOnly && !parallelBlock && !r.isMLA &&
			H <= fusedG32MaxHidden
		for l := range hls {
			h := &hls[l]
			if h.isMoE || h.qGate || !g32Fusable(h.q.kind, h.k.kind, h.v.kind, h.g.kind, h.u.kind) {
				r.fuseG32 = false
				break
			}
		}
	}
	// CUDA graphs (GOINFER_CUDA_GRAPHS): capture each layer's static launch segments now that fuseQKV is final (segA/segB
	// branch on it), so decode replays them instead of re-issuing the launches. Off, launchToken is unchanged. Incompatible
	// with the g4cap diagnostic, which syncs inside a segment. admitGraphs gates it: replay is bit-exact only under
	// EXCLUSIVE_PROCESS tenancy or MPS, so a shared-GPU box under DEFAULT declines to the live path
	// (docs/cuda-graphs-investigation.md).
	r.graphs = os.Getenv("GOINFER_CUDA_GRAPHS") != "" && !r.g4cap && !r.isMLA
	r.graphsSync = os.Getenv("GOINFER_CUDA_GRAPHS_SYNC") != "" // debug: serialize replays (bisect ordering hazards)
	r.graphMask = os.Getenv("GOINFER_CUDA_GRAPHS_ONLY")        // debug: replay only these segments (A/B/C), rest live
	if e := r.admitGraphs(); e != nil {
		r.Close()
		return declined(fmt.Errorf("cuda: %w", e))
	}
	// C′ compute/DMA overlap (see the overlap field). Not under graphs: a captured segB cannot record
	// the router event mid-segment and a captured segC cannot wait per miss.
	r.overlap = r.cacheExperts && !r.graphs && r.knobValue("GOINFER_MOE_DMA_OVERLAP") != "0"
	if r.overlap {
		if e := r.initOverlap(); e != nil {
			r.Close()
			return declined(fmt.Errorf("cuda: overlap: %w", e))
		}
	}
	b.resident = r
	return r, true, nil
}

// cudaKVSlotsRequest is how many resident KV slots a build asks checkKVFits for: the model's request
// (decoder.Model.ResidentKVSlotsRequest, already 1 for a family with recurrent state), forced to 1 in two cases the
// decoder cannot see. A Gated-DeltaNet resident mutates its state in place, outside any slot. Expert streaming
// (MoECacheExperts) takes whatever VRAM is left, so every extra KV slot is an expert-cache slot lost
// (docs/tasks/task-concurrency-2026-09.md).
func cudaKVSlotsRequest(m *decoder.Model, recurrent bool) int {
	if recurrent || m.MoECacheExperts() {
		return 1
	}
	return m.ResidentKVSlotsRequest()
}

// cudaBackend and cudaResident implement Close() error and io.Closer, the same spelling of "free this GPU resource" as
// gpu and metal, so callers can write generic cleanup; a signature drift back to a no-error Close breaks the build here.
var (
	_ io.Closer = (*cudaBackend)(nil)
	_ io.Closer = (*cudaResident)(nil)
)

// Close releases the resident backend's GPU resources. Propagates the resident's Close error (best-
// effort teardown returns nil today, but the contract is honored so a future failing release surfaces).
func (b *cudaBackend) Close() error {
	if b.resident != nil {
		return b.resident.Close()
	}
	return nil
}
