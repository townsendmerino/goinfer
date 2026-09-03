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
}

// Compile-time seam checks: cudaBackend must satisfy the decoder's backend +
// residency interfaces (catches signature drift against decoder/residency.go).
var (
	_ decoder.Backend          = (*cudaBackend)(nil)
	_ decoder.ResidencyBackend = (*cudaBackend)(nil)
)

// cudaBackend implements decoder.Backend + decoder.ResidencyBackend.
type cudaBackend struct {
	resident *cudaResident // set by BuildResident; shut down in Close
}

func (b *cudaBackend) Name() string { return "cuda" }

// MatmulBT is the staged (non-resident) path: dst[M,N] = a[M,K]·b[N,K]ᵀ. CUDA residency
// is decode-only and dense-only, so anything off that path (prefill matmuls, non-dense
// families, or a no-driver fallback) lands here — dispatched to the shared SIMD linalg
// kernels (same as the CPU backend), so `--backend cuda` is correct and reasonably fast
// even when the resident GPU path declines.
func (b *cudaBackend) MatmulBT(a, bmat, dst []float32, M, K, N int) {
	linalg.MatmulBT(a, bmat, dst, M, K, N)
}

// layerFusable reports whether one layer permits the fused super-kernels (M23). fQKV always reads
// Q/K/V, so those must be int4. fGU additionally reads gate/up/down as int4+f16-scales, but it runs
// only on dense-FFN layers — MoE layers take moeMLP and leave g/u/d unpacked (kind==""), so gate/up
// are exempt there. Requiring g/u/d on a MoE layer would wrongly strip fQKV from every MoE model;
// omitting them on a dense int8-gate/up layer would pass fGU a nil ws16 and crash the executor.
func layerFusable(qkvInt4, moe, guInt4 bool) bool {
	return qkvInt4 && (moe || guInt4)
}

// BuildResident builds a resident CUDA decoder from a loaded dense Model: host-packs the
// mixed int4/int8/f32 projections, spawns a LockOSThread-pinned CUDA executor, and uploads
// weights + KV scratch once, returning a *cudaResident whose Forward the decode loop drives.
// Declines gracefully (ok=false, no crash) when the driver is absent, dlopen fails, or a
// projection shape isn't residency-compatible — the decoder then uses the staged/CPU path.
// Callers gate on DecodeRunnerEligible (dense Qwen2/Llama) before reaching here.
func (b *cudaBackend) BuildResident(m *decoder.Model) (rf decoder.ResidentForward, ok bool, err error) {
	// Never crash the process on a missing/broken driver: recover → decline → fallback.
	defer func() {
		if p := recover(); p != nil {
			rf, ok, err = nil, false, fmt.Errorf("cuda: BuildResident recovered from panic: %v", p)
		}
	}()

	// The reason is printed UNCONDITIONALLY, not behind a debug flag. Declining moves the whole
	// forward to CPU, and v0.10.0's contract is that the runtime names the reason when it is not on
	// the fast path — a reason nobody can see does not satisfy that. It cost a 307-second 26B run to
	// learn "the experts do not fit VRAM", which the runtime knew at the moment it declined. One
	// line at load, not a debug stream: it is the same "zero means either" shape as a skip census
	// that prints nothing — a silent decline and a successful build look identical from outside.
	declined := func(e error) (decoder.ResidentForward, bool, error) {
		fmt.Fprintf(os.Stderr, "[cuda] resident path DECLINED (falling back to the staged/CPU path): %v\n", e)
		return nil, false, nil
	}

	w := m.Weights()
	if w == nil || len(w.Layers) == 0 {
		return nil, false, nil
	}

	// Admission: DecodeRunnerEligible was scoped to the RICHER WebGPU runner (QK-norm,
	// partial/scaled RoPE, sliding window, per-layer RoPE, MoE, MLA, SSM), so it is far too
	// permissive for this backend, which implements the plain dense Qwen2/Llama block ONLY.
	// Without this check the failure is silent — the feature is dropped and the logits are
	// wrong (e.g. Qwen3's QK-norm ignored; Mistral run full-attention past its window; a
	// partial-rotary model reading invFreq out of bounds because the rope kernel hardcodes
	// half = hd/2). Same bug class the Metal backend hit; the taxonomy lives in
	// decoder/features.go so all three backends share one source of truth.
	if missing := m.MissingResidentFeatures(decoder.ResidentBackendFeatures("cuda")); len(missing) > 0 {
		return declined(fmt.Errorf("arch needs unimplemented feature(s) %v", missing))
	}

	H, nLayers, nH, _, _, I, vocab := m.Dims() // nKV, hd are per-layer now (KVHeadsAtResident/HeadDimAtResident); model-level unused

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
	if isMoE && !isG4MoE {
		// Decline anything the dispatch does not implement, LOUDLY rather than by dropping it.
		// Each of these would otherwise be silent-wrong, which is the whole point of the
		// admission gate — and FeatMoE is one flag, so it cannot express these sub-shapes.
		//
		// The GATED shared expert (Qwen-MoE) is NOT declined here anymore: it is a derived
		// feature (FeatMoEGatedShared, decoder/features.go) that CUDA does not declare, so the
		// admission check above already declines it before this switch runs. Duplicating that
		// decline here would be a hand-coded copy that could drift from the taxonomy (the exact
		// class the hardware-matrix generator caught) — single source of truth instead.
		switch {
		case nE > 512:
			return declined(fmt.Errorf("MoE nE=%d exceeds moe_route's MOE_MAX_E=512", nE))
		case nGroup > 64:
			return declined(fmt.Errorf("MoE nGroup=%d exceeds moe_route's MOE_MAX_G=64", nGroup))
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
		if nE > 512 {
			return declined(fmt.Errorf("gemma4 MoE nE=%d exceeds moe_route's MOE_MAX_E=512", nE))
		}
	}

	// ---- host pack all weights (CPU; any incompatible shape → decline) ----
	type hlayer struct {
		q, k, v, o, g, u, d       hostW
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
	}
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
		case dnetOK:
			// The same family's SOFTMAX layer. Its weights live off lw.QProj (the family keeps
			// them in its own struct), and q_proj is DOUBLE WIDTH — [query ‖ gate] per head.
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
		default:
			proj = []projEnt{{&hl.q, &lw.QProj}, {&hl.k, &lw.KProj}, {&hl.o, &lw.OProj}}
			if !m.VFromKResident(l) { // K=V (attention_k_eq_v) global layers carry NO v_proj — V=v_norm(k)
				proj = append(proj, projEnt{&hl.v, &lw.VProj})
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
		for _, p := range proj {
			hw, e := packWeight(p.src)
			if e != nil {
				return declined(e)
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
					// Qwen-MoE's sigmoid-gated shared expert: out += sigmoid(SharedGate·h)·shared(h).
					// The KERNEL for this has been here all along — shared_gate_combine's ungated=0
					// branch, documented "gated (Qwen-MoE)" — so the feature was declined for a
					// missing [1,hidden] weight upload, not a missing kernel.
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
		if !dnetOK { // this family keeps its QK-norm weights off lw, packed with its projections above
			hl.qNorm, hl.kNorm = lw.QNorm, lw.KNorm
		}
		// Gemma sandwich: the extra post-attn / post-MLP norms. Required to be present when
		// the arch declares them — a silently-missing one would drop the norm, not error.
		if sandwich {
			hl.postAttnNorm, hl.postMLPNorm = lw.PostAttnNorm, lw.PostMLPNorm
			if len(hl.postAttnNorm) != H || len(hl.postMLPNorm) != H {
				return declined(fmt.Errorf("layer %d: arch declares sandwich norms but PostAttnNorm/PostMLPNorm are not len==hidden(%d) (got %d/%d)",
					l, H, len(hl.postAttnNorm), len(hl.postMLPNorm)))
			}
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
			// A DeltaNet layer has no attention geometry, no QK-norm and no rope table; the
			// checks below all read those. Its own shapes are validated at upload, where the
			// state buffers are sized from dnetParams.
			if len(hl.preNorm) == 0 || len(hl.postNorm) == 0 {
				return declined(fmt.Errorf("layer %d missing pre/pre-MLP norm", l))
			}
			hls[l] = hl
			continue
		}
		if hdL := m.HeadDimAtResident(l); m.HasQKNorm() && (len(hl.qNorm) != hdL || len(hl.kNorm) != hdL) {
			// Per-layer head_dim: Gemma 4's global layers have q_norm/k_norm of GlobalHeadDim (512),
			// not the model HeadDim (16 local) — validate against the layer's own width.
			return declined(fmt.Errorf("layer %d: arch claims QK-norm but QNorm/KNorm are not len==headDim(%d)", l, hdL))
		}
		if len(hl.preNorm) == 0 || len(hl.postNorm) == 0 {
			return declined(fmt.Errorf("layer %d missing pre/pre-MLP norm", l))
		}
		if lw.QBias != nil {
			hl.qb, hl.kb, hl.vb, hl.hasBias = lw.QBias, lw.KBias, lw.VBias, true
		}
		// Captured INDEPENDENTLY of QBias: the two travel together in Qwen2 but not in general —
		// GPT-2 carries an o_proj bias with no q/k/v bias, so folding this into the branch above
		// would silently drop it for exactly the families FeatOutBias exists for.
		//
		// N-34: this used to name gpt-oss as that example, saying it carries an o_proj bias
		// "with no q/k/v bias at all". It does carry them — gptoss_safetensors.go loads q/k/v
		// bias as REQUIRED (a missing one is an error) and decoder/testdata/gptoss_tiny.gguf
		// holds attn_q/k/v.bias. The code was right and the comment wrong, which is the
		// dangerous direction: it invited exactly the fold it warns against.
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
	r := &cudaResident{
		hidden: H, nLayers: nLayers, nH: nH, inter: I, vocab: vocab,
		eps: m.NormEps(), attnScale: m.AttnScale(), finalSoftcap: m.FinalLogitSoftcapResident(),
		qkNorm: m.HasQKNorm(), rmsAddOne: m.RMSAddOne(),
		act: int32(m.GatedActResident()), sandwich: m.SandwichNormResident(),
		moe: isMoE, nE: nE, topK: topK, moeInter: moeInter,
		moeScale: float32(moeScale), nGroup: nGroup, topkGroup: topkGroup,
		sharedInter: sharedInter,
		gemma4Moe:   isG4MoE, g4cap: os.Getenv("GOINFER_G4_CAPTURE") != "",
		// C′: DMA the routed int4 experts host→VRAM slots per token (device read, correct). The
		// path to running a model whose experts exceed VRAM. Off by default; byte-identical when off.
		cacheExperts: m.MoECacheExperts(),
		cacheProf:    os.Getenv("GOINFER_MOE_CACHE_PROF") != "",
		dnet:         dnetP,
		// Resolve the resident KV capacity HERE, at construction, not at the KV allocation site:
		// several buffers are sized from it earlier (the split-KV score scratch among them), and a
		// zero-value ctxCap makes those 0-byte allocations that fail the whole resident build.
		// cap = min(model context window, request); request 0 ⇒ the 4096 default, so a caller who
		// did not ask allocates exactly what they always did.
		ctxCap:      resolveCtxCap(m.ResidentContextRequest(), m.Config().MaxPositions),
		ctxExplicit: m.ResidentContextRequest() > 0,
	}
	if moeSig {
		r.moeSigmoid = 1
	}
	if moeNorm {
		r.moeNormTopK = 1
	}
	// C′ step 2: device slots per layer — an LRU cache of nSlots experts (clamped [topK, nE]).
	// VRAM is nLayers·nSlots·perExpert, so more slots trades VRAM for fewer per-token DMAs.
	// GOINFER_MOE_CACHE_SLOTS=N requests N explicitly.
	//
	// With caching ON and NO explicit request, ask for ALL experts and let allocSlots cap to
	// measured free VRAM. This reverses a default of topK, which was the worst possible setting for
	// the only situation this code runs in: at nSlots=topK the cache degenerates to fresh-loading
	// every routed expert every token — ~714 MB/token on the 26B, ~5 tok/s against ~17 at the 38
	// slots that fit. Nobody enables expert streaming to get the slow version of it, so the
	// conservative default was really deferring a VRAM decision that allocSlots already makes
	// properly: it measures free VRAM and caps-and-logs. An over-large request was never the hazard
	// it looked like either — post-C-24 an alloc panic on the executor becomes a DECLINE (→ staged
	// fallback), not a process kill.
	//
	// REVERTED to topK (2026-08-11). Defaulting to "ask for all, let allocSlots cap to free VRAM"
	// was correct in intent and WRONG in practice, because allocSlots' cap is not the safety net it
	// looks like: its headroom is a flat `marginBytes = 384 MB` described as covering "the
	// greedy-argmax readback + driver overhead" — per-token costs — while what it must actually
	// leave room for is everything the forward allocates AFTER it runs, which scales with layers,
	// context and vocab. On the real 26B it capped 128 slots to 34 (3.4 GB of 3.8 GB free) and the
	// warm forward then died with cuLaunchKernel: CUDA_ERROR_OUT_OF_MEMORY.
	//
	// Measured, same test, same box, only this default differing:
	//     GOINFER_MOE_CACHE_SLOTS=8  (this default) -> PASS 305s
	//     unset -> 128, capped to 34               -> FAIL 477s (OOM)
	//
	// RAISED (2026-08-20) to a BOUNDED multiple of topK. This is the "change about defaults" the
	// note above parked — its safety precondition (fix the margin, prove it on the 26B) was met by
	// A5/A7, and what remained was only that a default change should not ride along inside an
	// unrelated commit.
	//
	// WHY A BOUND AND NOT THE "ask for all, cap to VRAM" THE ACCESSOR DOCUMENTS. Measured on the
	// real Qwen3.6-35B-A3B (nE=256, topK=8), sweeping slots/layer:
	//
	//     8 (topK)      6.77 tok/s   ~0% hit   ~630 MB expert DMA/token
	//     48           10.09 tok/s   71.1%      265 MB
	//     76           10.32 tok/s   77.7%      205 MB
	//
	// The knee is around 48: cutting DMA 630→265 MB bought 1.49x, cutting it 265→205 bought 1.02x.
	// Past the knee, slots consume GB of VRAM for nothing — and every extra GB is more exposure to
	// the deferred local-memory reservations that made the previous "ask for all" attempt OOM after
	// allocSlots had already capped. A bound gets the whole win with a fraction of the risk, which
	// "all" cannot claim.
	//
	// 8*topK sits just above the measured knee. It is a HEURISTIC from one sweep (plus the 26B's
	// "~17 tok/s at the 38 slots that fit"), not a derived constant: where the knee falls depends on
	// the model's routing entropy. Erring above it costs VRAM the cap will reclaim if it is short;
	// erring below it costs throughput nothing reclaims.
	//
	// The floor stays topK — one token's routed set must be simultaneously resident — and allocSlots
	// still caps to measured free VRAM, so this can only ever ask for less than the reverted default.
	r.cacheSlots = topK
	r.cacheSlotsReq = topK
	if r.cacheExperts && nE > 0 {
		if d := 8 * topK; d > topK {
			r.cacheSlots = min(d, nE)
			r.cacheSlotsReq = r.cacheSlots
		}
	}
	if r.cacheExperts {
		// The request now comes from Options (--moe-cache-slots), and MoECacheSlotsRequest still
		// honours GOINFER_MOE_CACHE_SLOTS, so nothing that set the env var breaks.
		//
		// NOTE the accessor documents 0 as "ask for all, auto-cap to VRAM". That default is NOT
		// taken here: unset leaves topK in place, exactly as before this branch. Raising it is a
		// SEPARATE decision — main's comment above states its own precondition ("fixing the margin
		// FIRST and proving it on the 26B"), which A5 (6091e7a) and A7 have now met — and a change
		// of default belongs in a change about defaults, not in one promoting env vars to flags.
		// G-07: `> topK` silently floored a request of topK or less. A request BELOW topK cannot
		// be honoured — one token's own top-k must fit — but it was neither honoured nor
		// refused: cacheSlots stayed at the 8·topK default, which on a small fixture is nE, so
		// every expert gets a permanent slot and slot ≠ expert only by first-admit order. A gate
		// asking for "fewer slots than experts" therefore got the identity mapping and
		// discriminated by routing luck. Refuse what cannot be honoured; honour the rest exactly.
		if req := m.MoECacheSlotsRequest(); req > 0 && req < topK {
			// A hard error, NOT declined(): declined() swallows the error and falls back to the
			// staged path, which is right for a shape this backend does not implement and wrong
			// for an operator flag that cannot mean what it says. The user asked for something
			// impossible; tell them.
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
		// THESE MODULE AND PIPELINE HANDLES DO NOT SURVIVE DEVICE EXHAUSTION. Read this before
		// adding any path that recovers residency after memory pressure.
		//
		// Measured (A13, docs/QUEUE.md): once the device has been drained to exhaustion, a later
		// launch through a handle cached here returns SUCCESS and executes NOTHING — the output
		// buffer is left untouched and surfaces downstream as an all-zero result, e.g. a cosine of
		// exactly 0.000000. No CUDA call reports an error at any point, and free VRAM is back to
		// ~7.3 GB by then, so neither an error check nor a memory check will catch it.
		//
		// AND cuFuncGetAttribute WILL NOT DETECT IT. Queried across a poisoned and a clean run it
		// returns byte-identical valid values (maxThreadsPerBlock, numRegs, ptxVersion) because
		// those come from metadata that outlives the device code. A handle that answers is not a
		// handle that works.
		//
		// What restores it, measured 3/3: re-loading the module and re-resolving the function
		// IMMEDIATELY BEFORE the launch. Re-loading earlier does not — allocations performed
		// between the load and the launch re-invalidate it.
		//
		// goinfer does not hit this today only because BuildResident DECLINES on exhaustion
		// ((nil,false,nil)) and cudaBackend.MatmulBT then runs linalg.MatmulBT with no CUDA at all.
		// That decline is a safety property, not an incidental fallback. Any future residency
		// recovery must RE-LOAD rather than reuse what is cached here.
		//
		// THE FALSIFIER, which is the one sentence to carry away from all of this:
		//
		//	ANY CHANGE THAT DRIVES THE DEVICE TO REFUSAL AND THEN CONTINUES USING THE SAME CONTEXT
		//	BREAKS THIS.
		//
		// The whole tag rests on that single property holding across every shipped path — measured
		// path by path in docs/QUEUE.md A13 (prefill peaks 39.9x clear of the floor; unload frees
		// rather than exhausts; the cap search allocates nothing; the one path that DOES exhaust,
		// resident build, then declines and issues no CUDA). It is not a guarantee the type system
		// or any test can enforce for code that does not exist yet, so it is written here, where
		// someone adding a retry loop, an eviction-and-rebuild, or a "just try a smaller cache"
		// will be reading. If your change makes the device refuse and then keeps going on that
		// context, the failure will be silent zeros, not an error.
		gmod, e := r.dev.CompileLibrary(gemvFwdPTX)
		if e != nil {
			return e
		}
		glmod, e := r.dev.CompileLibrary(gluePTX)
		if e != nil {
			return e
		}
		// The generic quantized GEMVs come from aikit (gpu.QuantGEMVPTX), not from
		// gemv_fwd.ptx — the Phase-1b blob-split. aikit owns the quantized matmul on
		// the GPU exactly as linalg owns it on the CPU; gemv_fwd.cu keeps only the
		// LLM-specific kv_store / rope_kv. The kernels are the same instructions this
		// file shipped before the split, so decode stays bit-identical.
		qgemv, e := r.dev.NewQuantGEMV()
		if e != nil {
			return e
		}
		r.gemvW4, r.gemvW8 = qgemv.W4A8, qgemv.W8A8
		// kv_store and rope are NOT bound: the fused rope_kv below subsumes both (the Incr1
		// decode-fusion win). They were left bound and dead after that fusion shipped, JIT-compiled
		// into every model load and launched by nothing — see TestPipelineLint_boundKernelsAreLaunched.
		if r.ropeKV, e = r.dev.NewComputePipeline(gmod, "rope_kv"); e != nil {
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
		fns := []struct {
			dst  *Pipeline
			name string
		}{
			{&r.fRms, "rmsnorm_quant"}, {&r.fRmsF32, "rmsnorm_f32"}, {&r.fQ, "quant_vec"},
			{&r.fAttn, "attention"}, {&r.fSw, "glu_quant"}, {&r.fRes, "residual"},
		}
		for _, f := range fns {
			if *f.dst, e = r.dev.NewComputePipeline(glmod, f.name); e != nil {
				return e
			}
		}
		// argmax_reduce lives in its own module (argmax.ptx), off glue.ptx, so the C-14 index tie-break
		// fix didn't force a glue.ptx regen. (glue.ptx and all production PTX except moe.ptx are this
		// box's NVRTC 12.9.86; only moe.ptx + the bench kernels are the audited 12.6.85 — audit R-26.)
		// See cuda/argmax.cu.
		amod, e2 := r.dev.CompileLibrary(argmaxPTX)
		if e2 != nil {
			return e2
		}
		if r.fArg, e = r.dev.NewComputePipeline(amod, "argmax_reduce"); e != nil {
			return e
		}
		// Batched prefill kernels (weight-stationary M=len path). Own module; the audited PTX is
		// untouched. bGemv comes from gemv_w4a8_batched.ptx, the rest from prefill_batched.ptx.
		// gemvBatchedPTX is NO LONGER COMPILED HERE: its only entry, gemv_w4a8_batched, was bound
		// and never launched, so an entire PTX module was JIT-compiled on every model load to feed
		// a dead field. bGemvB dispatches int4 to bRN (gemv_w4a8_rn) unconditionally.
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
				// int8 batched GEMV (§C6): own module, own file; the audited PTX is untouched.
				if w8mod, e5 := r.dev.CompileLibrary(gemvW8BatchedPTX); e5 == nil {
					load(&r.bW8, w8mod, "gemv_w8a8_batched")
				} else {
					ok = false
				}
				load(&r.bRms, pbmod, "rmsnorm_quant_batched")
				load(&r.bQKN, pbmod, "qk_norm_batched")
				load(&r.bNormF32, pbmod, "rmsnorm_f32_batched")
				load(&r.bRopeKV, pbmod, "rope_kv_batched")
				load(&r.bAttn, pbmod, "attn_batched")
				load(&r.bQuant, pbmod, "quant_vec_batched")
				load(&r.bSw, pbmod, "glu_quant_batched")
				load(&r.bRes, pbmod, "residual_batched")
				r.prefillReady = ok
			}
		}
		// Campaign-A split-KV decode attention: a high-occupancy, bit-identical alternative to the A1
		// attn_batched(M=1) decode launch (it replaces exactly that launch, so it needs prefillReady).
		// Opt-in via GOINFER_SPLITKV_ATTN so it can be A/B'd and gated before default-on. Own module.
		// -1 ⇒ use the per-geometry table. Set before the load so a partial split-KV load can never
		// leave the zero value (0) sitting here meaning "always split".
		r.skMinKeys = -1
		// gpt-oss's clamped interleaved-SwiGLU expert epilogue, from its own module so the
		// audited glue.ptx/moe.ptx stay untouched. Loaded only for that family: every other
		// one keeps glu_quant, and launchGluSplitExpert branches on this pipeline being
		// populated. A load failure is not fatal — it simply leaves gpt-oss on the CPU path,
		// which is where it is today anyway.
		if alpha, limit, isGptOss := m.GptOssActResident(); isGptOss {
			if gmod, ge := r.dev.CompileLibrary(gptOssActPTX); ge == nil {
				// &r.field via a loader closure, matching every other pipeline here — the
				// pipeline lint keys on that form (`&r.<field>`), and a plain assignment
				// reads to it as a launch with no binding. That is not pedantry: an unbound
				// launch is a null-pipeline dispatch, which is what the lint exists to stop.
				loadG := func(dst *Pipeline, name string) {
					if pl, pe := r.dev.NewComputePipeline(gmod, name); pe == nil {
						*dst = pl
					}
				}
				loadG(&r.gptOssSw, "glu_quant_gptoss")
				// gpt-oss's ROUTER. moe.cu's moe_route means something different by "bias" —
				// it steers SELECTION only and takes the weight from the UNBIASED score, which
				// is right for DeepSeek/GLM and wrong here: gpt-oss softmaxes over the SELECTED
				// BIASED logits. Its own kernel comment calls running gpt-oss through moe_route
				// "plausible mixing weights that are simply not this model's — a silent quality
				// loss, not a crash". The kernel shipped 2026-08-18 and was never loaded, so
				// that is exactly what the resident path has been doing.
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
					// Default ON (bit-identical; gated per layer at runtime on the effective attended span
					// nWin ≥ splitkvThreshold(nH, hd), so geometries and depths it loses on are unaffected).
					// GOINFER_SPLITKV_ATTN=0 force-disables it (A/B / rollback); GOINFER_SPLITKV_MIN_KEYS
					// overrides the per-geometry threshold so the crossover is re-measurable without a
					// rebuild (0 ⇒ always split — the force-on arm).
					r.splitkvAttn = os.Getenv("GOINFER_SPLITKV_ATTN") != "0"
					if v, err := strconv.Atoi(os.Getenv("GOINFER_SPLITKV_MIN_KEYS")); err == nil && v >= 0 {
						r.skMinKeys = v
					}
				}
			}
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
		// router_f32 module: Gemma-4 MoE's own kernels, kept off the audited moe.ptx (this box's
		// 12.9 NVRTC would rewrite every moe.ptx kernel). Pure-f32 router GEMV + per-expert-scale
		// fold + weightless out-of-place norm + scalar-scale.
		if r.gemma4Moe {
			rmod, e2 := r.dev.CompileLibrary(routerF32PTX)
			if e2 != nil {
				return e2
			}
			for _, f := range []struct {
				dst  *Pipeline
				name string
			}{
				{&r.fRouterF32, "gemv_f32_f32"}, {&r.fScaleWgt, "scale_wgt_by_expert"},
				{&r.fRmsNW, "rmsnorm_nw"}, {&r.fScaleVec, "scale_vec"},
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
		r.stream = r.dev.NewCommandQueue()

		// gpt-oss per-layer device state: the attention sinks and the per-expert gate‖up
		// bias table. Uploaded once here rather than per token — they are weights, not
		// activations. Both are nil for every other family, and r.af/r.up32 are not called
		// at all in that case (Alloc(0) is an error, not a no-op).
		if _, _, isGptOss := m.GptOssActResident(); isGptOss {
			// N-10: the glue `attention` fallback (taken when !prefillReady) has NO sink
			// parameter in its signature at all — it cannot be passed one, only avoided. So a
			// gpt-oss model whose prefill_batched.ptx fails to JIT would decode through a
			// kernel that silently omits the learned per-head sink, with nothing declined and
			// nothing logged. decoder/features.go says the sink reaches both paths; this is
			// what makes that true, by refusing the case where it cannot.
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
				// The per-expert DOWN bias. Uploaded here with the others because it is a weight,
				// not an activation; consumed by gemv_w4a8_moe_wacc_bias INSIDE the router-weight
				// product. Never wired until 2026-08-31, which cost min cosine 0.75 vs 0.997.
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
				// &r.field via a loader closure, matching every other pipeline here — the
				// pipeline lint keys on that form, and a plain assignment reads to it as a
				// launch with no binding.
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
		r.layers = make([]cudaLayer, nLayers)
		for l := range nLayers {
			h := &hls[l]
			L := cudaLayer{
				idx:     l,
				preNorm: r.up32(h.preNorm), postNorm: r.up32(h.postNorm),
			}
			if h.isDeltaNet {
				// Gated-DeltaNet mixer layer: no q/k/v/o, no rope table, no KV cache. Two
				// persistent state buffers instead, both zeroed at build and re-zeroed per
				// generation — the recurrence COMPOUNDS, so unlike a KV cache the next sequence
				// cannot simply overwrite it.
				dp := r.dnet
				L.isDeltaNet = true
				L.dnQKV, L.dnZ, L.dnOut = r.upW(h.dnQKV), r.upW(h.dnZ), r.upW(h.dnOut)
				L.dnB, L.dnA = r.upW(h.dnB), r.upW(h.dnA)
				L.dnConvW, L.dnDtBias = r.up32(h.dnConvW), r.up32(h.dnDtBias)
				L.dnNegExpA, L.dnNormW = r.up32(h.dnNegExpA), r.up32(h.dnNormW)
				L.dnWin = r.up32(make([]float32, (dp.convK-1)*dp.convDim))
				L.dnState = r.up32(make([]float32, dp.stateElems))
			} else {
				L.q, L.k, L.o = r.upW(h.q), r.upW(h.k), r.upW(h.o) // v below (K=V layers have none)
				L.invF = r.up32(h.invFreq)
				L.mscale = float32(m.RopeMscaleLayer(l)) // YaRN attention_factor; 1.0 for every family without it
				L.qGate = h.qGate
			}
			if !h.isMoE {
				// A routed layer has no dense FFN to upload: its hostW's are empty, and
				// Alloc(0) is an error rather than a harmless no-op.
				L.g, L.u, L.d = r.upW(h.g), r.upW(h.u), r.upW(h.d)
			}
			if r.sandwich {
				L.postAttnNorm, L.postMLPNorm = r.up32(h.postAttnNorm), r.up32(h.postMLPNorm)
			}
			// Both of these are ATTENTION side tables: a DeltaNet mixer layer has neither, and
			// its nil slices would become 0-byte allocations (a hard error, not a no-op).
			if h.hasBias && !h.isDeltaNet {
				L.qb, L.kb, L.vb, L.hasBias = r.up32(h.qb), r.up32(h.kb), r.up32(h.vb), true
			}
			if h.hasOBias && !h.isDeltaNet {
				L.ob, L.hasOBias = r.up32(h.ob), true
			}
			if r.qkNorm && !h.isDeltaNet {
				L.qNorm, L.kNorm = r.up32(h.qNorm), r.up32(h.kNorm)
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
			// Per-layer attention geometry (9a-P2), read from the SAME accessors the CPU forward
			// uses (headDimAt/kvHeadsAt/isGlobalLayer via the *Resident wrappers) — never a
			// recomputed interleave, so the runner can't drift from runLayersGemma4. Uniform
			// families collapse to the model-level fields; Gemma 4's global layers report the wide
			// head / fewer KV heads / partial rotary.
			L.hd, L.nKV, L.rhalf = m.HeadDimAtResident(l), m.KVHeadsAtResident(l), m.RotaryDimAtResident(l)/2
			L.qDim, L.kvDim = nH*L.hd, L.nKV*L.hd
			if h.isDeltaNet {
				// A DeltaNet layer has no attention geometry to validate and no rope table to
				// bind. Leaving kvDim non-zero here would make the KV allocator below size a
				// cache this layer never reads — real VRAM, silently wasted, on the one family
				// that most needs it.
				L.hd, L.nKV, L.rhalf, L.qDim, L.kvDim = 0, 0, 0, 0, 0
				r.layers[l] = L
				continue
			}
			L.kEqV = m.VFromKResident(l)
			if !L.kEqV {
				L.v = r.upW(h.v) // non-K=V layers have a real v_proj weight; K=V derives V from k
			}
			// Per-layer rope-table invariant (9a-P2, the live version of ropeResidentCompatible):
			// rope_kv rotates L.rhalf pairs per head reading invFreq[0..rhalf), so the bound
			// per-layer table MUST have exactly rhalf entries. Gemma 4's global (rhalf=headDim/2,
			// tail zero-freq) and local (full) tables genuinely differ in length now — the generic
			// finalizeRoPE check can't see this, so assert it per layer, loudly, at build.
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
		if maxQDim != nH*maxHd {
			return fmt.Errorf("cuda: scratch maxQDim=%d != nH*maxHd=%d*%d=%d — per-layer geometry inconsistent with the accessors", maxQDim, nH, maxHd, nH*maxHd)
		}
		r.x, r.aSc, r.aq = r.af(H), r.af(1), r.ai(H/4)
		r.qB, r.kB, r.vB = r.af(maxQDim), r.af(maxKVDim), r.af(maxKVDim)
		r.kc, r.vc = make([]Buffer, nLayers), make([]Buffer, nLayers)
		// r.ctxCap was resolved at construction (several earlier buffers size from it). The fit check
		// belongs HERE, though: it needs the per-layer kvDims, and running it immediately before the
		// caches are allocated is what makes `free` mean "what is actually left for KV".
		if e := r.checkKVFits(); e != nil {
			return e
		}
		for l := range r.kc {
			// Each layer's KV cache is sized by ITS OWN kvDim (Gemma 4's local 2048 vs global
			// 1024), matching the pos*Ly.kvDim stride launchToken indexes it with. Cross-file
			// invariant guard (the CUDA twin of the webgpu one — now non-tautological, since two
			// layers genuinely differ): the kvDim the cache is SIZED with must equal the one the
			// accessors derive and launchToken INDEXES with. A future edit that sized from a stale
			// model-level kvDim while the launch indexed per-layer would index off the end into
			// garbage output, not a panic — so fail loudly at plan time.
			if r.layers[l].isDeltaNet {
				// No KV cache for a recurrent mixer. Allocating one would be pure waste on the
				// family least able to afford it: 3 of every 4 layers are this kind.
				continue
			}
			if want := m.KVHeadsAtResident(l) * m.HeadDimAtResident(l); r.layers[l].kvDim != want {
				return fmt.Errorf("cuda: layer %d KV cache kvDim=%d != nKV*hd=%d (accessor-derived) — geometry/cache-size mismatch", l, r.layers[l].kvDim, want)
			}
			r.kc[l], r.vc[l] = r.af(r.ctxCap*r.layers[l].kvDim), r.af(r.ctxCap*r.layers[l].kvDim)
		}
		r.cctx, r.cSc, r.cq = r.af(maxQDim), r.af(1), r.ai(maxQDim/4)
		// K=V (attention_k_eq_v) layers derive V = v_norm(k) by reusing qk_norm with a UNIT weight
		// [maxHd] (so x*inv*w = x*inv, scale-less; addOne=0). Allocate it only when needed.
		for l := range r.layers {
			if r.layers[l].kEqV {
				ones := make([]float32, maxHd)
				for i := range ones {
					ones[i] = 1.0
				}
				r.vNormUnit = r.up32(ones)
				break
			}
		}
		r.oO = r.af(H)
		r.mSc, r.mq = r.af(1), r.ai(H/4)
		// DENSE-FFN scratch, and only if the model HAS a dense FFN. A model whose every layer is
		// routed reports intermediate_size 0 — Qwen3.6-35B-A3B's config omits the key entirely —
		// and these become 0-byte allocations, which this driver rejects as "invalid length". The
		// dense branch of segBFFN is unreachable for such a model, so the buffers are simply never
		// needed; allocating them anyway was the only thing standing between it and residency.
		//
		// Nothing hit this before because every previously-resident MoE (Mixtral, GLM, Mellum) has
		// a real intermediate_size — dense prefix layers or a genuine dense width. Note that the
		// tiny qwen3_5_moe fixture does NOT reproduce it either: it carries intermediate_size 128
		// because HF's config defaults one in, so the fixture is less pure-MoE than the model it
		// stands for. That is a fixture-fidelity gap, recorded rather than silently fixed here.
		if I > 0 {
			r.gO, r.uO = r.af(I), r.af(I)
			r.dSc, r.dScr, r.dq = r.af(1), r.af(I), r.ai(I/4)
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
		if r.gemma4Moe { // parallel dense‖MoE branch scratch + the host zero slice to clear x2
			r.g4x1, r.g4x2, r.g4rn = r.af(H), r.af(H), r.af(H)
			r.g4zero = make([]float32, H)
		}
		r.dO, r.logits = r.af(H), r.af(vocab)
		r.argIdx, r.argVal = r.ai(1), r.af(1) // greedy fast-path readback (4 B vs 594 KB)
		if hb, e := gpu.NewHostBuffer[float32](r.dev, vocab); e != nil {
			return e
		} else {
			r.logitsPinned, r.logitsHost = hb, hb.Slice()
		}
		// A9-FIX: pay the DEFERRED first-launch reservation BEFORE the free reading that sizes the
		// cache, so the cap is correct by construction rather than covered by a margin.
		//
		// moe_route declares per-thread local scratch (two float[MOE_MAX_E]), and the driver backs
		// local memory for the device's occupancy the first time that kernel runs — not at module
		// load, which costs a measured 0 B. On an RTX 2070 SUPER that launch DEMANDS 289,013,760 B
		// and RETAINS 138,412,032 B. Both were invisible here: allocSlots reads free VRAM before any
		// kernel has run, so it sized the cache against memory that was about to be taken.
		//
		// Forcing it here is strictly better than enlarging slotMarginBytes, which is the fix
		// everyone reaches for first. The peak is 2.09x the residual, and it is TRANSIENT: paying it
		// now, while ~3.8 GB is still free, means the free reading below sees only the 132 MiB that
		// is actually retained. A margin bump would have to reserve the 275.6 MiB peak permanently
		// to cover something needed for microseconds — and it would bury a named consumer inside an
		// unnamed constant, so the next kernel with per-thread scratch reopens it silently.
		//
		// WHY BY NAME IS SAFE HERE, given that naming one member of a set is the sibling-drift shape:
		// the backing store is SHARED and sized by the largest kernel, measured — launching the whole
		// census gives a threshold and residual identical to moe_route alone, to the byte. So forcing
		// the maximum forces the pool for every kernel. That moe_route IS the maximum is not assumed:
		// TestKernelLocalMemoryCensus enumerates every entry point in every embedded module and fails
		// if any other kernel declares more, naming this site.
		//
		// REGIME: `max` was measured with sequential single-stream launch, which is what goinfer
		// does. Concurrent streams would reopen whether the bound is max or a sum.
		if r.moe && r.cacheExperts {
			// nE=1, k=1, nGroup=1 does the least work the kernel can do. It writes rIdx[0]/rWgt[0],
			// which every real token overwrites before reading, and reads rLogits as both logits and
			// bias — allocated above, uninitialised, and never observed: the kernel's OUTPUT is
			// discarded, only its allocation side effect is wanted.
			if e := r.stream.Launch(r.fRoute, onecfg(1, 0),
				Arg(r.rLogits), Arg(r.rLogits), Arg(r.rIdx), Arg(r.rWgt),
				gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1)),
				gpu.ArgValue(int32(0)), gpu.ArgValue(float32(1)),
				gpu.ArgValue(int32(1)), gpu.ArgValue(int32(1))); e != nil {
				return fmt.Errorf("pre-sizing warm-up of moe_route failed: %w", e)
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
		// A silent decline is right for a shape this backend simply does not implement — the staged
		// path serves it correctly, just slower. It is WRONG when the operator explicitly asked for a
		// resident context that does not fit: they asked for a capability, we cannot provide it, and
		// degrading quietly to the staged path means they discover it as a latency mystery under
		// load. Surface that one as a hard startup error naming the GB (errKVWontFit); everything
		// else keeps the historical decline, so the default path is byte-for-byte unchanged.
		if errors.Is(setupErr, errKVWontFit) {
			return nil, false, setupErr
		}
		return declined(setupErr)
	}
	// The fused path needs every projection it reads as int4: fQKV reads Q/K/V, and fGU reads
	// gate/up as int4 + f16-scales (ws16). fuseQKV gated on Q/K/V ALONE, but it also switches on
	// fGU — so an int4-QKV + int8-gate/up checkpoint passed fGU a nil ws16 and crashed the executor
	// goroutine (no recover → process dies). Require Q/K/V int4 always, plus gate/up/down int4 on
	// dense-FFN layers — the only layers fGU runs on. MoE layers take moeMLP and never pack g/u/d
	// (kind==""), so requiring them there would wrongly strip fQKV from every MoE model. This is the
	// measured q4_k_m reality (all layer projections int4, only the LM head int8); anything else
	// falls back to the unfused chain, which handles mixed quant correctly (M23).
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
	if os.Getenv("GOINFER_CUDA_NO_FUSE") != "" {
		r.fuseQKV = false
	}
	// CUDA graphs (GOINFER_CUDA_GRAPHS): capture each layer's static launch segments now that fuseQKV
	// is final (segA/segB branch on it), so decode replays them instead of re-issuing the launches.
	// Incompatible with the g4cap diagnostic (it syncs inside a segment). Off ⇒ launchToken is
	// byte-identical to before. Promotion to "on" is safe-gated in admitGraphs (tenancy + self-test):
	// replay is bit-exact only under EXCLUSIVE_PROCESS tenancy or MPS, so a shared-GPU box under DEFAULT
	// declines to the live path rather than silently mis-running (docs/cuda-graphs-investigation.md).
	r.graphs = os.Getenv("GOINFER_CUDA_GRAPHS") != "" && !r.g4cap
	r.graphsSync = os.Getenv("GOINFER_CUDA_GRAPHS_SYNC") != "" // debug: serialize replays (bisect ordering hazards)
	r.graphMask = os.Getenv("GOINFER_CUDA_GRAPHS_ONLY")        // debug: replay only these segments (A/B/C), rest live
	if e := r.admitGraphs(); e != nil {
		r.Close()
		return declined(fmt.Errorf("cuda: %w", e))
	}
	b.resident = r
	return r, true, nil
}

// cudaBackend and cudaResident standardize teardown on Close() error and satisfy io.Closer, matching
// gpu (webgpuBackend/Context/Resident* — Release() kept only as deprecated aliases) and metal (audit
// B-12): one spelling of "free this GPU resource" across all three modules, so callers can write
// generic cleanup. The assertions make the contract compiler-enforced (a signature drift back to a
// no-error Close breaks the build, not a caller months later).
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
