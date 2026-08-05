//go:build cuda

package cuda

import (
	"fmt"
	"io"
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

	declined := func(e error) (decoder.ResidentForward, bool, error) {
		if os.Getenv("GOINFER_RESIDENT_DEBUG") != "" {
			fmt.Fprintf(os.Stderr, "[cuda] BuildResident declined: %v\n", e)
		}
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
	// sharedUngated is intentionally dropped (_): CUDA admits only the UNGATED shared expert
	// (the gated one is declined upstream by the FeatMoEGatedShared admission check), so the
	// build never needs to branch on it.
	nE, topK, moeInter, sharedInter, moeSig, moeNorm, _, moeScale, nGroup, topkGroup, isMoE := m.MoEResidentParams()
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
		case nE > 256:
			return declined(fmt.Errorf("MoE nE=%d exceeds moe_route's MOE_MAX_E=256", nE))
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
		if nE > 256 {
			return declined(fmt.Errorf("gemma4 MoE nE=%d exceeds moe_route's MOE_MAX_E=256", nE))
		}
	}

	// ---- host pack all weights (CPU; any incompatible shape → decline) ----
	type hlayer struct {
		q, k, v, o, g, u, d       hostW
		qb, kb, vb                []float32
		preNorm, postNorm         []float32
		qNorm, kNorm              []float32
		postAttnNorm, postMLPNorm []float32 // Gemma sandwich (nil unless the arch declares it)
		invFreq                   []float32 // per-layer RoPE table
		window                    int32
		hasBias                   bool

		// MoE, per layer: GLM/DeepSeek run dense for the first FirstKDense layers and route
		// after, so this is keyed off the layer's own Experts (as decoder/mlp.go does), not
		// off the arch.
		isMoE            bool
		expGU, expDown   hostW
		router, routerBs []float32
		hasShared        bool
		shGU, shDown     hostW

		// Gemma-4 enable_moe_block (parallel dense‖MoE). g4moe routes the upload to gemma4MoeMLP's
		// fields: the dense branch reuses g/u/d (packed from mlpGate/up/down), the router reuses
		// router (RouterProjScaled, f32) + routerBs (zeros), the experts reuse expGU/expDown.
		g4moe                                               bool
		g4preFFN, g4postFFN1, g4preFFN2, g4postFFN2, g4post []float32
		perExpertScale                                      []float32
		layerScalar                                         float32
	}
	sandwich := m.SandwichNormResident()
	hls := make([]hlayer, nLayers)
	for l := 0; l < nLayers; l++ {
		lw := &w.Layers[l]
		var hl hlayer
		hl.isMoE = isMoE && lw.Experts != nil // same key as decoder/mlp.go; false on dense prefix layers
		g4b, isG4 := m.Gemma4MoEResidentLayer(l)
		hl.g4moe = isG4
		proj := []struct {
			dst *hostW
			src *linalg.WeightMat
		}{
			{&hl.q, &lw.QProj}, {&hl.k, &lw.KProj}, {&hl.o, &lw.OProj},
		}
		if !m.VFromKResident(l) { // K=V (attention_k_eq_v) global layers carry NO v_proj — V=v_norm(k)
			proj = append(proj, struct {
				dst *hostW
				src *linalg.WeightMat
			}{&hl.v, &lw.VProj})
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
			}
		}
		hl.preNorm, hl.postNorm = lw.PreAttnNorm, lw.PreMLPNorm
		hl.qNorm, hl.kNorm = lw.QNorm, lw.KNorm
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
		hl.invFreq = m.RopeInvFreqLayerResident(l) // Gemma 4: real per-layer table (gemma4InvFreq), not the generic one
		// Per-layer window: only LOCAL layers are windowed; global layers stay full causal.
		if m.LayerIsLocalResident(l) {
			hl.window = int32(m.SlidingWindowResident())
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
		cacheExperts: os.Getenv("GOINFER_MOE_CACHE_EXPERTS") != "",
	}
	if moeSig {
		r.moeSigmoid = 1
	}
	if moeNorm {
		r.moeNormTopK = 1
	}
	// C′ step 2: device slots per layer. Default topK (step-1 fresh-load, no cross-token reuse,
	// byte-identical); GOINFER_MOE_CACHE_SLOTS=N gives an LRU cache of N slots (clamped [topK, nE]).
	// VRAM for slots is nLayers·nSlots·perExpert, so a bigger N trades VRAM for fewer per-token DMAs.
	// CAUTION: too large OOMs at BuildResident (the alloc runs in the executor goroutine, so it
	// crashes rather than declining) — keep N within the free-VRAM budget (~40 for the 26B on 8 GB).
	r.cacheSlots = topK
	if r.cacheExperts {
		if v, err := strconv.Atoi(os.Getenv("GOINFER_MOE_CACHE_SLOTS")); err == nil && v > topK {
			r.cacheSlots = v
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
			r.ackCh <- j()
		}
	}()

	// setup job: create the context on the pinned thread, JIT kernels, upload everything.
	setupErr := r.do(func() error {
		var e error
		if r.dev, e = CreateSystemDefaultDevice(); e != nil {
			return e
		}
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
		if r.kvStore, e = r.dev.NewComputePipeline(gmod, "kv_store"); e != nil {
			return e
		}
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
			{&r.fRms, "rmsnorm_quant"}, {&r.fRmsF32, "rmsnorm_f32"}, {&r.fQ, "quant_vec"}, {&r.fRope, "rope"},
			{&r.fAttn, "attention"}, {&r.fSw, "glu_quant"}, {&r.fRes, "residual"},
			{&r.fArg, "argmax_reduce"},
		}
		for _, f := range fns {
			if *f.dst, e = r.dev.NewComputePipeline(glmod, f.name); e != nil {
				return e
			}
		}
		// Batched prefill kernels (weight-stationary M=len path). Own module; the audited PTX is
		// untouched. bGemv comes from gemv_w4a8_batched.ptx, the rest from prefill_batched.ptx.
		if bgmod, e2 := r.dev.CompileLibrary(gemvBatchedPTX); e2 == nil {
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
				load(&r.bGemv, bgmod, "gemv_w4a8_batched")
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
					r.skScoreBuf = r.af(r.nH * cudaCtxCap)
					r.skInvBuf = r.af(r.nH)
					// Default ON (bit-identical; gated at runtime on nKeys≥splitkvMinKeys so shallow
					// decode is unaffected). GOINFER_SPLITKV_ATTN=0 force-disables it (A/B / rollback).
					r.splitkvAttn = os.Getenv("GOINFER_SPLITKV_ATTN") != "0"
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

		r.layers = make([]cudaLayer, nLayers)
		for l := 0; l < nLayers; l++ {
			h := &hls[l]
			L := cudaLayer{
				q: r.upW(h.q), k: r.upW(h.k), o: r.upW(h.o), // v uploaded below only for non-K=V layers
				preNorm: r.up32(h.preNorm), postNorm: r.up32(h.postNorm),
				invF: r.up32(h.invFreq),
			}
			if !h.isMoE {
				// A routed layer has no dense FFN to upload: its hostW's are empty, and
				// Alloc(0) is an error rather than a harmless no-op.
				L.g, L.u, L.d = r.upW(h.g), r.upW(h.u), r.upW(h.d)
			}
			if r.sandwich {
				L.postAttnNorm, L.postMLPNorm = r.up32(h.postAttnNorm), r.up32(h.postMLPNorm)
			}
			if h.hasBias {
				L.qb, L.kb, L.vb, L.hasBias = r.up32(h.qb), r.up32(h.kb), r.up32(h.vb), true
			}
			if r.qkNorm {
				L.qNorm, L.kNorm = r.up32(h.qNorm), r.up32(h.kNorm)
			}
			if h.isMoE {
				L.isMoE = true
				L.routerW, L.routerB = r.up32(h.router), r.up32(h.routerBs)
				L.expGU, L.expDown = r.upExperts(h.expGU), r.upExperts(h.expDown)
				if h.hasShared {
					L.hasShared = true
					L.shGU, L.shDown = r.upW(h.shGU), r.upW(h.shDown)
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
		if maxQDim != nH*maxHd {
			return fmt.Errorf("cuda: scratch maxQDim=%d != nH*maxHd=%d*%d=%d — per-layer geometry inconsistent with the accessors", maxQDim, nH, maxHd, nH*maxHd)
		}
		r.x, r.aSc, r.aq = r.af(H), r.af(1), r.ai(H/4)
		r.qB, r.kB, r.vB = r.af(maxQDim), r.af(maxKVDim), r.af(maxKVDim)
		r.kc, r.vc = make([]Buffer, nLayers), make([]Buffer, nLayers)
		for l := range r.kc {
			// Each layer's KV cache is sized by ITS OWN kvDim (Gemma 4's local 2048 vs global
			// 1024), matching the pos*Ly.kvDim stride launchToken indexes it with. Cross-file
			// invariant guard (the CUDA twin of the webgpu one — now non-tautological, since two
			// layers genuinely differ): the kvDim the cache is SIZED with must equal the one the
			// accessors derive and launchToken INDEXES with. A future edit that sized from a stale
			// model-level kvDim while the launch indexed per-layer would index off the end into
			// garbage output, not a panic — so fail loudly at plan time.
			if want := m.KVHeadsAtResident(l) * m.HeadDimAtResident(l); r.layers[l].kvDim != want {
				return fmt.Errorf("cuda: layer %d KV cache kvDim=%d != nKV*hd=%d (accessor-derived) — geometry/cache-size mismatch", l, r.layers[l].kvDim, want)
			}
			r.kc[l], r.vc[l] = r.af(cudaCtxCap*r.layers[l].kvDim), r.af(cudaCtxCap*r.layers[l].kvDim)
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
		r.gO, r.uO = r.af(I), r.af(I)
		r.dSc, r.dScr, r.dq = r.af(1), r.af(I), r.ai(I/4)
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
		// C′: the core + KV + scratch are now up, so free VRAM reflects them — size the expert slot
		// cache to what actually fits (cap-and-log, never OOM) and allocate it.
		if e := r.allocSlots(); e != nil {
			return e
		}
		return r.setupErr
	})
	if setupErr != nil {
		r.Close()
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
