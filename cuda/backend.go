//go:build cuda

package cuda

import (
	"fmt"
	"os"
	"runtime"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

func init() {
	decoder.RegisterBackend("cuda", func() (decoder.Backend, error) {
		return &cudaBackend{drv: stubDriver{}}, nil
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
	drv      driver
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
	if missing := m.MissingResidentFeatures(decoder.ResidentBackendFeatures["cuda"]); len(missing) > 0 {
		return declined(fmt.Errorf("arch needs unimplemented feature(s) %v", missing))
	}

	H, nLayers, nH, nKV, hd, I, vocab := m.Dims()

	// ---- host pack all weights (CPU; any incompatible shape → decline) ----
	type hlayer struct {
		q, k, v, o, g, u, d hostW
		qb, kb, vb          []float32
		preNorm, postNorm   []float32
		qNorm, kNorm        []float32
		window              int32
		hasBias             bool
	}
	hls := make([]hlayer, nLayers)
	for l := 0; l < nLayers; l++ {
		lw := &w.Layers[l]
		var hl hlayer
		for _, p := range []struct {
			dst *hostW
			src *linalg.WeightMat
		}{
			{&hl.q, &lw.QProj}, {&hl.k, &lw.KProj}, {&hl.v, &lw.VProj}, {&hl.o, &lw.OProj},
			{&hl.g, &lw.GateProj}, {&hl.u, &lw.UpProj}, {&hl.d, &lw.DownProj},
		} {
			hw, e := packWeight(p.src)
			if e != nil {
				return declined(e)
			}
			*p.dst = hw
		}
		hl.preNorm, hl.postNorm = lw.PreAttnNorm, lw.PreMLPNorm
		hl.qNorm, hl.kNorm = lw.QNorm, lw.KNorm
		// Per-layer window: only LOCAL layers are windowed; global layers stay full causal.
		if m.LayerIsLocalResident(l) {
			hl.window = int32(m.SlidingWindowResident())
		}
		if m.HasQKNorm() && (len(hl.qNorm) != hd || len(hl.kNorm) != hd) {
			return declined(fmt.Errorf("layer %d: arch claims QK-norm but QNorm/KNorm are not len==headDim(%d)", l, hd))
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
		hidden: H, nLayers: nLayers, nH: nH, nKV: nKV, hd: hd, inter: I, vocab: vocab,
		qDim: nH * hd, kvDim: nKV * hd, half: hd / 2,
		eps: m.NormEps(), attnScale: m.AttnScale(),
		qkNorm: m.HasQKNorm(), rmsAddOne: m.RMSAddOne(),
	}
	kvDim := r.kvDim
	invFreq := m.RopeInvFreq()
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
		if e := gc.Init(); e != nil {
			return e
		}
		dev, e := gc.GetDevice(0)
		if e != nil {
			return e
		}
		if r.cx, e = dev.Primary(); e != nil {
			return e
		}
		gmod, e := r.cx.LoadModule(gemvFwdPTX)
		if e != nil {
			return e
		}
		glmod, e := r.cx.LoadModule(gluePTX)
		if e != nil {
			return e
		}
		if r.gemvW4, e = gmod.Function("gemv_w4a8_fwd"); e != nil {
			return e
		}
		if r.gemvW8, e = gmod.Function("gemv_w8a8_fwd"); e != nil {
			return e
		}
		if r.kvStore, e = gmod.Function("kv_store"); e != nil {
			return e
		}
		if r.ropeKV, e = gmod.Function("rope_kv"); e != nil {
			return e
		}
		qmod, e2 := r.cx.LoadModule(fusedQKVPTX)
		if e2 != nil {
			return e2
		}
		if r.fQKV, e = qmod.Function("fused_rms_qkv"); e != nil {
			return e
		}
		if r.fGU, e = qmod.Function("fused_rms_gu"); e != nil {
			return e
		}
		if r.fQKN, e = qmod.Function("qk_norm"); e != nil {
			return e
		}
		fns := []struct {
			dst  **gc.Function
			name string
		}{
			{&r.fRms, "rmsnorm_quant"}, {&r.fQ, "quant_vec"}, {&r.fRope, "rope"},
			{&r.fAttn, "attention"}, {&r.fSw, "swiglu_quant"}, {&r.fRes, "residual"},
			{&r.fArg, "argmax_reduce"},
		}
		for _, f := range fns {
			if *f.dst, e = glmod.Function(f.name); e != nil {
				return e
			}
		}
		if r.stream, e = r.cx.NewStream(); e != nil {
			return e
		}

		r.layers = make([]cudaLayer, nLayers)
		for l := 0; l < nLayers; l++ {
			h := &hls[l]
			L := cudaLayer{
				q: r.upW(h.q), k: r.upW(h.k), v: r.upW(h.v), o: r.upW(h.o),
				g: r.upW(h.g), u: r.upW(h.u), d: r.upW(h.d),
				preNorm: r.up32(h.preNorm), postNorm: r.up32(h.postNorm),
			}
			if h.hasBias {
				L.qb, L.kb, L.vb, L.hasBias = r.up32(h.qb), r.up32(h.kb), r.up32(h.vb), true
			}
			if r.qkNorm {
				L.qNorm, L.kNorm = r.up32(h.qNorm), r.up32(h.kNorm)
			}
			L.window = h.window
			r.layers[l] = L
		}
		r.lmW = r.upW(hlm)
		r.finalNorm = r.up32(hFinal)
		r.invF = r.up32(invFreq)

		r.x, r.aSc, r.aq = r.af(H), r.af(1), r.ai(H/4)
		r.qB, r.kB, r.vB = r.af(r.qDim), r.af(kvDim), r.af(kvDim)
		r.kc, r.vc = make([]*gc.Buffer[float32], nLayers), make([]*gc.Buffer[float32], nLayers)
		for l := range r.kc {
			r.kc[l], r.vc[l] = r.af(cudaCtxCap*kvDim), r.af(cudaCtxCap*kvDim)
		}
		r.cctx, r.cSc, r.cq = r.af(r.qDim), r.af(1), r.ai(r.qDim/4)
		r.oO = r.af(H)
		r.mSc, r.mq = r.af(1), r.ai(H/4)
		r.gO, r.uO = r.af(I), r.af(I)
		r.dSc, r.dScr, r.dq = r.af(1), r.af(I), r.ai(I/4)
		r.dO, r.logits = r.af(H), r.af(vocab)
		r.argIdx, r.argVal = r.ai(1), r.af(1) // greedy fast-path readback (4 B vs 594 KB)
		if hb, e := gc.AllocHost[float32](r.cx, vocab); e != nil {
			return e
		} else {
			r.logitsPinned, r.logitsHost = hb, hb.Slice()
		}
		return r.setupErr
	})
	if setupErr != nil {
		r.Close()
		return declined(setupErr)
	}
	// K1 needs every layer's Q/K/V to be int4 (the measured reality for q4_k_m: all layer
	// projections are int4, only the LM head is int8). Anything else falls back to the
	// unfused chain, so a mixed checkpoint stays correct.
	r.fuseQKV = true
	for l := range hls {
		if hls[l].q.kind != "int4" || hls[l].k.kind != "int4" || hls[l].v.kind != "int4" {
			r.fuseQKV = false
			break
		}
	}
	if os.Getenv("GOINFER_CUDA_NO_FUSE") != "" {
		r.fuseQKV = false
	}
	b.resident = r
	return r, true, nil
}

func (b *cudaBackend) Close() error {
	if b.resident != nil {
		_ = b.resident.Close()
	}
	return b.drv.Close()
}
