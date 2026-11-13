//go:build gpu

//lint:file-ignore SA1019 Release is Close's deprecated alias, called here as before; it is deprecated for new code only.

package webgpu

import (
	"errors"
	"fmt"
	"math"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/backend/webgpu/internal/kernels"
	"github.com/townsendmerino/goinfer/decoder"
)

// newRunLayer is layer i's RunLayer before any upload: its sliding-window flag, RoPE scale, scalar, and the per-layer attention
// geometry when it differs from the model's.
func (bs *webgpuBuild) newRunLayer(i int) kernels.RunLayer {
	m := bs.m
	lhd := m.HeadDimAtResident(i)
	lnKV := m.KVHeadsAtResident(i)
	lrot := m.RotaryDimAtResident(i)
	lrhalf := lrot / 2
	kEqV := m.VFromKResident(i)
	rl := kernels.RunLayer{
		IsLocal:     m.LayerIsLocalResident(i),     // sliding-window layer
		RopeScale:   float32(m.RopeMscaleLayer(i)), // per-layer YaRN mscale
		LayerScalar: m.Gemma4DenseLayerScalarAtResident(i),
		VNorm:       m.IsGemma4Resident() && !kernels.G4DropVNormForTest,
	}
	if kernels.G4DropLayerScalarForTest {
		rl.LayerScalar = 1
	}
	if lhd != bs.hd || lnKV != bs.nKV || lrhalf != bs.half || kEqV {
		rl.Ghd = lhd
		rl.GnKV = lnKV
		rl.Ghalf = lrhalf
		rl.GKEqV = kEqV
	}
	return rl
}

// nemotronLayer uploads Nemotron-H layer i: one pre-norm and exactly one op (Mamba-2 mixer, NoPE attention, relu² MLP or MoE),
// selected by the block kind. A kind this runner does not implement is an error, not a layer of nil weights.
func (bs *webgpuBuild) nemotronLayer(i int, lw *decoder.LayerWeights, rl *kernels.RunLayer) error {
	m := bs.m
	rd := bs.rd
	var e error
	// Nemotron-H single-op-per-block: ONE pre-norm (PreAttnNorm) + exactly one op +
	// residual. No FFN pairing, no multipliers. The resident layer-loop branches on
	// rl.nemoKind (mamba / NoPE-attn / relu²-MLP). int8 mamba projections (projF32);
	// attn/MLP projections take the model's native precision via proj (W8A8 or W4A8).
	if rl.AttnNorm, e = bs.up32(lw.PreAttnNorm); e != nil {
		return e
	}
	rl.MlpNorm = rl.AttnNorm // the relu²-MLP block reuses the single pre-norm
	mp := rd.rm.Mamba
	switch m.NemotronBlockKind(i) {
	case 0: // Mamba-2 mixer
		rl.NemoKind = kernels.NemoKMamba
		rl.IsMamba = true
		inP, cW, cB, aLog, dW, dtB, nW, outP := m.GraniteMambaWeights(i)
		if inQ, outQ := m.MambaProjInt8(i); inQ != nil { // a quantized load holds them as int8: projF32's codes
			if rl.MambaInProj, e = bs.projInt8(inQ, mp.ProjDim, bs.hidden); e != nil {
				return e
			}
			if rl.MambaOutProj, e = bs.projInt8(outQ, bs.hidden, mp.DInner); e != nil {
				return e
			}
		} else {
			if rl.MambaInProj, e = bs.projF32(inP, mp.ProjDim, bs.hidden); e != nil {
				return e
			}
			if rl.MambaOutProj, e = bs.projF32(outP, bs.hidden, mp.DInner); e != nil {
				return e
			}
		}
		if rl.MambaConvW, e = bs.up32(cW); e != nil {
			return e
		}
		if rl.MambaConvB, e = bs.up32(cB); e != nil {
			return e
		}
		headP := make([]float32, mp.NHeads*3)
		for h := 0; h < mp.NHeads; h++ {
			headP[h*3+0] = float32(-math.Exp(float64(aLog[h])))
			headP[h*3+1] = dtB[h]
			headP[h*3+2] = dW[h]
		}
		if rl.MambaHeadP, e = bs.up32(headP); e != nil {
			return e
		}
		if rl.MambaNormW, e = bs.up32(nW); e != nil {
			return e
		}
		if rl.MambaWin, e = bs.stateBuf((mp.DConv - 1) * mp.ConvDim); e != nil {
			return e
		}
		if rl.MambaSSM, e = bs.stateBuf(mp.NHeads * mp.Hp * mp.Dn); e != nil {
			return e
		}
	case 1: // NoPE GQA attention (identity rope via the zeroed invFreq)
		rl.NemoKind = kernels.NemoKAttn
		rl.InvFreq = bs.nemoZeroInvFreq
		kc, e1 := bs.c.NewKVCache(nil, bs.ctxCap*bs.kvDim)
		vc, e2 := bs.c.NewKVCache(nil, bs.ctxCap*bs.kvDim)
		if e1 != nil || e2 != nil {
			return fmt.Errorf("gpu: nemotron KV alloc (layer %d): %w", i, errors.Join(e1, e2))
		}
		bs.keep(kc.Release)
		bs.keep(vc.Release)
		rl.KCache, rl.VCache = (*kernels.DeviceBufferBuf(kc)), (*kernels.DeviceBufferBuf(vc))
		if rl.Q, e = bs.proj(&lw.QProj); e != nil {
			return e
		}
		if rl.K, e = bs.proj(&lw.KProj); e != nil {
			return e
		}
		if rl.V, e = bs.proj(&lw.VProj); e != nil {
			return e
		}
		if rl.O, e = bs.proj(&lw.OProj); e != nil {
			return e
		}
	case 2: // non-gated relu² MLP
		rl.NemoKind = kernels.NemoKMLP
		if rl.Up, e = bs.proj(&lw.UpProj); e != nil {
			return e
		}
		if rl.Down, e = bs.proj(&lw.DownProj); e != nil {
			return e
		}
	case 3: // MoE FFN
		// rd.rm.moe is already populated MODEL-level by MoEResidentParams (it reads arch.MoE generically, which Nemotron also sets);
		// this is the PER-LAYER weight build. There is no expGate/shGate/shGateW: this family's experts, and its shared expert, have
		// only up_proj/down_proj (see decoder/forward_nemotron.go).
		rl.NemoKind = kernels.NemoKMoE
		rl.IsMoE = true
		if rl.Router, e = bs.proj(&lw.Router); e != nil {
			return e
		}
		if len(lw.RouterBias) > 0 {
			if rl.RouterBias, e = bs.up32(lw.RouterBias); e != nil {
				return e
			}
		}
		if rl.ExpUp, e = bs.buildStacked(len(lw.Experts), func(x int) *linalg.WeightMat { return &lw.Experts[x].Up }, 1); e != nil {
			return e
		}
		bs.keep(rl.ExpUp.Release)
		if rl.ExpDown, e = bs.buildStacked(len(lw.Experts), func(x int) *linalg.WeightMat { return &lw.Experts[x].Down }, 1); e != nil {
			return e
		}
		bs.keep(rl.ExpDown.Release)
		// Gated on shInter > 0, like the generic branch below: n_shared_experts == 0 is a valid config (ValidateNemotron in
		// decoder/config's Config.ValidateNemotron rejects only the opposite combination), and the CPU path in decoder/forward_nemotron.go gates the add on
		// moe.SharedIntermediateDim > 0. Projecting a zero-value SharedExpert would fail with "unsupported projection precision" and
		// fall back to CPU with the wrong stated reason.
		if bs.shInter > 0 {
			if rl.ShUp, e = bs.proj(&lw.SharedExpert.Up); e != nil {
				return e
			}
			if rl.ShDown, e = bs.proj(&lw.SharedExpert.Down); e != nil {
				return e
			}
		}
	default:
		// No default would let an unhandled block kind append a layer with NIL weights, and decoderunner's gemv would
		// nil-dereference on the first token. The nemotron bypass above skips DecodeRunnerEligible, which decoder/residency.go calls
		// "the one predicate every backend's admission funnels through", so this switch is the last place the shape is checked.
		// Reachable through the exported ResidencyBackend.BuildResident.
		return fmt.Errorf("gpu: nemotron block kind %d at layer %d is not "+
			"implemented by this runner", m.NemotronBlockKind(i), i)
	}
	return nil
}

// deltaNetMixer uploads layer i's Gated-DeltaNet mixer into rl.
func (bs *webgpuBuild) deltaNetMixer(i int, rl *kernels.RunLayer) error {
	m := bs.m
	rd := bs.rd
	var e error
	// Gated-DeltaNet mixer layer. No KV cache and no q/k/v/o: the recurrence's state is the whole history, fixed-size and NOT
	// position-truncatable (which is why this family declines prefix reuse and batched ForwardN). The conv reuses the Mamba-2
	// ring window; DeltaNet's conv is bias-free, so every layer binds one shared zeroed convB rather than allocating an
	// all-zero bias per layer.
	rl.IsDeltaNet = true
	dp := rd.rm.Dnet
	qkv, z, outP, inB, inA, cW, dtB, negA, nW := m.Qwen35DeltaWeights(i)
	if rl.DnQKV, e = bs.proj(qkv); e != nil {
		return e
	}
	if rl.DnZ, e = bs.proj(z); e != nil {
		return e
	}
	if rl.DnOut, e = bs.proj(outP); e != nil {
		return e
	}
	// inB/inA are f32 on the CPU by deliberate choice (they feed the write/decay gates,
	// where the recurrence is most precision-sensitive). projF32 quantizes them to W8A8
	// for the resident path — the one place this port is coarser than the reference, and
	// the reason the parity gate scores the gates as their own stage.
	if rl.DnB, e = bs.projF32(inB, dp.Nv, bs.hidden); e != nil {
		return e
	}
	if rl.DnA, e = bs.projF32(inA, dp.Nv, bs.hidden); e != nil {
		return e
	}
	rl.MambaConvB = bs.dnZeroConvB
	if rl.MambaConvW, e = bs.up32(cW); e != nil {
		return e
	}
	if rl.DnDtBias, e = bs.up32(dtB); e != nil {
		return e
	}
	if rl.DnNegExpA, e = bs.up32(negA); e != nil { // ALREADY -exp(A_log); do not re-exp
		return e
	}
	if rl.DnNormW, e = bs.up32(nW); e != nil {
		return e
	}
	if rl.MambaWin, e = bs.stateBuf((dp.ConvK - 1) * dp.ConvDim); e != nil {
		return e
	}
	if rl.DnState, e = bs.stateBuf(dp.StateElems); e != nil {
		return e
	}
	return nil
}

// gatedAttnMixer uploads layer i's gated softmax attention (qwen3_5, qwen3_5_moe, qwen3_next) into rl.
func (bs *webgpuBuild) gatedAttnMixer(i int, lw *decoder.LayerWeights, rl *kernels.RunLayer) error {
	m := bs.m
	var e error
	// Gated SOFTMAX attention layer of the same family (qwen3_5, qwen3_5_moe, qwen3_next): ordinary GQA + per-head QK-norm
	// + partial RoPE, plus the output gate. q_proj stays fused at double width because it is quantized; the runner splits
	// the activation (runLayer.qGate). Olmo Hybrid's full-attention layer is plain (dnAttnGate=false) and takes the
	// generic branch below.
	rl.QGate = true
	kc, e1 := bs.c.NewKVCache(nil, bs.ctxCap*bs.kvDim)
	vc, e2 := bs.c.NewKVCache(nil, bs.ctxCap*bs.kvDim)
	if e1 != nil || e2 != nil {
		return fmt.Errorf("gpu: qwen35 KV alloc (layer %d): %w", i, errors.Join(e1, e2))
	}
	bs.keep(kc.Release)
	bs.keep(vc.Release)
	rl.KCache, rl.VCache = (*kernels.DeviceBufferBuf(kc)), (*kernels.DeviceBufferBuf(vc))
	qP, kP, vP, oP, qN, kN := m.Qwen35AttnWeights(i)
	if rl.Q, e = bs.proj(qP); e != nil {
		return e
	}
	if rl.K, e = bs.proj(kP); e != nil {
		return e
	}
	if rl.V, e = bs.proj(vP); e != nil {
		return e
	}
	if rl.O, e = bs.proj(oP); e != nil {
		return e
	}
	if rl.QNorm, e = bs.up32(qN); e != nil {
		return e
	}
	if rl.KNorm, e = bs.up32(kN); e != nil {
		return e
	}
	return nil
}

// graniteMambaMixer uploads layer i's Granite Mamba-2 mixer into rl.
func (bs *webgpuBuild) graniteMambaMixer(i int, rl *kernels.RunLayer) error {
	m := bs.m
	rd := bs.rd
	var e error
	// Mamba-2 SSM mixer layer: in/out_proj W8A8, conv/headP/normW f32, build-once {win, ssm} state. headP interleaves the
	// per-head [Aexp=-exp(aLog), dtBias, D]. ResidMul folds into out_proj. No KV cache, no q/k/v/o (the plan branches on
	// isMamba).
	rl.IsMamba = true
	mp := rd.rm.Mamba
	inP, cW, cB, aLog, dW, dtB, nW, outP := m.GraniteMambaWeights(i)
	// int8 projections: f16 projections do not recover the quality lost to the SSM's f32 (vs f64) exp feeding the router
	// (docs/ssm-int8-quality.md).
	if rl.MambaInProj, e = bs.projF32(inP, mp.ProjDim, bs.hidden); e != nil {
		return e
	}
	if rl.MambaOutProj, e = bs.projF32Mul(outP, bs.hidden, mp.DInner, bs.rmul); e != nil {
		return e
	}
	if rl.MambaConvW, e = bs.up32(cW); e != nil {
		return e
	}
	if rl.MambaConvB, e = bs.up32(cB); e != nil {
		return e
	}
	headP := make([]float32, mp.NHeads*3)
	for h := 0; h < mp.NHeads; h++ {
		headP[h*3+0] = float32(-math.Exp(float64(aLog[h])))
		headP[h*3+1] = dtB[h]
		headP[h*3+2] = dW[h]
	}
	if rl.MambaHeadP, e = bs.up32(headP); e != nil {
		return e
	}
	if rl.MambaNormW, e = bs.up32(nW); e != nil {
		return e
	}
	if rl.MambaWin, e = bs.stateBuf((mp.DConv - 1) * mp.ConvDim); e != nil {
		return e
	}
	if rl.MambaSSM, e = bs.stateBuf(mp.NHeads * mp.Hp * mp.Dn); e != nil {
		return e
	}
	return nil
}

// mlaMixer uploads layer i's latent attention (one compressed-latent cache in place of the K and V caches) into rl.
func (bs *webgpuBuild) mlaMixer(i int, rl *kernels.RunLayer) error {
	m := bs.m
	var e error
	// MLA: a single compressed-latent cache [ctxCap·latDim] replaces the per-head
	// K/V caches; the per-head K/V are rebuilt in rank-space, never materialized.
	latDim := bs.kvLoRA + bs.qkRope
	lc, e1 := bs.c.NewKVCache(nil, bs.ctxCap*latDim)
	if e1 != nil {
		return fmt.Errorf("gpu: residency latent alloc (layer %d): %w", i, e1)
	}
	bs.keep(lc.Release)
	rl.LatCache = (*kernels.DeviceBufferBuf(lc))
	qA, qANorm, qB, qProj, kvA, kvANorm, kvB, oProj := m.MLALayerWeights(i)
	qAQ, qBQ, qQ, kvAQ, oQ := m.MLALayerInt8(i)
	// mlaProj uploads a projection's int8 form when the load holds one (the codes projF32 makes from the f32), else the f32.
	mlaProj := func(q *linalg.WeightMat, f []float32, N, K int) (kernels.DecodeWeight, error) {
		if q.Rows() > 0 {
			return bs.projInt8(q, N, K)
		}
		return bs.projF32(f, N, K)
	}
	if bs.qLoRA > 0 { // q_a → norm → q_b LoRA bottleneck
		if rl.MlaQA, e = mlaProj(qAQ, qA, bs.qLoRA, bs.hidden); e != nil {
			return e
		}
		if rl.MlaQB, e = mlaProj(qBQ, qB, bs.nH*(bs.qkNope+bs.qkRope), bs.qLoRA); e != nil {
			return e
		}
		if rl.MlaQANorm, e = bs.up32(qANorm); e != nil {
			return e
		}
	} else { // direct q_proj (V2-Lite)
		if rl.MlaQ, e = mlaProj(qQ, qProj, bs.nH*(bs.qkNope+bs.qkRope), bs.hidden); e != nil {
			return e
		}
	}
	if rl.MlaKVA, e = mlaProj(kvAQ, kvA, latDim, bs.hidden); e != nil {
		return e
	}
	if rl.MlaKVANorm, e = bs.up32(kvANorm); e != nil {
		return e
	}
	if rl.MlaO, e = mlaProj(oQ, oProj, bs.hidden, bs.nH*bs.vHead); e != nil {
		return e
	}
	// kvB is [nH*(qkNope+vHead), kvLoRA] row-major (per head: k_nope rows ‖ v rows).
	// Slice into W_UKᵀ [nH, kvLoRA, qkNope] (transposed for the absorb GEMV) and
	// W_UV [nH, vHead, kvLoRA] (the lift, used as-is).
	hRow := bs.qkNope + bs.vHead
	wuk := make([]float32, bs.nH*bs.kvLoRA*bs.qkNope)
	wuv := make([]float32, bs.nH*bs.vHead*bs.kvLoRA)
	for h := range bs.nH {
		for d := range bs.qkNope {
			src := kvB[(h*hRow+d)*bs.kvLoRA : (h*hRow+d)*bs.kvLoRA+bs.kvLoRA]
			for cc := range bs.kvLoRA {
				wuk[(h*bs.kvLoRA+cc)*bs.qkNope+d] = src[cc] // transpose: [c][d] ← kvB[d][c]
			}
		}
		for ev := range bs.vHead {
			copy(wuv[(h*bs.vHead+ev)*bs.kvLoRA:(h*bs.vHead+ev)*bs.kvLoRA+bs.kvLoRA], kvB[(h*hRow+bs.qkNope+ev)*bs.kvLoRA:(h*hRow+bs.qkNope+ev)*bs.kvLoRA+bs.kvLoRA])
		}
	}
	if rl.MlaWUK, e = bs.up32(wuk); e != nil {
		return e
	}
	if rl.MlaWUV, e = bs.up32(wuv); e != nil {
		return e
	}
	return nil
}

// attnMixer uploads layer i's generic GQA attention into rl: its K and V caches, or the source layer's when the layer shares K/V,
// and the q/k/v/o projections.
func (bs *webgpuBuild) attnMixer(i int, lw *decoder.LayerWeights, rl *kernels.RunLayer) error {
	m := bs.m
	rd := bs.rd
	var e error
	// Per-layer KV geometry: each layer's cache is sized by its own nKV·hd, not a model-level kvDim. A layer without a
	// geometry override (rl.ghd == 0) falls back to the model values.
	lnKV, lhd, lkvDim := bs.nKV, bs.hd, bs.kvDim
	if rl.Ghd != 0 {
		lnKV, lhd, lkvDim = rl.GnKV, rl.Ghd, rl.GnKV*rl.Ghd
	}
	// A Gemma 4 E-model's tail layers attend over an earlier layer's K/V (Model.KVSrcAtResident): such a layer owns no
	// cache, binds its source's, and projects Q only.
	kvSrc := m.KVSrcAtResident(i)
	if kernels.G4KVSrcOffForTest && kvSrc != i {
		for j := kvSrc - 1; j >= 0; j-- {
			if m.LayerIsLocalResident(j) == m.LayerIsLocalResident(i) {
				kvSrc = j
				break
			}
		}
	}
	if kvSrc != i {
		if kvSrc < 0 || kvSrc >= len(rd.rm.Layers) {
			return fmt.Errorf("gpu: layer %d shares the K/V of layer %d, which is not built before it", i, kvSrc)
		}
		sl := rd.rm.Layers[kvSrc]
		if sl.KCache == nil || sl.Ghd != rl.Ghd || sl.GnKV != rl.GnKV || sl.GKEqV != rl.GKEqV {
			return fmt.Errorf("gpu: layer %d shares the K/V of layer %d, whose attention geometry differs", i, kvSrc)
		}
		rl.KCache, rl.VCache, rl.KScale, rl.VScale = sl.KCache, sl.VCache, sl.KScale, sl.VScale
		rl.KvShared = true
	} else {
		var kc, vc *DeviceBuffer
		var e1, e2 error
		switch {
		case bs.kvI8:
			var ks, vs *DeviceBuffer
			kc, ks, e1 = bs.c.NewKVCacheI8(nil, bs.ctxCap*lkvDim, lnKV, lhd)
			vc, vs, e2 = bs.c.NewKVCacheI8(nil, bs.ctxCap*lkvDim, lnKV, lhd)
			if e1 == nil && e2 == nil {
				bs.keep(ks.Release)
				bs.keep(vs.Release)
				rl.KScale, rl.VScale = (*kernels.DeviceBufferBuf(ks)), (*kernels.DeviceBufferBuf(vs))
			}
		case bs.kvF16:
			kc, e1 = bs.c.NewKVCacheF16(nil, bs.ctxCap*lkvDim)
			vc, e2 = bs.c.NewKVCacheF16(nil, bs.ctxCap*lkvDim)
		default:
			kc, e1 = bs.c.NewKVCache(nil, bs.ctxCap*lkvDim)
			vc, e2 = bs.c.NewKVCache(nil, bs.ctxCap*lkvDim)
		}
		if e1 != nil || e2 != nil {
			return fmt.Errorf("gpu: residency KV alloc (layer %d): %w", i, errors.Join(e1, e2))
		}
		// Cross-file invariant guard: the decode runner indexes this cache with strides of g.kvDim = gnKV·ghd up to ctxCap
		// positions. The alloc above derives lkvDim from the same tuple, so this holds by construction; a builder that sized the
		// cache from an independent source would index off the end into garbage output, not a panic, so assert it here, loudly, at
		// plan time.
		if want := bs.ctxCap * lkvDim; (*kernels.DeviceBufferN(kc)) < want || (*kernels.DeviceBufferN(vc)) < want {
			return fmt.Errorf("gpu: layer %d KV cache too small for its geometry: have k=%d v=%d elems, need ctxCap·nKV·hd = %d·%d·%d = %d", i, (*kernels.DeviceBufferN(kc)), (*kernels.DeviceBufferN(vc)), bs.ctxCap, lnKV, lhd, want)
		}
		bs.keep(kc.Release)
		bs.keep(vc.Release)
		rl.KCache, rl.VCache = (*kernels.DeviceBufferBuf(kc)), (*kernels.DeviceBufferBuf(vc))
	}
	if rl.Q, e = bs.proj(&lw.QProj); e != nil {
		return e
	}
	if !rl.KvShared {
		if rl.K, e = bs.proj(&lw.KProj); e != nil {
			return e
		}
		if !rl.GKEqV {
			if rl.V, e = bs.proj(&lw.VProj); e != nil {
				return e
			}
		}
	}
	if rl.O, e = bs.projMul(&lw.OProj, bs.rmul); e != nil {
		return e
	}
	return nil
}

// g4MoEFFN uploads layer i's Gemma 4 feed-forward, the parallel dense and MoE block (decoder's gemma4MoEFFN), into rl.
func (bs *webgpuBuild) g4MoEFFN(i int, g4b decoder.Gemma4MoEResidentBundle, rl *kernels.RunLayer) error {
	m := bs.m
	rd := bs.rd
	var e error
	g := &kernels.G4MoELayer{NE: g4b.NE, TopK: g4b.TopK, DenseInter: g4b.DenseInter, MoeInter: g4b.MoeInter}
	if g4b.NE <= 0 || g4b.TopK <= 0 || g4b.TopK > g4b.NE || len(g4b.ExpertsGateUp) != g4b.NE || len(g4b.ExpertsDown) != g4b.NE ||
		len(g4b.RouterProjScaled) != g4b.NE*bs.hidden || len(g4b.PerExpertScale) != g4b.NE {
		return fmt.Errorf("gpu: layer %d: Gemma 4 MoE block with %d experts (top %d), router %d values, %d per-expert scales, %d/%d expert projections",
			i, g4b.NE, g4b.TopK, len(g4b.RouterProjScaled), len(g4b.PerExpertScale), len(g4b.ExpertsGateUp), len(g4b.ExpertsDown))
	}
	for _, nb := range []struct {
		dst *(*wgpu.Buffer)
		v   []float32
	}{{&g.PreFFNNorm, g4b.PreFFNNorm}, {&g.PostFFNNorm1, g4b.PostFFNNorm1}, {&g.PreFFNNorm2, g4b.PreFFNNorm2},
		{&g.PostFFNNorm2, g4b.PostFFNNorm2}, {&g.PostFFNNorm, g4b.PostFFNNorm}} {
		if len(nb.v) != bs.hidden {
			return fmt.Errorf("gpu: layer %d: a Gemma 4 MoE block norm has %d weights, want %d", i, len(nb.v), bs.hidden)
		}
		if *nb.dst, e = bs.up32(nb.v); e != nil {
			return e
		}
	}
	// The router's norm has no weight. The runner's RMSNorm multiplies by 1+w when the arch adds one and by w otherwise, so
	// the weight that makes it weightless is 0 or 1.
	flat := make([]float32, bs.hidden)
	if !m.RMSAddOne() {
		for j := range flat {
			flat[j] = 1
		}
	}
	if g.RouterNorm, e = bs.up32(flat); e != nil {
		return e
	}
	if g.Router, e = bs.up32(g4b.RouterProjScaled); e != nil {
		return e
	}
	if g.PerExpertScale, e = bs.up32(g4b.PerExpertScale); e != nil {
		return e
	}
	if g.ZeroBias, e = bs.up32(make([]float32, g4b.NE)); e != nil {
		return e
	}
	if g.Gate, e = bs.proj(g4b.MlpGate); e != nil {
		return e
	}
	if g.Up, e = bs.proj(g4b.MlpUp); e != nil {
		return e
	}
	if g.Down, e = bs.proj(g4b.MlpDown); e != nil {
		return e
	}
	if g.ExpGateUp, e = bs.buildStacked(g4b.NE, func(x int) *linalg.WeightMat { return g4b.ExpertsGateUp[x] }, 1); e != nil {
		return e
	}
	bs.keep(g.ExpGateUp.Release)
	if g.ExpDown, e = bs.buildStacked(g4b.NE, func(x int) *linalg.WeightMat { return g4b.ExpertsDown[x] }, 1); e != nil {
		return e
	}
	bs.keep(g.ExpDown.Release)
	if (*kernels.ResidentStackedW8A8Rows(g.ExpGateUp)) != 2*g4b.MoeInter || (*kernels.ResidentStackedW8A8Rows(g.ExpDown)) != bs.hidden {
		return fmt.Errorf("gpu: layer %d: Gemma 4 expert projections have %d and %d rows, want %d and %d", i, (*kernels.ResidentStackedW8A8Rows(g.ExpGateUp)), (*kernels.ResidentStackedW8A8Rows(g.ExpDown)), 2*g4b.MoeInter, bs.hidden)
	}
	rl.G4moe, rd.rm.G4moe = g, true
	rl.LayerScalar = g4b.LayerScalar
	if kernels.G4DropLayerScalarForTest {
		rl.LayerScalar = 1
	}
	return nil
}

// sparseMoEFFN uploads a Mixtral-class sparse MoE feed-forward for layer i (the stacked experts, the router, gpt-oss's bias tables
// and the shared expert when the model has one) into rl.
func (bs *webgpuBuild) sparseMoEFFN(i int, lw *decoder.LayerWeights, rl *kernels.RunLayer) error {
	m := bs.m
	rd := bs.rd
	var e error
	rl.IsMoE = true
	if rl.Router, e = bs.proj(&lw.Router); e != nil {
		return e
	}
	if len(lw.RouterBias) > 0 {
		if rl.RouterBias, e = bs.up32(lw.RouterBias); e != nil {
			return e
		}
	}
	if rl.ExpGate, e = bs.buildStacked(len(lw.Experts), func(x int) *linalg.WeightMat { return &lw.Experts[x].Gate }, 1); e != nil {
		return e
	}
	bs.keep(rl.ExpGate.Release)
	if rl.ExpUp, e = bs.buildStacked(len(lw.Experts), func(x int) *linalg.WeightMat { return &lw.Experts[x].Up }, 1); e != nil {
		return e
	}
	bs.keep(rl.ExpUp.Release)
	if rl.ExpDown, e = bs.buildStacked(len(lw.Experts), func(x int) *linalg.WeightMat { return &lw.Experts[x].Down }, bs.rmul); e != nil {
		return e
	}
	bs.keep(rl.ExpDown.Release)
	// gpt-oss (FeatAttnSink): expert-major gate‖up and down bias tables.
	if rd.rm.Moe != nil && rd.rm.Moe.Gptoss {
		if rl.GateUpBias, e = bs.up32(m.GptOssExpertBiasResident(i)); e != nil {
			return e
		}
		if rl.DownBias, e = bs.up32(m.GptOssExpertDownBiasResident(i)); e != nil {
			return e
		}
	}
	if bs.shInter > 0 { // always-on shared expert (qwen2_moe gated / GLM ungated)
		if rl.ShGate, e = bs.proj(&lw.SharedExpert.Gate); e != nil {
			return e
		}
		if rl.ShUp, e = bs.proj(&lw.SharedExpert.Up); e != nil {
			return e
		}
		if rl.ShDown, e = bs.projMul(&lw.SharedExpert.Down, bs.rmul); e != nil {
			return e
		}
		if !bs.shUngated { // qwen2_moe: the [1,hidden] sigmoid gate
			if rl.ShGateW, e = bs.proj(&lw.SharedGate); e != nil {
				return e
			}
		}
	}
	return nil
}

// denseFFN uploads a dense gate/up/down feed-forward, and a Gemma 4 E-model's per-layer-embedding branch, into rl.
func (bs *webgpuBuild) denseFFN(lw *decoder.LayerWeights, rl *kernels.RunLayer) error {
	rd := bs.rd
	var e error
	if rl.Gate, e = bs.proj(&lw.GateProj); e != nil {
		return e
	}
	if rl.Up, e = bs.proj(&lw.UpProj); e != nil {
		return e
	}
	if rl.Down, e = bs.projMul(&lw.DownProj, bs.rmul); e != nil {
		return e
	}
	// Gemma 4 E-model: the per-layer-embedding branch's two projections and its norm.
	if rd.rm.PleDim > 0 {
		if rl.PleGate, e = bs.proj(&lw.PLEGate); e != nil {
			return e
		}
		if rl.PleProj, e = bs.proj(&lw.PLEProj); e != nil {
			return e
		}
		if rl.PostPLENorm, e = bs.up32(lw.PostPLENorm); e != nil {
			return e
		}
	}
	return nil
}

// layerTail uploads what every layer carries after its mixer and feed-forward: q/k/v and output biases, QK-norms, sandwich or
// post-only norms and the attention sinks. isG4MoE and postOnlyHere are the layer's norm placement, decided by the caller.
func (bs *webgpuBuild) layerTail(i int, lw *decoder.LayerWeights, rl *kernels.RunLayer, isG4MoE, postOnlyHere bool) error {
	m := bs.m
	lhd, lnKV := m.HeadDimAtResident(i), m.KVHeadsAtResident(i) // the layer's own head dim and KV heads, for the QK-norm width check
	var e error
	if len(lw.QBias) > 0 { // Qwen2 q/k/v bias
		if rl.QBias, e = bs.up32(lw.QBias); e != nil {
			return e
		}
		if rl.KBias, e = bs.up32(lw.KBias); e != nil {
			return e
		}
		if rl.VBias, e = bs.up32(lw.VBias); e != nil {
			return e
		}
	}
	if len(lw.QNorm) > 0 { // Qwen3/GLM per-head QK-norm, or Olmo 3/Olmo Hybrid whole-vector QK-norm (applied before RoPE)
		if bs.qkNormWhole {
			if len(lw.QNorm) != bs.nH*lhd || (!rl.KvShared && len(lw.KNorm) != lnKV*lhd) {
				return fmt.Errorf("gpu: layer %d: QKNormWhole weights len(QNorm)=%d len(KNorm)=%d, want %d/%d",
					i, len(lw.QNorm), len(lw.KNorm), bs.nH*lhd, lnKV*lhd)
			}
		}
		if rl.QNorm, e = bs.up32(lw.QNorm); e != nil {
			return e
		}
		if rl.KNorm, e = bs.up32(lw.KNorm); e != nil {
			return e
		}
	}
	// FeatOutBias (gpt-oss o_proj bias) and FeatSandwichNorm / FeatPostOnlyNorm (Gemma's / Olmo 3's / Olmo Hybrid's
	// post-sublayer norms): the same "len>0, else leave nil" gate as QBias/QNorm above.
	if len(lw.OBias) > 0 {
		if rl.OBias, e = bs.up32(lw.OBias); e != nil {
			return e
		}
	}
	if bs.sandwichOK || postOnlyHere { // both present together when the layer uses sandwich or post-only norms
		if len(lw.PostAttnNorm) != bs.hidden || (!isG4MoE && len(lw.PostMLPNorm) != bs.hidden) {
			return fmt.Errorf("gpu: layer %d: arch declares sandwich/postOnly norms but PostAttnNorm/PostMLPNorm are not len==hidden(%d) (got %d/%d)",
				i, bs.hidden, len(lw.PostAttnNorm), len(lw.PostMLPNorm))
		}
		if rl.PostAttnNorm, e = bs.up32(lw.PostAttnNorm); e != nil {
			return e
		}
		if !isG4MoE {
			if rl.PostMLPNorm, e = bs.up32(lw.PostMLPNorm); e != nil {
				return e
			}
		}
		if postOnlyNormOffForTest {
			rl.PostAttnNorm, rl.PostMLPNorm = nil, nil
		}
	}
	// FeatAttnSink (gpt-oss): attnSinks is ALWAYS bound (a one-element dummy when the layer has none) because WGSL bind groups
	// cannot bind a null storage buffer; Metal does the same, CUDA uses a null-pointer sentinel. DecodeRunner's noAttnSinks is a
	// separate runner-local fallback for hand-built test fixtures that skip BuildResident. The two buffers have different owners
	// and lifetimes (resident, shared across runners, vs per-runner), so they are deliberately not unified.
	if sinks := m.GptOssSinksResident(i); len(sinks) > 0 {
		if rl.AttnSinks, e = bs.up32(sinks); e != nil {
			return e
		}
		rl.HasSink = true
	} else {
		if rl.AttnSinks, e = bs.up32([]float32{0}); e != nil {
			return e
		}
	}
	return nil
}
