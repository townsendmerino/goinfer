//go:build gpu

package gpu

import (
	"fmt"
	"math"
	"os"

	"github.com/cogentcore/webgpu/wgpu"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

// GPU full-residency bridge: builds a resident DecodeRunner from a loaded
// decoder.Model (dense Qwen2/Llama) so decoder.Generate's per-token forward runs
// entirely on the device. The webgpuBackend satisfies decoder.ResidencyBackend;
// the decoder calls BuildResident when the arch is eligible, then routes the
// per-token forward through the returned ResidentForward (decoder/residency.go).

// The decoder's projections are now linalg.WeightMat (the aikit consolidation), so
// the bridge pulls a projection's resident arrays via its exported accessors
// (Kind/Rows/Cols/Int4/Int8) directly — no goinfer-local interface needed.

// uploadProj uploads one projection to the device at its native precision,
// returning a decodeWeight the DecodeRunner can GEMV. int4 and int8 only (the
// .giw cases); f32 is unsupported here (caller falls back).
func (c *Context) uploadProj(w *linalg.WeightMat) (decodeWeight, error) {
	N, K := w.Rows(), w.Cols()
	switch w.Kind() {
	case "int4":
		q4, q4s, group, _ := w.Int4()
		if group != w4a8GroupSize {
			return nil, fmt.Errorf("gpu: residency int4 group %d != %d", group, w4a8GroupSize)
		}
		// decoder packs 2 nibbles/byte (elem k → byte k>>1, low nibble if even);
		// UploadW4A8 takes one nibble (0..15) per element and re-packs to the GPU
		// layout. Unpack here — values (and so nibble−8) are preserved.
		nib := make([]uint8, N*K)
		for r := 0; r < N; r++ {
			row := q4[r*((K+1)/2):]
			dst := nib[r*K : r*K+K]
			for k := 0; k < K; k++ {
				b := row[k>>1]
				if k&1 == 0 {
					dst[k] = b & 0x0F
				} else {
					dst[k] = b >> 4
				}
			}
		}
		return c.UploadW4A8(nib, q4s, N, K)
	case "int8":
		q8, scales, _, _ := w.Int8()
		return c.UploadW8A8(q8, scales, N, K)
	case "f32":
		// Some projections stay f32 even under an int8 load (the MoE router is kept
		// full-precision for selection stability). Quantize row-wise to int8 here so the
		// resident GEMV can read it — the router logits feed a top-k, so int8 is ample.
		f32, ok := w.F32()
		if !ok {
			return nil, fmt.Errorf("gpu: residency f32 projection has no data")
		}
		q8, scales := linalg.QuantizeRowsInt8(f32, N, K)
		return c.UploadW8A8(q8, scales, N, K)
	default:
		return nil, fmt.Errorf("gpu: residency unsupported projection precision %q", w.Kind())
	}
}

// residentDecoder is the gpu side of decoder.ResidentForward: a persistent
// DecodeRunner + the runModel (for KV upload), built once per model.
type residentDecoder struct {
	c       *Context
	runner  *DecodeRunner
	rm      runModel
	nKV, hd int      // KV grouping — UploadKV needs it to per-head quantize int8
	keep    []func() // release the resident buffers (norms, biases, KV, projections)

	// Batched verify (ForwardN): extra DecodeRunner instances sharing rm (the same
	// resident weights + KV caches) with their own scratch/uniforms, built lazily up
	// to the largest K seen. batch[0] aliases runner. newRunner builds one more.
	newRunner func() (*DecodeRunner, error)
	batch     []*DecodeRunner
}

// BuildResident builds a resident DecodeRunner from m, or (nil,false,nil) when
// the arch is ineligible / a projection is f32 (caller uses the staged path).
func (b *webgpuBackend) BuildResident(m *decoder.Model) (decoder.ResidentForward, bool, error) {
	// Granite-4.0-H resident SSM path (P5b): buildable independent of the public
	// eligibility flag (which flips at P6) so it can be parity-gated first. Production
	// routing still goes through withResidency()'s DecodeRunnerEligible() check.
	_, _, _, _, _, _, _, _, _, granOK := m.GraniteResidentParams()
	_, _, _, _, _, _, _, nemoOK := m.NemotronResidentParams()
	if !m.DecodeRunnerEligible() && !granOK && !nemoOK {
		return nil, false, nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	c := b.ctx
	w := m.Weights()
	hidden, _, nH, nKV, hd, inter, vocab := m.Dims() // arch-backed (Cfg may be zero for GGUF/.giw)
	eps := m.NormEps()
	// f32 KV caps context at 16k (the proven 8 GB fit); f16 halves per-token KV
	// bytes → 32k (task-gpu-f16-kv.md); int8 quarters them → ~64k (task-gpu-kv-i8.md).
	kvF16 := m.KVCacheF16()
	kvI8 := m.KVCacheI8()
	ctxCap := 16384
	switch {
	case kvI8:
		ctxCap = 65536
	case kvF16:
		ctxCap = 32768
	}
	kvDim := nKV * hd

	rd := &residentDecoder{c: c, nKV: nKV, hd: hd}
	keepF := func(f func()) { rd.keep = append(rd.keep, f) }
	up32 := func(v []float32) (*wgpu.Buffer, error) {
		d, err := c.UploadF32(v)
		if err != nil {
			return nil, err
		}
		keepF(d.Release)
		return d.buf, nil
	}
	proj := func(pw *linalg.WeightMat) (decodeWeight, error) {
		dw, err := c.uploadProj(pw)
		if err != nil {
			return nil, err
		}
		switch t := dw.(type) {
		case *ResidentW4A8:
			keepF(t.Release)
		case *ResidentW8A8:
			keepF(t.Release)
		}
		return dw, nil
	}
	// projF32 quantizes a raw f32 [N,K] weight to int8 row-wise and uploads it (the MLA
	// projections are stored f32 in the decoder; the resident GEMVs are W8A8).
	projF32 := func(w []float32, N, K int) (decodeWeight, error) {
		q8, scales := linalg.QuantizeRowsInt8(w, N, K)
		rm, err := c.UploadW8A8(q8, scales, N, K)
		if err != nil {
			return nil, err
		}
		keepF(rm.Release)
		return rm, nil
	}

	// --- Granite multiplier folding (P5b): ResidMul → every residual-add weight scale
	// (o-proj / mamba out-proj / MoE expert-down / shared-down); LogitScale → 1/LogitScale
	// into the lm_head; EmbMul → the embedding (in decoder.embedResident). mult==1 ⇒ the
	// plain proj path, byte-identical for non-granite models. ---
	scaleCopy := func(s []float32, m float32) []float32 {
		out := make([]float32, len(s))
		for i, v := range s {
			out[i] = v * m
		}
		return out
	}
	projMul := func(pw *linalg.WeightMat, mult float32) (decodeWeight, error) {
		if mult == 1 {
			return proj(pw)
		}
		N, K := pw.Rows(), pw.Cols()
		var q8 []int8
		var scales []float32
		switch pw.Kind() {
		case "int8":
			q8, scales, _, _ = pw.Int8()
		case "f32":
			f32, ok := pw.F32()
			if !ok {
				return nil, fmt.Errorf("gpu: granite mult-fold f32 has no data")
			}
			q8, scales = linalg.QuantizeRowsInt8(f32, N, K)
		default:
			return nil, fmt.Errorf("gpu: granite mult-fold unsupported kind %q", pw.Kind())
		}
		rm, err := c.UploadW8A8(q8, scaleCopy(scales, mult), N, K)
		if err != nil {
			return nil, err
		}
		keepF(rm.Release)
		return rm, nil
	}
	projF32Mul := func(w []float32, N, K int, mult float32) (decodeWeight, error) {
		q8, scales := linalg.QuantizeRowsInt8(w, N, K)
		if mult != 1 {
			scales = scaleCopy(scales, mult)
		}
		rm, err := c.UploadW8A8(q8, scales, N, K)
		if err != nil {
			return nil, err
		}
		keepF(rm.Release)
		return rm, nil
	}
	// stateBuf allocates a build-once, zeroed, in-place-updatable Mamba state buffer
	// (Storage|CopyDst so Reset can re-zero it per generation).
	stateBuf := func(n int) (*wgpu.Buffer, error) {
		b2, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "mamba-state", Size: uint64(n * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst})
		if err != nil {
			return nil, err
		}
		c.queue.WriteBuffer(b2, 0, wgpu.ToBytes(make([]float32, n)))
		keepF(b2.Release)
		return b2, nil
	}

	fail := func(err error) (decoder.ResidentForward, bool, error) { rd.release(); return nil, false, err }

	// Per-layer RoPE (Lever C7): bind each layer the global or local invFreq table. Most
	// models share one (cache keyed on the global/local flag); Mellum has two (YaRN global,
	// default local). ropeHalf = rotaryDim/2 is uniform (the guard requires equal lengths).
	invFreq := m.RopeInvFreq()
	ropeInvCache := map[bool]*wgpu.Buffer{}
	ropeInvBuf := func(global bool, layer int) (*wgpu.Buffer, error) {
		if b, ok := ropeInvCache[global]; ok {
			return b, nil
		}
		b, err := up32(m.RopeInvFreqLayer(layer))
		if err != nil {
			return nil, err
		}
		ropeInvCache[global] = b
		return b, nil
	}
	finalNorm, err := up32(w.FinalNorm)
	if err != nil {
		return fail(err)
	}
	// Granite-4.0-H (P5b): the Mamba/attention hybrid. rmul folds ResidMul into every
	// residual-add weight; lmMult folds 1/LogitScale into the head. mambaP carries the
	// model-level SSM geometry (per-layer mixer-kind from m.GraniteMambaLayer).
	gNH, gHd, gDS, gNG, gK, _, gResidMul, gLogitScale, _, granOK := m.GraniteResidentParams()
	rmul := float32(1)
	lmMult := float32(1)
	if granOK {
		rmul = gResidMul
		if gLogitScale != 0 {
			lmMult = 1 / gLogitScale
		}
		gIn := gNH * gHd
		rd.rm.mamba = &mambaRunParams{
			nHeads: gNH, hp: gHd, dn: gDS, nGroups: gNG, dConv: gK,
			dInner: gIn, convDim: gIn + 2*gNG*gDS, projDim: 2*gIn + 2*gNG*gDS + gNH,
			gSize: gNG * gDS, repeat: gNH / gNG, normGroups: 1,
		}
	}
	// Nemotron-H (dense squared-ReLU hybrid): the same Mamba-2 geometry but the gated RMSNorm
	// is PER GROUP (normGroups = NGroups), and there are NO multipliers (rmul/lmMult stay 1).
	nNH, nHd, nDS, nNG, nK, nNormG, _, nemoOK := m.NemotronResidentParams()
	var nemoZeroInvFreq *wgpu.Buffer // NoPE: a zeroed invFreq makes the rope kernel an identity
	if nemoOK {
		nIn := nNH * nHd
		rd.rm.mamba = &mambaRunParams{
			nHeads: nNH, hp: nHd, dn: nDS, nGroups: nNG, dConv: nK,
			dInner: nIn, convDim: nIn + 2*nNG*nDS, projDim: 2*nIn + 2*nNG*nDS + nNH,
			gSize: nNG * nDS, repeat: nNH / nNG, normGroups: nNormG,
		}
		var ze error
		if nemoZeroInvFreq, ze = up32(make([]float32, hd/2)); ze != nil {
			return fail(ze)
		}
	}
	// LM head: tied (LMHead empty → the Embed matrix is the head) or separate.
	headW := &w.LMHead
	if w.LMHead.Rows() == 0 {
		headW = &w.Embed
	}
	lmHead, err := projMul(headW, lmMult)
	if err != nil {
		return fail(err)
	}
	// ropeHalf = len(invFreq) = rotaryDim/2 — drives the rope dispatch (partial RoPE,
	// Lever C5: GLM/Phi rotate only the first rotaryDim dims of each head).
	rd.rm.finalNorm, rd.rm.lmHead, rd.rm.ropeHalf, rd.rm.slidingWindow = finalNorm, lmHead, len(invFreq), m.SlidingWindowResident()

	// MoE (Lever C3c): Mixtral-class models route to stacked int8 experts on-device.
	// moeOK gates the per-layer FFN build below; the params are model-level.
	nExp, topK, moeInter, shInter, sig, normTopK, shUngated, rScale, nGroup, topkGroup, moeOK := m.MoEResidentParams()
	if moeOK {
		rd.rm.moe = &moeRunParams{
			nE: nExp, k: topK, inter: moeInter, sigmoid: sig, norm: normTopK, scale: float32(rScale),
			sharedInter: shInter, sharedUngated: shUngated, nGroup: nGroup, topkGroup: topkGroup,
		}
	}
	// MLA (Lever C4): DeepSeek/Kimi latent attention replaces the q/k/v/o block. mlaOK
	// gates the per-layer MLA build below; the geometry is model-level. attnScale
	// overrides the GQA 1/√HeadDim with the resolved qk_head_dim score scale.
	qLoRA, kvLoRA, qkNope, qkRope, vHead, interleave, mlaAttnScale, mlaRopeScale, mlaOK := m.MLAResidentParams()
	if mlaOK {
		rd.rm.mla = &mlaRunParams{
			qLoRARank: qLoRA, kvLoRARank: kvLoRA, qkNope: qkNope, qkRope: qkRope,
			vHead: vHead, interleave: interleave, ropeScale: float32(mlaRopeScale),
		}
	}
	// buildStacked packs one projection (gate/up/down) across all nE experts into a
	// resident stacked int8 buffer the indexed expert GEMV reads. int8 only — int4
	// experts aren't stacked yet (returns an error → silent staged fallback).
	buildStacked := func(lw *decoder.LayerWeights, get func(e int) *linalg.WeightMat, mult float32) (*ResidentStackedW8A8, error) {
		nE := len(lw.Experts)
		w0 := get(0)
		N, K := w0.Rows(), w0.Cols()
		q8 := make([][]int8, nE)
		sc := make([][]float32, nE)
		for e := 0; e < nE; e++ {
			w := get(e)
			if w.Kind() != "int8" {
				return nil, fmt.Errorf("gpu: MoE residency expert %d kind %q (int8 only)", e, w.Kind())
			}
			q, s, _, _ := w.Int8()
			if mult != 1 { // Granite ResidMul fold into the expert-down residual add
				s = scaleCopy(s, mult)
			}
			q8[e], sc[e] = q, s
		}
		return c.UploadStackedExperts(q8, sc, nE, N, K)
	}

	for i := range w.Layers {
		lw := &w.Layers[i]
		rl := runLayer{
			isLocal:   m.LayerIsLocalResident(i),     // sliding-window layer (Lever C6)
			ropeScale: float32(m.RopeMscaleLayer(i)), // per-layer YaRN mscale (Lever C7)
		}
		var e error
		if nemoOK {
			// Nemotron-H single-op-per-block: ONE pre-norm (PreAttnNorm) + exactly one op +
			// residual. No FFN pairing, no multipliers. The resident layer-loop branches on
			// rl.nemoKind (mamba / NoPE-attn / relu²-MLP). int8 mamba projections (projF32);
			// attn/MLP projections take the model's native precision via proj (W8A8 or W4A8).
			if rl.attnNorm, e = up32(lw.PreAttnNorm); e != nil {
				return fail(e)
			}
			rl.mlpNorm = rl.attnNorm // the relu²-MLP block reuses the single pre-norm
			mp := rd.rm.mamba
			switch m.NemotronBlockKind(i) {
			case 0: // Mamba-2 mixer
				rl.nemoKind = nemoKMamba
				rl.isMamba = true
				inP, cW, cB, aLog, dW, dtB, nW, outP := m.GraniteMambaWeights(i)
				if rl.mambaInProj, e = projF32(inP, mp.projDim, hidden); e != nil {
					return fail(e)
				}
				if rl.mambaOutProj, e = projF32(outP, hidden, mp.dInner); e != nil {
					return fail(e)
				}
				if rl.mambaConvW, e = up32(cW); e != nil {
					return fail(e)
				}
				if rl.mambaConvB, e = up32(cB); e != nil {
					return fail(e)
				}
				headP := make([]float32, mp.nHeads*3)
				for h := 0; h < mp.nHeads; h++ {
					headP[h*3+0] = float32(-math.Exp(float64(aLog[h])))
					headP[h*3+1] = dtB[h]
					headP[h*3+2] = dW[h]
				}
				if rl.mambaHeadP, e = up32(headP); e != nil {
					return fail(e)
				}
				if rl.mambaNormW, e = up32(nW); e != nil {
					return fail(e)
				}
				if rl.mambaWin, e = stateBuf((mp.dConv - 1) * mp.convDim); e != nil {
					return fail(e)
				}
				if rl.mambaSSM, e = stateBuf(mp.nHeads * mp.hp * mp.dn); e != nil {
					return fail(e)
				}
			case 1: // NoPE GQA attention (identity rope via the zeroed invFreq)
				rl.nemoKind = nemoKAttn
				rl.invFreq = nemoZeroInvFreq
				kc, e1 := c.NewKVCache(nil, ctxCap*kvDim)
				vc, e2 := c.NewKVCache(nil, ctxCap*kvDim)
				if e1 != nil || e2 != nil {
					return fail(fmt.Errorf("gpu: nemotron KV alloc (layer %d): %v %v", i, e1, e2))
				}
				keepF(kc.Release)
				keepF(vc.Release)
				rl.kCache, rl.vCache = kc.buf, vc.buf
				if rl.q, e = proj(&lw.QProj); e != nil {
					return fail(e)
				}
				if rl.k, e = proj(&lw.KProj); e != nil {
					return fail(e)
				}
				if rl.v, e = proj(&lw.VProj); e != nil {
					return fail(e)
				}
				if rl.o, e = proj(&lw.OProj); e != nil {
					return fail(e)
				}
			case 2: // non-gated relu² MLP
				rl.nemoKind = nemoKMLP
				if rl.up, e = proj(&lw.UpProj); e != nil {
					return fail(e)
				}
				if rl.down, e = proj(&lw.DownProj); e != nil {
					return fail(e)
				}
			}
			rd.rm.layers = append(rd.rm.layers, rl)
			continue
		}
		if rl.attnNorm, e = up32(lw.PreAttnNorm); e != nil {
			return fail(e)
		}
		if rl.mlpNorm, e = up32(lw.PreMLPNorm); e != nil {
			return fail(e)
		}
		if rl.invFreq, e = ropeInvBuf(m.LayerRopeGlobal(i), i); e != nil {
			return fail(e)
		}
		if granOK && m.GraniteMambaLayer(i) {
			// Mamba-2 SSM mixer layer (P5b): in/out_proj W8A8, conv/headP/normW f32, build-once
			// {win, ssm} state. headP interleaves the per-head [Aexp=-exp(aLog), dtBias, D].
			// ResidMul folds into out_proj. No KV cache, no q/k/v/o (the plan branches on isMamba).
			rl.isMamba = true
			mp := rd.rm.mamba
			inP, cW, cB, aLog, dW, dtB, nW, outP := m.GraniteMambaWeights(i)
			// DEFAULT int8: f16 on the mamba projections does NOT recover quality (measured
			// 63.9% vs int8 66.2% — the loss is the SSM f32-vs-f64 exp feeding the
			// hypersensitive 64-expert MoE router, not the projection precision; the f16
			// GEMV is proven accurate) and is ~2× slower, so int8 is strictly better.
			// GOINFER_SSM_F16MAMBA keeps the f16 path available for the record.
			if os.Getenv("GOINFER_SSM_F16MAMBA") != "" { // f16 (no quality gain; kept for experiments)
				var b *wgpu.Buffer
				if b, e = c.UploadF16Weight(inP, mp.projDim, hidden, 1); e != nil {
					return fail(e)
				}
				keepF(b.Release)
				rl.mambaInProjF16 = b
				if b, e = c.UploadF16Weight(outP, hidden, mp.dInner, rmul); e != nil {
					return fail(e)
				}
				keepF(b.Release)
				rl.mambaOutProjF16 = b
			} else { // int8 (default): faster, same quality
				if rl.mambaInProj, e = projF32(inP, mp.projDim, hidden); e != nil {
					return fail(e)
				}
				if rl.mambaOutProj, e = projF32Mul(outP, hidden, mp.dInner, rmul); e != nil {
					return fail(e)
				}
			}
			if rl.mambaConvW, e = up32(cW); e != nil {
				return fail(e)
			}
			if rl.mambaConvB, e = up32(cB); e != nil {
				return fail(e)
			}
			headP := make([]float32, mp.nHeads*3)
			for h := 0; h < mp.nHeads; h++ {
				headP[h*3+0] = float32(-math.Exp(float64(aLog[h])))
				headP[h*3+1] = dtB[h]
				headP[h*3+2] = dW[h]
			}
			if rl.mambaHeadP, e = up32(headP); e != nil {
				return fail(e)
			}
			if rl.mambaNormW, e = up32(nW); e != nil {
				return fail(e)
			}
			if rl.mambaWin, e = stateBuf((mp.dConv - 1) * mp.convDim); e != nil {
				return fail(e)
			}
			if rl.mambaSSM, e = stateBuf(mp.nHeads * mp.hp * mp.dn); e != nil {
				return fail(e)
			}
		} else if mlaOK {
			// MLA: a single compressed-latent cache [ctxCap·latDim] replaces the per-head
			// K/V caches; the per-head K/V are rebuilt in rank-space, never materialized.
			latDim := kvLoRA + qkRope
			lc, e1 := c.NewKVCache(nil, ctxCap*latDim)
			if e1 != nil {
				return fail(fmt.Errorf("gpu: residency latent alloc (layer %d): %v", i, e1))
			}
			keepF(lc.Release)
			rl.latCache = lc.buf
			qA, qANorm, qB, qProj, kvA, kvANorm, kvB, oProj := m.MLALayerWeights(i)
			if qLoRA > 0 { // q_a → norm → q_b LoRA bottleneck
				if rl.mlaQA, e = projF32(qA, qLoRA, hidden); e != nil {
					return fail(e)
				}
				if rl.mlaQB, e = projF32(qB, nH*(qkNope+qkRope), qLoRA); e != nil {
					return fail(e)
				}
				if rl.mlaQANorm, e = up32(qANorm); e != nil {
					return fail(e)
				}
			} else { // direct q_proj (V2-Lite)
				if rl.mlaQ, e = projF32(qProj, nH*(qkNope+qkRope), hidden); e != nil {
					return fail(e)
				}
			}
			if rl.mlaKVA, e = projF32(kvA, latDim, hidden); e != nil {
				return fail(e)
			}
			if rl.mlaKVANorm, e = up32(kvANorm); e != nil {
				return fail(e)
			}
			if rl.mlaO, e = projF32(oProj, hidden, nH*vHead); e != nil {
				return fail(e)
			}
			// kvB is [nH*(qkNope+vHead), kvLoRA] row-major (per head: k_nope rows ‖ v rows).
			// Slice into W_UKᵀ [nH, kvLoRA, qkNope] (transposed for the absorb GEMV) and
			// W_UV [nH, vHead, kvLoRA] (the lift, used as-is).
			hRow := qkNope + vHead
			wuk := make([]float32, nH*kvLoRA*qkNope)
			wuv := make([]float32, nH*vHead*kvLoRA)
			for h := 0; h < nH; h++ {
				for d := 0; d < qkNope; d++ {
					src := kvB[(h*hRow+d)*kvLoRA : (h*hRow+d)*kvLoRA+kvLoRA]
					for cc := 0; cc < kvLoRA; cc++ {
						wuk[(h*kvLoRA+cc)*qkNope+d] = src[cc] // transpose: [c][d] ← kvB[d][c]
					}
				}
				for ev := 0; ev < vHead; ev++ {
					copy(wuv[(h*vHead+ev)*kvLoRA:(h*vHead+ev)*kvLoRA+kvLoRA], kvB[(h*hRow+qkNope+ev)*kvLoRA:(h*hRow+qkNope+ev)*kvLoRA+kvLoRA])
				}
			}
			if rl.mlaWUK, e = up32(wuk); e != nil {
				return fail(e)
			}
			if rl.mlaWUV, e = up32(wuv); e != nil {
				return fail(e)
			}
		} else {
			var kc, vc *DeviceBuffer
			var e1, e2 error
			switch {
			case kvI8:
				var ks, vs *DeviceBuffer
				kc, ks, e1 = c.NewKVCacheI8(nil, ctxCap*kvDim, nKV, hd)
				vc, vs, e2 = c.NewKVCacheI8(nil, ctxCap*kvDim, nKV, hd)
				if e1 == nil && e2 == nil {
					keepF(ks.Release)
					keepF(vs.Release)
					rl.kScale, rl.vScale = ks.buf, vs.buf
				}
			case kvF16:
				kc, e1 = c.NewKVCacheF16(nil, ctxCap*kvDim)
				vc, e2 = c.NewKVCacheF16(nil, ctxCap*kvDim)
			default:
				kc, e1 = c.NewKVCache(nil, ctxCap*kvDim)
				vc, e2 = c.NewKVCache(nil, ctxCap*kvDim)
			}
			if e1 != nil || e2 != nil {
				return fail(fmt.Errorf("gpu: residency KV alloc (layer %d): %v %v", i, e1, e2))
			}
			keepF(kc.Release)
			keepF(vc.Release)
			rl.kCache, rl.vCache = kc.buf, vc.buf
			if rl.q, e = proj(&lw.QProj); e != nil {
				return fail(e)
			}
			if rl.k, e = proj(&lw.KProj); e != nil {
				return fail(e)
			}
			if rl.v, e = proj(&lw.VProj); e != nil {
				return fail(e)
			}
			if rl.o, e = projMul(&lw.OProj, rmul); e != nil {
				return fail(e)
			}
		}
		if moeOK && len(lw.Experts) > 0 { // Mixtral-class sparse MoE FFN
			rl.isMoE = true
			if rl.router, e = proj(&lw.Router); e != nil {
				return fail(e)
			}
			if len(lw.RouterBias) > 0 {
				if rl.routerBias, e = up32(lw.RouterBias); e != nil {
					return fail(e)
				}
			}
			if rl.expGate, e = buildStacked(lw, func(x int) *linalg.WeightMat { return &lw.Experts[x].Gate }, 1); e != nil {
				return fail(e)
			}
			keepF(rl.expGate.Release)
			if rl.expUp, e = buildStacked(lw, func(x int) *linalg.WeightMat { return &lw.Experts[x].Up }, 1); e != nil {
				return fail(e)
			}
			keepF(rl.expUp.Release)
			if rl.expDown, e = buildStacked(lw, func(x int) *linalg.WeightMat { return &lw.Experts[x].Down }, rmul); e != nil {
				return fail(e)
			}
			keepF(rl.expDown.Release)
			if shInter > 0 { // always-on shared expert (qwen2_moe gated / GLM ungated)
				if rl.shGate, e = proj(&lw.SharedExpert.Gate); e != nil {
					return fail(e)
				}
				if rl.shUp, e = proj(&lw.SharedExpert.Up); e != nil {
					return fail(e)
				}
				if rl.shDown, e = projMul(&lw.SharedExpert.Down, rmul); e != nil {
					return fail(e)
				}
				if !shUngated { // qwen2_moe: the [1,hidden] sigmoid gate
					if rl.shGateW, e = proj(&lw.SharedGate); e != nil {
						return fail(e)
					}
				}
			}
		} else {
			if rl.gate, e = proj(&lw.GateProj); e != nil {
				return fail(e)
			}
			if rl.up, e = proj(&lw.UpProj); e != nil {
				return fail(e)
			}
			if rl.down, e = projMul(&lw.DownProj, rmul); e != nil {
				return fail(e)
			}
		}
		if len(lw.QBias) > 0 { // Qwen2 q/k/v bias
			if rl.qBias, e = up32(lw.QBias); e != nil {
				return fail(e)
			}
			if rl.kBias, e = up32(lw.KBias); e != nil {
				return fail(e)
			}
			if rl.vBias, e = up32(lw.VBias); e != nil {
				return fail(e)
			}
		}
		if len(lw.QNorm) > 0 { // Qwen3/GLM per-head QK-norm (applied before RoPE)
			if rl.qNorm, e = up32(lw.QNorm); e != nil {
				return fail(e)
			}
			if rl.kNorm, e = up32(lw.KNorm); e != nil {
				return fail(e)
			}
		}
		rd.rm.layers = append(rd.rm.layers, rl)
	}
	_ = vocab // logits length is lmHead.nRows()

	rd.rm.kvF16 = kvF16 // the runner picks the f16 attn/store kernels off this
	rd.rm.kvI8 = kvI8   // …or the int8 kernels + scale binds
	scale, addOne := m.AttnScale(), m.RMSAddOne()
	if mlaOK { // MLA scores over qk_head_dim, not HeadDim — use the resolved score scale
		scale = float32(mlaAttnScale)
	}
	rd.newRunner = func() (*DecodeRunner, error) {
		return c.newDecodeRunner(rd.rm, hidden, nH, nKV, hd, inter, 0, eps, scale, addOne)
	}
	runner, err := rd.newRunner()
	if err != nil {
		return fail(err)
	}
	rd.runner = runner
	return rd, true, nil
}

func (rd *residentDecoder) Forward(embedding []float32, pos int) ([]float32, error) {
	return rd.runner.Run(embedding, pos)
}

// ForwardN runs K tokens at startPos..startPos+K-1 in one command buffer. It lazily
// grows a pool of K DecodeRunners that share rd.rm (the resident weights + KV caches)
// but own their scratch/uniforms, then records all K into a single submit (runBatch).
// Causal across the shared KV: row i's kv-store precedes row i+1's attention read.
func (rd *residentDecoder) ForwardN(embeddings [][]float32, startPos int) ([][]float32, error) {
	n := len(embeddings)
	if n == 0 {
		return nil, nil
	}
	if len(rd.batch) == 0 {
		rd.batch = append(rd.batch, rd.runner) // batch[0] aliases the M=1 runner
	}
	for len(rd.batch) < n {
		r, err := rd.newRunner()
		if err != nil {
			return nil, err
		}
		rd.batch = append(rd.batch, r)
	}
	return runBatch(rd.c, rd.batch, embeddings, startPos)
}

// TruncateTo is a no-op on the resident cache: it is positional and Forward sets
// nKeys=pos+1, so entries past pos are never read and get overwritten next round
// (see the ResidentForward.TruncateTo contract).
func (rd *residentDecoder) TruncateTo(pos int) {}

// UploadKV writes a layer's post-RoPE K and raw V (positions 0..n-1) into the
// resident caches — the prefill bridge. keys/vals are [n*kvDim] f32, packed to the
// cache's precision (f32 raw, f16 via packF16Pairs, int8 per-head + scales) so the
// upload matches what an on-device decode would have written. Currently unused (the
// stateless prefill seeds the caches via sequential Forward); kept for prefix-reuse.
func (rd *residentDecoder) UploadKV(layer int, keys, vals []float32) error {
	if layer < 0 || layer >= len(rd.rm.layers) {
		return fmt.Errorf("gpu: UploadKV layer %d out of range", layer)
	}
	l := rd.rm.layers[layer]
	switch {
	case rd.rm.kvI8:
		kw, ks := packKVInt8(keys, rd.nKV, rd.hd)
		vw, vs := packKVInt8(vals, rd.nKV, rd.hd)
		if err := rd.c.queue.WriteBuffer(l.kCache, 0, wgpu.ToBytes(kw)); err != nil {
			return err
		}
		if err := rd.c.queue.WriteBuffer(l.kScale, 0, wgpu.ToBytes(ks)); err != nil {
			return err
		}
		if err := rd.c.queue.WriteBuffer(l.vCache, 0, wgpu.ToBytes(vw)); err != nil {
			return err
		}
		return rd.c.queue.WriteBuffer(l.vScale, 0, wgpu.ToBytes(vs))
	case rd.rm.kvF16:
		if err := rd.c.queue.WriteBuffer(l.kCache, 0, wgpu.ToBytes(packF16Pairs(keys))); err != nil {
			return err
		}
		return rd.c.queue.WriteBuffer(l.vCache, 0, wgpu.ToBytes(packF16Pairs(vals)))
	default:
		if err := rd.c.queue.WriteBuffer(l.kCache, 0, wgpu.ToBytes(keys)); err != nil {
			return err
		}
		return rd.c.queue.WriteBuffer(l.vCache, 0, wgpu.ToBytes(vals))
	}
}

// Reset clears resident state for a fresh generation. KV positions are overwritten
// (caller tracks pos), but Mamba {win, ssm} state COMPOUNDS, so it must be re-zeroed
// each generation or the recurrence carries over from the prior sequence.
func (rd *residentDecoder) Reset() {
	if rd.rm.mamba == nil {
		return
	}
	mp := rd.rm.mamba
	winZ := make([]float32, (mp.dConv-1)*mp.convDim)
	ssmZ := make([]float32, mp.nHeads*mp.hp*mp.dn)
	for i := range rd.rm.layers {
		if lw := &rd.rm.layers[i]; lw.isMamba {
			rd.c.queue.WriteBuffer(lw.mambaWin, 0, wgpu.ToBytes(winZ))
			rd.c.queue.WriteBuffer(lw.mambaSSM, 0, wgpu.ToBytes(ssmZ))
		}
	}
}

func (rd *residentDecoder) Close() error { rd.release(); return nil }

func (rd *residentDecoder) release() {
	// batch[0] aliases rd.runner; release the extra verify runners (batch[1:]) — they
	// own scratch but share rm's weights/KV (freed via rd.keep below).
	for i := 1; i < len(rd.batch); i++ {
		rd.batch[i].Release()
	}
	rd.batch = nil
	if rd.runner != nil {
		rd.runner.Release()
		rd.runner = nil
	}
	for i := len(rd.keep) - 1; i >= 0; i-- {
		rd.keep[i]()
	}
	rd.keep = nil
}
