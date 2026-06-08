package decoder

import (
	"fmt"
	"math"
	"os"
)

var g4debug = os.Getenv("G4DEBUG") != "" // env-gated per-layer hidden-norm trace

// g4traceHidden, when non-nil, receives the residual stream after each layer
// (layer -1 = post-embedding) on every runLayersGemma4 call. Debug/test only:
// set it around the final token's forward to capture a last-position layer trace
// for diffing against the HF reference. The callback must copy h if it retains it.
var g4traceHidden func(layer int, h []float32)

// g4traceQKV, when non-nil, receives the post-norm/RoPE q, k, v for each layer on
// every runLayersGemma4 call (k/v nil on KV-shared layers). Debug/test only.
var g4traceQKV func(layer int, q, k, v []float32)

// runLayersGemma4 is the Gemma 4 (E-model) forward over one token. It diverges
// from the generic runLayers enough to warrant its own path: per-layer head_dim
// (256 local / 512 global), per-layer KV-head count, scale-less v-norm,
// cross-layer KV sharing (the last N layers reuse an earlier layer's KV),
// per-attention-type RoPE (proportional / partial-rotary on the global layers),
// the Per-Layer-Embedding (PLE) residual branch, and a per-layer output scalar.
// Returns the post-final-norm... no — returns the pre-final-norm hidden, matching
// runLayers; forward() applies the final norm + LM head + softcap (30).
//
// Buffers are allocated per call (parity-first; not the perf path).
func (m *Model) runLayersGemma4(id int, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	g4 := arch.gemma4
	be := m.be
	hidden := arch.HiddenDim
	nH := arch.NumHeads
	pleDim := g4.HiddenSizePerLayerInput
	pos := cache.Pos()

	// Embedding × √hidden (inputs_embeds).
	h := make([]float32, hidden)
	m.w.Embed.embedRow(id, h)
	es := float32(arch.EmbedScale)
	for i := range h {
		h[i] *= es
	}
	if g4traceHidden != nil {
		g4traceHidden(-1, h)
	}

	// RoPE tables: local (full rotary, base 10k) and global (proportional /
	// partial-rotary, base 1e6 — only the first GlobalRotaryDim dims rotate).
	localInv := gemma4InvFreq(arch.HeadDim, arch.HeadDim, arch.RoPELocalBase)
	globalInv := gemma4InvFreq(g4.GlobalHeadDim, g4.GlobalRotaryDim, arch.RoPEGlobalBase)

	// Per-Layer-Embedding inputs: (token_identity + context_aware) / √2, shaped
	// [NumLayers, pleDim]. token_identity = per_layer_token_embd[id] × √pleDim;
	// context_aware = norm( per_layer_model_proj(inputs_embeds) × hidden^-0.5 ).
	perLayer := make([]float32, arch.NumLayers*pleDim)
	if pleDim > 0 {
		m.w.PerLayerTokenEmbed.embedRow(id, perLayer)
		tScale := float32(math.Sqrt(float64(pleDim)))
		for i := range perLayer {
			perLayer[i] *= tScale
		}
		ctxAware := make([]float32, arch.NumLayers*pleDim)
		m.w.PerLayerModelProj.matmul(be, h, ctxAware, 1)
		cScale := float32(1.0 / math.Sqrt(float64(hidden)))
		for i := range ctxAware {
			ctxAware[i] *= cScale
		}
		inv2 := float32(1.0 / math.Sqrt2)
		for l := 0; l < arch.NumLayers; l++ {
			row := ctxAware[l*pleDim : (l+1)*pleDim]
			normalize(arch, row, m.w.PerLayerProjNorm, nil, pleDim) // RMSNorm over pleDim
			pl := perLayer[l*pleDim : (l+1)*pleDim]
			for i := range pl {
				pl[i] = (pl[i] + row[i]) * inv2
			}
		}
	}

	// Cross-layer KV sharing: layers ≥ firstShared reuse the KV of the last
	// non-shared layer of their type (sliding vs global).
	firstShared := arch.NumLayers - g4.SharedKVLayers
	lastSliding, lastGlobal := -1, -1
	for i := range firstShared {
		if arch.isGlobalLayer(i) {
			lastGlobal = i
		} else {
			lastSliding = i
		}
	}
	kvSrc := func(l int) int {
		if l < firstShared {
			return l
		}
		if arch.isGlobalLayer(l) {
			return lastGlobal
		}
		return lastSliding
	}

	for l := 0; l < arch.NumLayers; l++ {
		lw := &m.w.Layers[l]
		global := arch.isGlobalLayer(l)
		hd := arch.headDimAt(l)
		nKV := arch.kvHeadsAt(l)
		ffn := arch.ffnAt(l)
		invFreq := localInv
		if global {
			invFreq = globalInv
		}

		// --- attention sub-block (sandwich) ---
		normd := make([]float32, hidden)
		copy(normd, h)
		normalize(arch, normd, lw.PreAttnNorm, nil, hidden)

		q := make([]float32, nH*hd)
		lw.QProj.matmul(be, normd, q, 1)
		rmsNorm(q, lw.QNorm, nH, hd, arch.NormEps, arch.RMSAddOne)
		applyRoPE(q, nH, hd, pos, invFreq, 1.0)

		if l < firstShared { // owns its KV
			k := make([]float32, nKV*hd)
			lw.KProj.matmul(be, normd, k, 1)
			v := make([]float32, nKV*hd)
			if lw.VFromK { // attention_k_eq_v: V is v_norm(k_proj output)
				copy(v, k)
			} else {
				lw.VProj.matmul(be, normd, v, 1)
			}
			rmsNorm(k, lw.KNorm, nKV, hd, arch.NormEps, arch.RMSAddOne) // K: k_norm + RoPE
			applyRoPE(k, nKV, hd, pos, invFreq, 1.0)
			rmsNormNoWeight(v, nKV, hd, arch.NormEps) // V: scale-less v_norm, no RoPE
			cache.Append(l, k, v)
			if g4traceQKV != nil {
				g4traceQKV(l, q, k, v)
			}
		} else if g4traceQKV != nil {
			g4traceQKV(l, q, nil, nil)
		}
		src := kvSrc(l)
		keys, vals := cache.Keys(src), cache.Vals(src)
		kvDim := nKV * hd
		nKeys := len(keys) / kvDim
		ctx := make([]float32, nH*hd)
		start := cache.WindowStart(pos, global)
		gemma4Attend(q, ctx, keys, vals, nH, nKV, hd, start, nKeys, arch.AttnScale)

		attnOut := make([]float32, hidden)
		lw.OProj.matmul(be, ctx, attnOut, 1)
		normalize(arch, attnOut, lw.PostAttnNorm, nil, hidden) // post-attn (sandwich)
		for i := range h {
			h[i] += attnOut[i]
		}

		// --- MLP sub-block (sandwich), GeGLU at the per-layer width ---
		copy(normd, h)
		normalize(arch, normd, lw.PreMLPNorm, nil, hidden)
		gate := make([]float32, ffn)
		up := make([]float32, ffn)
		lw.GateProj.matmul(be, normd, gate, 1)
		lw.UpProj.matmul(be, normd, up, 1)
		for i := range gate {
			gate[i] = geluTanh(gate[i]) * up[i]
		}
		mlpOut := make([]float32, hidden)
		lw.DownProj.matmul(be, gate, mlpOut, 1)
		normalize(arch, mlpOut, lw.PostMLPNorm, nil, hidden) // post-FFN (sandwich)
		for i := range h {
			h[i] += mlpOut[i]
		}

		// --- PLE branch: gate→gelu→×per-layer-embedding→proj→norm→+residual ---
		if pleDim > 0 {
			pl := perLayer[l*pleDim : (l+1)*pleDim]
			px := make([]float32, pleDim)
			lw.PLEGate.matmul(be, h, px, 1)
			for i := range px {
				px[i] = geluTanh(px[i]) * pl[i]
			}
			pout := make([]float32, hidden)
			lw.PLEProj.matmul(be, px, pout, 1)
			normalize(arch, pout, lw.PostPLENorm, nil, hidden)
			for i := range h {
				h[i] += pout[i]
			}
		}

		// --- per-layer output scalar ---
		if lw.LayerScalar != 0 {
			for i := range h {
				h[i] *= lw.LayerScalar
			}
		}
		if g4debug {
			var ss float64
			for _, e := range h {
				ss += float64(e) * float64(e)
			}
			kind := "loc"
			if global {
				kind = "GLB"
			}
			vk := ""
			if lw.VFromK {
				vk = " kv=eq"
			}
			fmt.Printf("  L%-2d %s hd=%d nkv=%d scale=%.4f ||h||=%.3f%s\n", l, kind, hd, nKV, lw.LayerScalar, math.Sqrt(ss), vk)
		}
		if g4traceHidden != nil {
			g4traceHidden(l, h)
		}
	}
	cache.Advance()
	return h, nil
}

// gemma4InvFreq builds the RoPE inverse-frequency table. For full rotary,
// rotaryDim == headDim and the result is headDim/2 entries. For proportional
// (partial) rotary, rotaryDim < headDim: the first rotaryDim/2 entries are the
// real frequencies (spaced by 1/headDim, NOT 1/rotaryDim) and the rest are zero,
// so applyRoPE (which rotates len(invFreq) pairs split at headDim/2) leaves the
// trailing dims unrotated — matching HF's "proportional" RoPE.
func gemma4InvFreq(headDim, rotaryDim int, base float64) []float64 {
	half := headDim / 2
	inv := make([]float64, half)
	angles := rotaryDim / 2
	for i := range angles {
		inv[i] = 1.0 / math.Pow(base, float64(2*i)/float64(headDim))
	}
	return inv // entries [angles:half] stay 0 → no rotation there
}

// rmsNormNoWeight applies a scale-less RMSNorm per head (Gemma 4's v_norm:
// with_scale=False — normalize V by RMS, no learned weight).
func rmsNormNoWeight(x []float32, rows, dim int, eps float64) {
	for r := range rows {
		v := x[r*dim : r*dim+dim]
		var ss float64
		for _, e := range v {
			ss += float64(e) * float64(e)
		}
		inv := float32(1.0 / math.Sqrt(ss/float64(dim)+eps))
		for i := range v {
			v[i] *= inv
		}
	}
}

// gemma4Attend is attendQuery with explicit per-layer dims (head_dim and
// KV-head count vary by layer). scale is arch.AttnScale (1.0 — the q/k-norm
// weights absorb the scaling).
func gemma4Attend(q, ctx, keys, vals []float32, nH, nKV, hd, start, nKeys int, scale float64) {
	kvDim := nKV * hd
	group := nH / nKV
	clear(ctx)
	scores := make([]float32, nKeys)
	for qh := range nH {
		kvh := qh / group
		qHead := q[qh*hd : qh*hd+hd]
		maxS := math.Inf(-1)
		for s := start; s < nKeys; s++ {
			kHead := keys[s*kvDim+kvh*hd : s*kvDim+kvh*hd+hd]
			var dot float64
			for d := range hd {
				dot += float64(qHead[d]) * float64(kHead[d])
			}
			sc := dot * scale
			scores[s] = float32(sc)
			if sc > maxS {
				maxS = sc
			}
		}
		var sum float64
		for s := start; s < nKeys; s++ {
			e := math.Exp(float64(scores[s]) - maxS)
			scores[s] = float32(e)
			sum += e
		}
		inv := 1.0 / sum
		oHead := ctx[qh*hd : qh*hd+hd]
		for s := start; s < nKeys; s++ {
			w := float32(float64(scores[s]) * inv)
			vHead := vals[s*kvDim+kvh*hd : s*kvDim+kvh*hd+hd]
			for d := range hd {
				oHead[d] += w * vHead[d]
			}
		}
	}
}
