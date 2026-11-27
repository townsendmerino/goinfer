package cpuengine

import (
	"fmt"
	"math"
	"time"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/normact"
)

// RunLayers advances one decode step for token id at position cache.Pos(): it embeds the token, runs the block stack
// (appending this position's K/V to the cache), and returns the residual-stream hidden state after the final layer, before the
// final norm and LM head. Splitting it out lets prefill skip the (vocab-sized) LM head on every token but the last.
//
// The loop is generic over the Architecture descriptor: embedding scale, norm placement (Gemma's 4-norm sandwich vs Llama's
// pre-2), the (1+w) RMS offset, and the activation are all knobs. Gemma 3 is one descriptor:
//
//	h = Embed[id] * EmbedScale
//	for each layer l:
//	  n  = rmsNorm(h, PreAttnNorm)
//	  a  = attention(l, n, ...)
//	  if Sandwich4 { a = rmsNorm(a, PostAttnNorm) }
//	  h += a
//	  n2 = rmsNorm(h, PreMLPNorm)
//	  g  = ffn(n2, ...)
//	  if Sandwich4 { g = rmsNorm(g, PostMLPNorm) }
//	  h += g
func (m *Engine) RunLayers(id int, cache *Cache) ([]float32, error) {
	arch := m.W.Arch
	if m.W.Embed.Rows() == 0 {
		return nil, fmt.Errorf("decoder.Forward: weights not loaded %w [M1]", ErrNotImplemented)
	}
	if cache.Scr == nil { // caches from NewCache directly (tests); Generate uses NewCache
		cache.Knobs, cache.Scr = m.Knobs, NewDecodeScratch(arch, m.Knobs)
	}
	scr := cache.Scr
	hidden := arch.HiddenDim
	h := scr.h // residual stream (reused per stream; fully overwritten below)
	t0 := CostStart(cache)
	m.W.Embed.Row(id, h)                    // f32 copy, or int8 dequant when quantized
	if hw := arch.Inkling2Head; hw != nil { // Inkling 2 normalizes a token's embedding (an image row skips it)
		normact.RMS(h, hw.EmbedNorm, 1, hidden, arch.NormEps, false)
	}
	// Embedding scale (Gemma = √hidden; 0/1 = none). NOTE: HF computes this
	// normalizer as sqrt(hidden) cast to the model's dtype — bf16 for a bf16
	// checkpoint (≈25.25 here) — then multiplies. We use the f32 value
	// (≈25.2982). It matches our parity gate because the next op (PreAttnNorm
	// RMSNorm) divides out a global scalar, so the difference only survives in
	// the residual and stays well under the ≥1−1e-4 cosine bar. If that bar is
	// ever tightened past ~1e-5, round the scale to bf16.
	emb := m.Program().Embed
	if emb.Scale != 0 && emb.Scale != 1 {
		scale := float32(emb.Scale)
		for i := range h {
			h[i] *= scale
		}
	}
	// Learned absolute position embedding (GPT-2): add wpe[pos], where pos is
	// this token's absolute position (the cache advances on Append inside
	// attention, so cache.Pos() here is still this step's position).
	if emb.LearnedPos {
		pe := make([]float32, hidden)
		m.W.PosEmbed.Row(cache.Pos(), pe)
		for i := range h {
			h[i] += pe[i]
		}
	}
	if cache.Cost != nil {
		cache.Cost.Tokens(cache.CostPhase, 1)
		m.CountEmbed(cache, 1, emb.LearnedPos, t0)
	}
	return m.RunLayersFromEmbedPLE(h, id, cache)
}

// RunLayersFromEmbed runs the transformer layers over a precomputed
// residual-stream embedding h ([hidden]) and returns it (mutated in place). For
// a text token h is the embedding lookup (+ scale/pos) that runLayers just
// computed; for an IMAGE position (multimodal) it is the projected vision
// embedding the interleaver substitutes in place of a token-id lookup. Splitting
// it out is the embed-by-vector seam: text behavior is unchanged (runLayers is
// exactly embed-then-this), and image embeddings reach the decoder without
// passing through embed_tokens. Generic path only (gemma4/qwen35 have their own
// runLayers and grow the same seam when those families go multimodal).
func (m *Engine) RunLayersFromEmbed(h []float32, cache *Cache) ([]float32, error) {
	pad := -1
	if g := m.W.Arch.Gemma4; g != nil {
		pad = g.PadTokenID // a row with no token behind it takes the pad id's PLE, as HF's multimodal forward does
	}
	return m.RunLayersFromEmbedPLE(h, pad, cache)
}

// RunLayersFromEmbedPLE is runLayersFromEmbed with the token id behind h for Gemma 4's per-layer embedding: the id itself for a
// text row, the checkpoint's pad id for a multimodal row (docs/multimodal.md, the Gemma 4 vision entry).
func (m *Engine) RunLayersFromEmbedPLE(h []float32, pleID int, cache *Cache) ([]float32, error) {
	if cache.Scr == nil { // direct callers (tests / the interleaver) may skip runLayers
		cache.Knobs, cache.Scr = m.Knobs, NewDecodeScratch(m.W.Arch, m.Knobs)
	}
	return m.WalkToken(m.Program(), h, pleID, cache)
}

func AddResidual(h, sub []float32) {
	n := min(len(h), len(sub))
	if n == 0 {
		return
	}
	h = h[:n]
	sub = sub[:n]
	_ = h[n-1]
	_ = sub[n-1]
	i := 0
	for ; i+3 < n; i += 4 {
		_ = h[i+3]
		_ = sub[i+3]
		h[i] += sub[i]
		h[i+1] += sub[i+1]
		h[i+2] += sub[i+2]
		h[i+3] += sub[i+3]
	}
	for ; i < n; i++ {
		h[i] += sub[i]
	}
}

func AddResidual2(h, sub, sub2 []float32) {
	n := min(len(h), min(len(sub), len(sub2)))
	if n == 0 {
		return
	}
	h = h[:n]
	sub = sub[:n]
	sub2 = sub2[:n]
	_ = h[n-1]
	_ = sub[n-1]
	_ = sub2[n-1]
	i := 0
	for ; i+3 < n; i += 4 {
		_ = h[i+3]
		_ = sub[i+3]
		_ = sub2[i+3]
		h[i] += sub[i] + sub2[i]
		h[i+1] += sub[i+1] + sub2[i+1]
		h[i+2] += sub[i+2] + sub2[i+2]
		h[i+3] += sub[i+3] + sub2[i+3]
	}
	for ; i < n; i++ {
		h[i] += sub[i] + sub2[i]
	}
}

// NormalizeInto applies the architecture's normalization from src into dst over one row:
// LayerNorm (mean-centered, with bias) for GPT-2/NeoX, else RMSNorm. bias is
// ignored by RMSNorm (and nil for the Sandwich4 post-norms).
func NormalizeInto(arch *darch.Architecture, dst, src, weight, bias []float32, dim int) {
	if arch.Norm == darch.NormLayer {
		normact.LayerInto(dst, src, weight, bias, 1, dim, arch.NormEps)
		return
	}
	normact.RMSInto(dst, src, weight, 1, dim, arch.NormEps, arch.RMSAddOne)
}

// Normalize applies the architecture's normalization in place over one row.
func Normalize(arch *darch.Architecture, x, weight, bias []float32, dim int) {
	NormalizeInto(arch, x, x, weight, bias, dim)
}

// Forward runs runLayers then the final norm + LM head, returning the logit
// vector ([VocabSize]) for the next token. The head is the tied embedding
// (Gemma) or a separate lm_head (untied). Optional final
// logit soft-capping (Gemma 2; Gemma 3 = none).
func (m *Engine) Forward(id int, cache *Cache) ([]float32, error) {
	h, err := m.RunLayers(id, cache)
	if err != nil {
		return nil, err
	}
	if activeHooks().DecodeTiming {
		t0 := time.Now()
		lg := m.LogitsFromHidden(h, cache)
		activeHooks().DecodeTimingStats.add(func() *int64 {
			if st := activeHooks().DecodeTimingStats; st != nil {
				return st.Head
			}
			return nil
		}(), int64(time.Since(t0)))
		return lg, nil
	}
	return m.LogitsFromHidden(h, cache), nil
}

// ForwardCapture runs one forward for token id and returns the next-token logits plus the residual stream after each layer in
// `layers` (cloned): the read-only hidden-state seam a draft head reads (block drafters). The forward is byte-identical to
// forward(id); the captures are copies that never feed back. Layer indices are 0-based into [0, NumLayers); out[i]
// corresponds to layers[i].
//
// Every family's layers run through the program walker, which captures the residual after each layer (gate C of §4.8's
// seams PR holds it to the forward cut after that layer, bit for bit).

// ForwardCapture is ForwardCapture without the lifecycle guard (lifecycle.go).
func (m *Engine) ForwardCapture(id int, cache *Cache, layers []int) (logits []float32, hidden [][]float32, err error) {
	a := m.W.Arch
	for _, l := range layers {
		if l < 0 || l >= a.NumLayers {
			return nil, nil, fmt.Errorf("decoder.ForwardCapture: layer %d out of range [0,%d)", l, a.NumLayers)
		}
	}
	cache.CaptureLayers = layers
	cache.Captured = make([][]float32, len(layers))
	defer func() { cache.CaptureLayers, cache.Captured = nil, nil }()
	lg, ferr := m.Forward(id, cache)
	if ferr != nil {
		return nil, nil, ferr
	}
	return lg, cache.Captured, nil
}

// ForwardSubCapture runs one token and returns, per layer, the attention contribution and the
// MLP contribution to the residual (scr.sub after each sublayer's sandwich-norm, before the add).
// This is the finer seam ForwardCapture doesn't give: it separates the two sublayers, which is
// what localizing a channel's sign flip to attention-vs-MLP at a specific layer requires.
// Diagnostic — same byte-identical-output contract as ForwardCapture. A single-operation block (Nemotron-H) fills one of the
// two; gate S of §4.8's seams PR rebuilds each layer's residual from them. Refused where a layer has an operation they do not
// cover (program.subCaptureUncovered).

// ForwardFromEmbed is forward for a position whose residual-stream embedding is
// supplied directly (the multimodal embed-by-vector seam): it runs the layers
// from h and projects to logits, identical to forward(id) when h is that token's
// embedding.
func (m *Engine) ForwardFromEmbed(h []float32, cache *Cache) ([]float32, error) {
	h, err := m.RunLayersFromEmbed(h, cache)
	if err != nil {
		return nil, err
	}
	return m.LogitsFromHidden(h, cache), nil
}

// SoftcapParallel applies Gemma's final-logit softcap sc*tanh(x/sc) in place, fanned out via parallelElementwise. Every
// element is independent and math.Tanh is deterministic, so splitting the loop is byte-identical to the serial form
// (TestSoftcapParallel_bitIdentical). It mirrors metal/model.go's SoftcapParallel.
func SoftcapParallel(logits []float32, softcap float32) {
	sc := softcap
	parallelElementwise(len(logits), func(lo, hi int) {
		for j := lo; j < hi; j++ {
			logits[j] = sc * float32(math.Tanh(float64(logits[j]/sc)))
		}
	})
}

// LogitsFromHidden applies the final norm + LM head (tied embedding or separate
// lm_head) + optional Gemma logit soft-cap to a layer-stack output h, returning
// the next-token logits ([VocabSize]). One home for the head math, shared by
// forward and forwardFromEmbed.
func (m *Engine) LogitsFromHidden(h []float32, cache *Cache) []float32 {
	hp := &m.Program().Head
	b := m.W.FinalNormBias
	if !hp.Norm.Bias {
		b = nil
	}
	t0 := CostStart(cache)
	ApplyNorm(hp.Norm, h, h, m.W.FinalNorm, b, m.W.Arch.HiddenDim)
	if cache.Cost != nil {
		m.CountFinalNorm(cache, 1, b, t0)
		t0 = time.Now()
	}
	lg := m.LogitsFromNormed(h, cache)
	if cache.Cost != nil {
		m.CountHead(cache, 1, t0)
	}
	return lg
}

// LogitsFromNormed is logitsFromHidden for a hidden state that has ALREADY been through the final norm (the batched
// Qwen3.5 forward returns its rows normed): the LM head, softcap and logit scale, exactly as logitsFromHidden runs them.
func (m *Engine) LogitsFromNormed(h []float32, cache *Cache) []float32 {
	hp := &m.Program().Head
	logits := cache.Scr.logits // reused per stream; Matmul fully overwrites it
	if hp.Tied {
		MatmulInto(cache.Scr.WS, m.Be, &m.W.Embed, h, logits, 1) // tied: embedding doubles as the head
	} else {
		MatmulInto(cache.Scr.WS, m.Be, &m.W.LMHead, h, logits, 1) // separate output projection
	}
	if hp.Softcap > 0 {
		SoftcapParallel(logits, float32(hp.Softcap))
	}
	if hp.LogitScale != 0 && hp.LogitScale != 1 { // Granite logits_scaling: logits /= scale
		inv := float32(1 / hp.LogitScale)
		n := len(logits)
		if n > 0 {
			_ = logits[n-1]
			i := 0
			for ; i+3 < n; i += 4 {
				_ = logits[i+3]
				logits[i] *= inv
				logits[i+1] *= inv
				logits[i+2] *= inv
				logits[i+3] *= inv
			}
			for ; i < n; i++ {
				logits[i] *= inv
			}
		}
	}
	return logits
}
