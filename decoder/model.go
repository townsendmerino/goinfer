package decoder

import (
	"context"
	"fmt"
	"math"
	"slices"
)

// Model is a loaded Gemma 3 checkpoint plus the compute backend. Goroutine
// safety follows encoder.Model: Weights are immutable after Load; per-
// sequence state (the KV cache) is owned by each Generate call, so distinct
// sequences can run concurrently, but a single KVCache is not shared.
type Model struct {
	w      *Weights
	be     Backend
	eosIDs []int // end-of-sequence ids from config (generation stops on these)
}

// Options configures Load.
type Options struct {
	Backend string // "cpu" (default) or "webgpu"
	Quant   string // "" (f32), "int8" (weight-only per-row), "int8int8" (full int8×int8 W8A8), or "int4" (group-wise) (M8)
}

// Load reads a Gemma 3 snapshot (config.json + model.safetensors) from dir
// and selects a backend. The forward pass (M3) is implemented; the CPU
// backend is the default and the only one wired (webgpu falls back to CPU).
func Load(dir string, opts Options) (*Model, error) {
	be, beErr := NewBackend(opts.Backend)
	// beErr is non-nil for the not-yet-implemented webgpu fallback; keep the
	// (cpu) backend and surface the note rather than abort.

	// Resolve the quant mode first so the weights stream straight into the
	// chosen precision at load — no whole-model f32 spike (see loadWeights).
	var quant quantMode
	switch opts.Quant {
	case "", "f32":
		quant = quantNone
	case "int8":
		quant = quantInt8
	case "int8int8":
		quant = quantInt8I8
	case "int4":
		quant = quantInt4
	default:
		if be != nil {
			_ = be.Close()
		}
		return nil, fmt.Errorf("decoder.Load: unknown quant %q (have: int8, int8int8, int4)", opts.Quant)
	}

	w, err := loadWeights(dir, quant)
	if err != nil {
		if be != nil {
			_ = be.Close()
		}
		return nil, err
	}
	if beErr != nil {
		// webgpu requested but fell back — not fatal.
		fmt.Println(beErr)
	}
	return &Model{w: w, be: be, eosIDs: resolveEOSIDs(dir, &w.Cfg)}, nil
}

// Config exposes the loaded architecture config.
func (m *Model) Config() *Config { return &m.w.Cfg }

// NewCache allocates a KV cache sized for this model. capHint pre-sizes for
// a known max length (0 = grow on demand).
func (m *Model) NewCache(capHint int) *KVCache {
	a := m.w.arch
	return NewKVCache(a.NumLayers, a.NumKVHeads, a.HeadDim, a.SlidingWindow, capHint)
}

// runLayers advances one decode step for token id at position cache.Pos():
// it embeds the token, runs the block stack (appending this position's K/V to
// the cache), and returns the residual-stream hidden state after the final
// layer — BEFORE the final norm and LM head. Splitting it out lets prefill skip
// the (vocab-sized) LM head on every token but the last.
//
// The loop is generic over the Architecture descriptor (G0): embedding scale,
// norm placement (Gemma's 4-norm sandwich vs Llama's pre-2), the (1+w) RMS
// offset, and the activation are all knobs. Gemma 3 is one descriptor:
//
//	h = Embed[id] * EmbedScale
//	for each layer l:
//	  n  = rmsNorm(h, PreAttnNorm)
//	  a  = causalAttention(l, n, …)
//	  if Sandwich4 { a = rmsNorm(a, PostAttnNorm) }
//	  h += a
//	  n2 = rmsNorm(h, PreMLPNorm)
//	  g  = gatedMLP(n2, …)
//	  if Sandwich4 { g = rmsNorm(g, PostMLPNorm) }
//	  h += g
func (m *Model) runLayers(id int, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	if m.w.Embed.rows == 0 {
		return nil, fmt.Errorf("decoder.forward: weights not loaded %w [M1]", errNotImplemented)
	}
	hidden := arch.HiddenDim
	h := make([]float32, hidden)
	m.w.Embed.embedRow(id, h) // f32 copy, or int8 dequant when quantized
	// Embedding scale (Gemma = √hidden; 0/1 = none). NOTE: HF computes this
	// normalizer as sqrt(hidden) cast to the model's dtype — bf16 for a bf16
	// checkpoint (≈25.25 here) — then multiplies. We use the f32 value
	// (≈25.2982). It matches our parity gate because the next op (PreAttnNorm
	// RMSNorm) divides out a global scalar, so the difference only survives in
	// the residual and stays well under the ≥1−1e-4 cosine bar. If that bar is
	// ever tightened past ~1e-5, round the scale to bf16. See M3-forward.md.
	if arch.EmbedScale != 0 && arch.EmbedScale != 1 {
		scale := float32(arch.EmbedScale)
		for i := range h {
			h[i] *= scale
		}
	}
	// Learned absolute position embedding (GPT-2): add wpe[pos], where pos is
	// this token's absolute position (the cache advances on Append inside
	// attention, so cache.Pos() here is still this step's position).
	if arch.LearnedPosEmbed {
		pe := make([]float32, hidden)
		m.w.PosEmbed.embedRow(cache.Pos(), pe)
		for i := range h {
			h[i] += pe[i]
		}
	}
	sandwich := arch.NormPlacement == NormSandwich4
	for l := 0; l < arch.NumLayers; l++ {
		lw := &m.w.Layers[l]
		n := append([]float32(nil), h...)
		normalize(arch, n, lw.PreAttnNorm, lw.PreAttnNormBias, hidden)
		att, err := causalAttention(l, n, lw, arch, cache, m.be)
		if err != nil {
			return nil, err
		}
		if sandwich {
			normalize(arch, att, lw.PostAttnNorm, nil, hidden)
		}
		for i := range h {
			h[i] += att[i]
		}
		n2 := append([]float32(nil), h...)
		normalize(arch, n2, lw.PreMLPNorm, lw.PreMLPNormBias, hidden)
		g, err := mlp(n2, lw, arch, m.be)
		if err != nil {
			return nil, err
		}
		if sandwich {
			normalize(arch, g, lw.PostMLPNorm, nil, hidden)
		}
		for i := range h {
			h[i] += g[i]
		}
	}
	return h, nil
}

// normalize applies the architecture's normalization in place over one row:
// LayerNorm (mean-centered, with bias) for GPT-2/NeoX, else RMSNorm. bias is
// ignored by RMSNorm (and nil for the Sandwich4 post-norms).
func normalize(arch *Architecture, x, weight, bias []float32, dim int) {
	if arch.Norm == NormLayer {
		layerNorm(x, weight, bias, 1, dim, arch.NormEps)
		return
	}
	rmsNorm(x, weight, 1, dim, arch.NormEps, arch.RMSAddOne)
}

// forward runs runLayers then the final norm + LM head, returning the logit
// vector ([VocabSize]) for the next token. The head is the tied embedding
// (Gemma) or a separate lm_head (untied; multi-model-plan G2). Optional final
// logit soft-capping (Gemma 2; Gemma 3 = none).
func (m *Model) forward(id int, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	h, err := m.runLayers(id, cache)
	if err != nil {
		return nil, err
	}
	normalize(arch, h, m.w.FinalNorm, m.w.FinalNormBias, arch.HiddenDim)
	logits := make([]float32, arch.VocabSize)
	if arch.TiedLMHead {
		m.w.Embed.matmul(m.be, h, logits, 1) // tied: embedding doubles as the head
	} else {
		m.w.LMHead.matmul(m.be, h, logits, 1) // separate output projection
	}
	if arch.FinalLogitSoftcap > 0 {
		softcap := float32(arch.FinalLogitSoftcap)
		for i, v := range logits {
			logits[i] = softcap * float32(math.Tanh(float64(v/softcap)))
		}
	}
	return logits, nil
}

// Generate streams generated token ids over the returned channel until EOS,
// a stop id, maxTokens, or ctx cancellation. prompt is already-tokenized
// ids (the demo runs the tokenizer). The channel closes when generation
// ends; check Err after the range loop for a terminal error.
//
// Sampling is greedy at Temperature 0, else temperature/top-k/top-p (see
// Sampler). A SamplingParams.LogitProcessor, if set, masks each step's logits
// before sampling — the seam for constrained/structured decoding.
func (m *Model) Generate(ctx context.Context, prompt []int, maxTokens int, sp SamplingParams) (<-chan int, *Generation) {
	out := make(chan int)
	g := &Generation{}
	go func() {
		defer close(out)
		if len(prompt) == 0 {
			g.err = fmt.Errorf("decoder.Generate: empty prompt")
			return
		}
		cache := m.NewCache(len(prompt) + maxTokens)
		sampler := NewSampler(sp)
		// Prefill: run every prompt token through the cache. Only the last
		// token needs logits (to seed the first generated token), so the rest
		// skip the vocab-sized LM head via runLayers.
		for _, id := range prompt[:len(prompt)-1] {
			if _, err := m.runLayers(id, cache); err != nil {
				g.err = err
				return
			}
		}
		logits, err := m.forward(prompt[len(prompt)-1], cache)
		if err != nil {
			g.err = err
			return
		}
		// Decode loop.
		var generated []int
		for range maxTokens {
			select {
			case <-ctx.Done():
				g.err = ctx.Err()
				return
			default:
			}
			// Constrained decoding: let the processor mask this step's logits
			// (based on what's been generated) before sampling and the stop check.
			if sp.LogitProcessor != nil {
				sp.LogitProcessor(generated, logits)
			}
			next, err := sampler.Sample(logits)
			if err != nil {
				g.err = err
				return
			}
			if m.isStop(next, sp) {
				return
			}
			out <- next
			generated = append(generated, next)
			if logits, err = m.forward(next, cache); err != nil {
				g.err = err
				return
			}
		}
	}()
	return out, g
}

// isStop reports whether id ends generation: a checkpoint EOS id (from
// config) or a caller-supplied stop id (SamplingParams.StopIDs, e.g.
// <end_of_turn> for chat).
func (m *Model) isStop(id int, sp SamplingParams) bool {
	if slices.Contains(m.eosIDs, id) {
		return true
	}
	return slices.Contains(sp.StopIDs, id)
}

// Generation carries the terminal status of a Generate stream.
type Generation struct{ err error }

// Err returns the error that ended the stream, or nil if it ended cleanly
// (EOS / stop / maxTokens). Read it after the channel closes.
func (g *Generation) Err() error { return g.err }
