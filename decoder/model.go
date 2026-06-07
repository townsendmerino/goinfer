package decoder

import (
	"context"
	"fmt"
	"math"
	"slices"
	"strings"
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
	LoRA    string // optional PEFT adapter dir (adapter_config.json + adapter_model.safetensors), merged into the base at load. Safetensors base only.
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
	quant, err := parseQuant(opts.Quant)
	if err != nil {
		closeBackend(be)
		return nil, err
	}

	// Optional LoRA adapter, merged into the base weights at load (safetensors base
	// only — PEFT targets HF module names).
	var lora *loraAdapter
	if opts.LoRA != "" {
		if strings.HasSuffix(dir, ".gguf") {
			closeBackend(be)
			return nil, fmt.Errorf("decoder: LoRA merge needs a safetensors base, not a .gguf")
		}
		if lora, err = loadLoRA(opts.LoRA); err != nil {
			closeBackend(be)
			return nil, err
		}
		defer lora.close()
	}

	w, err := loadWeights(dir, quant, lora)
	if err != nil {
		closeBackend(be)
		return nil, err
	}
	if beErr != nil {
		// webgpu requested but fell back — not fatal.
		fmt.Println(beErr)
	}
	return &Model{w: w, be: be, eosIDs: resolveEOSIDs(dir, &w.Cfg)}, nil
}

// parseQuant maps Options.Quant to the internal quantMode.
func parseQuant(q string) (quantMode, error) {
	switch q {
	case "", "f32":
		return quantNone, nil
	case "int8":
		return quantInt8, nil
	case "int8int8":
		return quantInt8I8, nil
	case "int4":
		return quantInt4, nil
	default:
		return quantNone, fmt.Errorf("decoder: unknown quant %q (have: int8, int8int8, int4)", q)
	}
}

// closeBackend closes a backend on a load-error path (nil-safe).
func closeBackend(be Backend) {
	if be != nil {
		_ = be.Close()
	}
}

// Config exposes the loaded architecture config.
func (m *Model) Config() *Config { return &m.w.Cfg }

// NewCache allocates a KV cache sized for this model. capHint pre-sizes for
// a known max length (0 = grow on demand).
func (m *Model) NewCache(capHint int) *KVCache {
	a := m.w.arch
	c := NewKVCache(a.NumLayers, a.NumKVHeads, a.HeadDim, a.SlidingWindow, capHint)
	c.scr = newDecodeScratch(a)
	if a.gemma4 != nil {
		c.manualPos = true // gemma4's last layer is KV-shared; pos advances via Advance()
	}
	return c
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
	if arch.gemma4 != nil { // Gemma 4: per-layer head_dim, KV-sharing, PLE — own path.
		return m.runLayersGemma4(id, cache)
	}
	if cache.scr == nil { // caches from NewKVCache directly (tests); Generate uses NewCache
		cache.scr = newDecodeScratch(arch)
	}
	scr := cache.scr
	hidden := arch.HiddenDim
	h := scr.h                // residual stream (reused per stream; fully overwritten below)
	m.w.Embed.embedRow(id, h) // f32 copy, or int8 dequant when quantized
	// Embedding scale (Gemma = √hidden; 0/1 = none). NOTE: HF computes this
	// normalizer as sqrt(hidden) cast to the model's dtype — bf16 for a bf16
	// checkpoint (≈25.25 here) — then multiplies. We use the f32 value
	// (≈25.2982). It matches our parity gate because the next op (PreAttnNorm
	// RMSNorm) divides out a global scalar, so the difference only survives in
	// the residual and stays well under the ≥1−1e-4 cosine bar. If that bar is
	// ever tightened past ~1e-5, round the scale to bf16.
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
		copy(scr.norm, h)
		normalize(arch, scr.norm, lw.PreAttnNorm, lw.PreAttnNormBias, hidden)
		if err := causalAttention(l, scr.norm, scr.sub, lw, arch, cache, m.be); err != nil {
			return nil, err
		}
		if sandwich {
			normalize(arch, scr.sub, lw.PostAttnNorm, nil, hidden)
		}
		for i := range h {
			h[i] += scr.sub[i]
		}
		copy(scr.norm, h)
		normalize(arch, scr.norm, lw.PreMLPNorm, lw.PreMLPNormBias, hidden)
		if err := mlp(scr.norm, scr.sub, lw, arch, m.be, scr); err != nil {
			return nil, err
		}
		if sandwich {
			normalize(arch, scr.sub, lw.PostMLPNorm, nil, hidden)
		}
		for i := range h {
			h[i] += scr.sub[i]
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
// (Gemma) or a separate lm_head (untied). Optional final
// logit soft-capping (Gemma 2; Gemma 3 = none).
func (m *Model) forward(id int, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	h, err := m.runLayers(id, cache)
	if err != nil {
		return nil, err
	}
	normalize(arch, h, m.w.FinalNorm, m.w.FinalNormBias, arch.HiddenDim)
	logits := cache.scr.logits // reused per stream; matmul fully overwrites it
	if arch.TiedLMHead {
		m.w.Embed.matmulInto(cache.scr.ws, m.be, h, logits, 1) // tied: embedding doubles as the head
	} else {
		m.w.LMHead.matmulInto(cache.scr.ws, m.be, h, logits, 1) // separate output projection
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
		sampler.Observe(prompt...) // so repetition penalties see the prompt
		// Prefill the prompt and seed the first token's logits. On the batched
		// archs this runs the layers at M=len(prompt) in one pass (each weight
		// streamed once — ~1.7–2× faster TTFT than sequential), LM head on the
		// last position only.
		logits, err := m.prefillLogits(prompt, cache)
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
			info, err := sampler.SampleWithInfo(logits)
			if err != nil {
				g.err = err
				return
			}
			next := info.ID
			if m.isStop(next, sp) {
				return
			}
			if sp.Logprobs {
				g.Logprobs = append(g.Logprobs, info)
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

// Generation carries the terminal status of a Generate stream. Spec is non-nil
// for GenerateSpeculative and carries acceptance telemetry.
type Generation struct {
	err  error
	Spec *SpecStats
	// Logprobs holds one entry per emitted token (in order) when
	// SamplingParams.Logprobs was set — the chosen token's log-probability and
	// any requested top alternatives. Complete once the stream has closed.
	Logprobs []SampleInfo
}

// Err returns the error that ended the stream, or nil if it ended cleanly
// (EOS / stop / maxTokens). Read it after the channel closes.
func (g *Generation) Err() error { return g.err }
