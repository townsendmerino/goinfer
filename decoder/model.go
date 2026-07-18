package decoder

import (
	"context"
	"fmt"
	"math"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/townsendmerino/aikit/mmap"
	"github.com/townsendmerino/goinfer/internal/giw"
)

// decodeTiming env-gates per-component decode timing (GOINFER_DECODE_TIMING=1) so
// the residency path's per-token cost can be decomposed without touching the hot
// path otherwise.
var decodeTiming = os.Getenv("GOINFER_DECODE_TIMING") != ""

// Model is a loaded Gemma 3 checkpoint plus the compute backend. Goroutine
// safety follows encoder.Model: Weights are immutable after Load; per-
// sequence state (the KV cache) is owned by each Generate call, so distinct
// sequences can run concurrently, but a single KVCache is not shared.
type Model struct {
	w          *Weights
	be         Backend
	eosIDs     []int           // end-of-sequence ids from config (generation stops on these)
	resident   ResidentForward // GPU full-residency decode path (webgpu + eligible arch); nil ⇒ staged/CPU
	kvF16      bool            // residency KV cache precision request (Options.KVPrecision == "f16")
	kvPrecI8   bool            // residency KV cache int8 request (Options.KVPrecision == "i8") — GPU
	kvI8       bool            // CPU KV cache int8 storage request (Options.KVQuant == "i8") — CPU staged path
	mmap       []byte          // .giw mmap region the int8/int4 weights alias; munmap'd by Close (nil off the .giw mmap path)
	pager      *expertPager    // MoE expert demand-paging over the mapping (Options.StreamWeights); nil = all-resident
	layerPager *layerPager     // dense per-layer streaming over the mapping (Options.StreamWeights); nil = all-resident
	quant      string          // the requested Options.Quant for a direct load ("" for a prequant .giw → Quant() derives from kinds)

	// adapters holds compute-time LoRA adapters loaded against this base (#7),
	// keyed by name. Each costs only its low-rank A/B bytes; they share the one
	// resident base. A stream activates one via Session.UseAdapter / cache.lora.
	adapters map[string]*loraRuntime
}

// KVCacheF16 reports whether the GPU residency path should use an f16 KV cache
// (Options.KVPrecision == "f16"): 2× context (32k) on the same VRAM, lossy. The
// residency builder reads it; off the residency path it has no effect.
func (m *Model) KVCacheF16() bool { return m.kvF16 }

// KVCacheI8 reports whether the GPU residency path should use an int8 KV cache
// (Options.KVPrecision == "i8"): 4× vs f32 / 2× vs f16, ~64k context on the 8 GB
// card. Lossy; f32 + f16 paths unchanged. Distinct from KVQuant (the CPU cache).
func (m *Model) KVCacheI8() bool { return m.kvPrecI8 }

// Options configures Load.
type Options struct {
	Backend string // "cpu" (default) or "webgpu"
	Quant   string // "" (f32), "int8" (weight-only per-row), "int8int8" (full int8×int8 W8A8), or "int4" (group-wise) (M8)
	LoRA    string // optional PEFT adapter dir (adapter_config.json + adapter_model.safetensors), merged into the base at load. Safetensors base only.
	// KVPrecision selects the GPU residency KV cache precision: "" / "f32"
	// (default, bit-exact, 16k context cap), "f16" (lossy, 2× context to 32k), or
	// "i8" (lossy, 4× vs f32 → ~64k context). Ignored off the residency path. See
	// task-gpu-f16-kv.md / task-gpu-kv-i8.md.
	KVPrecision string
	// KVQuant selects the CPU KV cache storage precision: "" / "f32" (default,
	// bit-exact) or "i8" (per-(position,KV-head) symmetric int8, 4× smaller +
	// SDOT decode). Lossy, opt-in; excluded on MoE / gemma4 / qwen3_5_moe in v1.
	// See task-cpu-kv-quant.md.
	KVQuant string
	// StreamWeights enables on-demand weight residency (idea #2): for an mmap-backed
	// .giw MoE model, expert weights are paged out of the mapping under a RAM budget
	// (WeightCacheBytes) instead of all held resident. Bit-exact (read-only re-fault);
	// trades RAM for cold-miss fault latency. No-op for non-MoE / non-.giw models.
	StreamWeights bool
	// WeightCacheBytes is the resident-bytes budget for streamed weights (0 = auto,
	// ~half of available RAM). Only meaningful with StreamWeights.
	WeightCacheBytes int64
	// EmbedInt4 relaxes the int8 pin on the token-embedding/LM-head table in int4
	// mode, storing it at int4 too — halving the single largest resident tensor on a
	// big-vocab small model. Lossy + opt-in (~2.3 pts top-1, mostly on rare tokens);
	// default off keeps the bit-exact int8 pin. GGUF load path only.
	EmbedInt4 bool
}

// Load reads a Gemma 3 snapshot (config.json + model.safetensors) from dir
// and selects a backend. The forward pass (M3) is implemented; the CPU
// backend is the default and the only one wired (webgpu falls back to CPU).
func Load(dir string, opts Options) (*Model, error) {
	be, beErr := NewBackend(opts.Backend)
	// beErr is non-nil for the not-yet-implemented webgpu fallback; keep the
	// (cpu) backend and surface the note rather than abort.

	// Prequant bundle (.giw): the weights are already quantized and serialized, so
	// they alias straight out of the file — no GGUF/safetensors load, no requant
	// (opts.Quant/LoRA do not apply). The file is mmap'd read-only so the aliased
	// int8/int4 weights are pageable (faulted from the page cache, evictable) rather
	// than copied to the heap — the substrate the streaming / expert-paging policies
	// build on. giw.Read splits the weight blob from the metadata-GGUF tokenizer; the
	// mapping is held on the Model and released by Close.
	if strings.HasSuffix(dir, ".giw") {
		data, rerr := mmap.MapReadOnly(dir)
		if rerr != nil {
			closeBackend(be)
			return nil, fmt.Errorf("decoder: mmap .giw: %w", rerr)
		}
		weightsBlob, _, gerr := giw.Read(data)
		if gerr != nil {
			_ = mmap.Unmap(data)
			closeBackend(be)
			return nil, fmt.Errorf("decoder: parse .giw bundle: %w", gerr)
		}
		w, lerr := LoadSerializedWeights(weightsBlob)
		if lerr != nil {
			_ = mmap.Unmap(data)
			closeBackend(be)
			return nil, lerr
		}
		if beErr != nil {
			fmt.Fprintln(os.Stderr, beErr)
		}
		m := &Model{w: w, be: be, mmap: data, eosIDs: w.Cfg.EOSIDs(), kvF16: opts.KVPrecision == "f16", kvPrecI8: opts.KVPrecision == "i8", kvI8: opts.KVQuant == "i8"}
		if opts.StreamWeights {
			// MoE → expert demand-paging (#2); dense → per-layer streaming (#4).
			if w.arch.MoE != nil {
				if m.pager = newExpertPager(w, data, opts.WeightCacheBytes); m.pager != nil {
					fmt.Fprintln(os.Stderr, "decoder: "+pagerSummary(m.pager))
				} else {
					fmt.Fprintln(os.Stderr, "decoder: --stream-weights ignored (no mmap-backed MoE experts to page)")
				}
			} else if m.layerPager = newLayerPager(w, data, opts.WeightCacheBytes); m.layerPager != nil {
				fmt.Fprintln(os.Stderr, "decoder: "+layerPagerSummary(m.layerPager))
			} else {
				fmt.Fprintln(os.Stderr, "decoder: --stream-weights ignored (model fits the budget, or no mmap-backed layer weights)")
			}
		}
		return m.withResidency(), nil
	}

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

	w, err := loadWeights(dir, quant, opts.EmbedInt4, lora)
	if err != nil {
		closeBackend(be)
		return nil, err
	}
	if beErr != nil {
		// webgpu requested but fell back — not fatal.
		fmt.Fprintln(os.Stderr, beErr)
	}
	if opts.StreamWeights {
		// Weight streaming pages weights out of the read-only mmap, which only the
		// .giw path provides; a GGUF/safetensors load dequantizes into the heap, so
		// there's nothing to page. Make the no-op visible rather than silently
		// running fully resident (prequant to .giw with cmd/prequant to use it).
		fmt.Fprintln(os.Stderr, "decoder: --stream-weights ignored — weights are heap-resident; prequant to .giw (cmd/prequant) to enable streaming")
	}
	return (&Model{w: w, be: be, quant: opts.Quant, eosIDs: resolveEOSIDs(dir, &w.Cfg), kvF16: opts.KVPrecision == "f16", kvPrecI8: opts.KVPrecision == "i8", kvI8: opts.KVQuant == "i8"}).withResidency(), nil
}

// Validate checks the stringly-typed knobs against their allowed values, so an
// invalid enum (e.g. -kv-quant=int8 instead of i8) is a clear load-time error
// rather than a silent fall-through to the default. Callers building Options from
// untrusted input (cmd/serve's per-model flags) should call it before Load; Load
// itself does not, to keep existing direct callers unchanged. Path/arch-dependent
// rules (StreamWeights needs a .giw; KVQuant i8 excludes MoE) stay runtime checks.
func (o Options) Validate() error {
	if _, err := parseQuant(o.Quant); err != nil {
		return err
	}
	switch o.Backend {
	case "", "cpu", "webgpu", "cuda", "metal":
		// Accepting the NAME is not a claim that the backend is built in: an unregistered
		// one falls back to CPU with a note (NewBackend). Rejecting it here instead meant
		// `serve --backend cuda|metal` failed at flag-validation even when the module WAS
		// compiled in (-tags cuda / -tags metal), because Validate runs before registration
		// is ever consulted.
	default:
		return fmt.Errorf("decoder: invalid backend %q (cpu | webgpu | cuda | metal)", o.Backend)
	}
	switch o.KVPrecision {
	case "", "f32", "f16", "i8":
	default:
		return fmt.Errorf("decoder: invalid -kv %q (f32 | f16 | i8)", o.KVPrecision)
	}
	switch o.KVQuant {
	case "", "f32", "i8":
	default:
		return fmt.Errorf("decoder: invalid -kv-quant %q (f32 | i8)", o.KVQuant)
	}
	return nil
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
	case "int4mix":
		return quantInt4Mix, nil
	default:
		return quantNone, fmt.Errorf("decoder: unknown quant %q (have: int8, int8int8, int4, int4mix)", q)
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

// Close releases backend resources (GPU resident buffers + the backend) and
// unmaps the .giw mapping if the model was loaded from a prequant bundle. A no-op
// for the CPU backend with no mapping. Safe to call once after the model is done;
// the weights must not be touched afterward (the mapping is gone).
func (m *Model) Close() error {
	if m.resident != nil {
		_ = m.resident.Close()
		m.resident = nil
	}
	var err error
	if m.be != nil {
		err = m.be.Close()
	}
	if m.mmap != nil {
		_ = mmap.Unmap(m.mmap)
		m.mmap = nil
	}
	for _, rt := range m.adapters { // release each compute-time adapter's mmap (#7)
		rt.close()
	}
	m.adapters = nil
	return err
}

// NewCache allocates a KV cache sized for this model. capHint pre-sizes for
// a known max length (0 = grow on demand).
func (m *Model) NewCache(capHint int) *KVCache {
	a := m.w.arch
	c := NewKVCache(a.NumLayers, a.NumKVHeads, a.HeadDim, a.SlidingWindow, capHint)
	c.scr = newDecodeScratch(a)
	// int8 KV storage (opt-in, Options.KVQuant=="i8"): the uniform dense families
	// only — MoE routes attention through the acc64 kernel for bit-stable expert
	// routing (quantized KV would reopen that), and gemma4/qwen3_5_moe have their
	// own forward. Must precede enableRings so local layers inherit the mode.
	if m.kvI8 && a.gemma4 == nil && a.qwen35 == nil && a.granite == nil && a.nemotron == nil && a.MoE == nil {
		c.setQuant(kvI8, capHint)
	}
	// Ring-buffer storage on sliding-window (local) layers: keep only the W most
	// recent positions, the only ones a future query can read. Restricted to the
	// uniform-stride families whose forward uses attendQuery/attendBatchedHeads;
	// gemma4 (per-layer widths + KV-sharing) and qwen3_5_moe (linear attention)
	// have their own forward and keep append-forever for now (a later increment).
	// See docs/completed/task-kv-ring-eviction.md.
	if a.gemma4 == nil && a.qwen35 == nil && a.granite == nil && a.nemotron == nil {
		c.enableRings(a.SlidingWindow, a.isGlobalLayer)
	}
	if a.gemma4 != nil {
		c.manualPos = true // gemma4's last layer is KV-shared; pos advances via Advance()
	}
	if a.qwen35 != nil {
		// Hybrid cache: KV for the softmax layers + a recurrent DeltaState for each
		// linear layer. manualPos because the linear layers never Append.
		c.manualPos = true
		c.delta = make([]*deltaState, a.NumLayers)
		for l := 0; l < a.NumLayers; l++ {
			if a.isLinearLayer(l) {
				c.delta[l] = newDeltaState(*a.qwen35)
			}
		}
	}
	if a.granite != nil {
		// Hybrid cache: KV for the attention layers + a Mamba-2 recurrent state for
		// each mamba layer. manualPos because the mamba layers never Append.
		c.manualPos = true
		c.mamba = make([]*mamba2State, a.NumLayers)
		mp := mamba2Params{NHeads: a.granite.NHeads, HeadDim: a.granite.HeadDim, DState: a.granite.DState, NGroups: a.granite.NGroups, DConv: a.granite.DConv, Hidden: a.HiddenDim}
		for l := 0; l < a.NumLayers; l++ {
			if a.isMambaLayer(l) {
				c.mamba[l] = newMamba2State(mp)
			}
		}
	}
	if a.nemotron != nil {
		// Single-op-block hybrid: only attention layers touch KV (manualPos), mamba
		// layers carry a Mamba-2 recurrent state, mlp layers neither.
		c.manualPos = true
		c.mamba = make([]*mamba2State, a.NumLayers)
		mp := mamba2Params{NHeads: a.nemotron.NHeads, HeadDim: a.nemotron.HeadDim, DState: a.nemotron.DState, NGroups: a.nemotron.NGroups, DConv: a.nemotron.DConv, Hidden: a.HiddenDim}
		for l := 0; l < a.NumLayers; l++ {
			if a.nemotron.blockKind[l] == nemoMamba {
				c.mamba[l] = newMamba2State(mp)
			}
		}
	}
	if a.mla != nil {
		// DeepSeek MLA: the cache is the per-layer compressed latent, not full K/V.
		// manualPos because the forward appends via AppendLatent (not the standard
		// Append) and advances pos once per token after the full layer sweep.
		c.manualPos = true
		c.mlaLatent = make([][]float32, a.NumLayers)
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
	if m.w.Embed.Rows() == 0 {
		return nil, fmt.Errorf("decoder.forward: weights not loaded %w [M1]", errNotImplemented)
	}
	if arch.gemma4 != nil { // Gemma 4: per-layer head_dim, KV-sharing, PLE — own path.
		return m.runLayersGemma4(id, cache)
	}
	if arch.qwen35 != nil { // qwen3_5_moe: Gated DeltaNet / softmax hybrid — own path.
		return m.runLayersQwen35(id, cache)
	}
	if arch.granite != nil { // granitemoehybrid: Mamba-2 / softmax hybrid — own path.
		return m.runLayersGranite(id, cache)
	}
	if arch.nemotron != nil { // nemotron_h: single-op-per-block hybrid — own path.
		return m.runLayersNemotron(id, cache)
	}
	if arch.mla != nil { // deepseek_v2/v3: Multi-head Latent Attention — own path.
		return m.runLayersDeepseek(id, cache)
	}
	if arch.llama4 != nil { // llama4_text: iRoPE (per-layer RoPE/NoPE + L2 QK-norm + attn-temp) — own path.
		return m.runLayersLlama4(id, cache)
	}
	if cache.scr == nil { // caches from NewKVCache directly (tests); Generate uses NewCache
		cache.scr = newDecodeScratch(arch)
	}
	scr := cache.scr
	hidden := arch.HiddenDim
	h := scr.h           // residual stream (reused per stream; fully overwritten below)
	m.w.Embed.Row(id, h) // f32 copy, or int8 dequant when quantized
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
		m.w.PosEmbed.Row(cache.Pos(), pe)
		for i := range h {
			h[i] += pe[i]
		}
	}
	return m.runLayersFromEmbed(h, cache)
}

// runLayersFromEmbed runs the transformer layers over a precomputed
// residual-stream embedding h ([hidden]) and returns it (mutated in place). For
// a text token h is the embedding lookup (+ scale/pos) that runLayers just
// computed; for an IMAGE position (multimodal) it is the projected vision
// embedding the interleaver substitutes in place of a token-id lookup. Splitting
// it out is the embed-by-vector seam: text behavior is unchanged (runLayers is
// exactly embed-then-this), and image embeddings reach the decoder without
// passing through embed_tokens. Generic path only (gemma4/qwen35 have their own
// runLayers and grow the same seam when those families go multimodal).
func (m *Model) runLayersFromEmbed(h []float32, cache *KVCache) ([]float32, error) {
	arch := m.w.arch
	if cache.scr == nil { // direct callers (tests / the interleaver) may skip runLayers
		cache.scr = newDecodeScratch(arch)
	}
	scr := cache.scr
	hidden := arch.HiddenDim
	sandwich := arch.NormPlacement == NormSandwich4
	if m.layerPager != nil {
		defer m.layerPager.finishLayers()
	}
	for l := 0; l < arch.NumLayers; l++ {
		if m.layerPager != nil {
			m.layerPager.enterLayer(l) // prefetch l+1, release the layer behind (#4)
		}
		lw := &m.w.Layers[l]
		var ld *loraLayerDelta // compute-time LoRA deltas for this layer (#7); nil when no adapter is active
		if cache.lora != nil {
			ld = &cache.lora.layers[l]
		}
		copy(scr.norm, h)
		normalize(arch, scr.norm, lw.PreAttnNorm, lw.PreAttnNormBias, hidden)
		if err := causalAttention(l, scr.norm, scr.sub, lw, arch, cache, m.be, ld); err != nil {
			return nil, err
		}
		if cache.subCapture { // scr.ctx is the pre-o-proj context; scr.sub is not yet overwritten
			cache.subCtx[l] = append(cache.subCtx[l][:0], scr.ctx...)
		}
		if sandwich {
			normalize(arch, scr.sub, lw.PostAttnNorm, nil, hidden)
		}
		if cache.subCapture { // scr.sub is now the attention contribution about to hit the residual
			cache.subAttn[l] = append(cache.subAttn[l][:0], scr.sub...)
		}
		for i := range h {
			h[i] += scr.sub[i]
		}
		copy(scr.norm, h)
		normalize(arch, scr.norm, lw.PreMLPNorm, lw.PreMLPNormBias, hidden)
		if err := mlp(scr.norm, scr.sub, lw, arch, m.be, scr, m.pager, ld); err != nil {
			return nil, err
		}
		if cache.subCapture { // scr.sub is the down output BEFORE the post-MLP sandwich norm
			cache.subMLPpre[l] = append(cache.subMLPpre[l][:0], scr.sub...)
		}
		if sandwich {
			normalize(arch, scr.sub, lw.PostMLPNorm, nil, hidden)
		}
		if cache.subCapture { // scr.sub is now the MLP contribution about to hit the residual
			cache.subMLP[l] = append(cache.subMLP[l][:0], scr.sub...)
		}
		for i := range h {
			h[i] += scr.sub[i]
		}
		// Read-only hidden-state seam (05): copy this layer's output residual stream
		// when requested. A copy (not a reference) — h is mutated by later layers.
		if cache.captureLayers != nil {
			for i, cl := range cache.captureLayers {
				if cl == l {
					cache.captured[i] = append(cache.captured[i][:0], h...)
				}
			}
		}
	}
	return h, nil
}

// embedToken writes the residual-stream embedding for token id into dst ([hidden])
// — the embed_tokens row with the architecture's embedding scale + GPT-2 learned
// positional embedding applied, i.e. exactly the vector runLayers feeds into
// runLayersFromEmbed. Exposed so the multimodal interleaver can build a mixed
// text/image embedding sequence (text positions via this, image positions via
// the projected vision features) and drive runLayersFromEmbed per position.
func (m *Model) embedToken(id int, dst []float32) {
	arch := m.w.arch
	m.w.Embed.Row(id, dst)
	if arch.EmbedScale != 0 && arch.EmbedScale != 1 {
		scale := float32(arch.EmbedScale)
		for i := range dst {
			dst[i] *= scale
		}
	}
	// NOTE: GPT-2 learned positional embedding is position-dependent, so it is
	// applied in runLayers (which knows cache.Pos()), not here. embedToken is the
	// position-independent token embedding; the interleaver adds pos if a future
	// learned-pos family goes multimodal (none today do).
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
	h, err := m.runLayers(id, cache)
	if err != nil {
		return nil, err
	}
	return m.logitsFromHidden(h, cache), nil
}

// ForwardCapture runs one forward for token id and returns the next-token logits
// PLUS the residual stream after each layer in `layers` (cloned) — the read-only
// hidden-state seam an EAGLE-3 draft head fuses (05). The forward is byte-identical
// to forward(id): the captures are copies that never feed back. Layer indices are
// 0-based into [0, NumLayers); out[i] corresponds to layers[i]. Generic decode path
// (the families with their own runLayers don't capture yet — return an error there
// rather than silently producing nothing).
func (m *Model) ForwardCapture(id int, cache *KVCache, layers []int) (logits []float32, hidden [][]float32, err error) {
	a := m.w.arch
	if a.gemma4 != nil || a.qwen35 != nil || a.granite != nil || a.nemotron != nil || a.mla != nil || a.llama4 != nil {
		return nil, nil, fmt.Errorf("decoder.ForwardCapture: hidden-state seam not wired for arch %q (own runLayers)", a.Name)
	}
	for _, l := range layers {
		if l < 0 || l >= a.NumLayers {
			return nil, nil, fmt.Errorf("decoder.ForwardCapture: layer %d out of range [0,%d)", l, a.NumLayers)
		}
	}
	cache.captureLayers = layers
	cache.captured = make([][]float32, len(layers))
	defer func() { cache.captureLayers, cache.captured = nil, nil }()
	lg, ferr := m.forward(id, cache)
	if ferr != nil {
		return nil, nil, ferr
	}
	return lg, cache.captured, nil
}

// ForwardSubCapture runs one token and returns, per layer, the attention contribution and the
// MLP contribution to the residual (scr.sub after each sublayer's sandwich-norm, before the add).
// This is the finer seam ForwardCapture doesn't give: it separates the two sublayers, which is
// what localizing a channel's sign flip to attention-vs-MLP at a specific layer requires.
// Diagnostic — same byte-identical-output contract as ForwardCapture. Not wired for own-forward
// families (they don't route through runLayersFromEmbed's uniform block).
func (m *Model) ForwardSubCapture(id int, cache *KVCache) (attn, mlp, ctx, mlpPre [][]float32, err error) {
	a := m.w.arch
	if a.gemma4 != nil || a.qwen35 != nil || a.granite != nil || a.nemotron != nil || a.mla != nil || a.llama4 != nil {
		return nil, nil, nil, nil, fmt.Errorf("decoder.ForwardSubCapture: not wired for arch %q (own runLayers)", a.Name)
	}
	nL := a.NumLayers
	cache.subCapture = true
	cache.subAttn = make([][]float32, nL)
	cache.subMLP = make([][]float32, nL)
	cache.subMLPpre = make([][]float32, nL)
	cache.subCtx = make([][]float32, nL)
	defer func() {
		cache.subCapture = false
		cache.subAttn, cache.subMLP, cache.subMLPpre, cache.subCtx = nil, nil, nil, nil
	}()
	if _, ferr := m.forward(id, cache); ferr != nil {
		return nil, nil, nil, nil, ferr
	}
	return cache.subAttn, cache.subMLP, cache.subCtx, cache.subMLPpre, nil
}

// forwardFromEmbed is forward for a position whose residual-stream embedding is
// supplied directly (the multimodal embed-by-vector seam): it runs the layers
// from h and projects to logits, identical to forward(id) when h is that token's
// embedding. The generic path only (gemma4/qwen35 keep their own forward).
func (m *Model) forwardFromEmbed(h []float32, cache *KVCache) ([]float32, error) {
	h, err := m.runLayersFromEmbed(h, cache)
	if err != nil {
		return nil, err
	}
	return m.logitsFromHidden(h, cache), nil
}

// logitsFromHidden applies the final norm + LM head (tied embedding or separate
// lm_head) + optional Gemma logit soft-cap to a layer-stack output h, returning
// the next-token logits ([VocabSize]). One home for the head math, shared by
// forward and forwardFromEmbed.
func (m *Model) logitsFromHidden(h []float32, cache *KVCache) []float32 {
	arch := m.w.arch
	normalize(arch, h, m.w.FinalNorm, m.w.FinalNormBias, arch.HiddenDim)
	logits := cache.scr.logits // reused per stream; matmul fully overwrites it
	if arch.TiedLMHead {
		matmulInto(cache.scr.ws, m.be, &m.w.Embed, h, logits, 1) // tied: embedding doubles as the head
	} else {
		matmulInto(cache.scr.ws, m.be, &m.w.LMHead, h, logits, 1) // separate output projection
	}
	if arch.FinalLogitSoftcap > 0 {
		softcap := float32(arch.FinalLogitSoftcap)
		for i, v := range logits {
			logits[i] = softcap * float32(math.Tanh(float64(v/softcap)))
		}
	}
	if arch.LogitScale != 0 && arch.LogitScale != 1 { // Granite logits_scaling: logits /= scale
		inv := float32(1 / arch.LogitScale)
		for i := range logits {
			logits[i] *= inv
		}
	}
	return logits
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
	cache := m.NewCache(len(prompt) + maxTokens)
	go func() {
		defer close(out)
		m.generateInto(ctx, out, g, cache, prompt, 0, maxTokens, sp, nil)
	}()
	return out, g
}

// generateInto is the shared prefill+decode loop behind Model.Generate and
// Session.Generate. It assumes cache already holds prompt[:prefillFrom] (0 for a
// fresh generation), prefills prompt[prefillFrom:] (always ≥1 token — the seed,
// whose last position's logits start the decode), then decodes up to maxTokens,
// streaming each id to out. It does NOT close out: the caller owns the channel so
// it can run post-generation bookkeeping (e.g. a Session reconciling its token
// list) before the consumer observes the close. commit, if non-nil, is invoked
// with each id once its forward has committed that position to the cache — the
// seam Session uses to track exactly what the cache holds. Terminal status lands
// on g.err.
func (m *Model) generateInto(ctx context.Context, out chan<- int, g *Generation, cache *KVCache, prompt []int, prefillFrom, maxTokens int, sp SamplingParams, commit func(int)) {
	if len(prompt) == 0 {
		g.err = fmt.Errorf("decoder.Generate: empty prompt")
		return
	}
	sampler := NewSampler(sp)
	sampler.Observe(prompt...) // repetition penalties see the whole prompt, reused prefix included
	// Prefill the (divergent suffix of the) prompt and seed the first token's
	// logits. On the batched archs this runs the layers at M=len in one pass (each
	// weight streamed once — ~1.7–2× faster TTFT than sequential), LM head on the
	// last position only. Reuse means len here is the suffix, not the whole prompt.
	// GPU full-residency decode (webgpu + eligible arch + plain stateless
	// Generate). Sessions (commit != nil) and prefix-reuse (prefillFrom > 0) keep
	// the CPU/staged path. Prefill = option (a): run the prompt through the
	// resident DecodeRunner sequentially to build its GPU KV (also warms the
	// pipelines); the last token's logits seed decode. O(prompt-len) GPU Runs —
	// fast for typical prompts since a GPU Run ≫ a CPU int4 forward (the K/V-upload
	// bridge stays for future prefix-reuse; batched on-device prefill, the
	// long-prompt fix, is deferred).
	useGPU := m.resident != nil && prefillFrom == 0 && commit == nil
	gpuPos := 0
	var logits []float32
	var err error
	if useGPU {
		prefilled := false
		// Batched prefill (optional Prefiller): ingest the whole prompt in one pass — much
		// faster TTFT for long prompts. Declines (falls back) for prompts past the backend's
		// cap; absent for backends without a batched forward, and skipped for tiny prompts.
		if pf, ok := m.resident.(Prefiller); ok && len(prompt) >= 8 {
			embs := make([][]float32, len(prompt))
			for i, id := range prompt {
				embs[i] = m.embedResident(id)
			}
			if lg, perr := pf.PrefillLast(embs, 0); perr == nil {
				logits, prefilled = lg, true
				gpuPos = len(prompt)
			}
		}
		if !prefilled {
			for i, id := range prompt {
				if logits, err = m.resident.Forward(m.embedResident(id), i); err != nil {
					g.err = err
					return
				}
			}
			gpuPos = len(prompt)
		}
	} else {
		if logits, err = m.prefillLogits(prompt[prefillFrom:], cache); err != nil {
			g.err = err
			return
		}
	}
	// Greedy fast path: when the resident can pick the argmax on-device AND the sampler's
	// choice is exactly argmax(raw logits) with nothing else reading them, skip the
	// full-logits readback (594 KB/token at a 151936 vocab). Emitted tokens are identical;
	// GOINFER_NO_GREEDY_FASTPATH forces the logits path (escape hatch / A-B check).
	// fastNext >= 0 means "the resident already picked the next token"; the first token
	// still comes from the prefill logits through the sampler.
	fastNext := -1
	greedyRF, hasGreedy := m.resident.(ResidentGreedy)
	fastGreedy := useGPU && hasGreedy && sp.LogitProcessor == nil &&
		sampler.ArgmaxEquivalent() && os.Getenv("GOINFER_NO_GREEDY_FASTPATH") == ""

	// Decode loop.
	var generated []int
	var tProc, tSample, tEmbed, tFwd time.Duration
	var nFwd int
	for range maxTokens {
		select {
		case <-ctx.Done():
			g.err = ctx.Err()
			return
		default:
		}
		var t0 time.Time
		if decodeTiming {
			t0 = time.Now()
		}
		var next int
		var info SampleInfo
		if fastNext >= 0 {
			// The resident already picked this token's argmax on-device (greedy fast
			// path): nothing reads logits this step, so there is nothing to process or
			// sample. Identical to the logits path — guarded by ArgmaxEquivalent.
			next = fastNext
		} else {
			// Constrained decoding: let the processor mask this step's logits
			// (based on what's been generated) before sampling and the stop check.
			if sp.LogitProcessor != nil {
				sp.LogitProcessor(generated, logits)
			}
			if decodeTiming {
				tProc += time.Since(t0)
				t0 = time.Now()
			}
			var serr error
			info, serr = sampler.SampleWithInfo(logits)
			if serr != nil {
				g.err = serr
				return
			}
			if decodeTiming {
				tSample += time.Since(t0)
			}
			next = info.ID
		}
		if m.isStop(next, sp) {
			break
		}
		// ArgmaxEquivalent excludes Logprobs, so the fast path never reaches this.
		if sp.Logprobs {
			g.Logprobs = append(g.Logprobs, info)
		}
		out <- next
		generated = append(generated, next)
		if useGPU {
			var emb []float32
			if decodeTiming {
				t0 = time.Now()
				emb = m.embedResident(next)
				tEmbed += time.Since(t0)
				t0 = time.Now()
			} else {
				emb = m.embedResident(next)
			}
			if fastGreedy {
				// Greedy fast path: the resident picks the argmax on-device and returns
				// just the id, skipping the full-logits readback.
				fastNext, err = greedyRF.ForwardArgmax(emb, gpuPos)
			} else {
				logits, err = m.resident.Forward(emb, gpuPos)
			}
			gpuPos++
		} else {
			logits, err = m.forward(next, cache)
		}
		if decodeTiming {
			tFwd += time.Since(t0)
			nFwd++
		}
		if err != nil {
			g.err = err
			return
		}
		if commit != nil {
			commit(next)
		}
	}
	if decodeTiming && nFwd > 0 {
		ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 / float64(nFwd) }
		fmt.Printf("DECODE TIMING (%d tok, gpu=%v): forward %.1f ms | sample %.2f ms | logitProc %.2f ms | embed %.2f ms /token\n",
			nFwd, useGPU, ms(tFwd), ms(tSample), ms(tProc), ms(tEmbed))
	}
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
