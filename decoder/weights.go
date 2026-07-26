package decoder

import (
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// LayerWeights bundles one decoder block's tensors. Matrices follow
// PyTorch's [out, in] row-major layout so MatmulBT is A·Bᵀ with no
// transpose copy, matching encoder.LayerWeights.
//
// Gemma 3 specifics reflected here: separate gate/up projections (GeGLU),
// pre- AND post-norms on both the attention and MLP sub-blocks, and
// optional QK-norm weights.
type LayerWeights struct {
	// Attention projections (matmul'd → linalg.WeightMat, quantizable).
	QProj linalg.WeightMat // [NumHeads*HeadDim, HiddenDim]
	KProj linalg.WeightMat // [NumKVHeads*HeadDim, HiddenDim]
	VProj linalg.WeightMat // [NumKVHeads*HeadDim, HiddenDim]
	OProj linalg.WeightMat // [HiddenDim, NumHeads*HeadDim]
	// Projection biases ([out]; Qwen2 q/k/v only). Nil when the family/checkpoint
	// has no bias.
	QBias []float32 // [NumHeads*HeadDim]
	KBias []float32 // [NumKVHeads*HeadDim]
	VBias []float32 // [NumKVHeads*HeadDim]
	OBias []float32 // [HiddenDim] attention output-projection bias (GPT-2)
	// Norms (elementwise → stay f32). The *Bias slices are set only for
	// LayerNorm families (GPT-2); RMSNorm leaves them nil.
	QNorm           []float32 // [HeadDim] QK-norm on queries (Gemma 3)
	KNorm           []float32 // [HeadDim] QK-norm on keys (Gemma 3)
	PreAttnNorm     []float32 // [HiddenDim] input norm
	PreAttnNormBias []float32 // [HiddenDim] LayerNorm bias (GPT-2 ln_1)
	PostAttnNorm    []float32 // [HiddenDim] norm after attention, before residual add

	// MLP: projections quantizable, norms f32. UpBias/DownBias are set only for
	// the non-gated MLP (GPT-2 c_fc/c_proj); gated families leave them nil.
	GateProj       linalg.WeightMat // [IntermediateDim, HiddenDim] (unused for non-gated MLP)
	UpProj         linalg.WeightMat // [IntermediateDim, HiddenDim]
	UpBias         []float32        // [IntermediateDim] (GPT-2 c_fc bias)
	DownProj       linalg.WeightMat // [HiddenDim, IntermediateDim]
	DownBias       []float32        // [HiddenDim] (GPT-2 c_proj bias)
	PreMLPNorm     []float32        // [HiddenDim]
	PreMLPNormBias []float32        // [HiddenDim] LayerNorm bias (GPT-2 ln_2)
	PostMLPNorm    []float32        // [HiddenDim]

	// Mixture-of-experts FFN (Mixtral; set only when arch.MoE != nil). Router
	// scores experts; each expert is a gated (SwiGLU) MLP. The dense GateProj/
	// UpProj/DownProj above are unused in that case.
	Router     linalg.WeightMat // [NumExperts, HiddenDim] router/gate logits
	RouterBias []float32        // [NumExperts] DeepSeek/GLM e_score_correction_bias added to scores for top-k SELECTION (nil = none)
	Experts    []expertWeights  // [NumExperts] gated MLPs

	// Shared expert (Qwen-MoE / Qwen2-MoE; set when arch.MoE.SharedIntermediateDim
	// > 0). An always-on gated MLP scaled by sigmoid(SharedGate·h).
	SharedExpert expertWeights    // gated SwiGLU MLP at SharedIntermediateDim
	SharedGate   linalg.WeightMat // [1, HiddenDim] → sigmoid scalar gate

	// Gemma 4 per-layer extras (zero for every other family). The PLE branch runs
	// after the MLP residual: gate→gelu→×per-layer-embedding→proj→norm→+residual.
	PLEGate     linalg.WeightMat // [HiddenSizePerLayerInput, HiddenDim] (GGUF inp_gate)
	PLEProj     linalg.WeightMat // [HiddenDim, HiddenSizePerLayerInput] (GGUF proj)
	PostPLENorm []float32        // [HiddenDim] post_per_layer_input_norm (GGUF post_norm)
	LayerScalar float32          // per-layer output scalar (GGUF layer_output_scale); 1 if absent
	KVShared    bool             // carries no k/v and reuses an earlier layer's KV (Gemma 4 E-models)
	VFromK      bool             // attention_k_eq_v: no v_proj — V is v_norm(k_proj output) (Gemma 4 12B global layers)

	// qwen3_5_moe per-layer attention. Exactly one is set on the hybrid's layers:
	// delta on the Gated DeltaNet (linear) layers, qattn on the softmax layers.
	// Both nil for every other family. Stored f32 (the qwen3_5_moe forward is
	// parity-first, like Gemma 4's); the MoE FFN above is shared by both kinds.
	delta *deltaNetWeights
	qattn *qwenAttnWeights

	// Granite-4.0-H Mamba-2 mixer weights, set only on the mamba layers (the
	// attention layers use QProj/KProj/VProj/OProj above). nil for every other
	// family. Stored f32 (parity-first, like the qwen35 hybrid).
	mamba *mamba2Weights

	// DeepSeek MLA attention weights (deepseek_v2 / deepseek_v3). nil for every
	// other family. Stored f32 (parity-first). The FFN side (dense prefix / MoE /
	// shared expert) reuses the generic Router/Experts/SharedExpert fields above.
	mla *mlaWeights
}

// qwenAttnWeights holds a qwen3_5_moe softmax layer's gated attention, f32.
// q_proj is DOUBLE-WIDTH — per head it emits [query ‖ gate]; the forward splits
// it and gates the attention output by sigmoid(gate). See docs/qwen3_5_moe.md.
type qwenAttnWeights struct {
	qProj []float32 // [numHeads*headDim*2, hidden]  (query ‖ gate per head)
	kProj []float32 // [numKVHeads*headDim, hidden]
	vProj []float32 // [numKVHeads*headDim, hidden]
	oProj []float32 // [hidden, numHeads*headDim]
	qNorm []float32 // [headDim]
	kNorm []float32 // [headDim]
}

// mlaWeights holds one DeepSeek MLA layer's projections, stored f32 (parity-first,
// like qwenAttnWeights). Queries route through the q_a/q_b LoRA bottleneck when
// qAProj != nil, else the direct qProj (V2-Lite). K/V down-project to the latent
// (kvAProj: the kv_lora_rank latent ‖ the qk_rope_head_dim rope key) and up-project
// per head from the normalized latent (kvBProj → k_nope ‖ v per head).
type mlaWeights struct {
	qAProj       []float32 // [q_lora_rank, hidden]                 (nil ⇒ direct qProj)
	qALayernorm  []float32 // [q_lora_rank]
	qBProj       []float32 // [numHeads*qk_head_dim, q_lora_rank]
	qProj        []float32 // [numHeads*qk_head_dim, hidden]        (V2-Lite direct path)
	kvAProj      []float32 // [kv_lora_rank+qk_rope_head_dim, hidden]
	kvALayernorm []float32 // [kv_lora_rank]
	kvBProj      []float32 // [numHeads*(qk_nope+v_head_dim), kv_lora_rank]
	oProj        []float32 // [hidden, numHeads*v_head_dim]
}

// expertWeights is one MoE expert: a gated (SwiGLU) MLP with no biases.
// Mixtral names these w1=gate, w3=up, w2=down.
type expertWeights struct {
	Gate linalg.WeightMat // [IntermediateDim, HiddenDim] (w1)
	Up   linalg.WeightMat // [IntermediateDim, HiddenDim] (w3)
	Down linalg.WeightMat // [HiddenDim, IntermediateDim] (w2)
}

// Weights is the immutable per-checkpoint bundle. Embeddings are TIED:
// Embed doubles as the LM head (logits = h · Embedᵀ), so there is no
// separate output projection tensor.
type Weights struct {
	Cfg           Config
	arch          *Architecture    // resolved descriptor the forward pass reads
	Embed         linalg.WeightMat // [VocabSize, HiddenDim] — input embedding (AND tied LM head when LMHead unset)
	LMHead        linalg.WeightMat // [VocabSize, HiddenDim] — separate output head (untied families); zero value when tied
	PosEmbed      linalg.WeightMat // [MaxPositions, HiddenDim] — learned position embedding (GPT-2); zero value otherwise
	FinalNorm     []float32        // [HiddenDim] final norm before the LM head
	FinalNormBias []float32        // [HiddenDim] final LayerNorm bias (GPT-2 ln_f)
	Layers        []LayerWeights

	// Gemma 4 PLE inputs (zero for every other family). The per-layer input is
	// (token_identity + context_aware)/√2: token_identity from PerLayerTokenEmbed,
	// context_aware from PerLayerModelProj normed by PerLayerProjNorm.
	PerLayerTokenEmbed linalg.WeightMat // [VocabSize, NumLayers*HiddenSizePerLayerInput] (per_layer_token_embd)
	PerLayerModelProj  linalg.WeightMat // [NumLayers*HiddenSizePerLayerInput, HiddenDim] (per_layer_model_proj)
	PerLayerProjNorm   []float32        // [HiddenSizePerLayerInput] (per_layer_proj_norm)

	st      *embed.SafetensorsFile // retained so alias-backed slices stay valid
	backing []byte                 // serialized-weights blob aliased by q8/q4 arrays (LoadSerializedWeights); keeps it reachable

	schema *tensorSchema // resolved tensor-name schema (safetensors loads); used by compute-time LoRA (#7) to map projections → adapter deltas. nil for GGUF/serialized loads.
}

// matmulWeights returns every quantizable matrix in the bundle (the projections,
// the embedding, and the untied head if present); norms stay f32.
func (w *Weights) matmulWeights() []*linalg.WeightMat {
	ms := []*linalg.WeightMat{&w.Embed}
	if w.LMHead.Rows() > 0 {
		ms = append(ms, &w.LMHead)
	}
	for i := range w.Layers {
		l := &w.Layers[i]
		ms = append(ms, &l.QProj, &l.KProj, &l.VProj, &l.OProj, &l.GateProj, &l.UpProj, &l.DownProj)
		if l.Router.Rows() > 0 {
			ms = append(ms, &l.Router)
		}
		for e := range l.Experts {
			ex := &l.Experts[e]
			ms = append(ms, &ex.Gate, &ex.Up, &ex.Down)
		}
		if l.SharedExpert.Gate.Rows() > 0 {
			ms = append(ms, &l.SharedExpert.Gate, &l.SharedExpert.Up, &l.SharedExpert.Down, &l.SharedGate)
		}
		if l.PLEGate.Rows() > 0 { // Gemma 4 PLE branch
			ms = append(ms, &l.PLEGate, &l.PLEProj)
		}
	}
	if w.PerLayerTokenEmbed.Rows() > 0 { // Gemma 4 model-level PLE inputs
		ms = append(ms, &w.PerLayerTokenEmbed, &w.PerLayerModelProj)
	}
	return ms
}

// LoadWeights reads config.json + model.safetensors from a real on-disk
// directory (the HF snapshot layout). The .safetensors blob is mmapped
// (not heap-copied) so the 270M's ~340 MB bf16 checkpoint stays in the OS
// page cache — same M8 path as encoder.LoadWeights.
//
// NOTE: this widens bf16/f16 weights to f32 on load (BFloat16sToF32 /
// Float16sToF32 allocate), which roughly doubles resident RAM vs keeping
// the tensors bf16. That's the M1 correctness-first choice; the
// half-the-RAM route is per-tile widen inside matmul.
// TODO(M8): bf16-resident matmul tiling to
// drop the widen-on-load 2× memory cost for the 1B+ checkpoints.
//
// Use LoadWeightsFromFS for fs.FS-backed (MapFS, embed.FS) paths — that
// route stays heap-backed because fs.FS doesn't expose a file descriptor.
func LoadWeights(dir string) (*Weights, error) { return loadWeights(dir, quantNone, false, nil) }

// parallelLayers runs fn over the n layer indices across a worker pool, so the
// per-tensor dequant + re-quant (independent per layer — distinct linalg.WeightMat
// slots, read-only source) fans out across cores. The first error stops further
// work and is returned. Transient memory scales with the worker count (each
// in-flight layer briefly holds its dequantized f32); GOMAXPROCS workers on a
// machine that can hold the model is the right trade.
func parallelLayers(n int, fn func(i int) error) error {
	if n <= 1 {
		if n == 1 {
			return fn(0)
		}
		return nil
	}
	workers := min(runtime.GOMAXPROCS(0), n)
	var (
		next     int
		mu       sync.Mutex
		firstErr error
		wg       sync.WaitGroup
	)
	grab := func() (int, bool) {
		mu.Lock()
		defer mu.Unlock()
		if next >= n || firstErr != nil {
			return 0, false
		}
		i := next
		next++
		return i, true
	}
	for range workers {
		wg.Go(func() {
			for {
				i, ok := grab()
				if !ok {
					return
				}
				if err := fn(i); err != nil {
					mu.Lock()
					if firstErr == nil {
						firstErr = err
					}
					mu.Unlock()
					return
				}
			}
		})
	}
	wg.Wait()
	return firstErr
}

// loadWeights is the quant-aware internal load. When quant is int8/int4, each
// matmul tensor is quantized the moment it is read and its f32 backing is freed
// before the next tensor loads — so the transient footprint is the quantized
// model plus one tensor's f32, not the whole model in f32. That is what lets a
// big quantized checkpoint load in a quarter (int8) or eighth (int4) of the RAM
// the load-everything-then-quantize path needed. The forward output is identical
// to quantizing after load; only the peak memory differs.
func loadWeights(dir string, quant quantMode, embedInt4 bool, lora *loraAdapter) (*Weights, error) {
	if strings.HasSuffix(dir, ".gguf") {
		return loadGGUFWeights(dir, quant, embedInt4) // quantized llama.cpp checkpoint (G7); LoRA guarded in Load
	}
	if quant == quantInt4Mix {
		return nil, fmt.Errorf("decoder: int4mix is GGUF-only (got safetensors %s)", dir)
	}
	// embedInt4 is wired only through the GGUF path for now (its target is big-vocab
	// small .gguf models); a safetensors load keeps the int8 pin.
	cfg, err := loadConfig(os.DirFS(dir), "config.json")
	if err != nil {
		return nil, err
	}
	arch, schema, err := resolveArchitecture(cfg) // selects + validates the family descriptor
	if err != nil {
		return nil, err
	}
	st, err := openCheckpointMmap(dir)
	if err != nil {
		return nil, err
	}
	return buildWeightsFromSafetensors(cfg, arch, schema, st, quant, lora)
}

const shardIndexFile = "model.safetensors.index.json"

// openCheckpointMmap mmaps the checkpoint weights: the multi-shard set named by
// model.safetensors.index.json when present (anything above ~2B params ships
// this way — Gemma 3 4B/12B/27B, every Llama ≥7B), else the single
// model.safetensors. Either way the returned file resolves Tensor() uniformly.
func openCheckpointMmap(dir string) (*embed.SafetensorsFile, error) {
	indexPath := filepath.Join(dir, shardIndexFile)
	if _, err := os.Stat(indexPath); err == nil {
		st, err := embed.OpenSafetensorsShardedMmap(indexPath)
		if err != nil {
			return nil, fmt.Errorf("decoder: open sharded safetensors: %w", err)
		}
		return st, nil
	}
	st, err := embed.OpenSafetensorsMmap(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("decoder: open safetensors: %w", err)
	}
	return st, nil
}

// LoadWeightsFromFS mirrors encoder.LoadWeightsFromFS: reads config.json +
// model.safetensors from fsys/dir, validates every tensor's shape against
// Cfg, and returns the populated bundle. Heap-backed (fs.ReadFile); use
// LoadWeights for the mmap path on a real directory.
func LoadWeightsFromFS(fsys fs.FS, dir string) (*Weights, error) {
	return loadWeightsFromFS(fsys, dir, quantNone)
}

// loadWeightsFromFS is the quant-aware internal counterpart of loadWeights for
// fs.FS-backed paths (see loadWeights for the streaming-quant rationale).
func loadWeightsFromFS(fsys fs.FS, dir string, quant quantMode) (*Weights, error) {
	cfg, err := loadConfig(fsys, path.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	arch, schema, err := resolveArchitecture(cfg)
	if err != nil {
		return nil, err
	}
	st, err := openCheckpointFromFS(fsys, dir)
	if err != nil {
		return nil, err
	}
	return buildWeightsFromSafetensors(cfg, arch, schema, st, quant, nil)
}

// openCheckpointFromFS is the fs.FS counterpart of openCheckpointMmap (heap):
// sharded when an index.json is present, else the single file.
func openCheckpointFromFS(fsys fs.FS, dir string) (*embed.SafetensorsFile, error) {
	indexPath := path.Join(dir, shardIndexFile)
	if _, err := fs.Stat(fsys, indexPath); err == nil {
		st, err := embed.OpenSafetensorsShardedFromFS(fsys, indexPath)
		if err != nil {
			return nil, fmt.Errorf("decoder: open sharded safetensors: %w", err)
		}
		return st, nil
	}
	st, err := embed.OpenSafetensorsFromFS(fsys, path.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("decoder: open safetensors: %w", err)
	}
	return st, nil
}

// buildWeightsFromSafetensors fills a *Weights from an already-opened
// SafetensorsFile, shape-validating every tensor in gemma3TensorSchema
// against Cfg. Factored out so the heap (fs.FS) and mmap paths share one
// tensor-name + shape contract — a schema change is one edit, not two.
// Mirrors encoder.buildWeightsFromSafetensors.
func buildWeightsFromSafetensors(cfg *Config, arch *Architecture, s *tensorSchema, st *embed.SafetensorsFile, quant quantMode, lora *loraAdapter) (*Weights, error) {
	if arch.Name == "gpt2" {
		if lora != nil {
			return nil, fmt.Errorf("decoder: LoRA merge unsupported for the gpt2 (Conv1D/fused-QKV) layout")
		}
		return buildGPT2Weights(cfg, arch, st, quant) // Conv1D layout + fused QKV need a dedicated path
	}
	if arch.granite != nil {
		if lora != nil {
			return nil, fmt.Errorf("decoder: LoRA merge unsupported for the granitemoehybrid (Mamba-2 + fused-MoE) layout")
		}
		return buildGraniteWeights(cfg, arch, st, quant) // per-layer mamba/attention + fused experts
	}
	if arch.nemotron != nil {
		if lora != nil {
			return nil, fmt.Errorf("decoder: LoRA merge unsupported for the nemotron_h (single-op-block) layout")
		}
		return buildNemotronWeights(cfg, arch, st, quant) // per-layer mamba | attention | mlp
	}
	if arch.Name == "phi3" {
		if lora != nil {
			return nil, fmt.Errorf("decoder: LoRA merge unsupported for the phi3 (fused qkv/gate_up) layout")
		}
		return buildPhi3Weights(cfg, arch, st, quant) // split fused qkv_proj + gate_up_proj → generic forward
	}
	if arch.llama4 != nil {
		if lora != nil {
			return nil, fmt.Errorf("decoder: LoRA merge unsupported for the llama4_text (iRoPE + fused-expert) layout")
		}
		return buildLlama4Weights(cfg, arch, st, quant) // per-layer dense/MoE + transposed fused experts
	}
	// LoRA merge-at-load validation is deferred until after tn is defined (below) so it validates
	// against the SAME prefixed names merge actually looks up (M18).
	hd := cfg.HiddenDim
	headDim := arch.HeadDim           // resolved (Llama configs may omit head_dim; arch derives it)
	qDim := cfg.NumHeads * headDim    // query projection rows
	kvDim := cfg.NumKVHeads * headDim // key/value projection rows (narrower under GQA)

	w := &Weights{Cfg: *cfg, arch: arch, st: st, schema: s, Layers: make([]LayerWeights, cfg.NumLayers)}

	// The released qwen3_5_moe ships as a VL model: its TEXT DECODER lives under
	// model.language_model.* with the MoE experts stored as fused+stacked tensors
	// (mlp.experts.gate_up_proj / down_proj), whereas the tiny text-only golden
	// uses flat model.* + per-expert tensors. Auto-detect both from the tensor
	// index so one loader serves both — the vision tower (model.visual.*) and MTP
	// heads (mtp.*) are simply never requested. (This loads the VL model's text
	// decoder, NOT Qwen3.6-VL multimodal support.)
	have := make(map[string]bool)
	for _, n := range st.Names() {
		have[n] = true
	}
	modelPrefix := ""
	if have["model.language_model.embed_tokens.weight"] {
		modelPrefix = "language_model."
	}
	// Gemma 3 VL (transformers ≥5.10) wraps the WHOLE text decoder as
	// language_model.* — i.e. language_model.model.* + language_model.lm_head.* —
	// rather than qwen35's model.language_model.*. A uniform top-level prefix on
	// every requested name handles it; the vision_tower.* / multi_modal_projector.*
	// tensors are simply never requested (that's vision/, not the text decoder).
	topPrefix := ""
	if have["language_model.model.embed_tokens.weight"] {
		topPrefix = "language_model."
	}
	// A decoder-as-embedder checkpoint (Qwen3-Embedding, and the same shape for embeddinggemma)
	// ships the BASE model — Qwen3Model, no LM head — whose tensors carry NO "model." prefix at
	// all: embed_tokens.weight, norm.weight, layers.N.*. (The config still says
	// architectures: ["Qwen3ForCausalLM"], so the naming is the only tell.) Detect it from the
	// index and strip the prefix off the canonical names. tie_word_embeddings is true on these, so
	// the absent lm_head is expected — the embedding doubles as the head, which an embedder never
	// runs anyway. See docs/task-decoder-as-embedder.md.
	stripModel := !have["model.embed_tokens.weight"] && have["embed_tokens.weight"]
	mp := func(n string) string { // inject the prefix after the leading "model."
		if modelPrefix != "" && strings.HasPrefix(n, "model.") {
			n = "model." + modelPrefix + n[len("model."):]
		}
		if stripModel {
			n = strings.TrimPrefix(n, "model.")
		}
		return topPrefix + n
	}
	tn := func(i int, suf string) string {
		if stripModel {
			return topPrefix + fmt.Sprintf("%slayers.%d.%s", modelPrefix, i, suf)
		}
		return topPrefix + fmt.Sprintf("model.%slayers.%d.%s", modelPrefix, i, suf)
	}
	fusedExperts := arch.MoE != nil && have[tn(0, "mlp.experts.gate_up_proj")]

	// LoRA merge-at-load validation (M18). Deferred to here so it sees tn — the SAME prefixed name
	// (language_model.* / model.language_model.* on VL checkpoints, or model.-stripped) that loadProj
	// looks the delta up by. Two silent-no-op classes this closes:
	//   - a VL-prefixed base validated clean against bare names, then merge no-op'd every prefixed
	//     tensor (deltas are unprefixed) — now it fails loudly instead;
	//   - qwen35/mla load attention via loadQwen35Attn/loadDeepseekAttn, OUTSIDE loadProj, so merge
	//     never touches their attention deltas — reject rather than half-merge.
	if lora != nil {
		if arch.qwen35 != nil || arch.mla != nil {
			return nil, fmt.Errorf("decoder: LoRA merge unsupported for %s (attention projections load outside the generic merge path)", arch.Name)
		}
		if err := lora.validateTargets(cfg.NumLayers, s, tn); err != nil {
			return nil, err
		}
	}

	// GPTQ/AWQ checkpoints ship their projections as packed int4 (qweight/…);
	// resolve the params once. nil ⇒ a normal f32/bf16 checkpoint.
	qc, err := parseQuantConfig(cfg.QuantizationConfig)
	if err != nil {
		return nil, err
	}

	// loadMatQ loads a matmul weight and quantizes it immediately (the
	// streaming-quant memory win). Used for tensors that aren't LoRA targets
	// (MoE experts, router); the LoRA-mergeable projections go through loadProj.
	loadMatQ := func(name string, rows, cols int) (linalg.WeightMat, error) {
		m, merr := loadMat(st, name, rows, cols)
		if merr == nil {
			m = quantizeWM(m, quant)
		}
		return m, merr
	}
	// loadProj loads a (per-layer) attention/MLP projection to f32 — a GPTQ/AWQ
	// reconstruction when the checkpoint is pre-quantized, else a plain weight
	// load — merges any LoRA delta into it, then quantizes to the requested
	// resident format (freeing the f32 before the next tensor; the streaming-quant
	// memory win). The LoRA merge must happen here, on the f32, before quantization.
	loadProj := func(name string, out, in int) (linalg.WeightMat, error) {
		var data []float32
		var derr error
		switch {
		case qc == nil:
			data, derr = st.TensorF32(name, out, in)
			// TensorF32 may alias the read-only mmap for F32 tensors; the in-place
			// merge needs a writable copy (only for the targeted tensors).
			if derr == nil && lora.has(name) {
				data = append([]float32(nil), data...)
			}
		case qc.method == "awq":
			data, derr = awqReconstruct(st, strings.TrimSuffix(name, ".weight"), in, out)
		default:
			data, derr = gptqReconstruct(st, strings.TrimSuffix(name, ".weight"), in, out)
		}
		if derr != nil {
			return linalg.WeightMat{}, derr
		}
		if derr = lora.merge(name, data, out, in); derr != nil {
			return linalg.WeightMat{}, derr
		}
		m := linalg.WrapF32(data, out, in)
		m = quantizeWM(m, quant)
		return m, nil
	}

	// Input embedding + final norm. The embedding is the (tied or untied) LM
	// head, so it is logit-critical — quantize it with the embedding policy
	// (int8 even in int4 mode), not the projection mode.
	if w.Embed, err = loadMat(st, mp(s.Embed), cfg.VocabSize, hd); err != nil {
		return nil, err
	}
	w.Embed = quantizeWM(w.Embed, quant.embedding())
	if w.FinalNorm, err = st.TensorF32(mp(s.FinalNorm), hd); err != nil {
		return nil, err
	}
	// LM head: separate tensor when the family/checkpoint is untied, else the
	// tied embedding serves as the head. Determined by tensor presence so a
	// checkpoint that ties despite its family default still loads.
	arch.TiedLMHead = true
	if s.LMHead != "" {
		if head, herr := loadMat(st, s.LMHead, cfg.VocabSize, hd); herr == nil {
			head = quantizeWM(head, quant.embedding())
			w.LMHead = head
			arch.TiedLMHead = false
		}
	}

	// optNorm loads a [HiddenDim] norm whose schema suffix may be empty (the
	// Post*Norm tensors are absent for Pre2 families).
	optNorm := func(i int, suffix string) ([]float32, error) {
		if suffix == "" {
			return nil, nil
		}
		return st.TensorF32(tn(i, suffix), hd)
	}

	// Layers load in parallel — each is independent over the read-only mmap, and
	// the per-tensor dequant/re-quant is the cost.
	loadLayer := func(i int) error {
		l := &w.Layers[i]
		var err error
		// qwen3_5_moe: per-layer kind decides the attention tensor set (Gated
		// DeltaNet vs gated softmax); both share the MoE FFN loaded below. Stored
		// f32 (parity-first forward). Other families take the generic path.
		if arch.qwen35 != nil {
			if err = loadQwen35Attn(st, i, l, arch, hd, tn); err != nil {
				return err
			}
		} else if arch.mla != nil {
			if err = loadDeepseekAttn(st, i, l, arch, hd, tn); err != nil {
				return err
			}
		} else {
			// Attention projections ([out, in] row-major).
			if l.QProj, err = loadProj(tn(i, s.QProj), qDim, hd); err != nil {
				return err
			}
			if l.KProj, err = loadProj(tn(i, s.KProj), kvDim, hd); err != nil {
				return err
			}
			if l.VProj, err = loadProj(tn(i, s.VProj), kvDim, hd); err != nil {
				return err
			}
			if l.OProj, err = loadProj(tn(i, s.OProj), hd, qDim); err != nil {
				return err
			}
			// Projection bias (Qwen2 q/k/v; o_proj stays biasless). Gated on QKVBias so
			// a family whose schema lists the bias suffix but whose config disables it
			// (GLM's attention_bias) doesn't demand the absent tensor.
			if arch.QKVBias && s.QBias != "" {
				if l.QBias, err = st.TensorF32(tn(i, s.QBias), qDim); err != nil {
					return err
				}
				if l.KBias, err = st.TensorF32(tn(i, s.KBias), kvDim); err != nil {
					return err
				}
				if l.VBias, err = st.TensorF32(tn(i, s.VBias), kvDim); err != nil {
					return err
				}
			}
			// QK-norm (Gemma 3, Qwen3): RMSNorm over head_dim. Gated on arch.QKNorm so a
			// family whose schema lists the suffix but whose config disables it (GLM's
			// use_qk_norm=false) doesn't demand the absent tensor.
			if arch.QKNorm && s.QNorm != "" {
				if l.QNorm, err = st.TensorF32(tn(i, s.QNorm), headDim); err != nil {
					return err
				}
				if l.KNorm, err = st.TensorF32(tn(i, s.KNorm), headDim); err != nil {
					return err
				}
			}
		}
		// Block norms — Pre2 has only Pre*; Sandwich4 adds Post*.
		if l.PreAttnNorm, err = optNorm(i, s.PreAttnNorm); err != nil {
			return err
		}
		if l.PostAttnNorm, err = optNorm(i, s.PostAttnNorm); err != nil {
			return err
		}
		if l.PreMLPNorm, err = optNorm(i, s.PreMLPNorm); err != nil {
			return err
		}
		if l.PostMLPNorm, err = optNorm(i, s.PostMLPNorm); err != nil {
			return err
		}
		// FFN: sparse MoE (Mixtral) or dense gated MLP. The schema's MoE name
		// templates carry a %d for the expert index. GLM's first_k_dense_replace
		// prefix layers (i < FirstKDense) are dense and fall through to the gated MLP
		// below — leaving l.Experts nil, which the forward path keys off.
		if arch.MoE != nil && i >= arch.FirstKDense {
			if l.Router, err = loadMat(st, tn(i, s.Router), arch.MoE.NumExperts, hd); err != nil {
				return err
			}
			// DeepSeek/GLM e_score_correction_bias: steers the top-k selection. Only
			// the sigmoid-scored (noaux_tc) routers carry it — GLM and DeepSeek-V3.
			// DeepSeek-V2 (softmax/greedy, e.g. V2-Lite) has no such tensor, so gate
			// the load on RouterSigmoid even though the schema lists the suffix.
			if s.RouterBias != "" && arch.MoE.RouterSigmoid {
				if l.RouterBias, err = st.TensorF32(tn(i, s.RouterBias), arch.MoE.NumExperts); err != nil {
					return err
				}
			}
			expInter := arch.MoE.IntermediateDim // expert FFN width (Mellum: moe_intermediate_size)
			if fusedExperts {
				// Real qwen3_5_moe: all experts in two stacked 3-D tensors.
				if l.Experts, err = loadFusedExperts(st, tn(i, "mlp.experts.gate_up_proj"), tn(i, "mlp.experts.down_proj"), arch.MoE.NumExperts, expInter, hd, quant); err != nil {
					return err
				}
			} else {
				l.Experts = make([]expertWeights, arch.MoE.NumExperts)
				for e := 0; e < arch.MoE.NumExperts; e++ {
					ex := &l.Experts[e]
					if ex.Gate, err = loadMatQ(tn(i, fmt.Sprintf(s.ExpertGate, e)), expInter, hd); err != nil {
						return err
					}
					if ex.Up, err = loadMatQ(tn(i, fmt.Sprintf(s.ExpertUp, e)), expInter, hd); err != nil {
						return err
					}
					if ex.Down, err = loadMatQ(tn(i, fmt.Sprintf(s.ExpertDown, e)), hd, expInter); err != nil {
						return err
					}
				}
			}
			// Shared expert (Qwen2-MoE): an always-on gated MLP + a scalar sigmoid
			// gate. s.SharedGate/Up/Down are the expert's gate/up/down_proj;
			// s.SharedExpertGate is the [1, hidden] sigmoid gate.
			if arch.MoE.SharedIntermediateDim > 0 {
				sInter := arch.MoE.SharedIntermediateDim
				if l.SharedExpert.Gate, err = loadMatQ(tn(i, s.SharedGate), sInter, hd); err != nil {
					return err
				}
				if l.SharedExpert.Up, err = loadMatQ(tn(i, s.SharedUp), sInter, hd); err != nil {
					return err
				}
				if l.SharedExpert.Down, err = loadMatQ(tn(i, s.SharedDown), hd, sInter); err != nil {
					return err
				}
				// Sigmoid gate (Qwen2-MoE). GLM/DeepSeek add the shared expert ungated
				// (SharedExpertGate empty); moeMLP reads moe.SharedUngated, not l.SharedGate.
				if s.SharedExpertGate != "" {
					if l.SharedGate, err = loadMat(st, tn(i, s.SharedExpertGate), 1, hd); err != nil {
						return err
					}
				}
			}
			return nil
		}
		// Gated MLP (GeGLU / SwiGLU — same weights, activation differs).
		if l.GateProj, err = loadProj(tn(i, s.GateProj), cfg.IntermediateDim, hd); err != nil {
			return err
		}
		if l.UpProj, err = loadProj(tn(i, s.UpProj), cfg.IntermediateDim, hd); err != nil {
			return err
		}
		if l.DownProj, err = loadProj(tn(i, s.DownProj), hd, cfg.IntermediateDim); err != nil {
			return err
		}
		return nil
	}
	if err := parallelLayers(cfg.NumLayers, loadLayer); err != nil {
		return nil, err
	}
	return w, nil
}

// loadQwen35Attn loads one qwen3_5_moe layer's attention tensors as f32 (the
// parity-first forward uses plain matvec): the Gated DeltaNet set on linear
// layers (linear_attn.*), the gated-softmax set on the rest (self_attn.*, with a
// double-width q_proj — query ‖ gate per head). The MoE FFN is loaded by the
// shared path. See docs/qwen3_5_moe.md.
func loadQwen35Attn(st *embed.SafetensorsFile, i int, l *LayerWeights, arch *Architecture, hidden int, tn func(int, string) string) error {
	g := arch.qwen35
	var err error
	nm := func(suf string) string { return tn(i, suf) }
	if arch.isLinearLayer(i) {
		keyDim, valueDim := g.KeyHeadDim*g.NumKeyHeads, g.ValueHeadDim*g.NumValueHeads
		convDim := 2*keyDim + valueDim
		d := &deltaNetWeights{}
		if d.inProjQKV, err = st.TensorF32(nm("linear_attn.in_proj_qkv.weight"), convDim, hidden); err != nil {
			return err
		}
		if d.inProjZ, err = st.TensorF32(nm("linear_attn.in_proj_z.weight"), valueDim, hidden); err != nil {
			return err
		}
		if d.inProjB, err = st.TensorF32(nm("linear_attn.in_proj_b.weight"), g.NumValueHeads, hidden); err != nil {
			return err
		}
		if d.inProjA, err = st.TensorF32(nm("linear_attn.in_proj_a.weight"), g.NumValueHeads, hidden); err != nil {
			return err
		}
		if d.convW, err = st.TensorF32(nm("linear_attn.conv1d.weight"), convDim, 1, g.ConvKernel); err != nil {
			return err
		}
		if d.dtBias, err = st.TensorF32(nm("linear_attn.dt_bias"), g.NumValueHeads); err != nil {
			return err
		}
		aLog, aerr := st.TensorF32(nm("linear_attn.A_log"), g.NumValueHeads)
		if aerr != nil {
			return aerr
		}
		d.negExpA = negExpAFromLog(aLog) // store −exp(A_log) (GGUF bakes this; here we compute it)
		if d.normW, err = st.TensorF32(nm("linear_attn.norm.weight"), g.ValueHeadDim); err != nil {
			return err
		}
		if d.outProj, err = st.TensorF32(nm("linear_attn.out_proj.weight"), hidden, valueDim); err != nil {
			return err
		}
		l.delta = d
		return nil
	}
	hd := arch.HeadDim
	kvd := arch.NumKVHeads * hd
	a := &qwenAttnWeights{}
	if a.qProj, err = st.TensorF32(nm("self_attn.q_proj.weight"), arch.NumHeads*hd*2, hidden); err != nil {
		return err
	}
	if a.kProj, err = st.TensorF32(nm("self_attn.k_proj.weight"), kvd, hidden); err != nil {
		return err
	}
	if a.vProj, err = st.TensorF32(nm("self_attn.v_proj.weight"), kvd, hidden); err != nil {
		return err
	}
	if a.oProj, err = st.TensorF32(nm("self_attn.o_proj.weight"), hidden, arch.NumHeads*hd); err != nil {
		return err
	}
	if a.qNorm, err = st.TensorF32(nm("self_attn.q_norm.weight"), hd); err != nil {
		return err
	}
	if a.kNorm, err = st.TensorF32(nm("self_attn.k_norm.weight"), hd); err != nil {
		return err
	}
	l.qattn = a
	return nil
}

// loadDeepseekAttn loads one DeepSeek MLA layer's attention tensors as f32 (the
// parity-first forward uses plain matvec). q-LoRA (q_a_proj→norm→q_b_proj) when
// arch.mla.QLoRARank > 0, else a direct q_proj (V2-Lite). The KV down-proj
// (kv_a_proj_with_mqa) emits the latent ‖ the shared rope key; kv_b_proj reconstructs
// per-head k_nope ‖ v from the normalized latent.
func loadDeepseekAttn(st *embed.SafetensorsFile, i int, l *LayerWeights, arch *Architecture, hidden int, tn func(int, string) string) error {
	p := arch.mla
	qkHeadDim := p.qkHeadDim()
	qOut := arch.NumHeads * qkHeadDim
	kvUp := arch.NumHeads * (p.QKNopeHeadDim + p.VHeadDim)
	nm := func(suf string) string { return tn(i, suf) }
	w := &mlaWeights{}
	var err error
	if p.QLoRARank > 0 {
		if w.qAProj, err = st.TensorF32(nm("self_attn.q_a_proj.weight"), p.QLoRARank, hidden); err != nil {
			return err
		}
		if w.qALayernorm, err = st.TensorF32(nm("self_attn.q_a_layernorm.weight"), p.QLoRARank); err != nil {
			return err
		}
		if w.qBProj, err = st.TensorF32(nm("self_attn.q_b_proj.weight"), qOut, p.QLoRARank); err != nil {
			return err
		}
	} else {
		if w.qProj, err = st.TensorF32(nm("self_attn.q_proj.weight"), qOut, hidden); err != nil {
			return err
		}
	}
	if w.kvAProj, err = st.TensorF32(nm("self_attn.kv_a_proj_with_mqa.weight"), p.KVLoRARank+p.QKRopeHeadDim, hidden); err != nil {
		return err
	}
	if w.kvALayernorm, err = st.TensorF32(nm("self_attn.kv_a_layernorm.weight"), p.KVLoRARank); err != nil {
		return err
	}
	if w.kvBProj, err = st.TensorF32(nm("self_attn.kv_b_proj.weight"), kvUp, p.KVLoRARank); err != nil {
		return err
	}
	if w.oProj, err = st.TensorF32(nm("self_attn.o_proj.weight"), hidden, arch.NumHeads*p.VHeadDim); err != nil {
		return err
	}
	l.mla = w
	return nil
}

// loadFusedExperts unpacks the real qwen3_5_moe MoE FFN, which stores all experts
// as two stacked 3-D tensors instead of per-expert weights: gate_up_proj
// [nExpert, 2*inter, hidden] (gate ‖ up concatenated on the output/row axis) and
// down_proj [nExpert, hidden, inter]. Splits the fused gate_up and de-stacks per
// expert (the safetensors analogue of the GGUF stackedExperts path), then
// quantizes each into the resident format. Each expert slice is copied so the two
// big 3-D arrays are released after the layer rather than aliased.
func loadFusedExperts(st *embed.SafetensorsFile, gateUpName, downName string, nExpert, inter, hidden int, quant quantMode) ([]expertWeights, error) {
	gu, err := st.TensorF32(gateUpName, nExpert, 2*inter, hidden)
	if err != nil {
		return nil, err
	}
	down, err := st.TensorF32(downName, nExpert, hidden, inter)
	if err != nil {
		return nil, err
	}
	guStride, downStride, half := 2*inter*hidden, hidden*inter, inter*hidden
	experts := make([]expertWeights, nExpert)
	for e := range experts {
		guE := gu[e*guStride : (e+1)*guStride]
		dnE := down[e*downStride : (e+1)*downStride]
		gate := linalg.WrapF32(append([]float32(nil), guE[:half]...), inter, hidden)
		up := linalg.WrapF32(append([]float32(nil), guE[half:]...), inter, hidden)
		dn := linalg.WrapF32(append([]float32(nil), dnE...), hidden, inter)
		gate = quantizeWM(gate, quant)
		up = quantizeWM(up, quant)
		dn = quantizeWM(dn, quant)
		experts[e] = expertWeights{Gate: gate, Up: up, Down: dn}
	}
	return experts, nil
}

// loadMat loads + shape-validates a [rows, cols] matrix (via the aikit
// embed.SafetensorsFile.TensorF32 typed read — F32/BF16/F16 dispatch + shape
// check) and wraps it as a (f32) linalg.WeightMat ready for matmul/quantization.
func loadMat(st *embed.SafetensorsFile, name string, rows, cols int) (linalg.WeightMat, error) {
	data, err := st.TensorF32(name, rows, cols)
	if err != nil {
		return linalg.WeightMat{}, err
	}
	return linalg.WrapF32(data, rows, cols), nil
}

// tensorName returns the HF safetensors key for a per-layer tensor. Kept in
// one place so the M1 loader and any future schema bump touch one function.
func tensorName(layer int, suffix string) string {
	return fmt.Sprintf("model.layers.%d.%s", layer, suffix)
}

// tensorSchema maps each weight role to its safetensors key (per-layer roles
// are suffixes for tensorName). An empty string means the tensor is ABSENT for
// this family — e.g. Pre2 families have no Post*Norm, and a tied head has no
// LMHead. The per-family adapter (registry.go) picks the schema; the loader is
// schema-driven so one buildWeightsFromSafetensors serves every family.
//
// The norm roles are by POSITION in the block, not by HF tensor name: Gemma's
// "post_attention_layernorm" is a post-attn norm (PostAttnNorm) while Qwen's
// same-named tensor is the pre-MLP norm (PreMLPNorm) — exactly why the schema
// is per-family.
type tensorSchema struct {
	Embed     string
	LMHead    string // "" = tied (use Embed)
	FinalNorm string
	// per-layer suffixes passed to tensorName(layer, suffix); "" = absent
	QProj, KProj, VProj, OProj string
	QBias, KBias, VBias        string // "" = no projection bias (Qwen2 sets q/k/v)
	QNorm, KNorm               string // "" = no QK-norm
	PreAttnNorm, PostAttnNorm  string
	GateProj, UpProj, DownProj string
	PreMLPNorm, PostMLPNorm    string
	// MoE (Mixtral): router + per-expert gate/up/down. The Expert* templates
	// contain a single %d for the expert index. Empty ⇒ dense FFN.
	Router                           string
	RouterBias                       string // "" = none; DeepSeek/GLM e_score_correction_bias [NumExperts]
	ExpertGate, ExpertUp, ExpertDown string
	// Shared expert (Qwen2-MoE): an always-on gated MLP + a sigmoid gate. Empty ⇒
	// no shared expert.
	SharedGate, SharedUp, SharedDown string
	SharedExpertGate                 string
}

// gemma3TensorSchema: tied head, 4-norm sandwich, QK-norm.
var gemma3TensorSchema = tensorSchema{
	Embed:        "model.embed_tokens.weight",
	LMHead:       "", // tied
	FinalNorm:    "model.norm.weight",
	QProj:        "self_attn.q_proj.weight",
	KProj:        "self_attn.k_proj.weight",
	VProj:        "self_attn.v_proj.weight",
	OProj:        "self_attn.o_proj.weight",
	QNorm:        "self_attn.q_norm.weight",
	KNorm:        "self_attn.k_norm.weight",
	PreAttnNorm:  "input_layernorm.weight",
	PostAttnNorm: "post_attention_layernorm.weight",
	GateProj:     "mlp.gate_proj.weight",
	UpProj:       "mlp.up_proj.weight",
	DownProj:     "mlp.down_proj.weight",
	PreMLPNorm:   "pre_feedforward_layernorm.weight",
	PostMLPNorm:  "post_feedforward_layernorm.weight",
}

// qwen3TensorSchema: separate lm_head, 2-norm Pre2 (input_layernorm pre-attn,
// post_attention_layernorm pre-MLP), QK-norm, SwiGLU. Llama/Mistral/Qwen2 reuse
// this minus QNorm/KNorm (and Qwen2 adds q/k/v bias — a later add).
var qwen3TensorSchema = tensorSchema{
	Embed:        "model.embed_tokens.weight",
	LMHead:       "lm_head.weight",
	FinalNorm:    "model.norm.weight",
	QProj:        "self_attn.q_proj.weight",
	KProj:        "self_attn.k_proj.weight",
	VProj:        "self_attn.v_proj.weight",
	OProj:        "self_attn.o_proj.weight",
	QNorm:        "self_attn.q_norm.weight",
	KNorm:        "self_attn.k_norm.weight",
	PreAttnNorm:  "input_layernorm.weight",
	PostAttnNorm: "", // Pre2: no post-attn norm
	GateProj:     "mlp.gate_proj.weight",
	UpProj:       "mlp.up_proj.weight",
	DownProj:     "mlp.down_proj.weight",
	PreMLPNorm:   "post_attention_layernorm.weight", // HF name; positionally the pre-MLP norm
	PostMLPNorm:  "",                                // Pre2: no post-MLP norm
}

// llamaTensorSchema: Llama-2/3 dense. Identical to qwen3TensorSchema except
// Llama has no QK-norm tensors (QNorm/KNorm empty) — RoPE applies to raw q/k.
// Same Pre2 norm layout and SwiGLU MLP; LM head tied (small) or untied (8B+),
// resolved from lm_head.weight presence at load.
var llamaTensorSchema = tensorSchema{
	Embed:        "model.embed_tokens.weight",
	LMHead:       "lm_head.weight",
	FinalNorm:    "model.norm.weight",
	QProj:        "self_attn.q_proj.weight",
	KProj:        "self_attn.k_proj.weight",
	VProj:        "self_attn.v_proj.weight",
	OProj:        "self_attn.o_proj.weight",
	QNorm:        "", // no QK-norm
	KNorm:        "",
	PreAttnNorm:  "input_layernorm.weight",
	PostAttnNorm: "", // Pre2: no post-attn norm
	GateProj:     "mlp.gate_proj.weight",
	UpProj:       "mlp.up_proj.weight",
	DownProj:     "mlp.down_proj.weight",
	PreMLPNorm:   "post_attention_layernorm.weight", // HF name; positionally the pre-MLP norm
	PostMLPNorm:  "",                                // Pre2: no post-MLP norm
}

// mixtralTensorSchema: Mixtral — the llama attention/norm names with a sparse
// MoE FFN in place of the dense gate/up/down. Router + 8 experts (w1=gate,
// w3=up, w2=down) per layer. No QK-norm, no bias, untied head.
var mixtralTensorSchema = tensorSchema{
	Embed:       "model.embed_tokens.weight",
	LMHead:      "lm_head.weight",
	FinalNorm:   "model.norm.weight",
	QProj:       "self_attn.q_proj.weight",
	KProj:       "self_attn.k_proj.weight",
	VProj:       "self_attn.v_proj.weight",
	OProj:       "self_attn.o_proj.weight",
	PreAttnNorm: "input_layernorm.weight",
	PreMLPNorm:  "post_attention_layernorm.weight",
	Router:      "block_sparse_moe.gate.weight",
	ExpertGate:  "block_sparse_moe.experts.%d.w1.weight",
	ExpertUp:    "block_sparse_moe.experts.%d.w3.weight",
	ExpertDown:  "block_sparse_moe.experts.%d.w2.weight",
}

// mellumTensorSchema: Mellum2 — the llama attention/norm names with a sparse
// MoE FFN, but using the standard HF Qwen/Llama MoE naming (mlp.gate router,
// mlp.experts.E.{gate,up,down}_proj) rather than Mixtral's block_sparse_moe /
// w1,w3,w2. Every layer is MoE (mlp_layer_types all "sparse"); no QK-norm, no
// bias, untied head. The experts use moe_intermediate_size (MoEConfig.IntermediateDim).
var mellumTensorSchema = tensorSchema{
	Embed:       "model.embed_tokens.weight",
	LMHead:      "lm_head.weight",
	FinalNorm:   "model.norm.weight",
	QProj:       "self_attn.q_proj.weight",
	KProj:       "self_attn.k_proj.weight",
	VProj:       "self_attn.v_proj.weight",
	OProj:       "self_attn.o_proj.weight",
	QNorm:       "self_attn.q_norm.weight", // Mellum has QK-norm (per-head RMSNorm)
	KNorm:       "self_attn.k_norm.weight",
	PreAttnNorm: "input_layernorm.weight",
	PreMLPNorm:  "post_attention_layernorm.weight", // HF name; positionally the pre-MLP norm
	Router:      "mlp.gate.weight",
	ExpertGate:  "mlp.experts.%d.gate_proj.weight",
	ExpertUp:    "mlp.experts.%d.up_proj.weight",
	ExpertDown:  "mlp.experts.%d.down_proj.weight",
}

// gpt2TensorSchema is a marker — GPT-2's fused c_attn + Conv1D weight layout
// don't fit the per-suffix schema, so buildGPT2Weights handles its tensors
// directly. Kept non-empty so the adapter returns a valid (if unused) schema.
var gpt2TensorSchema = tensorSchema{Embed: "wte.weight"}

// qwen2TensorSchema: Qwen2/Qwen2.5 dense. Identical to llamaTensorSchema plus
// the q/k/v projection biases Qwen2 carries (o_proj stays biasless), and still
// no QK-norm (that arrived in Qwen3). Pre2 norms, SwiGLU; tied head on the small
// models / untied on the large, resolved from lm_head.weight presence at load.
var qwen2TensorSchema = tensorSchema{
	Embed:        "model.embed_tokens.weight",
	LMHead:       "lm_head.weight",
	FinalNorm:    "model.norm.weight",
	QProj:        "self_attn.q_proj.weight",
	KProj:        "self_attn.k_proj.weight",
	VProj:        "self_attn.v_proj.weight",
	OProj:        "self_attn.o_proj.weight",
	QBias:        "self_attn.q_proj.bias",
	KBias:        "self_attn.k_proj.bias",
	VBias:        "self_attn.v_proj.bias",
	QNorm:        "", // no QK-norm
	KNorm:        "",
	PreAttnNorm:  "input_layernorm.weight",
	PostAttnNorm: "", // Pre2: no post-attn norm
	GateProj:     "mlp.gate_proj.weight",
	UpProj:       "mlp.up_proj.weight",
	DownProj:     "mlp.down_proj.weight",
	PreMLPNorm:   "post_attention_layernorm.weight", // HF name; positionally the pre-MLP norm
	PostMLPNorm:  "",                                // Pre2: no post-MLP norm
}

// qwen2MoeTensorSchema: qwen2 attention (q/k/v bias) with the FFN replaced by a
// sparse MoE (router mlp.gate + per-expert mlp.experts.%d.*) plus an always-on
// shared expert (mlp.shared_expert.* + the mlp.shared_expert_gate sigmoid gate).
// qwen35TensorSchema covers the qwen3_5_moe SOFTMAX layers (QK-norm, no bias) +
// the routed/shared MoE common to every layer. The Gated DeltaNet (linear) layers
// carry an entirely different tensor set (in_proj_qkv/z/a/b, conv1d, A_log,
// dt_bias, norm, out_proj) that the current tensorSchema can't express; loading
// those — and pinning the exact fused-expert tensor names against a real
// checkpoint — is Phase 4 (see docs/qwen3_5_moe.md). Used today only for
// descriptor resolution.
var qwen35TensorSchema = tensorSchema{
	Embed:            "model.embed_tokens.weight",
	LMHead:           "lm_head.weight",
	FinalNorm:        "model.norm.weight",
	QProj:            "self_attn.q_proj.weight",
	KProj:            "self_attn.k_proj.weight",
	VProj:            "self_attn.v_proj.weight",
	OProj:            "self_attn.o_proj.weight",
	QNorm:            "self_attn.q_norm.weight",
	KNorm:            "self_attn.k_norm.weight",
	PreAttnNorm:      "input_layernorm.weight",
	PreMLPNorm:       "post_attention_layernorm.weight",
	Router:           "mlp.gate.weight",
	ExpertGate:       "mlp.experts.%d.gate_proj.weight",
	ExpertUp:         "mlp.experts.%d.up_proj.weight",
	ExpertDown:       "mlp.experts.%d.down_proj.weight",
	SharedGate:       "mlp.shared_expert.gate_proj.weight",
	SharedUp:         "mlp.shared_expert.up_proj.weight",
	SharedDown:       "mlp.shared_expert.down_proj.weight",
	SharedExpertGate: "mlp.shared_expert_gate.weight",
}

var qwen2MoeTensorSchema = tensorSchema{
	Embed:            "model.embed_tokens.weight",
	LMHead:           "lm_head.weight",
	FinalNorm:        "model.norm.weight",
	QProj:            "self_attn.q_proj.weight",
	KProj:            "self_attn.k_proj.weight",
	VProj:            "self_attn.v_proj.weight",
	OProj:            "self_attn.o_proj.weight",
	QBias:            "self_attn.q_proj.bias",
	KBias:            "self_attn.k_proj.bias",
	VBias:            "self_attn.v_proj.bias",
	PreAttnNorm:      "input_layernorm.weight",
	PreMLPNorm:       "post_attention_layernorm.weight",
	Router:           "mlp.gate.weight",
	ExpertGate:       "mlp.experts.%d.gate_proj.weight",
	ExpertUp:         "mlp.experts.%d.up_proj.weight",
	ExpertDown:       "mlp.experts.%d.down_proj.weight",
	SharedGate:       "mlp.shared_expert.gate_proj.weight",
	SharedUp:         "mlp.shared_expert.up_proj.weight",
	SharedDown:       "mlp.shared_expert.down_proj.weight",
	SharedExpertGate: "mlp.shared_expert_gate.weight",
}

// glm4moeTensorSchema: GLM-4.5/4.6 (glm4_moe). qwen3-like attention (per-head
// QK-norm, NO q/k/v bias) over a DeepSeek-style MoE — router gate + an
// e_score_correction_bias, per-expert gate/up/down, and an always-on shared expert
// under the PLURAL "shared_experts" name with NO sigmoid gate (SharedExpertGate
// empty ⇒ the ungated path). The first_k_dense_replace prefix layers carry the
// dense mlp.{gate,up,down}_proj instead; the loader uses those suffixes on
// i < FirstKDense and the MoE ones on i ≥ FirstKDense.
var glm4moeTensorSchema = tensorSchema{
	Embed:     "model.embed_tokens.weight",
	LMHead:    "lm_head.weight",
	FinalNorm: "model.norm.weight",
	QProj:     "self_attn.q_proj.weight",
	KProj:     "self_attn.k_proj.weight",
	VProj:     "self_attn.v_proj.weight",
	OProj:     "self_attn.o_proj.weight",
	// q/k/v bias (o_proj biasless), loaded only when arch.QKVBias (attention_bias).
	QBias:       "self_attn.q_proj.bias",
	KBias:       "self_attn.k_proj.bias",
	VBias:       "self_attn.v_proj.bias",
	QNorm:       "self_attn.q_norm.weight",
	KNorm:       "self_attn.k_norm.weight",
	PreAttnNorm: "input_layernorm.weight",
	PreMLPNorm:  "post_attention_layernorm.weight",
	// dense prefix (first_k_dense_replace) layers use these
	GateProj: "mlp.gate_proj.weight",
	UpProj:   "mlp.up_proj.weight",
	DownProj: "mlp.down_proj.weight",
	// MoE layers use these
	Router:     "mlp.gate.weight",
	RouterBias: "mlp.gate.e_score_correction_bias",
	ExpertGate: "mlp.experts.%d.gate_proj.weight",
	ExpertUp:   "mlp.experts.%d.up_proj.weight",
	ExpertDown: "mlp.experts.%d.down_proj.weight",
	SharedGate: "mlp.shared_experts.gate_proj.weight",
	SharedUp:   "mlp.shared_experts.up_proj.weight",
	SharedDown: "mlp.shared_experts.down_proj.weight",
	// SharedExpertGate empty: GLM adds the shared expert ungated.
}

// deepseekTensorSchema: DeepSeek-V2/V3 (MLA). The attention tensors are the MLA set
// (q_a/q_b/kv_a/kv_b + the two latent layernorms), loaded by loadDeepseekAttn rather
// than the QProj/KProj/VProj/OProj suffixes here — so those stay empty. The FFN side is
// identical to GLM: dense prefix (first_k_dense_replace), DeepSeekMoE (router gate +
// e_score_correction_bias, per-expert gate/up/down), and an ungated shared expert under
// the plural "shared_experts" name.
var deepseekTensorSchema = tensorSchema{
	Embed:       "model.embed_tokens.weight",
	LMHead:      "lm_head.weight",
	FinalNorm:   "model.norm.weight",
	PreAttnNorm: "input_layernorm.weight",
	PreMLPNorm:  "post_attention_layernorm.weight",
	// dense prefix (first_k_dense_replace) layers use these
	GateProj: "mlp.gate_proj.weight",
	UpProj:   "mlp.up_proj.weight",
	DownProj: "mlp.down_proj.weight",
	// MoE layers use these
	Router:     "mlp.gate.weight",
	RouterBias: "mlp.gate.e_score_correction_bias",
	ExpertGate: "mlp.experts.%d.gate_proj.weight",
	ExpertUp:   "mlp.experts.%d.up_proj.weight",
	ExpertDown: "mlp.experts.%d.down_proj.weight",
	SharedGate: "mlp.shared_experts.gate_proj.weight",
	SharedUp:   "mlp.shared_experts.up_proj.weight",
	SharedDown: "mlp.shared_experts.down_proj.weight",
	// SharedExpertGate empty: DeepSeek adds the shared expert ungated.
}

// graniteTensorSchema is a marker — Granite-4.0-H's per-layer-kind tensors (Mamba-2
// mixer vs attention) and fused-stacked GraniteMoe experts don't fit the per-suffix
// schema, so buildGraniteWeights loads them directly. Kept non-empty for a valid
// (unused) schema return.
var graniteTensorSchema = tensorSchema{Embed: "model.embed_tokens.weight"}

// phi3TensorSchema is a marker — Phi-3/Phi-4's fused qkv_proj + gate_up_proj are split by
// buildPhi3Weights into the standard fields, after which the generic llama forward runs.
var phi3TensorSchema = tensorSchema{Embed: "model.embed_tokens.weight"}

// llama4TensorSchema is a marker — Llama 4's per-layer dense/MoE FFN, fused+transposed
// batched experts, and parameter-free L2 QK-norm are loaded directly by buildLlama4Weights.
var llama4TensorSchema = tensorSchema{Embed: "model.embed_tokens.weight"}

// nemotronTensorSchema is a marker — Nemotron-H's single-op-per-block layout
// (per-layer one of mamba/attention/mlp under a "mixer" prefix) is loaded directly
// by buildNemotronWeights.
var nemotronTensorSchema = tensorSchema{Embed: "backbone.embedding.weight"}

// conv1DTransposed loads a GPT-2 Conv1D weight and returns it transposed to the
// [out, in] row-major layout the rest of the decoder (MatmulBT) expects. GPT-2
// stores these weights as [in, out] (nn.Conv1D, not nn.Linear), so a plain load
// would compute the wrong product.
func conv1DTransposed(st *embed.SafetensorsFile, name string, in, out int) ([]float32, error) {
	src, err := st.TensorF32(name, in, out) // [in, out] row-major
	if err != nil {
		return nil, err
	}
	dst := make([]float32, in*out)
	for i := range in {
		row := src[i*out : i*out+out]
		for o := range out {
			dst[o*in+i] = row[o]
		}
	}
	return dst, nil
}

// buildGPT2Weights loads a GPT-2 checkpoint. GPT-2 diverges from the
// schema-driven families on three axes the generic loader can't express: the
// q/k/v projection is a single fused c_attn tensor (split into thirds here), all
// projection weights use the Conv1D [in, out] layout (transposed on load), and
// it carries a learned position table (wpe) plus LayerNorm biases. Tensor names
// are the flat h.N.* / wte / wpe / ln_f scheme.
func buildGPT2Weights(cfg *Config, arch *Architecture, st *embed.SafetensorsFile, quant quantMode) (*Weights, error) {
	hidden, inter, vocab := arch.HiddenDim, arch.IntermediateDim, arch.VocabSize
	w := &Weights{Cfg: *cfg, arch: arch, st: st, Layers: make([]LayerWeights, arch.NumLayers)}
	var err error

	// maybeQuant streams a matmul weight to per-row int8 when quant is set,
	// freeing its f32 (see loadWeights). The Conv1D projections are built with
	// newWeightMat (post-transpose), so the quantization is applied here rather
	// than in a loader closure.
	maybeQuant := func(m linalg.WeightMat) linalg.WeightMat {
		m = quantizeWM(m, quant)
		return m
	}

	// Token + learned position embeddings (wte doubles as the tied LM head).
	// wte is the tied head → logit-critical, so quantize with the embedding
	// policy (int8 even in int4 mode); wpe is a positional lookup table added to
	// the embedding, never matmul'd → stays f32.
	if w.Embed, err = loadMat(st, "wte.weight", vocab, hidden); err != nil {
		return nil, err
	}
	w.Embed = quantizeWM(w.Embed, quant.embedding())
	if w.PosEmbed, err = loadMat(st, "wpe.weight", arch.MaxPositions, hidden); err != nil {
		return nil, err
	}
	// Final LayerNorm (weight + bias).
	if w.FinalNorm, err = st.TensorF32("ln_f.weight", hidden); err != nil {
		return nil, err
	}
	if w.FinalNormBias, err = st.TensorF32("ln_f.bias", hidden); err != nil {
		return nil, err
	}

	for i := 0; i < arch.NumLayers; i++ {
		l := &w.Layers[i]
		p := fmt.Sprintf("h.%d.", i)

		// ln_1 (pre-attention LayerNorm).
		if l.PreAttnNorm, err = st.TensorF32(p+"ln_1.weight", hidden); err != nil {
			return nil, err
		}
		if l.PreAttnNormBias, err = st.TensorF32(p+"ln_1.bias", hidden); err != nil {
			return nil, err
		}

		// Fused c_attn → q/k/v. Conv1D weight is [hidden, 3*hidden]; transpose to
		// [3*hidden, hidden] then split the rows into thirds.
		qkv, cerr := conv1DTransposed(st, p+"attn.c_attn.weight", hidden, 3*hidden)
		if cerr != nil {
			return nil, cerr
		}
		l.QProj = maybeQuant(linalg.WrapF32(qkv[0:hidden*hidden], hidden, hidden))
		l.KProj = maybeQuant(linalg.WrapF32(qkv[hidden*hidden:2*hidden*hidden], hidden, hidden))
		l.VProj = maybeQuant(linalg.WrapF32(qkv[2*hidden*hidden:3*hidden*hidden], hidden, hidden))
		qkvB, berr := st.TensorF32(p+"attn.c_attn.bias", 3*hidden)
		if berr != nil {
			return nil, berr
		}
		l.QBias, l.KBias, l.VBias = qkvB[0:hidden], qkvB[hidden:2*hidden], qkvB[2*hidden:3*hidden]

		// Attention output projection (Conv1D [hidden, hidden]) + bias.
		oData, oerr := conv1DTransposed(st, p+"attn.c_proj.weight", hidden, hidden)
		if oerr != nil {
			return nil, oerr
		}
		l.OProj = maybeQuant(linalg.WrapF32(oData, hidden, hidden))
		if l.OBias, err = st.TensorF32(p+"attn.c_proj.bias", hidden); err != nil {
			return nil, err
		}

		// ln_2 (pre-MLP LayerNorm).
		if l.PreMLPNorm, err = st.TensorF32(p+"ln_2.weight", hidden); err != nil {
			return nil, err
		}
		if l.PreMLPNormBias, err = st.TensorF32(p+"ln_2.bias", hidden); err != nil {
			return nil, err
		}

		// Non-gated MLP: c_fc (up, [hidden, inter]) → gelu → c_proj (down,
		// [inter, hidden]), both Conv1D, both with bias.
		upData, uerr := conv1DTransposed(st, p+"mlp.c_fc.weight", hidden, inter)
		if uerr != nil {
			return nil, uerr
		}
		l.UpProj = maybeQuant(linalg.WrapF32(upData, inter, hidden))
		if l.UpBias, err = st.TensorF32(p+"mlp.c_fc.bias", inter); err != nil {
			return nil, err
		}
		downData, derr := conv1DTransposed(st, p+"mlp.c_proj.weight", inter, hidden)
		if derr != nil {
			return nil, derr
		}
		l.DownProj = maybeQuant(linalg.WrapF32(downData, hidden, inter))
		if l.DownBias, err = st.TensorF32(p+"mlp.c_proj.bias", hidden); err != nil {
			return nil, err
		}
	}
	arch.TiedLMHead = true // GPT-2 has no separate lm_head tensor
	return w, nil
}

// buildGraniteWeights loads a Granite-4.0-H (granitemoehybrid) checkpoint: per layer
// either a Mamba-2 mixer (f32, parity-first, like the qwen35 hybrid) or GQA
// attention, plus a GraniteMoe routed+shared MoE on EVERY layer. The fused
// GraniteMoe experts (input_linear [E, 2·inter, hidden] = gate‖up, output_linear
// [E, hidden, inter] = down) are exactly the loadFusedExperts layout, so the routed
// FFN reuses moeMLP unchanged once split; the ungated shared_mlp loads into
// SharedExpert. Embeddings/experts/attention quantize; the Mamba-2 mixer stays f32.
func buildGraniteWeights(cfg *Config, arch *Architecture, st *embed.SafetensorsFile, quant quantMode) (*Weights, error) {
	hidden, vocab, inter := arch.HiddenDim, arch.VocabSize, arch.IntermediateDim
	g := arch.granite
	w := &Weights{Cfg: *cfg, arch: arch, st: st, Layers: make([]LayerWeights, arch.NumLayers)}
	var err error
	if w.Embed, err = loadMat(st, "model.embed_tokens.weight", vocab, hidden); err != nil {
		return nil, err
	}
	w.Embed = quantizeWM(w.Embed, quant.embedding())
	if w.FinalNorm, err = st.TensorF32("model.norm.weight", hidden); err != nil {
		return nil, err
	}
	arch.TiedLMHead = true
	if head, herr := loadMat(st, "lm_head.weight", vocab, hidden); herr == nil {
		w.LMHead = quantizeWM(head, quant.embedding())
		arch.TiedLMHead = false
	}

	dInner := g.NHeads * g.HeadDim
	convDim := dInner + 2*g.NGroups*g.DState
	projDim := 2*dInner + 2*g.NGroups*g.DState + g.NHeads
	hd := arch.HeadDim
	qDim, kvDim := arch.NumHeads*hd, arch.NumKVHeads*hd

	loadLayer := func(l int) error {
		lw := &w.Layers[l]
		tn := func(suf string) string { return fmt.Sprintf("model.layers.%d.%s", l, suf) }
		var e error
		if lw.PreAttnNorm, e = st.TensorF32(tn("input_layernorm.weight"), hidden); e != nil {
			return e
		}
		if lw.PreMLPNorm, e = st.TensorF32(tn("post_attention_layernorm.weight"), hidden); e != nil {
			return e
		}
		// Sequence mixer: Mamba-2 (f32) or GQA attention (quantized).
		if arch.isMambaLayer(l) {
			mw := &mamba2Weights{}
			if mw.inProj, e = st.TensorF32(tn("mamba.in_proj.weight"), projDim, hidden); e != nil {
				return e
			}
			if mw.convW, e = st.TensorF32(tn("mamba.conv1d.weight"), convDim, 1, g.DConv); e != nil {
				return e
			}
			if mw.convB, e = st.TensorF32(tn("mamba.conv1d.bias"), convDim); e != nil {
				return e
			}
			if mw.aLog, e = st.TensorF32(tn("mamba.A_log"), g.NHeads); e != nil {
				return e
			}
			if mw.d, e = st.TensorF32(tn("mamba.D"), g.NHeads); e != nil {
				return e
			}
			if mw.dtBias, e = st.TensorF32(tn("mamba.dt_bias"), g.NHeads); e != nil {
				return e
			}
			if mw.normW, e = st.TensorF32(tn("mamba.norm.weight"), dInner); e != nil {
				return e
			}
			if mw.outProj, e = st.TensorF32(tn("mamba.out_proj.weight"), hidden, dInner); e != nil {
				return e
			}
			lw.mamba = mw
		} else {
			if lw.QProj, e = loadMat(st, tn("self_attn.q_proj.weight"), qDim, hidden); e != nil {
				return e
			}
			if lw.KProj, e = loadMat(st, tn("self_attn.k_proj.weight"), kvDim, hidden); e != nil {
				return e
			}
			if lw.VProj, e = loadMat(st, tn("self_attn.v_proj.weight"), kvDim, hidden); e != nil {
				return e
			}
			if lw.OProj, e = loadMat(st, tn("self_attn.o_proj.weight"), hidden, qDim); e != nil {
				return e
			}
			lw.QProj, lw.KProj = quantizeWM(lw.QProj, quant), quantizeWM(lw.KProj, quant)
			lw.VProj, lw.OProj = quantizeWM(lw.VProj, quant), quantizeWM(lw.OProj, quant)
		}
		// MoE FFN (every layer): router (f32) + fused routed experts + ungated shared.
		if lw.Router, e = loadMat(st, tn("block_sparse_moe.router.layer.weight"), arch.MoE.NumExperts, hidden); e != nil {
			return e
		}
		if lw.Experts, e = loadFusedExperts(st, tn("block_sparse_moe.input_linear.weight"), tn("block_sparse_moe.output_linear.weight"), arch.MoE.NumExperts, inter, hidden, quant); e != nil {
			return e
		}
		sInter := arch.MoE.SharedIntermediateDim
		gu, gerr := st.TensorF32(tn("shared_mlp.input_linear.weight"), 2*sInter, hidden)
		if gerr != nil {
			return gerr
		}
		dn, derr := st.TensorF32(tn("shared_mlp.output_linear.weight"), hidden, sInter)
		if derr != nil {
			return derr
		}
		half := sInter * hidden
		lw.SharedExpert.Gate = quantizeWM(linalg.WrapF32(append([]float32(nil), gu[:half]...), sInter, hidden), quant)
		lw.SharedExpert.Up = quantizeWM(linalg.WrapF32(append([]float32(nil), gu[half:]...), sInter, hidden), quant)
		lw.SharedExpert.Down = quantizeWM(linalg.WrapF32(append([]float32(nil), dn...), hidden, sInter), quant)
		return nil
	}
	if err := parallelLayers(arch.NumLayers, loadLayer); err != nil {
		return nil, err
	}
	return w, nil
}

// buildNemotronWeights loads a Nemotron-H (nemotron_h) checkpoint: a single-op-per-
// block stack where each layer (under a "mixer" prefix) is a Mamba-2 mixer (f32,
// parity-first, reusing the Granite conventions), a NoPE GQA attention, or a
// non-gated relu² MLP — keyed by arch.nemotron.blockKind. Plain RMSNorm per layer.
func buildNemotronWeights(cfg *Config, arch *Architecture, st *embed.SafetensorsFile, quant quantMode) (*Weights, error) {
	hidden, vocab, inter := arch.HiddenDim, arch.VocabSize, arch.IntermediateDim
	np := arch.nemotron
	w := &Weights{Cfg: *cfg, arch: arch, st: st, Layers: make([]LayerWeights, arch.NumLayers)}
	var err error
	if w.Embed, err = loadMat(st, "backbone.embedding.weight", vocab, hidden); err != nil {
		return nil, err
	}
	w.Embed = quantizeWM(w.Embed, quant.embedding())
	if w.FinalNorm, err = st.TensorF32("backbone.norm_f.weight", hidden); err != nil {
		return nil, err
	}
	arch.TiedLMHead = true
	if head, herr := loadMat(st, "lm_head.weight", vocab, hidden); herr == nil {
		w.LMHead = quantizeWM(head, quant.embedding())
		arch.TiedLMHead = false
	}

	dInner := np.NHeads * np.HeadDim
	convDim := dInner + 2*np.NGroups*np.DState
	projDim := 2*dInner + 2*np.NGroups*np.DState + np.NHeads
	hd := arch.HeadDim
	qDim, kvDim := arch.NumHeads*hd, arch.NumKVHeads*hd

	loadLayer := func(l int) error {
		lw := &w.Layers[l]
		tn := func(suf string) string { return fmt.Sprintf("backbone.layers.%d.%s", l, suf) }
		var e error
		// The single pre-op RMSNorm (held in PreAttnNorm regardless of op kind).
		if lw.PreAttnNorm, e = st.TensorF32(tn("norm.weight"), hidden); e != nil {
			return e
		}
		switch np.blockKind[l] {
		case nemoMamba:
			mw := &mamba2Weights{}
			if mw.inProj, e = st.TensorF32(tn("mixer.in_proj.weight"), projDim, hidden); e != nil {
				return e
			}
			if mw.convW, e = st.TensorF32(tn("mixer.conv1d.weight"), convDim, 1, np.DConv); e != nil {
				return e
			}
			if mw.convB, e = st.TensorF32(tn("mixer.conv1d.bias"), convDim); e != nil {
				return e
			}
			if mw.aLog, e = st.TensorF32(tn("mixer.A_log"), np.NHeads); e != nil {
				return e
			}
			if mw.d, e = st.TensorF32(tn("mixer.D"), np.NHeads); e != nil {
				return e
			}
			if mw.dtBias, e = st.TensorF32(tn("mixer.dt_bias"), np.NHeads); e != nil {
				return e
			}
			if mw.normW, e = st.TensorF32(tn("mixer.norm.weight"), dInner); e != nil {
				return e
			}
			if mw.outProj, e = st.TensorF32(tn("mixer.out_proj.weight"), hidden, dInner); e != nil {
				return e
			}
			lw.mamba = mw
		case nemoAttn:
			if lw.QProj, e = loadMat(st, tn("mixer.q_proj.weight"), qDim, hidden); e != nil {
				return e
			}
			if lw.KProj, e = loadMat(st, tn("mixer.k_proj.weight"), kvDim, hidden); e != nil {
				return e
			}
			if lw.VProj, e = loadMat(st, tn("mixer.v_proj.weight"), kvDim, hidden); e != nil {
				return e
			}
			if lw.OProj, e = loadMat(st, tn("mixer.o_proj.weight"), hidden, qDim); e != nil {
				return e
			}
			lw.QProj, lw.KProj = quantizeWM(lw.QProj, quant), quantizeWM(lw.KProj, quant)
			lw.VProj, lw.OProj = quantizeWM(lw.VProj, quant), quantizeWM(lw.OProj, quant)
		case nemoMLP:
			if lw.UpProj, e = loadMat(st, tn("mixer.up_proj.weight"), inter, hidden); e != nil {
				return e
			}
			if lw.DownProj, e = loadMat(st, tn("mixer.down_proj.weight"), hidden, inter); e != nil {
				return e
			}
			lw.UpProj, lw.DownProj = quantizeWM(lw.UpProj, quant), quantizeWM(lw.DownProj, quant)
		}
		return nil
	}
	if err := parallelLayers(arch.NumLayers, loadLayer); err != nil {
		return nil, err
	}
	return w, nil
}

// buildPhi3Weights loads a Phi-3 / Phi-4 (phi3) checkpoint. Phi-3 is the llama skeleton
// (RMSNorm no-offset, Pre2, SwiGLU, NeoX RoPE, no QK-norm, no bias, untied head) with two
// FUSED tensors that this loader splits into the standard LayerWeights fields so the
// generic runLayers handles the forward unchanged: self_attn.qkv_proj is q‖k‖v stacked by
// output rows (split at NumHeads*HeadDim, then +NumKVHeads*HeadDim), and mlp.gate_up_proj
// is gate‖up (split in half). The fused tensors load to f32, slice by rows, and quantize
// per the resident mode (the GPT-2 fused-QKV precedent).
func buildPhi3Weights(cfg *Config, arch *Architecture, st *embed.SafetensorsFile, quant quantMode) (*Weights, error) {
	hidden, inter, vocab := arch.HiddenDim, arch.IntermediateDim, arch.VocabSize
	hd := arch.HeadDim
	qDim, kvDim := arch.NumHeads*hd, arch.NumKVHeads*hd
	w := &Weights{Cfg: *cfg, arch: arch, st: st, Layers: make([]LayerWeights, arch.NumLayers)}
	var err error
	if w.Embed, err = loadMat(st, "model.embed_tokens.weight", vocab, hidden); err != nil {
		return nil, err
	}
	w.Embed = quantizeWM(w.Embed, quant.embedding())
	if w.FinalNorm, err = st.TensorF32("model.norm.weight", hidden); err != nil {
		return nil, err
	}
	arch.TiedLMHead = true
	if head, herr := loadMat(st, "lm_head.weight", vocab, hidden); herr == nil {
		w.LMHead = quantizeWM(head, quant.embedding())
		arch.TiedLMHead = false
	}

	loadLayer := func(i int) error {
		l := &w.Layers[i]
		p := fmt.Sprintf("model.layers.%d.", i)
		var e error
		if l.PreAttnNorm, e = st.TensorF32(p+"input_layernorm.weight", hidden); e != nil {
			return e
		}
		if l.PreMLPNorm, e = st.TensorF32(p+"post_attention_layernorm.weight", hidden); e != nil {
			return e
		}
		// Fused qkv_proj [qDim+2*kvDim, hidden] → Q ‖ K ‖ V by output rows.
		qkv, qerr := st.TensorF32(p+"self_attn.qkv_proj.weight", qDim+2*kvDim, hidden)
		if qerr != nil {
			return qerr
		}
		l.QProj = quantizeWM(linalg.WrapF32(qkv[0:qDim*hidden], qDim, hidden), quant)
		l.KProj = quantizeWM(linalg.WrapF32(qkv[qDim*hidden:(qDim+kvDim)*hidden], kvDim, hidden), quant)
		l.VProj = quantizeWM(linalg.WrapF32(qkv[(qDim+kvDim)*hidden:(qDim+2*kvDim)*hidden], kvDim, hidden), quant)
		if l.OProj, e = loadMat(st, p+"self_attn.o_proj.weight", hidden, qDim); e != nil {
			return e
		}
		l.OProj = quantizeWM(l.OProj, quant)
		// Fused gate_up_proj [2*inter, hidden] → gate ‖ up (SwiGLU: down(silu(gate)·up)).
		gu, gerr := st.TensorF32(p+"mlp.gate_up_proj.weight", 2*inter, hidden)
		if gerr != nil {
			return gerr
		}
		l.GateProj = quantizeWM(linalg.WrapF32(gu[0:inter*hidden], inter, hidden), quant)
		l.UpProj = quantizeWM(linalg.WrapF32(gu[inter*hidden:2*inter*hidden], inter, hidden), quant)
		if l.DownProj, e = loadMat(st, p+"mlp.down_proj.weight", hidden, inter); e != nil {
			return e
		}
		l.DownProj = quantizeWM(l.DownProj, quant)
		return nil
	}
	if err := parallelLayers(arch.NumLayers, loadLayer); err != nil {
		return nil, err
	}
	return w, nil
}

// buildLlama4Weights loads a Llama 4 text decoder (llama4_text). Attention is separate
// q/k/v/o (no fusion, no bias; the L2 QK-norm is parameter-free, applied in the forward).
// The FFN is per-layer: dense layers (arch.llama4.isMoE[l] false) use feed_forward.{gate,up,
// down}_proj at intermediate_size_mlp; MoE layers use feed_forward.router + a SHARED expert
// (feed_forward.shared_expert.*) + batched routed experts. The routed experts are stored
// FUSED and TRANSPOSED for a bmm — gate_up_proj is [nE, hidden, 2*inter] and down_proj is
// [nE, inter, hidden] ([in, out] per expert) — so each expert is transposed to goinfer's
// [out, in] WeightMat and the gate‖up halves split out.
func buildLlama4Weights(cfg *Config, arch *Architecture, st *embed.SafetensorsFile, quant quantMode) (*Weights, error) {
	hidden, vocab := arch.HiddenDim, arch.VocabSize
	hd := arch.HeadDim
	qDim, kvDim := arch.NumHeads*hd, arch.NumKVHeads*hd
	denseInter := arch.IntermediateDim   // intermediate_size_mlp
	expInter := arch.MoE.IntermediateDim // intermediate_size (routed + shared)
	nE := arch.MoE.NumExperts
	lp := arch.llama4
	w := &Weights{Cfg: *cfg, arch: arch, st: st, Layers: make([]LayerWeights, arch.NumLayers)}
	var err error
	if w.Embed, err = loadMat(st, "model.embed_tokens.weight", vocab, hidden); err != nil {
		return nil, err
	}
	w.Embed = quantizeWM(w.Embed, quant.embedding())
	if w.FinalNorm, err = st.TensorF32("model.norm.weight", hidden); err != nil {
		return nil, err
	}
	arch.TiedLMHead = true
	if head, herr := loadMat(st, "lm_head.weight", vocab, hidden); herr == nil {
		w.LMHead = quantizeWM(head, quant.embedding())
		arch.TiedLMHead = false
	}

	loadLayer := func(i int) error {
		l := &w.Layers[i]
		p := fmt.Sprintf("model.layers.%d.", i)
		var e error
		if l.PreAttnNorm, e = st.TensorF32(p+"input_layernorm.weight", hidden); e != nil {
			return e
		}
		if l.PreMLPNorm, e = st.TensorF32(p+"post_attention_layernorm.weight", hidden); e != nil {
			return e
		}
		// Attention: separate q/k/v/o, no bias.
		if l.QProj, e = loadMat(st, p+"self_attn.q_proj.weight", qDim, hidden); e != nil {
			return e
		}
		if l.KProj, e = loadMat(st, p+"self_attn.k_proj.weight", kvDim, hidden); e != nil {
			return e
		}
		if l.VProj, e = loadMat(st, p+"self_attn.v_proj.weight", kvDim, hidden); e != nil {
			return e
		}
		if l.OProj, e = loadMat(st, p+"self_attn.o_proj.weight", hidden, qDim); e != nil {
			return e
		}
		l.QProj, l.KProj = quantizeWM(l.QProj, quant), quantizeWM(l.KProj, quant)
		l.VProj, l.OProj = quantizeWM(l.VProj, quant), quantizeWM(l.OProj, quant)
		if !lp.isMoE[i] {
			// Dense FFN at intermediate_size_mlp.
			if l.GateProj, e = loadMat(st, p+"feed_forward.gate_proj.weight", denseInter, hidden); e != nil {
				return e
			}
			if l.UpProj, e = loadMat(st, p+"feed_forward.up_proj.weight", denseInter, hidden); e != nil {
				return e
			}
			if l.DownProj, e = loadMat(st, p+"feed_forward.down_proj.weight", hidden, denseInter); e != nil {
				return e
			}
			l.GateProj, l.UpProj, l.DownProj = quantizeWM(l.GateProj, quant), quantizeWM(l.UpProj, quant), quantizeWM(l.DownProj, quant)
			return nil
		}
		// MoE: router + ungated shared expert + batched fused routed experts.
		if l.Router, e = loadMat(st, p+"feed_forward.router.weight", nE, hidden); e != nil {
			return e
		}
		if l.SharedExpert.Gate, e = loadMat(st, p+"feed_forward.shared_expert.gate_proj.weight", expInter, hidden); e != nil {
			return e
		}
		if l.SharedExpert.Up, e = loadMat(st, p+"feed_forward.shared_expert.up_proj.weight", expInter, hidden); e != nil {
			return e
		}
		if l.SharedExpert.Down, e = loadMat(st, p+"feed_forward.shared_expert.down_proj.weight", hidden, expInter); e != nil {
			return e
		}
		l.SharedExpert.Gate = quantizeWM(l.SharedExpert.Gate, quant)
		l.SharedExpert.Up = quantizeWM(l.SharedExpert.Up, quant)
		l.SharedExpert.Down = quantizeWM(l.SharedExpert.Down, quant)
		// Routed experts: gate_up_proj [nE, hidden, 2*inter], down_proj [nE, inter, hidden]
		// ([in, out] per expert) → transpose each to [out, in] + split gate‖up.
		gu, gerr := st.TensorF32(p+"feed_forward.experts.gate_up_proj", nE, hidden, 2*expInter)
		if gerr != nil {
			return gerr
		}
		dn, derr := st.TensorF32(p+"feed_forward.experts.down_proj", nE, expInter, hidden)
		if derr != nil {
			return derr
		}
		l.Experts = make([]expertWeights, nE)
		for ex := range nE {
			guBase, dnBase := ex*hidden*2*expInter, ex*expInter*hidden
			gate := make([]float32, expInter*hidden)
			up := make([]float32, expInter*hidden)
			down := make([]float32, hidden*expInter)
			for r := range expInter {
				for h := range hidden {
					gate[r*hidden+h] = gu[guBase+h*2*expInter+r]
					up[r*hidden+h] = gu[guBase+h*2*expInter+expInter+r]
				}
			}
			for h := range hidden {
				for ii := range expInter {
					down[h*expInter+ii] = dn[dnBase+ii*hidden+h]
				}
			}
			l.Experts[ex] = expertWeights{
				Gate: quantizeWM(linalg.WrapF32(gate, expInter, hidden), quant),
				Up:   quantizeWM(linalg.WrapF32(up, expInter, hidden), quant),
				Down: quantizeWM(linalg.WrapF32(down, hidden, expInter), quant),
			}
		}
		return nil
	}
	if err := parallelLayers(arch.NumLayers, loadLayer); err != nil {
		return nil, err
	}
	return w, nil
}
