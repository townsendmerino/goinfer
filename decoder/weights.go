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
	// GProj is Laguna's attention output gate (g_proj), applied to the attention
	// context BEFORE OProj as ctx *= softplus(GProj·h). Rows are NumHeads for the
	// "per-head" granularity or NumHeads*HeadDim for "per-element". Zero-valued
	// (Rows()==0) for every other family. See applyAttnGate.
	GProj linalg.WeightMat // [NumHeads | NumHeads*HeadDim, HiddenDim]
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
	RouterBias []float32        // [NumExperts] DeepSeek/GLM e_score_correction_bias added to scores for top-k SELECTION (nil = none). gpt-oss reuses this as a plain router-LOGIT bias (added before top-k, its own forward).
	Experts    []expertWeights  // [NumExperts] gated MLPs

	// AttnSinks is gpt-oss's per-head learned attention-sink logit ([NumHeads]):
	// an extra term in each head's softmax denominator (exp(sink_h)) with no value,
	// bleeding attention mass. nil for every other family. Consumed by gptOssAttention.
	AttnSinks []float32

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

	// LFM2 gated short-convolution mixer weights, set ONLY on the conv layers (the
	// attention layers use QProj/KProj/VProj/OProj above and leave this nil). nil for
	// every other family.
	shortConv *shortConvWeights

	// Granite-4.0-H Mamba-2 mixer weights, set only on the mamba layers (the
	// attention layers use QProj/KProj/VProj/OProj above). nil for every other
	// family. Stored f32 (parity-first, like the qwen35 hybrid).
	mamba *mamba2Weights

	// DeepSeek MLA attention weights (deepseek_v2 / deepseek_v3). nil for every
	// other family. Stored f32 (parity-first). The FFN side (dense prefix / MoE /
	// shared expert) reuses the generic Router/Experts/SharedExpert fields above.
	mla *mlaWeights

	// Gemma 4 26B-A4B parallel dense+MoE FFN sub-block (enable_moe_block). Set only
	// on gemma4 layers when arch.MoE != nil; nil for the dense E2B/E4B/12B variants
	// and every other family. The router/experts do NOT fit the generic Router/
	// Experts fields (weightless-norm + learned-scale router, per-expert scale,
	// gelu-tanh fused-gate_up experts, three parallel-branch norms), so gemma4
	// carries its own struct consumed by gemma4MoEFFN. Stored f32 (parity-first).
	gemma4moe *gemma4MoEWeights
}

// qwenAttnWeights holds a qwen3_5_moe softmax layer's gated attention, f32.
// q_proj is DOUBLE-WIDTH — per head it emits [query ‖ gate]; the forward splits
// it and gates the attention output by sigmoid(gate). See docs/qwen3_5_moe.md.
type qwenAttnWeights struct {
	// Quantizable as of 2026-08-19 — same reason as deltaNetWeights: these were f32 regardless of
	// Options.Quant, so the 16 softmax layers of a 27.8B Qwen3.8 streamed 6.7 GB per token that
	// int4 should have made ~1.7 GB. WeightMat stays f32 when no quant is requested.
	qProj linalg.WeightMat // [numHeads*headDim*2, hidden]  (query ‖ gate per head)
	kProj linalg.WeightMat // [numKVHeads*headDim, hidden]
	vProj linalg.WeightMat // [numKVHeads*headDim, hidden]
	oProj linalg.WeightMat // [hidden, numHeads*headDim]
	qNorm []float32        // [headDim]
	kNorm []float32        // [headDim]
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

// expertWeights is one MoE expert: a gated (SwiGLU) MLP. Mixtral names these
// w1=gate, w3=up, w2=down and has no biases; gpt-oss adds a per-expert bias on
// each projection (the *Bias slices, nil for every other family).
type expertWeights struct {
	Gate linalg.WeightMat // [IntermediateDim, HiddenDim] (w1)
	Up   linalg.WeightMat // [IntermediateDim, HiddenDim] (w3)
	Down linalg.WeightMat // [HiddenDim, IntermediateDim] (w2)

	// gpt-oss per-expert biases (nil elsewhere).
	GateBias []float32 // [IntermediateDim]
	UpBias   []float32 // [IntermediateDim]
	DownBias []float32 // [HiddenDim]
}

// Weights is the immutable per-checkpoint bundle. Embeddings are TIED:
// Embed doubles as the LM head (logits = h · Embedᵀ), so there is no
// separate output projection tensor.
type Weights struct {
	Cfg  Config
	arch *Architecture // resolved descriptor the forward pass reads
	// bakedQuant is the resolved quant label recorded in the .giw header (v5+): "int4" |
	// "int4mix" | "int8int8" | "int8" | "native". Non-empty only for a v5 .giw loaded with the
	// field present; Model.Quant() prefers it over re-inferring from tensor kinds. Empty for a
	// direct GGUF/safetensors load, a pre-v5 bundle, or a streamed v5 bundle (which records "")
	// — those fall back to quantLabel() inference.
	bakedQuant    string
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
		// GProj is deliberately EXCLUDED from quantization, for the same reason Router
		// is treated carefully: it is tiny (one row per head — [64, hidden] against
		// q_proj's [8192, hidden], well under 1% of a layer's attention weights) so
		// quantizing it buys nothing, and it feeds a softplus whose output MULTIPLIES
		// the whole attention context. Quantization error there scales every channel of
		// every head rather than perturbing one projection's output additively.
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

// isLogitTable reports whether m is one of the embedding-class tensors — the token embedding, the
// LM head, or the Gemma-4 model-level PLE embeddings. In int4 mode these are pinned to int8 by
// DEFAULT (logit-critical; the EmbedInt4 knob relaxes them), so their precision is orthogonal to the
// int4-vs-int4mix distinction and must be excluded from the quant classification (T1-6). This is the
// single definition of that exclusion.
func (w *Weights) isLogitTable(m *linalg.WeightMat) bool {
	return m == &w.Embed || m == &w.LMHead || m == &w.PerLayerTokenEmbed || m == &w.PerLayerModelProj
}

// bodyMatmulWeights is matmulWeights minus the logit tables: the attention/FFN projections, experts,
// and routers — exactly the matmuls whose precision the chosen quant (int4|int4mix|int8|int8int8)
// determines. It is the one list quantLabel classifies over (and, through quantLabel, the value the
// .giw header records at bake time), so the "which tensors define the quant" fact lives in ONE place
// and cannot drift the way the T1-6 label did. (The cuda batched-prefill gate — nonBatchableKind,
// cuda/prefill.go — inspects the *resident* per-layer projections, a different type in a different
// module; it agrees on excluding the logit tables but cannot share this host-side function.)
func (w *Weights) bodyMatmulWeights() []*linalg.WeightMat {
	all := w.matmulWeights()
	body := make([]*linalg.WeightMat, 0, len(all))
	for _, m := range all {
		if !w.isLogitTable(m) {
			body = append(body, m)
		}
	}
	return body
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
	// One atomic add per model load, so the fit guard's test can OBSERVE that a refused load
	// allocated nothing rather than infer it from an error string. Inferring is how a guard that
	// fires after the allocation still looks correct (docs/task-first-hour.md, R3).
	weightAllocs.Add(1)
	if strings.HasSuffix(dir, ".gguf") {
		return loadGGUFWeights(dir, quant, embedInt4) // quantized llama.cpp checkpoint (G7); LoRA guarded in Load
	}
	if quant == quantInt4Mix {
		return nil, fmt.Errorf("decoder: int4mix is GGUF-only (got safetensors %s)", dir)
	}
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
	// buildWeightsFromSafetensors retains st (the WeightMats MAY alias its mmap) ONLY on success —
	// on any of its ~40 error returns st would otherwise leak the mapping + fd. A serve process
	// probing candidate dirs, or retrying a load of a checkpoint with one missing tensor,
	// accumulates GBs of address space — the exact leak Model.Close exists to avoid (audit M-08).
	w, err := buildWeightsFromSafetensors(cfg, arch, schema, st, quant, embedInt4, lora)
	if err != nil {
		_ = st.Close()
		return nil, err
	}
	// P13: release the SOURCE mapping now when nothing can alias it, instead of holding it for
	// the model's whole life. The mapping is the bf16 checkpoint — 55.6 GB for a 27B — and the
	// quantized weights the decode actually reads are a separate, much smaller allocation. Holding
	// the source means dead pages compete with hot weights for page cache: measured 46.8 GB RSS
	// against GGUF's 24.5 GB for an IDENTICAL 17.9 GB Go heap, and 1.69x slower decode.
	if why := mmapAliasRisk(st); why == "" && os.Getenv("GOINFER_P13_OFF") == "" {
		_ = st.Close()
		w.st = nil
	}
	return w, nil
}

// mmapAliasRisk reports the first tensor whose dtype could leave a slice ALIASING the mapping,
// or "" when none can and the source is therefore safe to close at end of load.
//
// The rule comes from aikit's reader, not from inspection of checkpoints. BF16 and F16 are widened
// into a fresh slice by TensorF32/SubF32 — aikit's own comment: "the result then does not alias the
// file". Every other dtype is served by reinterpretLE, which takes a "zero-copy view" whenever the
// payload is aligned, and alignment is the common case. So a checkpoint of BF16/F16 tensors leaves
// nothing pointing into the mapping, and any other dtype might.
//
// Deliberately conservative in two ways. It keys on what the FILE holds rather than on which
// tensors this load happened to read, and it does not try to reason about quant mode — int8/int4
// drop the f32 after quantizing while quantNone keeps it (see streamExperts), and encoding that
// interaction here would put a second, subtler rule in a second place. A false "risky" costs the
// old behaviour; a false "safe" is a use-after-free.
func mmapAliasRisk(st *embed.SafetensorsFile) string {
	for _, n := range st.Names() {
		t, err := st.Tensor(n)
		if err != nil {
			return n // header entry we cannot classify: assume the worst
		}
		if riskyDType(t.DType) {
			return n
		}
	}
	return ""
}

// riskyDType reports whether a stored dtype can be handed back as a slice aliasing the mapping.
// BF16/F16 are always widened into fresh storage; every other dtype may be a zero-copy view.
func riskyDType(dt string) bool {
	switch dt {
	case "BF16", "F16":
		return false
	default:
		return true
	}
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
	w, err := buildWeightsFromSafetensors(cfg, arch, schema, st, quant, false, nil)
	if err != nil {
		_ = st.Close() // st is retained only on success; close it on error so the mapping/fd doesn't leak (M-08)
		return nil, err
	}
	return w, nil
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
func buildWeightsFromSafetensors(cfg *Config, arch *Architecture, s *tensorSchema, st *embed.SafetensorsFile, quant quantMode, embedInt4 bool, lora *loraAdapter) (*Weights, error) {
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
	if arch.Name == "internlm2" {
		return buildInternLM2Weights(cfg, arch, st, quant) // renamed tensors + GROUPED fused wqkv
	}
	if arch.llama4 != nil {
		if lora != nil {
			return nil, fmt.Errorf("decoder: LoRA merge unsupported for the llama4_text (iRoPE + fused-expert) layout")
		}
		return buildLlama4Weights(cfg, arch, st, quant) // per-layer dense/MoE + transposed fused experts
	}
	if arch.gptoss != nil {
		// gpt-oss safetensors: MXFP4 experts as paired U8 *_blocks/*_scales tensors with
		// an INTERLEAVED gate_up, per-expert biases, and per-head attention sinks. The
		// GGUF path reads the same family through a different layout entirely (llama.cpp's
		// converter separates gate/up and re-packs the nibbles), so it gets its own loader
		// rather than a shared one with branches.
		return buildGptOssWeights(cfg, arch, st, quant)
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
	// runs anyway. See docs/completed/task-decoder-as-embedder.md.
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
		case qc.method == "fp8":
			// Block-quantized fp8 keeps the full tensor NAME (scales are name+"_scale_inv"),
			// unlike gptq/awq which trim ".weight" and rebuild from a family of suffixes.
			data, derr = fp8Reconstruct(st, name, in, out, qc.blockR, qc.blockC)
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
	w.Embed = quantizeWM(w.Embed, quant.embeddingWith(embedInt4))
	if w.FinalNorm, err = st.TensorF32(mp(s.FinalNorm), hd); err != nil {
		return nil, err
	}
	// LM head: separate tensor when the family/checkpoint is untied, else the
	// tied embedding serves as the head. Determined by tensor presence so a
	// checkpoint that ties despite its family default still loads.
	arch.TiedLMHead = true
	if s.LMHead != "" {
		if head, herr := loadMat(st, s.LMHead, cfg.VocabSize, hd); herr == nil {
			head = quantizeWM(head, quant.embeddingWith(embedInt4))
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
		if arch.lfm2 != nil && arch.isConvLayer(i) {
			// LFM2 conv layer: the gated short-conv mixer REPLACES attention, so there is
			// no q/k/v/o and no QK-norm to load — reading them would fail on tensors the
			// checkpoint does not contain. The FFN below is shared with the attention
			// layers and still loads. Attention layers fall through to the generic path,
			// which lfm2TensorSchema already describes.
			if err = loadLFM2Conv(st, i, l, arch, hd, tn); err != nil {
				return err
			}
		} else if arch.qwen35 != nil && (arch.isLinearLayer(i) || !arch.qwen35.PlainFullAttn) {
			// PlainFullAttn (Olmo Hybrid): its full-attention layers are NOT
			// qwen3.5's own double-width gated softmax attention, so only the
			// linear (DeltaNet) layers come through here; the rest fall through
			// to the generic tensorSchema-driven path below (olmo3's own shape).
			if err = loadQwen35Attn(st, i, l, arch, hd, tn, loadMatQ, quant); err != nil {
				return err
			}
		} else if arch.mla != nil {
			if err = loadDeepseekAttn(st, i, l, arch, hd, tn); err != nil {
				return err
			}
		} else if arch.gemma4 != nil {
			// Gemma 4 attention has PER-LAYER widths: the global (full-attention) layers
			// use a wider head (global_head_dim, e.g. 512 vs 16 local) and their own KV
			// head count, so qDim/kvDim/head_dim differ by layer. attention_k_eq_v makes
			// the 12B global layers reuse K as V (no v_proj → VFromK); the E-models and
			// the MoE tiny carry v_proj on every layer.
			ahd := arch.headDimAt(i)
			aKV := arch.kvHeadsAt(i)
			aqDim, akvDim := arch.NumHeads*ahd, aKV*ahd
			if l.QProj, err = loadProj(tn(i, s.QProj), aqDim, hd); err != nil {
				return err
			}
			if l.KProj, err = loadProj(tn(i, s.KProj), akvDim, hd); err != nil {
				return err
			}
			if arch.gemma4.KVShared && arch.isGlobalLayer(i) {
				l.VFromK = true // V = v_norm(k_proj output); no v_proj tensor
			} else if l.VProj, err = loadProj(tn(i, s.VProj), akvDim, hd); err != nil {
				return err
			}
			if l.OProj, err = loadProj(tn(i, s.OProj), hd, aqDim); err != nil {
				return err
			}
			if l.QNorm, err = st.TensorF32(tn(i, s.QNorm), ahd); err != nil {
				return err
			}
			if l.KNorm, err = st.TensorF32(tn(i, s.KNorm), ahd); err != nil {
				return err
			}
		} else {
			// Attention projections ([out, in] row-major). qDim is per-LAYER: Laguna's XS
			// generations vary the query head count by layer type (48 on full-attention,
			// 64 on sliding), which the real checkpoint shows as q_proj [6144,2048] on
			// layer 0 and [8192,2048] on layer 1. headsAt collapses to NumHeads for every
			// other family, leaving lqDim == qDim.
			lqDim := arch.headsAt(i) * headDim
			if l.QProj, err = loadProj(tn(i, s.QProj), lqDim, hd); err != nil {
				return err
			}
			if l.KProj, err = loadProj(tn(i, s.KProj), kvDim, hd); err != nil {
				return err
			}
			if l.VProj, err = loadProj(tn(i, s.VProj), kvDim, hd); err != nil {
				return err
			}
			if l.OProj, err = loadProj(tn(i, s.OProj), hd, lqDim); err != nil {
				return err
			}
			// Laguna attention output gate. Its row count SELECTS the granularity —
			// arch.headsAt(i) rows is per-head, headsAt(i)*headDim is per-element — so
			// both valid shapes are accepted and anything else is a hard error rather
			// than a silently mis-shaped gate. See applyAttnGate for why the checkpoint,
			// not config.gating, is authoritative here.
			if s.GProj != "" {
				perHead, perElem := arch.headsAt(i), lqDim
				rows := perHead
				if arch.laguna != nil && !arch.laguna.GatePerHead {
					rows = perElem
				}
				if l.GProj, err = loadProj(tn(i, s.GProj), rows, hd); err != nil {
					if rows == perHead {
						rows = perElem
					} else {
						rows = perHead
					}
					if l.GProj, err = loadProj(tn(i, s.GProj), rows, hd); err != nil {
						return fmt.Errorf("decoder(laguna): layer %d g_proj is neither per-head [%d,%d] nor per-element [%d,%d]: %w", i, perHead, hd, perElem, hd, err)
					}
				}
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
			// use_qk_norm=false) doesn't demand the absent tensor. QKNormWhole (Olmo 3/Olmo
			// Hybrid) normalizes the FULL projected vector, so its weight tensor is
			// per-layer-query-dim/kv-dim wide instead of head_dim wide.
			if arch.QKNorm && s.QNorm != "" {
				qNormLen, kNormLen := headDim, headDim
				if arch.QKNormWhole {
					qNormLen, kNormLen = lqDim, kvDim
				}
				if l.QNorm, err = st.TensorF32(tn(i, s.QNorm), qNormLen); err != nil {
					return err
				}
				if l.KNorm, err = st.TensorF32(tn(i, s.KNorm), kNormLen); err != nil {
					return err
				}
			}
		}
		// Block norms — Pre2 has only Pre*; Sandwich4 adds Post*. A hybrid whose norm
		// tensor NAMES (not just placement) differ between linear and full-attention
		// layers — Olmo Hybrid — uses the *Linear suffixes on isLinearLayer(i)
		// layers instead (see their own comment on tensorSchema).
		preAttn, postAttn, preMLP, postMLP := s.PreAttnNorm, s.PostAttnNorm, s.PreMLPNorm, s.PostMLPNorm
		hasLinearNormOverride := s.PreAttnNormLinear != "" || s.PostAttnNormLinear != "" || s.PreMLPNormLinear != "" || s.PostMLPNormLinear != ""
		if hasLinearNormOverride && arch.isLinearLayer(i) {
			preAttn, postAttn, preMLP, postMLP = s.PreAttnNormLinear, s.PostAttnNormLinear, s.PreMLPNormLinear, s.PostMLPNormLinear
		}
		if l.PreAttnNorm, err = optNorm(i, preAttn); err != nil {
			return err
		}
		if l.PostAttnNorm, err = optNorm(i, postAttn); err != nil {
			return err
		}
		if l.PreMLPNorm, err = optNorm(i, preMLP); err != nil {
			return err
		}
		if l.PostMLPNorm, err = optNorm(i, postMLP); err != nil {
			return err
		}
		// Gemma 4 FFN: the dense gated MLP is always present (the parallel dense
		// branch), plus — when enable_moe_block — the MoE sub-block (own router/
		// experts/norms). layer_scalar is a per-layer output multiplier. This owns
		// gemma4's FFN load because its MoE shape doesn't fit the generic Router/
		// Experts path below. (PLE is loaded by the shared code above via the
		// model-level PerLayer tensors; the per-layer inp_gate/proj come from the
		// GGUF path — the safetensors E-models with PLE are Phase 4.)
		if arch.gemma4 != nil {
			if l.GateProj, err = loadProj(tn(i, s.GateProj), cfg.IntermediateDim, hd); err != nil {
				return err
			}
			if l.UpProj, err = loadProj(tn(i, s.UpProj), cfg.IntermediateDim, hd); err != nil {
				return err
			}
			if l.DownProj, err = loadProj(tn(i, s.DownProj), hd, cfg.IntermediateDim); err != nil {
				return err
			}
			// layer_scalar (a [1] buffer). Absent ⇒ 1.0 (no scaling).
			if ls, lerr := st.TensorF32(tn(i, "layer_scalar"), 1); lerr == nil {
				l.LayerScalar = ls[0]
			} else {
				l.LayerScalar = 1
			}
			if arch.MoE != nil {
				if l.gemma4moe, err = loadGemma4MoE(st, i, cfg, arch, hd, l, quant, tn); err != nil {
					return err
				}
			}
			return nil
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
// mkQ builds a quantized-if-requested WeightMat, so this loader honours Options.Quant like every
// other family. Passed in rather than rebuilt here because the quant resolution lives in
// buildWeights with the rest of the load.
func loadQwen35Attn(st *embed.SafetensorsFile, i int, l *LayerWeights, arch *Architecture, hidden int,
	tn func(int, string) string, mkQ func(string, int, int) (linalg.WeightMat, error), quant quantMode) error {
	g := arch.qwen35
	var err error
	nm := func(suf string) string { return tn(i, suf) }
	if arch.isLinearLayer(i) {
		keyDim, valueDim := g.KeyHeadDim*g.NumKeyHeads, g.ValueHeadDim*g.NumValueHeads
		convDim := 2*keyDim + valueDim
		d := &deltaNetWeights{}
		if g.FusedDeltaNetProj {
			qkvzRows := 2*keyDim + 2*valueDim
			qkvz, e := st.TensorF32(nm("linear_attn.in_proj_qkvz.weight"), qkvzRows, hidden)
			if e != nil {
				return e
			}
			ba, e := st.TensorF32(nm("linear_attn.in_proj_ba.weight"), 2*g.NumValueHeads, hidden)
			if e != nil {
				return e
			}
			qkvF, zF := splitQwen3NextQKVZ(qkvz, g, hidden)
			d.inProjQKV = quantizeWM(linalg.WrapF32(qkvF, convDim, hidden), quant)
			d.inProjZ = quantizeWM(linalg.WrapF32(zF, valueDim, hidden), quant)
			d.inProjB, d.inProjA = splitQwen3NextBA(ba, g, hidden)
		} else if g.SeparateQKVProj {
			// Olmo Hybrid: q_proj/k_proj/v_proj are three fully independent tensors —
			// more unfused than qwen3.5's own pre-concatenated in_proj_qkv. Each
			// nn.Linear's weight is already [out_features, hidden] with out_features
			// in head-major order (view(...,-1,head_dim) is a reshape, not a
			// reorder), so vertically stacking the three f32 matrices in q,k,v order
			// reproduces the SAME flat [convDim, hidden] layout in_proj_qkv already
			// has — a plain row concat, no per-group interleaving like
			// splitQwen3NextQKVZ needs for the head-grouped fused case.
			qf, e := st.TensorF32(nm("linear_attn.q_proj.weight"), keyDim, hidden)
			if e != nil {
				return e
			}
			kf, e := st.TensorF32(nm("linear_attn.k_proj.weight"), keyDim, hidden)
			if e != nil {
				return e
			}
			vf, e := st.TensorF32(nm("linear_attn.v_proj.weight"), valueDim, hidden)
			if e != nil {
				return e
			}
			qkv := make([]float32, 0, convDim*hidden)
			qkv = append(qkv, qf...)
			qkv = append(qkv, kf...)
			qkv = append(qkv, vf...)
			d.inProjQKV = quantizeWM(linalg.WrapF32(qkv, convDim, hidden), quant)
			if d.inProjZ, err = mkQ(nm("linear_attn.g_proj.weight"), valueDim, hidden); err != nil {
				return err
			}
			if d.inProjB, err = st.TensorF32(nm("linear_attn.b_proj.weight"), g.NumValueHeads, hidden); err != nil {
				return err
			}
			if d.inProjA, err = st.TensorF32(nm("linear_attn.a_proj.weight"), g.NumValueHeads, hidden); err != nil {
				return err
			}
		} else {
			if d.inProjQKV, err = mkQ(nm("linear_attn.in_proj_qkv.weight"), convDim, hidden); err != nil {
				return err
			}
			if d.inProjZ, err = mkQ(nm("linear_attn.in_proj_z.weight"), valueDim, hidden); err != nil {
				return err
			}
			if d.inProjB, err = st.TensorF32(nm("linear_attn.in_proj_b.weight"), g.NumValueHeads, hidden); err != nil {
				return err
			}
			if d.inProjA, err = st.TensorF32(nm("linear_attn.in_proj_a.weight"), g.NumValueHeads, hidden); err != nil {
				return err
			}
		}
		if g.SeparateConv {
			// Olmo Hybrid: q_conv1d/k_conv1d/v_conv1d, split at the SAME q/k/v
			// channel boundaries as q_proj/k_proj/v_proj (keyDim, keyDim,
			// valueDim rows) — see SeparateConv's own comment.
			qc, e := st.TensorF32(nm("linear_attn.q_conv1d.weight"), keyDim, 1, g.ConvKernel)
			if e != nil {
				return e
			}
			kc, e := st.TensorF32(nm("linear_attn.k_conv1d.weight"), keyDim, 1, g.ConvKernel)
			if e != nil {
				return e
			}
			vc, e := st.TensorF32(nm("linear_attn.v_conv1d.weight"), valueDim, 1, g.ConvKernel)
			if e != nil {
				return e
			}
			conv := make([]float32, 0, convDim*g.ConvKernel)
			conv = append(conv, qc...)
			conv = append(conv, kc...)
			conv = append(conv, vc...)
			d.convW = conv
		} else if d.convW, err = st.TensorF32(nm("linear_attn.conv1d.weight"), convDim, 1, g.ConvKernel); err != nil {
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
		// DeltaNetNormSuffix/DeltaNetOutProjSuffix (Olmo Hybrid: o_norm/o_proj instead of
		// qwen3.5's norm/out_proj) — see their own comment on qwen35Params.
		normSuffix := "linear_attn.norm.weight"
		if g.DeltaNetNormSuffix != "" {
			normSuffix = g.DeltaNetNormSuffix
		}
		outProjSuffix := "linear_attn.out_proj.weight"
		if g.DeltaNetOutProjSuffix != "" {
			outProjSuffix = g.DeltaNetOutProjSuffix
		}
		if d.normW, err = st.TensorF32(nm(normSuffix), g.ValueHeadDim); err != nil {
			return err
		}
		if d.outProj, err = mkQ(nm(outProjSuffix), hidden, valueDim); err != nil {
			return err
		}
		l.delta = d
		return nil
	}
	hd := arch.HeadDim
	kvd := arch.NumKVHeads * hd
	a := &qwenAttnWeights{}
	if a.qProj, err = mkQ(nm("self_attn.q_proj.weight"), arch.NumHeads*hd*2, hidden); err != nil {
		return err
	}
	if a.kProj, err = mkQ(nm("self_attn.k_proj.weight"), kvd, hidden); err != nil {
		return err
	}
	if a.vProj, err = mkQ(nm("self_attn.v_proj.weight"), kvd, hidden); err != nil {
		return err
	}
	if a.oProj, err = mkQ(nm("self_attn.o_proj.weight"), hidden, arch.NumHeads*hd); err != nil {
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
// quantizes each into the resident format.
//
// P-11: streamed via Tensor.SubF32, one expert at a time — the same win
// streamExperts already banks for gemma4's fused experts (the bf16-26B transient
// fix). The two whole-tensor TensorF32 reads this used to do materialized
// nExpert*2*inter*hidden + nExpert*hidden*inter f32 floats before touching a
// single expert (~3 GB transient at Qwen3.6-35B-A3B shapes, per layer, times
// however many layers parallelLayers has in flight at once) — exactly the
// per-layer materialization streamExperts exists to avoid, just not routed
// through it because this loader predates the split of that helper out. Same
// external behavior (same expertWeights per index, same quantization), smaller
// peak.
func loadFusedExperts(st *embed.SafetensorsFile, gateUpName, downName string, nExpert, inter, hidden int, quant quantMode) ([]expertWeights, error) {
	guT, err := st.Tensor(gateUpName)
	if err != nil {
		return nil, err
	}
	dnT, err := st.Tensor(downName)
	if err != nil {
		return nil, err
	}
	guStride, downStride, half := 2*inter*hidden, hidden*inter, inter*hidden
	if guT.Elements() != nExpert*guStride {
		return nil, fmt.Errorf("experts %q: %d elements, want %d (=%d×%d×%d)", gateUpName, guT.Elements(), nExpert*guStride, nExpert, 2*inter, hidden)
	}
	if dnT.Elements() != nExpert*downStride {
		return nil, fmt.Errorf("experts %q: %d elements, want %d (=%d×%d×%d)", downName, dnT.Elements(), nExpert*downStride, nExpert, hidden, inter)
	}
	experts := make([]expertWeights, nExpert)
	for e := range experts {
		guE, err := guT.SubF32(e*guStride, guStride)
		if err != nil {
			return nil, err
		}
		dnE, err := dnT.SubF32(e*downStride, downStride)
		if err != nil {
			return nil, err
		}
		gate := quantizeWM(linalg.WrapF32(append([]float32(nil), guE[:half]...), inter, hidden), quant)
		up := quantizeWM(linalg.WrapF32(append([]float32(nil), guE[half:]...), inter, hidden), quant)
		dn := quantizeWM(linalg.WrapF32(append([]float32(nil), dnE...), hidden, inter), quant)
		experts[e] = expertWeights{Gate: gate, Up: up, Down: dn}
	}
	return experts, nil
}

// loadGemma4MoE loads one gemma4 layer's parallel dense+MoE FFN sub-block
// (enable_moe_block) into a gemma4MoEWeights, consumed by gemma4MoEFFN. The dense
// branch MLP (l.GateProj/UpProj/DownProj) and its sandwich norms (l.PreMLPNorm =
// pre_feedforward_layernorm, l.PostMLPNorm = the JOINT post_feedforward_layernorm)
// are already loaded by the caller — this aliases them and loads the MoE-specific
// tensors: the three parallel-branch norms, the weightless-norm/learned-scale
// router + per-expert scale, and the fused gelu-tanh experts (gate_up ‖ down).
//
// Router weights stay f32 (loadMat, no quant — the router is logit-critical); the
// experts quantize at load through the layer's quant mode like every other family
// (router-f32 / experts-int4). The experts stream one at a time via Tensor.SubF32
// (§4): each expert's slice is widened/quantized on its own, so a bf16 26B-A4B never
// materializes the whole [128, 2*inter, hidden] gate_up (a ~2 GB/layer transient) —
// only one expert's f32 at a time.
func loadGemma4MoE(st *embed.SafetensorsFile, i int, cfg *Config, arch *Architecture, hidden int, l *LayerWeights, quant quantMode, tn func(int, string) string) (*gemma4MoEWeights, error) {
	m := arch.MoE
	nm := func(s string) string { return tn(i, s) }
	w := &gemma4MoEWeights{
		preFFNNorm:  l.PreMLPNorm,  // pre_feedforward_layernorm   (dense pre-norm)
		postFFNNorm: l.PostMLPNorm, // post_feedforward_layernorm  (joint post-norm on x1+x2)
		mlpGate:     l.GateProj,
		mlpUp:       l.UpProj,
		mlpDown:     l.DownProj,
		layerScalar: l.LayerScalar,
		denseInter:  cfg.IntermediateDim,
		moeInter:    m.IntermediateDim,
		nE:          m.NumExperts,
		topK:        m.TopK,
	}
	var err error
	if w.postFFNNorm1, err = st.TensorF32(nm("post_feedforward_layernorm_1.weight"), hidden); err != nil {
		return nil, err
	}
	if w.preFFNNorm2, err = st.TensorF32(nm("pre_feedforward_layernorm_2.weight"), hidden); err != nil {
		return nil, err
	}
	if w.postFFNNorm2, err = st.TensorF32(nm("post_feedforward_layernorm_2.weight"), hidden); err != nil {
		return nil, err
	}
	if w.routerProj, err = loadMat(st, nm("router.proj.weight"), m.NumExperts, hidden); err != nil {
		return nil, err
	}
	if w.routerScale, err = st.TensorF32(nm("router.scale"), hidden); err != nil {
		return nil, err
	}
	if w.perExpertScale, err = st.TensorF32(nm("router.per_expert_scale"), m.NumExperts); err != nil {
		return nil, err
	}
	// Fused experts: gate_up [E, 2*moeInter, hidden] (gate ‖ up on the row axis),
	// down [E, hidden, moeInter]. Streamed + quantized per expert.
	guT, err := st.Tensor(nm("experts.gate_up_proj"))
	if err != nil {
		return nil, err
	}
	dnT, err := st.Tensor(nm("experts.down_proj"))
	if err != nil {
		return nil, err
	}
	if w.expertsGateUp, err = streamExperts(guT, m.NumExperts, 2*m.IntermediateDim, hidden, quant); err != nil {
		return nil, err
	}
	if w.expertsDown, err = streamExperts(dnT, m.NumExperts, hidden, m.IntermediateDim, quant); err != nil {
		return nil, err
	}
	return w, nil
}

// streamExperts slices a fused [nExpert, rows, cols] safetensors tensor into per-
// expert WeightMats, widening + quantizing ONE expert at a time via Tensor.SubF32
// so the whole 3-D f32 is never materialized (the bf16-26B transient win). Each
// expert passes through quantizeWM: int8/int4 produces owned quantized data and the
// f32 slice is dropped; f32/quantNone leaves an f32 WeightMat aliasing the mapping
// (pageable, valid while w.st is retained) — no heap copy.
func streamExperts(t embed.Tensor, nExpert, rows, cols int, quant quantMode) ([]linalg.WeightMat, error) {
	stride := rows * cols
	if t.Elements() != nExpert*stride {
		return nil, fmt.Errorf("experts %q: %d elements, want %d (=%d×%d×%d)", t.Name, t.Elements(), nExpert*stride, nExpert, rows, cols)
	}
	out := make([]linalg.WeightMat, nExpert)
	for e := range nExpert {
		f32, err := t.SubF32(e*stride, stride)
		if err != nil {
			return nil, err
		}
		if fakeQuantScheme != "" && fakeQuantExpertsOnly { // DIAGNOSTIC (default-off): fake-4-bit the EXPERTS only; see fakequant.go
			out[e] = fakeInt4WM(f32, rows, cols, fakeQuantScheme)
			continue
		}
		out[e] = quantizeWM(linalg.WrapF32(f32, rows, cols), quant)
	}
	return out, nil
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
	// GProj is Laguna's attention output gate (g_proj). "" = the family has none.
	GProj                      string
	QBias, KBias, VBias        string // "" = no projection bias (Qwen2 sets q/k/v)
	QNorm, KNorm               string // "" = no QK-norm
	PreAttnNorm, PostAttnNorm  string
	GateProj, UpProj, DownProj string
	PreMLPNorm, PostMLPNorm    string
	// PreAttnNormLinear/PostAttnNormLinear/PreMLPNormLinear/PostMLPNormLinear
	// override the corresponding suffix above on layers where isLinearLayer(i) is
	// true. "" (every family so far) ⇒ no override, same suffix on every layer.
	// Olmo Hybrid needs this: its two decoder-layer CLASSES each independently
	// define an attribute literally NAMED "post_attention_layernorm", but in
	// DIFFERENT POSITIONAL ROLES — a post-attn norm on full-attention layers
	// (NormPostOnly), a pre-MLP norm on linear/DeltaNet layers (NormPre2) — and
	// only its linear layers carry an input_layernorm tensor at all (full-attention
	// layers have none, per NormPostOnly). One static per-family schema can't route
	// one on-disk name to two different LayerWeights fields depending on layer
	// kind; these four give linear layers their own suffix set instead.
	PreAttnNormLinear, PostAttnNormLinear, PreMLPNormLinear, PostMLPNormLinear string
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

// lfm2TensorSchema: LFM2/LFM2.5. Tied head, Pre2 norms under LFM2's own names
// (operator_norm before the mixer, ffn_norm before the FFN), per-head RMSNorm on Q and K,
// SwiGLU under llama's w1/w2/w3 naming, and attention output as out_proj rather than o_proj.
//
// The attention entries apply to the 8 attention layers only; the 22 conv layers have none of
// them and instead carry conv.{in_proj,conv,out_proj}, which this schema cannot express (it has
// no conv roles) and buildLFM2Weights loads directly — the same division Granite uses for its
// Mamba tensors.
//
// FinalNorm is embedding_norm, not model.norm: LFM2 normalises before the tied LM head under a
// name no other family here uses, so a copy-paste of "model.norm.weight" would fail to load
// rather than load the wrong thing — which is the better failure, but worth naming.
var lfm2TensorSchema = tensorSchema{
	Embed:       "model.embed_tokens.weight",
	LMHead:      "", // tied (tie_word_embeddings true on every released checkpoint)
	FinalNorm:   "model.embedding_norm.weight",
	QProj:       "self_attn.q_proj.weight",
	KProj:       "self_attn.k_proj.weight",
	VProj:       "self_attn.v_proj.weight",
	OProj:       "self_attn.out_proj.weight",
	QNorm:       "self_attn.q_layernorm.weight",
	KNorm:       "self_attn.k_layernorm.weight",
	PreAttnNorm: "operator_norm.weight",
	GateProj:    "feed_forward.w1.weight",
	UpProj:      "feed_forward.w3.weight",
	DownProj:    "feed_forward.w2.weight",
	PreMLPNorm:  "ffn_norm.weight",
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

// cohereTensorSchema: Cohere / Command-R — llama attention/MLP tensor names, but
// the parallel block has ONE norm per layer (input_layernorm, shared by attn and
// MLP), so PreMLPNorm/PostAttnNorm/PostMLPNorm are all empty. No biases anywhere;
// no QK-norm (cohere1 Phase 1). Embeddings are tied (LMHead empty ⇒ use Embed).
var cohereTensorSchema = tensorSchema{
	Embed:        "model.embed_tokens.weight",
	LMHead:       "", // tied
	FinalNorm:    "model.norm.weight",
	QProj:        "self_attn.q_proj.weight",
	KProj:        "self_attn.k_proj.weight",
	VProj:        "self_attn.v_proj.weight",
	OProj:        "self_attn.o_proj.weight",
	QNorm:        "", // no QK-norm (Phase 1)
	KNorm:        "",
	PreAttnNorm:  "input_layernorm.weight", // the single shared parallel-block norm
	PostAttnNorm: "",                       // parallel: no post-attn norm
	GateProj:     "mlp.gate_proj.weight",
	UpProj:       "mlp.up_proj.weight",
	DownProj:     "mlp.down_proj.weight",
	PreMLPNorm:   "", // parallel: MLP reads the shared input norm, no separate pre-MLP norm
	PostMLPNorm:  "",
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
// qwen35DenseTensorSchema is Qwen3.8's (model_type qwen3_5): identical to the MoE sibling's
// except the router/expert names give way to a plain SwiGLU. Kept as its own value rather
// than mutating the MoE schema, so the MoE families are untouched by this addition.
var qwen35DenseTensorSchema = tensorSchema{
	Embed:       "model.embed_tokens.weight",
	LMHead:      "lm_head.weight",
	FinalNorm:   "model.norm.weight",
	QProj:       "self_attn.q_proj.weight",
	KProj:       "self_attn.k_proj.weight",
	VProj:       "self_attn.v_proj.weight",
	OProj:       "self_attn.o_proj.weight",
	QNorm:       "self_attn.q_norm.weight",
	KNorm:       "self_attn.k_norm.weight",
	PreAttnNorm: "input_layernorm.weight",
	PreMLPNorm:  "post_attention_layernorm.weight",
	GateProj:    "mlp.gate_proj.weight",
	UpProj:      "mlp.up_proj.weight",
	DownProj:    "mlp.down_proj.weight",
}

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

// qwen3MoeTensorSchema: qwen3's attention tensor names (per-head q_norm/k_norm,
// no q/k/v bias) with the FFN replaced on every layer by a sparse MoE (router
// mlp.gate + per-expert mlp.experts.%d.*) — no shared expert (Shared* fields
// left empty; arch.MoE.SharedIntermediateDim stays 0, which both the
// safetensors and GGUF loaders already read as "no shared expert").
var qwen3MoeTensorSchema = tensorSchema{
	Embed:       "model.embed_tokens.weight",
	LMHead:      "lm_head.weight",
	FinalNorm:   "model.norm.weight",
	QProj:       "self_attn.q_proj.weight",
	KProj:       "self_attn.k_proj.weight",
	VProj:       "self_attn.v_proj.weight",
	OProj:       "self_attn.o_proj.weight",
	QNorm:       "self_attn.q_norm.weight",
	KNorm:       "self_attn.k_norm.weight",
	PreAttnNorm: "input_layernorm.weight",
	PreMLPNorm:  "post_attention_layernorm.weight",
	Router:      "mlp.gate.weight",
	ExpertGate:  "mlp.experts.%d.gate_proj.weight",
	ExpertUp:    "mlp.experts.%d.up_proj.weight",
	ExpertDown:  "mlp.experts.%d.down_proj.weight",
}

// olmo3TensorSchema: Olmo 3 — QKNormWhole's q_norm/k_norm are whole-vector but the TENSOR NAME
// is identical to the per-head convention; only NormPostOnly's PreAttnNorm/PreMLPNorm being empty
// (no input_layernorm tensor exists at all — confirmed by instantiating Olmo3ForCausalLM
// directly) and PostAttnNorm/PostMLPNorm mapping to post_attention_layernorm/
// post_feedforward_layernorm actually differs from qwen3TensorSchema.
var olmo3TensorSchema = tensorSchema{
	Embed:        "model.embed_tokens.weight",
	LMHead:       "lm_head.weight",
	FinalNorm:    "model.norm.weight",
	QProj:        "self_attn.q_proj.weight",
	KProj:        "self_attn.k_proj.weight",
	VProj:        "self_attn.v_proj.weight",
	OProj:        "self_attn.o_proj.weight",
	QNorm:        "self_attn.q_norm.weight",
	KNorm:        "self_attn.k_norm.weight",
	PreAttnNorm:  "", // NormPostOnly: no input_layernorm tensor exists
	PostAttnNorm: "post_attention_layernorm.weight",
	GateProj:     "mlp.gate_proj.weight",
	UpProj:       "mlp.up_proj.weight",
	DownProj:     "mlp.down_proj.weight",
	PreMLPNorm:   "", // NormPostOnly: no pre-MLP norm tensor exists
	PostMLPNorm:  "post_feedforward_layernorm.weight",
}

// olmoHybridTensorSchema: Olmo Hybrid's full-attention layers only (its linear/DeltaNet layers
// are loaded entirely by loadQwen35Attn, gated on PlainFullAttn — see qwen35Params). Identical to
// olmo3TensorSchema for those layers (verified against the real modeling_olmo_hybrid.py:
// OlmoHybridAttention literally inherits Olmo3Attention's behavior), plus the *Linear norm
// overrides: linear layers carry an input_layernorm the full-attention layers lack, and reuse the
// SAME on-disk name as the full-attention layers' PostAttnNorm ("post_attention_layernorm.weight")
// for a DIFFERENT role — pre-MLP norm, not post-attn norm — because the two decoder-layer classes
// each independently define an attribute with that name (see PreAttnNormLinear's own comment).
var olmoHybridTensorSchema = tensorSchema{
	Embed:              "model.embed_tokens.weight",
	LMHead:             "lm_head.weight", // tie_word_embeddings false on the released checkpoint
	FinalNorm:          "model.norm.weight",
	QProj:              "self_attn.q_proj.weight",
	KProj:              "self_attn.k_proj.weight",
	VProj:              "self_attn.v_proj.weight",
	OProj:              "self_attn.o_proj.weight",
	QNorm:              "self_attn.q_norm.weight",
	KNorm:              "self_attn.k_norm.weight",
	PreAttnNorm:        "", // NormPostOnly (full-attention layers): no input_layernorm tensor
	PostAttnNorm:       "post_attention_layernorm.weight",
	GateProj:           "mlp.gate_proj.weight",
	UpProj:             "mlp.up_proj.weight",
	DownProj:           "mlp.down_proj.weight",
	PreMLPNorm:         "", // NormPostOnly (full-attention layers): no pre-MLP norm tensor
	PostMLPNorm:        "post_feedforward_layernorm.weight",
	PreAttnNormLinear:  "input_layernorm.weight",          // NormPre2 (linear layers)
	PostAttnNormLinear: "",                                // NormPre2: no post-attn norm
	PreMLPNormLinear:   "post_attention_layernorm.weight", // SAME name, pre-MLP role here
	PostMLPNormLinear:  "",                                // NormPre2: no post-MLP norm
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

// internlm2TensorSchema is a MARKER, like phi3's: InternLM2's fused, GROUPED wqkv cannot be
// expressed as per-tensor suffixes, so buildInternLM2Weights loads this family directly and
// the generic schema-driven path is never used. The names are recorded here anyway so the
// family's layout is discoverable from the same place every other family's is.
var internlm2TensorSchema = tensorSchema{
	Embed:       "model.tok_embeddings.weight",
	LMHead:      "output.weight",
	FinalNorm:   "model.norm.weight",
	OProj:       "attention.wo.weight",
	PreAttnNorm: "attention_norm.weight",
	PreMLPNorm:  "ffn_norm.weight",
	GateProj:    "feed_forward.w1.weight", // llama's original naming: w1 gate, w3 up, w2 down
	UpProj:      "feed_forward.w3.weight",
	DownProj:    "feed_forward.w2.weight",
	// QProj/KProj/VProj intentionally empty: they live inside attention.wqkv.
}

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
	// The released NVIDIA checkpoints name the embedding backbone.embeddings.weight;
	// transformers' own NemotronH names it backbone.embedding.weight, which is the only
	// spelling the tiny fixture (built by instantiating the config) ever produced. Every
	// other tensor name agrees, so this one word is the whole delta — try both.
	if w.Embed, err = loadMat(st, "backbone.embedding.weight", vocab, hidden); err != nil {
		if w.Embed, err = loadMat(st, "backbone.embeddings.weight", vocab, hidden); err != nil {
			return nil, err
		}
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
		case nemoMoE:
			// Nemotron 3 Nano's MoE FFN. Tensor names verified against the real safetensors
			// index (nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16), not assumed from the
			// generic MoE schema: "mixer.gate.weight" (router), "mixer.gate.
			// e_score_correction_bias" (selection bias), "mixer.experts.{i}.
			// {up,down}_proj.weight" (per-expert, NO gate_proj — non-gated relu²) and
			// "mixer.shared_experts.{up,down}_proj.weight" (the always-on shared expert,
			// also non-gated). Router stays f32/unquantized like every other family
			// (logit-critical); experts and the shared expert quantize like the plain
			// nemoMLP case above.
			moe := arch.MoE
			if lw.Router, e = loadMat(st, tn("mixer.gate.weight"), moe.NumExperts, hidden); e != nil {
				return e
			}
			if lw.RouterBias, e = st.TensorF32(tn("mixer.gate.e_score_correction_bias"), moe.NumExperts); e != nil {
				return e
			}
			lw.Experts = make([]expertWeights, moe.NumExperts)
			for ei := range lw.Experts {
				ex := &lw.Experts[ei]
				if ex.Up, e = loadMat(st, tn(fmt.Sprintf("mixer.experts.%d.up_proj.weight", ei)), moe.IntermediateDim, hidden); e != nil {
					return e
				}
				if ex.Down, e = loadMat(st, tn(fmt.Sprintf("mixer.experts.%d.down_proj.weight", ei)), hidden, moe.IntermediateDim); e != nil {
					return e
				}
				ex.Up, ex.Down = quantizeWM(ex.Up, quant), quantizeWM(ex.Down, quant)
			}
			if moe.SharedIntermediateDim > 0 {
				se := &lw.SharedExpert
				if se.Up, e = loadMat(st, tn("mixer.shared_experts.up_proj.weight"), moe.SharedIntermediateDim, hidden); e != nil {
					return e
				}
				if se.Down, e = loadMat(st, tn("mixer.shared_experts.down_proj.weight"), hidden, moe.SharedIntermediateDim); e != nil {
					return e
				}
				se.Up, se.Down = quantizeWM(se.Up, quant), quantizeWM(se.Down, quant)
			}
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

// lagunaTensorSchema: Laguna (poolside). Read from the REAL Laguna-XS.2 checkpoint
// index rather than inferred from modeling_laguna.py, because the two disagree in
// two places that matter:
//
//  1. The module allocates FUSED 3D expert parameters (LagunaExperts holds
//     gate_up_proj [E, 2*inter, hidden] and down_proj [E, hidden, inter]), but the
//     shipped checkpoint stores PER-EXPERT 2D tensors — 9984 = 39 MoE layers × 256
//     experts of each. HF re-packs them at load via its conversion mapping. The
//     per-expert form is what goinfer already reads, so the Expert* templates apply
//     unchanged and no stacked-expert handling is needed.
//
//  2. The module names the shared expert self.shared_experts (PLURAL, as GLM and
//     DeepSeek do), but the checkpoint keys are mlp.shared_expert.* (SINGULAR).
//
// RouterBias is likewise the SHIPPED spelling: the bias lives under mlp.experts.*
// on disk and HF's _checkpoint_conversion_mapping rewrites it to mlp.gate.* at
// load, so reading the checkpoint directly means taking the experts spelling.
//
// The dense prefix layers (mlp_only_layers) use the plain GateProj/UpProj/DownProj
// names at the model's intermediate_size; the MoE layers use the Expert*/Shared*
// names at moe_intermediate_size. See docs/task-laguna.md.
var lagunaTensorSchema = tensorSchema{
	Embed:       "model.embed_tokens.weight",
	LMHead:      "lm_head.weight",
	FinalNorm:   "model.norm.weight",
	QProj:       "self_attn.q_proj.weight",
	KProj:       "self_attn.k_proj.weight",
	VProj:       "self_attn.v_proj.weight",
	OProj:       "self_attn.o_proj.weight",
	GProj:       "self_attn.g_proj.weight",
	QNorm:       "self_attn.q_norm.weight",
	KNorm:       "self_attn.k_norm.weight",
	PreAttnNorm: "input_layernorm.weight",
	PreMLPNorm:  "post_attention_layernorm.weight",
	// dense prefix (mlp_only_layers) layers
	GateProj: "mlp.gate_proj.weight",
	UpProj:   "mlp.up_proj.weight",
	DownProj: "mlp.down_proj.weight",
	// MoE layers
	Router:     "mlp.gate.weight",
	RouterBias: "mlp.experts.e_score_correction_bias",
	ExpertGate: "mlp.experts.%d.gate_proj.weight",
	ExpertUp:   "mlp.experts.%d.up_proj.weight",
	ExpertDown: "mlp.experts.%d.down_proj.weight",
	SharedGate: "mlp.shared_expert.gate_proj.weight",
	SharedUp:   "mlp.shared_expert.up_proj.weight",
	SharedDown: "mlp.shared_expert.down_proj.weight",
	// SharedExpertGate empty: Laguna adds the shared expert with no outer sigmoid gate.
}
