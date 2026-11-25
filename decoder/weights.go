package decoder

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder/quant"
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
	// (Rows()==0) for every other family. See applyGateRow.
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
	// bleeding attention mass. nil for every other family. Consumed by attendSinksToken.
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

	// DeepSeek MLA attention weights (deepseek_v2 / deepseek_v3, and Bailing Hybrid's own
	// full-attention layers). nil for every other family. Stored f32 (parity-first). The FFN
	// side (dense prefix / MoE / shared expert) reuses the generic Router/Experts/SharedExpert
	// fields above.
	mla *mlaWeights

	// Bailing Hybrid (Ling 3.0) KDA attention weights, set on the linear layers (the
	// non-linear layers use mla above). nil for every other family.
	kda *kdaWeights

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
	// Quantizable: these stay f32 when no quant is requested.
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
	gProj        []float32 // [numHeads] (head_wise) or [numHeads*v_head_dim] (element_wise); nil ⇒ no gate (every DeepSeek family)
	// qAQ, qBQ, qQ, kvAQ and oQ are those projections as per-row int8 on a load armed for it (mlaProjInt8, quantizeProjections)
	// or read from a .giw that carries them, their f32 slices then nil.
	qAQ, qBQ, qQ, kvAQ, oQ linalg.WeightMat
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
	// prof records how long each phase of this load took, when the loader instrumented it (the GGUF path does); nil otherwise.
	// It rides on Weights because that is the value the loader already returns.
	prof *LoadProfile

	Cfg  Config
	arch *Architecture // resolved descriptor the forward pass reads

	// bakedQuant is the resolved quant label recorded in the .giw header: "int4" | "int4mix" | "int8int8" | "int8" |
	// "native". Model.Quant() prefers it over re-inferring from tensor kinds. Empty for a direct GGUF/safetensors load
	// and for a streamed bundle (which records ""); those fall back to quantLabel() inference.
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

	schema *tensorSchema // resolved tensor-name schema (safetensors loads); used by compute-time LoRA to map projections → adapter deltas. nil for GGUF/serialized loads.

	// kquant is where each int8 matmul of a q4k GGUF load came from when the file stores it as Q5_K or Q6_K, keyed by its int8
	// rows' first element; ggufPath is that file, and kquantFile the load's own mapping of it, held until the model's resident
	// build has run (Model.releaseKQuantFile). All are empty for any other load. KQuantReader serves them.
	kquant     map[*int8]kquantSrc
	ggufPath   string
	kquantFile *embed.GGUFFile
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
		// GProj is deliberately excluded from quantization: it is tiny (one row per head, well under 1% of a layer's attention
		// weights), so quantizing buys nothing, and it feeds a softplus whose output multiplies the whole attention context, so
		// quantization error there scales every channel of every head rather than perturbing one projection additively.
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

// repackedOnlyInt4Count reports how many of w's matmulWeights() tensors are int4-resident with no canonical bytes
// (IsInt4() true, Int4()'s ok false): a kind-5 .giw tensor, or an in-RAM repacked-only build that reached a .giw
// round-trip (no writer produces one today). decoder.Load's .giw branch uses it to refuse such a file under a backend
// that needs canonical bytes.
func repackedOnlyInt4Count(w *Weights) int {
	n := 0
	for _, m := range w.matmulWeights() {
		if !m.IsInt4() {
			continue
		}
		if _, _, _, ok := m.Int4F16(); !ok {
			n++
		}
	}
	return n
}

// isLogitTable reports whether m is an embedding-class tensor: the token embedding, the LM head, or Gemma 4's model-level
// PLE embeddings. In int4 mode these are pinned to int8 by default (logit-critical; the EmbedInt4 knob relaxes them), so
// their precision is orthogonal to the int4-vs-int4mix distinction and must be excluded from the quant classification.
// This is the single definition of that exclusion.
func (w *Weights) isLogitTable(m *linalg.WeightMat) bool {
	return m == &w.Embed || m == &w.LMHead || m == &w.PerLayerTokenEmbed || m == &w.PerLayerModelProj
}

// bodyMatmulWeights is matmulWeights minus the logit tables: the attention/FFN projections, experts and routers, exactly
// the matmuls whose precision the chosen quant (int4|int4mix|int8|int8int8) determines. It is the one list quantLabel
// classifies over (and, through quantLabel, the value the .giw header records at bake time), so "which tensors define the
// quant" lives in one place. cuda's nonBatchableKind also excludes the logit tables but inspects resident per-layer
// projections, a different type in another module, so it cannot share this host-side function.
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

// LoadWeights reads config.json and the model.safetensors blob (or sharded set) from an on-disk directory in the HF
// snapshot layout. The blob is mmapped, not heap-copied, so a large checkpoint stays in the OS page cache. bf16 and f16
// weights are widened to f32 on load, which roughly doubles resident RAM against keeping them bf16; per-tile widening
// inside matmul would avoid that and is not implemented.
//
// Use LoadWeightsFromFS for fs.FS-backed paths (MapFS, embed.FS); those stay heap-backed because fs.FS exposes no file
// descriptor.
func LoadWeights(dir string) (*Weights, error) {
	return loadWeights(dir, weightLoadOpts{needCanonical: true})
}

// errLoadAborted is parallelLayers' sentinel when abort closes mid-build, the swap tripwire's load-time consumer.
// decoder.Load does not know why abort closed (a swap watch it neither owns nor
// imports); the caller that armed the watch checks errors.Is(err, errLoadAborted) and wraps it with its own reason and
// pricing detail (see Options.LoadAbort).
var errLoadAborted = errors.New("decoder: load aborted")

// parallelLayers runs fn over the n layer indices across a worker pool, so the per-tensor dequant and re-quant
// (independent per layer: distinct WeightMat slots, read-only source) fans out across cores. The first error stops
// further work and is returned. Transient memory scales with the worker count (each in-flight layer briefly holds its
// dequantized f32); GOMAXPROCS workers is the right trade on a machine that can hold the model.
//
// abort, if non-nil, is checked between layers (at grab, never inside a layer in flight): once closed, no new fn(i)
// starts, in-flight ones finish, and the call returns errLoadAborted, unless an fn(i) returned a real build error, which
// always wins over a concurrent abort. A nil abort blocks forever in a select, so it is a zero-cost no-op for callers
// that pass none.
func parallelLayers(n int, abort <-chan struct{}, fn func(i int) error) error {
	if n <= 1 {
		if n == 1 {
			select {
			case <-abort:
				return errLoadAborted
			default:
			}
			return fn(0)
		}
		return nil
	}
	workers := min(runtime.GOMAXPROCS(0), n)
	var (
		next     int
		mu       sync.Mutex
		firstErr error
		aborted  bool
		wg       sync.WaitGroup
	)
	grab := func() (int, bool) {
		mu.Lock()
		defer mu.Unlock()
		if next >= n || firstErr != nil {
			return 0, false
		}
		select {
		case <-abort:
			aborted = true
			return 0, false
		default:
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
	if firstErr != nil {
		return firstErr // a real build error always wins over a concurrent abort
	}
	if aborted {
		return errLoadAborted
	}
	return nil
}

// loadWeights is the quant-aware internal load. When quant is int8/int4, each
// matmul tensor is quantized the moment it is read and its f32 backing is freed
// before the next tensor loads — so the transient footprint is the quantized
// model plus one tensor's f32, not the whole model in f32. That is what lets a
// big quantized checkpoint load in a quarter (int8) or eighth (int4) of the RAM
// the load-everything-then-quantize path needed. The forward output is identical
// to quantizing after load; only the peak memory differs.
func loadWeights(dir string, o weightLoadOpts) (*Weights, error) {
	quant, embedInt4, needCanonical, skipRow4, lora, abort := o.quant, o.embedInt4, o.needCanonical, o.skipRow4, o.lora, o.abort
	// One atomic add per model load, so the fit guard's test can observe that a refused load allocated nothing rather than
	// infer it from an error string (a guard that fires after the allocation still looks correct that way).
	weightAllocs.Add(1)
	if strings.HasSuffix(dir, ".gguf") {
		return loadGGUFWeights(dir, quant, embedInt4, needCanonical, skipRow4, abort) // quantized llama.cpp checkpoint; LoRA guarded in Load
	}
	if quant == quantInt4Mix {
		return nil, fmt.Errorf("decoder: int4mix is GGUF-only (got safetensors %s)", dir)
	}
	if quant == quantQ4K {
		return nil, fmt.Errorf("decoder: q4k keeps a GGUF's Q4_K tensors as they are, so it is GGUF-only (got safetensors %s)", dir)
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
	// buildWeightsFromSafetensors retains st (the WeightMats may alias its mmap) only on success; on any error return st
	// would leak the mapping and fd. A serve process probing candidate dirs, or retrying a load of a checkpoint with one
	// missing tensor, would accumulate address space, the leak Model.Close exists to avoid.
	w, err := buildWeightsFromSafetensors(cfg, arch, schema, st, o)
	if err == nil {
		// Every family's loader, not only the generic one, answers for the adapter here: a delta the load never merged is
		// refused, which covers the loaders that take no lora at all.
		err = lora.checkAllMerged()
	}
	if err != nil {
		_ = st.Close()
		return nil, err
	}
	// Release the source mapping now when nothing can alias it, instead of holding it for the model's whole life. The mapping
	// is the bf16 checkpoint, while the quantized weights decode reads are a separate, much smaller allocation; holding the
	// source lets dead pages compete with hot weights for page cache (higher RSS and slower decode).
	if why := mmapAliasRisk(st); why == "" && !islandsMayAliasSource(w) && os.Getenv("GOINFER_P13_OFF") == "" {
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

// islandsMayAliasSource reports whether a bf16 island may be the checkpoint's own bits: stIsland keeps a BF16 tensor as a
// zero-copy view of the mapping, which mmapAliasRisk's dtype rule does not see, so a load holding any bf16 island keeps the
// source mapped. Conservative: an island islandMat made bf16 on the heap keeps it too.
func islandsMayAliasSource(w *Weights) bool {
	for i := range w.Layers {
		if sc := w.Layers[i].shortConv; sc != nil {
			for _, m := range []*linalg.WeightMat{&sc.inProj, &sc.outProj} {
				if _, bf16, ok := m.Half16(); ok && bf16 {
					return true
				}
			}
		}
	}
	return false
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

// openCheckpointMmap mmaps the checkpoint weights: the multi-shard set named by model.safetensors.index.json when
// present, else the single model.safetensors. Either way the returned file resolves Tensor() uniformly.
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

// LoadWeightsFromFS reads config.json and model.safetensors from fsys/dir, validates every tensor's shape against Cfg,
// and returns the populated bundle. It is heap-backed (fs.ReadFile); use LoadWeights for the mmap path on a real
// directory.
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
	w, err := buildWeightsFromSafetensors(cfg, arch, schema, st, weightLoadOpts{quant: quant, needCanonical: true})
	if err != nil {
		_ = st.Close() // st is retained only on success; close it on error so the mapping/fd doesn't leak
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

// buildWeightsFromSafetensors fills a *Weights from an already-opened SafetensorsFile, shape-validating every tensor
// against Cfg through the family's tensor schema. The heap (fs.FS) and mmap paths share it so the tensor-name and shape
// contract is one edit, not two.
func buildWeightsFromSafetensors(cfg *Config, arch *Architecture, s *tensorSchema, st *embed.SafetensorsFile, o weightLoadOpts) (*Weights, error) {
	return buildWeightsFromSafetensorsTo(cfg, arch, s, st, o.quant, o.embedInt4, o.needCanonical, o.skipRow4, o.lora, nil, "", o.abort)
}

// weightLoadOpts is how loadWeights and buildWeightsFromSafetensors build a checkpoint's weights: the quantization,
// whether the embedding goes int4, whether the canonical int4 layout is kept beside a repacked one (needCanonical) or
// the row4 fallback skipped (skipRow4), the LoRA adapter to merge, and the load's abort channel.
type weightLoadOpts struct {
	quant                              quantMode
	embedInt4, needCanonical, skipRow4 bool
	lora                               *loraAdapter
	abort                              <-chan struct{}
}

// errDirNoStream is buildWeightsFromSafetensorsTo's answer, given a sink, for a family whose dedicated builder does not
// stream yet; StreamTranscodeDir's caller then takes the resident transcode instead.
type errDirNoStream struct{ family string }

func (e errDirNoStream) Error() string {
	return fmt.Sprintf("decoder: the %s safetensors loader does not stream layer by layer", e.family)
}

// IsDirNoStream reports whether err is StreamTranscodeDir declining a family whose loader does not stream.
func IsDirNoStream(err error) bool {
	_, ok := errors.AsType[errDirNoStream](err)
	return ok
}

// loadEmbedTable loads a [rows, cols] logit table (the token embedding or an untied head) at the embedding precision mode:
// whole and then quantized for the resident build, a row at a time when streaming (a 256k x 4096 table is 4.2 GB as f32,
// the largest single allocation of a transcode). Both give the same bytes.
func loadEmbedTable(st *embed.SafetensorsFile, name string, rows, cols int, mode quantMode, needCanonical bool, sink *giwWriter) (linalg.WeightMat, error) {
	if sink == nil {
		m, err := loadMat(st, name, rows, cols)
		if err != nil {
			return m, err
		}
		return quantizeEmbedWM(m, mode, needCanonical), nil
	}
	if r := quant.ForTensor(st, nil, name); r != nil {
		if rs, ok := r.(quant.RowStreamReconstructor); ok {
			rowInto, err := rs.RowReader2D(st, name, rows, cols)
			if err != nil {
				return linalg.WeightMat{}, err
			}
			return streamQuantizedEmbed(rows, cols, mode, needCanonical, !repackForDecode(sink), 1, streamEmbedWorkers(sink), func() rowReader { return rowInto })
		}
		data, err := r.ReconstructMat2D(st, name, rows, cols)
		if err != nil {
			return linalg.WeightMat{}, err
		}
		readRow := func(r int, dst []float32) error {
			copy(dst, data[r*cols:(r+1)*cols])
			return nil
		}
		return streamQuantizedEmbed(rows, cols, mode, needCanonical, !repackForDecode(sink), 1, streamEmbedWorkers(sink), func() rowReader { return readRow })
	}
	t, err := st.Tensor(name)
	if err != nil {
		return linalg.WeightMat{}, err
	}
	if len(t.Shape) != 2 || t.Shape[0] != rows || t.Shape[1] != cols {
		return linalg.WeightMat{}, fmt.Errorf("decoder: %q shape %v, want [%d %d]", name, t.Shape, rows, cols)
	}
	// Read in blocks of rows: one bf16->f32 conversion per block instead of per row (a 262k-row table is 262k calls
	// otherwise). A block of the widest table here is about 10 MB, and each row range holds one (its reader's own).
	const block = 1024
	return streamQuantizedEmbed(rows, cols, mode, needCanonical, !repackForDecode(sink), block, streamEmbedWorkers(sink), func() rowReader {
		var buf []float32
		base := -1
		return func(r int, dst []float32) error {
			if base < 0 || r < base || r >= base+block {
				base = r - r%block
				var rerr error
				if buf, rerr = t.SubF32(base*cols, min(block, rows-base)*cols); rerr != nil {
					return rerr
				}
			}
			copy(dst, buf[(r-base)*cols:(r-base+1)*cols])
			return nil
		}
	})
}

// finishLayers runs a builder's per-layer loads. Resident (sink nil): in parallel. Streaming: the bundle head (every
// global exists by now), then the layers through streamLayers, a window at a time, so the model is never resident at
// once; the head records no quant label, as StreamTranscodeGGUF's does.
func finishLayers(w *Weights, arch *Architecture, loadLayer func(i int) error, sink *giwWriter, id string, abort <-chan struct{}) (*Weights, error) {
	n := len(w.Layers)
	if sink == nil {
		if err := parallelLayers(n, abort, loadLayer); err != nil {
			return nil, err
		}
		return w, nil
	}
	sink.arch = arch
	if err := sink.writeHeadGlobals(w, id); err != nil {
		return nil, err
	}
	// The logit tables and Gemma 4's per-layer token table are written; nothing after the head reads them.
	w.Embed, w.LMHead, w.PerLayerTokenEmbed = linalg.WeightMat{}, linalg.WeightMat{}, linalg.WeightMat{}
	if err := streamLayers(w, arch, sink, loadLayer); err != nil {
		return nil, err
	}
	return w, nil
}

// buildWeightsFromSafetensorsTo is buildWeightsFromSafetensors with an optional sink. With sink nil it is the resident
// build. With a sink (StreamTranscodeDir) the generic builder writes the bundle head once the globals exist, then loads,
// writes and frees one layer at a time, and loads the embedding (and an untied head) a row at a time, so peak memory is
// about the globals plus one layer rather than the whole model. The dedicated builders stream their layers the same way
// (their embedding still loads whole); a family whose builder does not stream returns errDirNoStream.
func buildWeightsFromSafetensorsTo(cfg *Config, arch *Architecture, s *tensorSchema, st *embed.SafetensorsFile, quant quantMode, embedInt4, needCanonical, skipRow4 bool, lora *loraAdapter, sink *giwWriter, id string, abort <-chan struct{}) (*Weights, error) {
	skipRow4 = skipRow4 || !repackForDecode(sink) // every builder below takes it from here
	if sink != nil {
		switch {
		case arch.Name == "gpt2", arch.Name == "internlm2", arch.gptoss != nil:
			return nil, errDirNoStream{arch.Name}
		}
	}
	if arch.Name == "gpt2" {
		if lora != nil {
			return nil, fmt.Errorf("decoder: LoRA merge unsupported for the gpt2 (Conv1D/fused-QKV) layout")
		}
		return buildGPT2Weights(cfg, arch, st, quant, embedInt4, needCanonical, skipRow4) // Conv1D layout + fused QKV need a dedicated path
	}
	if arch.granite != nil {
		if lora != nil {
			return nil, fmt.Errorf("decoder: LoRA merge unsupported for the granitemoehybrid (Mamba-2 + fused-MoE) layout")
		}
		return buildGraniteWeights(cfg, arch, st, quant, embedInt4, needCanonical, skipRow4, sink, id, abort) // per-layer mamba/attention + fused experts
	}
	if arch.nemotron != nil {
		if lora != nil {
			return nil, fmt.Errorf("decoder: LoRA merge unsupported for the nemotron_h (single-op-block) layout")
		}
		return buildNemotronWeights(cfg, arch, st, quant, embedInt4, needCanonical, skipRow4, sink, id, abort) // per-layer mamba | attention | mlp
	}
	if arch.Name == "phi3" {
		if lora != nil {
			return nil, fmt.Errorf("decoder: LoRA merge unsupported for the phi3 (fused qkv/gate_up) layout")
		}
		return buildPhi3Weights(cfg, arch, st, quant, embedInt4, needCanonical, skipRow4, sink, id, abort) // split fused qkv_proj + gate_up_proj → generic forward
	}
	if arch.Name == "glm_ocr" {
		if lora != nil {
			return nil, fmt.Errorf("decoder: LoRA merge unsupported for the glm_ocr (fused gate_up) layout")
		}
		return buildGlmOcrWeights(cfg, arch, st, quant, embedInt4, needCanonical, skipRow4, sink, id, abort) // split fused gate_up_proj; the MTP layer is never requested
	}
	if arch.Name == "internlm2" {
		return buildInternLM2Weights(cfg, arch, st, quant, embedInt4, needCanonical, skipRow4) // renamed tensors + GROUPED fused wqkv
	}
	if arch.Name == "spark2_5" {
		if lora != nil {
			return nil, fmt.Errorf("decoder: LoRA merge unsupported for the spark2_5 (fused qkv) layout")
		}
		return buildSpark25Weights(cfg, arch, st, quant, embedInt4, needCanonical, skipRow4, sink, id, abort) // split fused q_k_v_proj → generic forward
	}
	if arch.llama4 != nil {
		if lora != nil {
			return nil, fmt.Errorf("decoder: LoRA merge unsupported for the llama4_text (iRoPE + fused-expert) layout")
		}
		return buildLlama4Weights(cfg, arch, st, quant, embedInt4, needCanonical, skipRow4, sink, id, abort) // per-layer dense/MoE + transposed fused experts
	}
	if arch.gptoss != nil {
		// gpt-oss safetensors: MXFP4 experts as paired U8 *_blocks/*_scales tensors with
		// an INTERLEAVED gate_up, per-expert biases, and per-head attention sinks. The
		// GGUF path reads the same family through a different layout entirely (llama.cpp's
		// converter separates gate/up and re-packs the nibbles), so it gets its own loader
		// rather than a shared one with branches.
		return buildGptOssWeights(cfg, arch, st, quant, embedInt4, needCanonical, skipRow4)
	}
	// LoRA merge-at-load validation is deferred until tn is defined (below) so it validates the same prefixed names merge
	// looks up.
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
	// Qwen3-ASR (and Qwen3-Omni) keep the whole text decoder under thinker.*: thinker.model.* and thinker.lm_head.weight, beside thinker.audio_tower.* (the audio encoder is audio/, not this).
	if have["thinker.model.embed_tokens.weight"] {
		topPrefix = "thinker."
	}
	// A decoder-as-embedder checkpoint (Qwen3-Embedding, and the same shape for embeddinggemma)
	// ships the BASE model — Qwen3Model, no LM head — whose tensors carry NO "model." prefix at
	// all: embed_tokens.weight, norm.weight, layers.N.*. (The config still says
	// architectures: ["Qwen3ForCausalLM"], so the naming is the only tell.) Detect it from the
	// index and strip the prefix off the canonical names. tie_word_embeddings is true on these, so
	// the absent lm_head is expected — the embedding doubles as the head, which an embedder never
	// runs anyway.
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
	fusedExperts := arch.MoE != nil && arch.FirstKDense < arch.NumLayers && have[tn(arch.FirstKDense, "mlp.experts.gate_up_proj")]
	switchMLPExperts := arch.MoE != nil && arch.FirstKDense < arch.NumLayers && have[tn(arch.FirstKDense, "mlp.switch_mlp.gate_proj.weight")]

	// LoRA merge-at-load validation, deferred to here so it sees tn: the same prefixed name (language_model.* or
	// model.language_model.* on VL checkpoints, or model.-stripped) that loadProj looks the delta up by. It closes two
	// silent-no-op classes: a VL-prefixed base that validated clean against bare names and then merged nothing, and mla
	// attention, which loads outside loadProj, so merge never touches its deltas (rejected rather than half-merged).
	//
	// qwen35 loads its attention and DeltaNet projections through loadQwen35Attn, which merges. Its tensor names live in
	// that function across four layouts (plain in_proj_qkv, qwen3_next's fused in_proj_qkvz, Olmo Hybrid's separate q/k/v,
	// Olmo's plain full attention), so a list here would be a second copy that could drift. It skips this name check and is
	// covered by checkAllMerged in loadWeights, which compares the adapter with what the load merged.
	if lora != nil {
		if arch.mla != nil {
			return nil, fmt.Errorf("decoder: LoRA merge unsupported for %s (attention projections load outside the generic merge path)", arch.Name)
		}
		if arch.qwen35 == nil {
			if err := lora.validateTargets(cfg.NumLayers, s, tn); err != nil {
				return nil, err
			}
		}
	}

	// Pre-quantized safetensors checkpoints (GPTQ/AWQ/FP8/NVFP4/MLX);
	// resolve the params once. nil ⇒ a normal f32/bf16 checkpoint.
	qc, err := parseCheckpointQuantConfig(cfg)
	if err != nil {
		return nil, err
	}

	// loadMatQ loads a matmul weight and quantizes it immediately (the streaming-quant memory win), for tensors that are not
	// LoRA targets (MoE experts, router); the LoRA-mergeable projections go through loadProj. With skipRow4, experts, router
	// and everything loadQwen35Attn and loadBailingKDA route through this (as mkQ) skip the arm64 row4 side-copy.
	loadMatQ := func(name string, rows, cols int) (linalg.WeightMat, error) {
		m, merr := loadMat(st, name, rows, cols)
		if merr == nil {
			m = quantizeLayerWM(m, quant, needCanonical, skipRow4)
		}
		return m, merr
	}
	// loadProj loads a per-layer attention/MLP projection to f32 (a GPTQ/AWQ/fp8/nvfp4/mlx reconstruction when the checkpoint is
	// pre-quantized, else a plain load), merges any LoRA delta into it, then quantizes to the requested resident format,
	// freeing the f32 before the next tensor. The LoRA merge must happen here, on the f32, before quantization. batched marks
	// a projection reached through the batched W4A8 dispatch (attention Q/K/V, MLP gate/up): quantizeBatchedProjWM instead of
	// quantizeWM, so it may build repacked-only int4 when this load's backend allows it. See that function for the safety
	// argument and why o_proj/down_proj are not marked this way.
	loadProj := func(name string, out, in int, batched bool) (linalg.WeightMat, error) {
		data, fresh, derr := reconstructSafetensorsMat2D(st, qc, name, out, in)
		if derr != nil {
			return linalg.WeightMat{}, derr
		}
		// TensorF32 may alias the read-only mmap for F32 tensors; the in-place
		// merge needs a writable copy (only for the targeted tensors).
		if !fresh && lora.has(name) {
			data = slices.Clone(data)
		}
		if derr = lora.merge(name, data, out, in); derr != nil {
			return linalg.WeightMat{}, derr
		}
		m := linalg.WrapF32(data, out, in)
		if batched {
			m = quantizeBatchedProjWM(m, quant, needCanonical, skipRow4)
		} else {
			// o_proj, down_proj and router: everything loadProj reaches that is not a batched Q/K/V/gate/up tensor.
			m = quantizeLayerWM(m, quant, needCanonical, skipRow4)
		}
		return m, nil
	}

	// loadF32Merged reads an f32 tensor the way st.TensorF32 does and merges its LoRA delta, if it has one, into a writable
	// copy (TensorF32 may alias the read-only mmap). loadMatMerged is loadMatQ with that merge before quantizing. Both equal
	// loadMatQ / TensorF32 for a tensor with no delta, so a load without an adapter is unchanged. They serve the projections
	// loaded outside loadProj (qwen35's).
	loadF32Merged := func(name string, shape ...int) ([]float32, error) {
		data, derr := st.TensorF32(name, shape...)
		if derr != nil || !lora.has(name) {
			return data, derr
		}
		if len(shape) != 2 {
			return nil, fmt.Errorf("decoder(lora): %q is rank %d; only a rank-2 weight can take a LoRA delta", name, len(shape))
		}
		data = slices.Clone(data)
		if derr = lora.merge(name, data, shape[0], shape[1]); derr != nil {
			return nil, derr
		}
		return data, nil
	}
	loadMatMerged := func(name string, rows, cols int) (linalg.WeightMat, error) {
		if !lora.has(name) {
			return loadMatQ(name, rows, cols)
		}
		data, derr := loadF32Merged(name, rows, cols)
		if derr != nil {
			return linalg.WeightMat{}, derr
		}
		m := linalg.WrapF32(data, rows, cols)
		return quantizeLayerWM(m, quant, needCanonical, skipRow4), nil
	}

	// Input embedding + final norm. The embedding is the (tied or untied) LM
	// head, so it is logit-critical — quantize it with the embedding policy
	// (int8 even in int4 mode), not the projection mode.
	embedTable := func(name string, rows int) (linalg.WeightMat, error) {
		return loadEmbedTable(st, name, rows, hd, quant.embeddingWith(embedInt4), needCanonical, sink)
	}
	if w.Embed, err = embedTable(mp(s.Embed), cfg.VocabSize); err != nil {
		return nil, err
	}
	if w.FinalNorm, err = st.TensorF32(mp(s.FinalNorm), hd); err != nil {
		return nil, err
	}
	// LM head: a separate tensor when the family or checkpoint is untied, else the tied embedding serves as the head.
	// Determined by tensor presence, so a checkpoint that ties despite its family default still loads. The head sits under
	// the checkpoint's top-level prefix like every other tensor (Voxtral: language_model.lm_head.weight, untied); looked up
	// bare it is never found and the embedding silently becomes the head.
	arch.TiedLMHead = true
	if s.LMHead != "" {
		if head, herr := embedTable(topPrefix+s.LMHead, cfg.VocabSize); herr == nil {
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

	b := &stBuild{cfg: cfg, arch: arch, s: s, st: st, quant: quant, needCanonical: needCanonical, skipRow4: skipRow4, sink: sink, abort: abort, hd: hd, headDim: headDim, qDim: qDim, kvDim: kvDim, w: w, have: have, tn: tn, fusedExperts: fusedExperts, switchMLPExperts: switchMLPExperts, loadMatQ: loadMatQ, loadProj: loadProj, loadF32Merged: loadF32Merged, loadMatMerged: loadMatMerged, optNorm: optNorm, mp: mp, id: id}
	return b.finish()
}

// loadMat loads + shape-validates a [rows, cols] matrix (via the aikit
// embed.SafetensorsFile.TensorF32 typed read — F32/BF16/F16 dispatch + shape
// check) and wraps it as a (f32) linalg.WeightMat ready for matmul/quantization.
func loadMat(st *embed.SafetensorsFile, name string, rows, cols int) (linalg.WeightMat, error) {
	data, _, err := reconstructSafetensorsMat2D(st, nil, name, rows, cols)
	if err != nil {
		return linalg.WeightMat{}, err
	}
	return linalg.WrapF32(data, rows, cols), nil
}

// tensorName returns the HF safetensors key for a per-layer tensor, kept in one place so a schema change touches one
// function.
func tensorName(layer int, suffix string) string {
	return fmt.Sprintf("model.layers.%d.%s", layer, suffix)
}
