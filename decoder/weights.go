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
)

// LayerWeights bundles one decoder block's tensors. Matrices follow
// PyTorch's [out, in] row-major layout so MatmulBT is A·Bᵀ with no
// transpose copy, matching encoder.LayerWeights.
//
// Gemma 3 specifics reflected here: separate gate/up projections (GeGLU),
// pre- AND post-norms on both the attention and MLP sub-blocks, and
// optional QK-norm weights.
type LayerWeights struct {
	// Attention projections (matmul'd → weightMat, quantizable).
	QProj weightMat // [NumHeads*HeadDim, HiddenDim]
	KProj weightMat // [NumKVHeads*HeadDim, HiddenDim]
	VProj weightMat // [NumKVHeads*HeadDim, HiddenDim]
	OProj weightMat // [HiddenDim, NumHeads*HeadDim]
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
	GateProj       weightMat // [IntermediateDim, HiddenDim] (unused for non-gated MLP)
	UpProj         weightMat // [IntermediateDim, HiddenDim]
	UpBias         []float32 // [IntermediateDim] (GPT-2 c_fc bias)
	DownProj       weightMat // [HiddenDim, IntermediateDim]
	DownBias       []float32 // [HiddenDim] (GPT-2 c_proj bias)
	PreMLPNorm     []float32 // [HiddenDim]
	PreMLPNormBias []float32 // [HiddenDim] LayerNorm bias (GPT-2 ln_2)
	PostMLPNorm    []float32 // [HiddenDim]

	// Mixture-of-experts FFN (Mixtral; set only when arch.MoE != nil). Router
	// scores experts; each expert is a gated (SwiGLU) MLP. The dense GateProj/
	// UpProj/DownProj above are unused in that case.
	Router  weightMat       // [NumExperts, HiddenDim] router/gate logits
	Experts []expertWeights // [NumExperts] gated MLPs

	// Shared expert (Qwen-MoE / Qwen2-MoE; set when arch.MoE.SharedIntermediateDim
	// > 0). An always-on gated MLP scaled by sigmoid(SharedGate·h).
	SharedExpert expertWeights // gated SwiGLU MLP at SharedIntermediateDim
	SharedGate   weightMat     // [1, HiddenDim] → sigmoid scalar gate

	// Gemma 4 per-layer extras (zero for every other family). The PLE branch runs
	// after the MLP residual: gate→gelu→×per-layer-embedding→proj→norm→+residual.
	PLEGate     weightMat // [HiddenSizePerLayerInput, HiddenDim] (GGUF inp_gate)
	PLEProj     weightMat // [HiddenDim, HiddenSizePerLayerInput] (GGUF proj)
	PostPLENorm []float32 // [HiddenDim] post_per_layer_input_norm (GGUF post_norm)
	LayerScalar float32   // per-layer output scalar (GGUF layer_output_scale); 1 if absent
	KVShared    bool      // carries no k/v and reuses an earlier layer's KV (Gemma 4 E-models)
}

// expertWeights is one MoE expert: a gated (SwiGLU) MLP with no biases.
// Mixtral names these w1=gate, w3=up, w2=down.
type expertWeights struct {
	Gate weightMat // [IntermediateDim, HiddenDim] (w1)
	Up   weightMat // [IntermediateDim, HiddenDim] (w3)
	Down weightMat // [HiddenDim, IntermediateDim] (w2)
}

// Weights is the immutable per-checkpoint bundle. Embeddings are TIED:
// Embed doubles as the LM head (logits = h · Embedᵀ), so there is no
// separate output projection tensor.
type Weights struct {
	Cfg           Config
	arch          *Architecture // resolved descriptor the forward pass reads
	Embed         weightMat     // [VocabSize, HiddenDim] — input embedding (AND tied LM head when LMHead unset)
	LMHead        weightMat     // [VocabSize, HiddenDim] — separate output head (untied families); zero value when tied
	PosEmbed      weightMat     // [MaxPositions, HiddenDim] — learned position embedding (GPT-2); zero value otherwise
	FinalNorm     []float32     // [HiddenDim] final norm before the LM head
	FinalNormBias []float32     // [HiddenDim] final LayerNorm bias (GPT-2 ln_f)
	Layers        []LayerWeights

	// Gemma 4 PLE inputs (zero for every other family). The per-layer input is
	// (token_identity + context_aware)/√2: token_identity from PerLayerTokenEmbed,
	// context_aware from PerLayerModelProj normed by PerLayerProjNorm.
	PerLayerTokenEmbed weightMat // [VocabSize, NumLayers*HiddenSizePerLayerInput] (per_layer_token_embd)
	PerLayerModelProj  weightMat // [NumLayers*HiddenSizePerLayerInput, HiddenDim] (per_layer_model_proj)
	PerLayerProjNorm   []float32 // [HiddenSizePerLayerInput] (per_layer_proj_norm)

	st      *embed.SafetensorsFile // retained so alias-backed slices stay valid
	backing []byte                 // serialized-weights blob aliased by q8/q4 arrays (LoadSerializedWeights); keeps it reachable
}

// matmulWeights returns every quantizable matrix in the bundle (the projections,
// the embedding, and the untied head if present); norms stay f32.
func (w *Weights) matmulWeights() []*weightMat {
	ms := []*weightMat{&w.Embed}
	if w.LMHead.rows > 0 {
		ms = append(ms, &w.LMHead)
	}
	for i := range w.Layers {
		l := &w.Layers[i]
		ms = append(ms, &l.QProj, &l.KProj, &l.VProj, &l.OProj, &l.GateProj, &l.UpProj, &l.DownProj)
		if l.Router.rows > 0 {
			ms = append(ms, &l.Router)
		}
		for e := range l.Experts {
			ex := &l.Experts[e]
			ms = append(ms, &ex.Gate, &ex.Up, &ex.Down)
		}
		if l.SharedExpert.Gate.rows > 0 {
			ms = append(ms, &l.SharedExpert.Gate, &l.SharedExpert.Up, &l.SharedExpert.Down, &l.SharedGate)
		}
		if l.PLEGate.rows > 0 { // Gemma 4 PLE branch
			ms = append(ms, &l.PLEGate, &l.PLEProj)
		}
	}
	if w.PerLayerTokenEmbed.rows > 0 { // Gemma 4 model-level PLE inputs
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
// TODO(M8): bf16-resident matmul tiling (gemma-decoder-plan §2/§8) to
// drop the widen-on-load 2× memory cost for the 1B+ checkpoints.
//
// Use LoadWeightsFromFS for fs.FS-backed (MapFS, embed.FS) paths — that
// route stays heap-backed because fs.FS doesn't expose a file descriptor.
func LoadWeights(dir string) (*Weights, error) { return loadWeights(dir, quantNone) }

// parallelLayers runs fn over the n layer indices across a worker pool, so the
// per-tensor dequant + re-quant (independent per layer — distinct weightMat
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
func loadWeights(dir string, quant quantMode) (*Weights, error) {
	if strings.HasSuffix(dir, ".gguf") {
		return loadGGUFWeights(dir, quant) // quantized llama.cpp checkpoint (G7)
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
	return buildWeightsFromSafetensors(cfg, arch, schema, st, quant)
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
	return buildWeightsFromSafetensors(cfg, arch, schema, st, quant)
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
func buildWeightsFromSafetensors(cfg *Config, arch *Architecture, s *tensorSchema, st *embed.SafetensorsFile, quant quantMode) (*Weights, error) {
	if arch.Name == "gpt2" {
		return buildGPT2Weights(cfg, arch, st, quant) // Conv1D layout + fused QKV need a dedicated path
	}
	hd := cfg.HiddenDim
	headDim := arch.HeadDim           // resolved (Llama configs may omit head_dim; arch derives it)
	qDim := cfg.NumHeads * headDim    // query projection rows
	kvDim := cfg.NumKVHeads * headDim // key/value projection rows (narrower under GQA)

	w := &Weights{Cfg: *cfg, arch: arch, st: st, Layers: make([]LayerWeights, cfg.NumLayers)}

	// GPTQ/AWQ checkpoints ship their projections as packed int4 (qweight/…);
	// resolve the params once. nil ⇒ a normal f32/bf16 checkpoint.
	qc, err := parseQuantConfig(cfg.QuantizationConfig)
	if err != nil {
		return nil, err
	}

	// loadMatQ loads a matmul weight and, when quant is set, quantizes it to
	// per-row int8 immediately — freeing the f32 before the next tensor loads
	// (the streaming-quant memory win; see loadWeights). Norms still use loadF32
	// and stay f32.
	loadMatQ := func(name string, rows, cols int) (weightMat, error) {
		m, merr := loadMat(st, name, rows, cols)
		if merr == nil {
			m.quantize(quant)
		}
		return m, merr
	}
	// loadProj loads a (per-layer) attention/MLP projection: a GPTQ/AWQ
	// reconstruction when the checkpoint is pre-quantized, else a plain weight
	// load. Either way the result is then streamed through the requested resident
	// quant. (Embeddings/norms/LM head are not quantized — they keep loadMat/loadF32.)
	loadProj := func(name string, out, in int) (weightMat, error) {
		if qc == nil {
			return loadMatQ(name, out, in)
		}
		base := strings.TrimSuffix(name, ".weight")
		var data []float32
		var derr error
		if qc.method == "awq" {
			data, derr = awqReconstruct(st, base, in, out)
		} else {
			data, derr = gptqReconstruct(st, base, in, out)
		}
		if derr != nil {
			return weightMat{}, derr
		}
		m := newWeightMat(data, out, in)
		m.quantize(quant)
		return m, nil
	}

	// Input embedding + final norm. The embedding is the (tied or untied) LM
	// head, so it is logit-critical — quantize it with the embedding policy
	// (int8 even in int4 mode), not the projection mode.
	if w.Embed, err = loadMat(st, s.Embed, cfg.VocabSize, hd); err != nil {
		return nil, err
	}
	w.Embed.quantize(quant.embedding())
	if w.FinalNorm, err = loadF32(st, s.FinalNorm, []int{hd}); err != nil {
		return nil, err
	}
	// LM head: separate tensor when the family/checkpoint is untied, else the
	// tied embedding serves as the head. Determined by tensor presence so a
	// checkpoint that ties despite its family default still loads.
	arch.TiedLMHead = true
	if s.LMHead != "" {
		if head, herr := loadMat(st, s.LMHead, cfg.VocabSize, hd); herr == nil {
			head.quantize(quant.embedding())
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
		return loadF32(st, tensorName(i, suffix), []int{hd})
	}

	// Layers load in parallel — each is independent over the read-only mmap, and
	// the per-tensor dequant/re-quant is the cost.
	loadLayer := func(i int) error {
		l := &w.Layers[i]
		var err error
		// Attention projections ([out, in] row-major).
		if l.QProj, err = loadProj(tensorName(i, s.QProj), qDim, hd); err != nil {
			return err
		}
		if l.KProj, err = loadProj(tensorName(i, s.KProj), kvDim, hd); err != nil {
			return err
		}
		if l.VProj, err = loadProj(tensorName(i, s.VProj), kvDim, hd); err != nil {
			return err
		}
		if l.OProj, err = loadProj(tensorName(i, s.OProj), hd, qDim); err != nil {
			return err
		}
		// Projection bias (Qwen2 q/k/v; o_proj stays biasless). Absent → empty suffix.
		if s.QBias != "" {
			if l.QBias, err = loadF32(st, tensorName(i, s.QBias), []int{qDim}); err != nil {
				return err
			}
			if l.KBias, err = loadF32(st, tensorName(i, s.KBias), []int{kvDim}); err != nil {
				return err
			}
			if l.VBias, err = loadF32(st, tensorName(i, s.VBias), []int{kvDim}); err != nil {
				return err
			}
		}
		// QK-norm (Gemma 3, Qwen3): RMSNorm over head_dim. Absent → empty suffix.
		if s.QNorm != "" {
			if l.QNorm, err = loadF32(st, tensorName(i, s.QNorm), []int{headDim}); err != nil {
				return err
			}
			if l.KNorm, err = loadF32(st, tensorName(i, s.KNorm), []int{headDim}); err != nil {
				return err
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
		// templates carry a %d for the expert index.
		if arch.MoE != nil {
			if l.Router, err = loadMat(st, tensorName(i, s.Router), arch.MoE.NumExperts, hd); err != nil {
				return err
			}
			expInter := arch.MoE.IntermediateDim // expert FFN width (Mellum: moe_intermediate_size)
			l.Experts = make([]expertWeights, arch.MoE.NumExperts)
			for e := 0; e < arch.MoE.NumExperts; e++ {
				ex := &l.Experts[e]
				if ex.Gate, err = loadMatQ(tensorName(i, fmt.Sprintf(s.ExpertGate, e)), expInter, hd); err != nil {
					return err
				}
				if ex.Up, err = loadMatQ(tensorName(i, fmt.Sprintf(s.ExpertUp, e)), expInter, hd); err != nil {
					return err
				}
				if ex.Down, err = loadMatQ(tensorName(i, fmt.Sprintf(s.ExpertDown, e)), hd, expInter); err != nil {
					return err
				}
			}
			// Shared expert (Qwen2-MoE): an always-on gated MLP + a scalar sigmoid
			// gate. s.SharedGate/Up/Down are the expert's gate/up/down_proj;
			// s.SharedExpertGate is the [1, hidden] sigmoid gate.
			if arch.MoE.SharedIntermediateDim > 0 {
				sInter := arch.MoE.SharedIntermediateDim
				if l.SharedExpert.Gate, err = loadMatQ(tensorName(i, s.SharedGate), sInter, hd); err != nil {
					return err
				}
				if l.SharedExpert.Up, err = loadMatQ(tensorName(i, s.SharedUp), sInter, hd); err != nil {
					return err
				}
				if l.SharedExpert.Down, err = loadMatQ(tensorName(i, s.SharedDown), hd, sInter); err != nil {
					return err
				}
				if l.SharedGate, err = loadMat(st, tensorName(i, s.SharedExpertGate), 1, hd); err != nil {
					return err
				}
			}
			return nil
		}
		// Gated MLP (GeGLU / SwiGLU — same weights, activation differs).
		if l.GateProj, err = loadProj(tensorName(i, s.GateProj), cfg.IntermediateDim, hd); err != nil {
			return err
		}
		if l.UpProj, err = loadProj(tensorName(i, s.UpProj), cfg.IntermediateDim, hd); err != nil {
			return err
		}
		if l.DownProj, err = loadProj(tensorName(i, s.DownProj), hd, cfg.IntermediateDim); err != nil {
			return err
		}
		return nil
	}
	if err := parallelLayers(cfg.NumLayers, loadLayer); err != nil {
		return nil, err
	}
	return w, nil
}

// loadF32 fetches a tensor, shape-validates it against want, and decodes it
// to []float32 dispatching on DType: F32 is a zero-copy view; BF16/F16 are
// widened (allocating). One clear "shape %v != want %v" error otherwise.
// Mirrors encoder.loadF32, extended with the bf16/f16 dispatch.
func loadF32(st *embed.SafetensorsFile, name string, want []int) ([]float32, error) {
	t, err := st.Tensor(name)
	if err != nil {
		return nil, fmt.Errorf("decoder: tensor %q: %w", name, err)
	}
	if !shapeEqual(t.Shape, want) {
		return nil, fmt.Errorf("decoder: tensor %q shape %v != want %v", name, t.Shape, want)
	}
	var data []float32
	switch t.DType {
	case "F32":
		data, err = t.Float32s()
	case "BF16":
		data, err = t.BFloat16sToF32()
	case "F16":
		data, err = t.Float16sToF32()
	default:
		return nil, fmt.Errorf("decoder: tensor %q unsupported dtype %q (want F32/BF16/F16)", name, t.DType)
	}
	if err != nil {
		return nil, fmt.Errorf("decoder: tensor %q decode: %w", name, err)
	}
	return data, nil
}

// loadMat loads + shape-validates a [rows, cols] matrix and wraps it as a
// (f32) weightMat ready for matmul/quantization. Mirrors loadF32 for the
// matmul'd projections; the norms keep loadF32 (they stay f32).
func loadMat(st *embed.SafetensorsFile, name string, rows, cols int) (weightMat, error) {
	data, err := loadF32(st, name, []int{rows, cols})
	if err != nil {
		return weightMat{}, err
	}
	return newWeightMat(data, rows, cols), nil
}

func shapeEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
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

// conv1DTransposed loads a GPT-2 Conv1D weight and returns it transposed to the
// [out, in] row-major layout the rest of the decoder (MatmulBT) expects. GPT-2
// stores these weights as [in, out] (nn.Conv1D, not nn.Linear), so a plain load
// would compute the wrong product.
func conv1DTransposed(st *embed.SafetensorsFile, name string, in, out int) ([]float32, error) {
	src, err := loadF32(st, name, []int{in, out}) // [in, out] row-major
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
	maybeQuant := func(m weightMat) weightMat {
		m.quantize(quant)
		return m
	}

	// Token + learned position embeddings (wte doubles as the tied LM head).
	// wte is the tied head → logit-critical, so quantize with the embedding
	// policy (int8 even in int4 mode); wpe is a positional lookup table added to
	// the embedding, never matmul'd → stays f32.
	if w.Embed, err = loadMat(st, "wte.weight", vocab, hidden); err != nil {
		return nil, err
	}
	w.Embed.quantize(quant.embedding())
	if w.PosEmbed, err = loadMat(st, "wpe.weight", arch.MaxPositions, hidden); err != nil {
		return nil, err
	}
	// Final LayerNorm (weight + bias).
	if w.FinalNorm, err = loadF32(st, "ln_f.weight", []int{hidden}); err != nil {
		return nil, err
	}
	if w.FinalNormBias, err = loadF32(st, "ln_f.bias", []int{hidden}); err != nil {
		return nil, err
	}

	for i := 0; i < arch.NumLayers; i++ {
		l := &w.Layers[i]
		p := fmt.Sprintf("h.%d.", i)

		// ln_1 (pre-attention LayerNorm).
		if l.PreAttnNorm, err = loadF32(st, p+"ln_1.weight", []int{hidden}); err != nil {
			return nil, err
		}
		if l.PreAttnNormBias, err = loadF32(st, p+"ln_1.bias", []int{hidden}); err != nil {
			return nil, err
		}

		// Fused c_attn → q/k/v. Conv1D weight is [hidden, 3*hidden]; transpose to
		// [3*hidden, hidden] then split the rows into thirds.
		qkv, cerr := conv1DTransposed(st, p+"attn.c_attn.weight", hidden, 3*hidden)
		if cerr != nil {
			return nil, cerr
		}
		l.QProj = maybeQuant(newWeightMat(qkv[0:hidden*hidden], hidden, hidden))
		l.KProj = maybeQuant(newWeightMat(qkv[hidden*hidden:2*hidden*hidden], hidden, hidden))
		l.VProj = maybeQuant(newWeightMat(qkv[2*hidden*hidden:3*hidden*hidden], hidden, hidden))
		qkvB, berr := loadF32(st, p+"attn.c_attn.bias", []int{3 * hidden})
		if berr != nil {
			return nil, berr
		}
		l.QBias, l.KBias, l.VBias = qkvB[0:hidden], qkvB[hidden:2*hidden], qkvB[2*hidden:3*hidden]

		// Attention output projection (Conv1D [hidden, hidden]) + bias.
		oData, oerr := conv1DTransposed(st, p+"attn.c_proj.weight", hidden, hidden)
		if oerr != nil {
			return nil, oerr
		}
		l.OProj = maybeQuant(newWeightMat(oData, hidden, hidden))
		if l.OBias, err = loadF32(st, p+"attn.c_proj.bias", []int{hidden}); err != nil {
			return nil, err
		}

		// ln_2 (pre-MLP LayerNorm).
		if l.PreMLPNorm, err = loadF32(st, p+"ln_2.weight", []int{hidden}); err != nil {
			return nil, err
		}
		if l.PreMLPNormBias, err = loadF32(st, p+"ln_2.bias", []int{hidden}); err != nil {
			return nil, err
		}

		// Non-gated MLP: c_fc (up, [hidden, inter]) → gelu → c_proj (down,
		// [inter, hidden]), both Conv1D, both with bias.
		upData, uerr := conv1DTransposed(st, p+"mlp.c_fc.weight", hidden, inter)
		if uerr != nil {
			return nil, uerr
		}
		l.UpProj = maybeQuant(newWeightMat(upData, inter, hidden))
		if l.UpBias, err = loadF32(st, p+"mlp.c_fc.bias", []int{inter}); err != nil {
			return nil, err
		}
		downData, derr := conv1DTransposed(st, p+"mlp.c_proj.weight", inter, hidden)
		if derr != nil {
			return nil, derr
		}
		l.DownProj = maybeQuant(newWeightMat(downData, hidden, inter))
		if l.DownBias, err = loadF32(st, p+"mlp.c_proj.bias", []int{hidden}); err != nil {
			return nil, err
		}
	}
	arch.TiedLMHead = true // GPT-2 has no separate lm_head tensor
	return w, nil
}
