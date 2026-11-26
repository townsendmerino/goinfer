package decoder

import (
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"

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
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/decoder/quant"
)

// repackedOnlyInt4Count reports how many of w's matmulWeights() tensors are int4-resident with no canonical bytes
// (IsInt4() true, Int4()'s ok false): a kind-5 .giw tensor, or an in-RAM repacked-only build that reached a .giw
// round-trip (no writer produces one today). decoder.Load's .giw branch uses it to refuse such a file under a backend
// that needs canonical bytes.
func repackedOnlyInt4Count(w *darch.Weights) int {
	n := 0
	for _, m := range w.MatmulWeights() {
		if !m.IsInt4() {
			continue
		}
		if _, _, _, ok := m.Int4F16(); !ok {
			n++
		}
	}
	return n
}

// LoadWeights reads config.json and the model.safetensors blob (or sharded set) from an on-disk directory in the HF
// snapshot layout. The blob is mmapped, not heap-copied, so a large checkpoint stays in the OS page cache. bf16 and f16
// weights are widened to f32 on load, which roughly doubles resident RAM against keeping them bf16; per-tile widening
// inside matmul would avoid that and is not implemented.
//
// Use LoadWeightsFromFS for fs.FS-backed paths (MapFS, embed.FS); those stay heap-backed because fs.FS exposes no file
// descriptor.
func LoadWeights(dir string) (*darch.Weights, error) {
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
func loadWeights(dir string, o weightLoadOpts) (*darch.Weights, error) {
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
		w.ST = nil
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
func islandsMayAliasSource(w *darch.Weights) bool {
	for i := range w.Layers {
		if sc := w.Layers[i].ShortConv; sc != nil {
			for _, m := range []*linalg.WeightMat{&sc.InProj, &sc.OutProj} {
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
func LoadWeightsFromFS(fsys fs.FS, dir string) (*darch.Weights, error) {
	return loadWeightsFromFS(fsys, dir, quantNone)
}

// loadWeightsFromFS is the quant-aware internal counterpart of loadWeights for
// fs.FS-backed paths (see loadWeights for the streaming-quant rationale).
func loadWeightsFromFS(fsys fs.FS, dir string, quant quantMode) (*darch.Weights, error) {
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
func buildWeightsFromSafetensors(cfg *Config, arch *darch.Architecture, s *archparts.TensorSchema, st *embed.SafetensorsFile, o weightLoadOpts) (*darch.Weights, error) {
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
func finishLayers(w *darch.Weights, arch *darch.Architecture, loadLayer func(i int) error, sink *giwWriter, id string, abort <-chan struct{}) (*darch.Weights, error) {
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
func buildWeightsFromSafetensorsTo(cfg *Config, arch *darch.Architecture, s *archparts.TensorSchema, st *embed.SafetensorsFile, quant quantMode, embedInt4, needCanonical, skipRow4 bool, lora *loraAdapter, sink *giwWriter, id string, abort <-chan struct{}) (*darch.Weights, error) {
	skipRow4 = skipRow4 || !repackForDecode(sink) // every builder below takes it from here
	if sink != nil {
		switch {
		case arch.Name == "gpt2", arch.Name == "internlm2", arch.GPTOSS != nil:
			return nil, errDirNoStream{arch.Name}
		}
	}
	if arch.Name == "gpt2" {
		if lora != nil {
			return nil, fmt.Errorf("decoder: LoRA merge unsupported for the gpt2 (Conv1D/fused-QKV) layout")
		}
		return buildGPT2Weights(cfg, arch, st, quant, embedInt4, needCanonical, skipRow4) // Conv1D layout + fused QKV need a dedicated path
	}
	if arch.Granite != nil {
		if lora != nil {
			return nil, fmt.Errorf("decoder: LoRA merge unsupported for the granitemoehybrid (Mamba-2 + fused-MoE) layout")
		}
		return buildGraniteWeights(cfg, arch, st, quant, embedInt4, needCanonical, skipRow4, sink, id, abort) // per-layer mamba/attention + fused experts
	}
	if arch.Nemotron != nil {
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
	if arch.Llama4 != nil {
		if lora != nil {
			return nil, fmt.Errorf("decoder: LoRA merge unsupported for the llama4_text (iRoPE + fused-expert) layout")
		}
		return buildLlama4Weights(cfg, arch, st, quant, embedInt4, needCanonical, skipRow4, sink, id, abort) // per-layer dense/MoE + transposed fused experts
	}
	if arch.GPTOSS != nil {
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

	w := &darch.Weights{Cfg: *cfg, Arch: arch, ST: st, Schema: s, Layers: make([]darch.LayerWeights, cfg.NumLayers)}

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
		if arch.MLA != nil {
			return nil, fmt.Errorf("decoder: LoRA merge unsupported for %s (attention projections load outside the generic merge path)", arch.Name)
		}
		if arch.Qwen35 == nil {
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
