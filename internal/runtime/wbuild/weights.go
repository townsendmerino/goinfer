package wbuild

import (
	"fmt"
	"slices"
	"strings"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/decoder/config"
	"github.com/townsendmerino/goinfer/decoder/quant"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
)

// loadEmbedTable loads a [rows, cols] logit table (the token embedding or an untied head) at the embedding precision mode:
// whole and then quantized for the resident build, a row at a time when streaming (a 256k x 4096 table is 4.2 GB as f32,
// the largest single allocation of a transcode). Both give the same bytes.
func loadEmbedTable(st *embed.SafetensorsFile, name string, rows, cols int, mode QuantMode, needCanonical bool, sink *StreamWriter, h *Hooks) (linalg.WeightMat, error) {
	if sink == nil {
		m, err := LoadMat(st, name, rows, cols)
		if err != nil {
			return m, err
		}
		return h.QuantizeEmbedWM(m, mode, needCanonical), nil
	}
	if r := quant.ForTensor(st, nil, name); r != nil {
		if rs, ok := r.(quant.RowStreamReconstructor); ok {
			rowInto, err := rs.RowReader2D(st, name, rows, cols)
			if err != nil {
				return linalg.WeightMat{}, err
			}
			return h.StreamQuantizedEmbed(rows, cols, mode, needCanonical, !RepackForDecode(sink), 1, StreamEmbedWorkers(sink), func() RowReader { return rowInto })
		}
		data, err := r.ReconstructMat2D(st, name, rows, cols)
		if err != nil {
			return linalg.WeightMat{}, err
		}
		readRow := func(r int, dst []float32) error {
			copy(dst, data[r*cols:(r+1)*cols])
			return nil
		}
		return h.StreamQuantizedEmbed(rows, cols, mode, needCanonical, !RepackForDecode(sink), 1, StreamEmbedWorkers(sink), func() RowReader { return readRow })
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
	return h.StreamQuantizedEmbed(rows, cols, mode, needCanonical, !RepackForDecode(sink), block, StreamEmbedWorkers(sink), func() RowReader {
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
func finishLayers(w *darch.Weights, arch *darch.Architecture, loadLayer func(i int) error, sink *StreamWriter, id string, abort <-chan struct{}) (*darch.Weights, error) {
	n := len(w.Layers)
	if sink == nil {
		if err := ParallelLayers(n, abort, loadLayer); err != nil {
			return nil, err
		}
		return w, nil
	}
	sink.Arch = arch
	if err := sink.WriteHeadGlobals(w, id); err != nil {
		return nil, err
	}
	// The logit tables and Gemma 4's per-layer token table are written; nothing after the head reads them.
	w.Embed, w.LMHead, w.PerLayerTokenEmbed = linalg.WeightMat{}, linalg.WeightMat{}, linalg.WeightMat{}
	if err := StreamLayers(w, arch, sink, loadLayer); err != nil {
		return nil, err
	}
	return w, nil
}

// BuildWeightsFromSafetensorsTo is buildWeightsFromSafetensors with an optional sink. With sink nil it is the resident
// build. With a sink (StreamTranscodeDir) the generic builder writes the bundle head once the globals exist, then loads,
// writes and frees one layer at a time, and loads the embedding (and an untied head) a row at a time, so peak memory is
// about the globals plus one layer rather than the whole model. The dedicated builders stream their layers the same way
// (their embedding still loads whole); a family whose builder does not stream returns errDirNoStream.
func BuildWeightsFromSafetensorsTo(cfg *config.Config, arch *darch.Architecture, s *archparts.TensorSchema, st *embed.SafetensorsFile, quant QuantMode, embedInt4, needCanonical, skipRow4 bool, lora LoRAMerger, sink *StreamWriter, id string, abort <-chan struct{}, h Hooks) (*darch.Weights, error) {
	skipRow4 = skipRow4 || !RepackForDecode(sink) // every builder below takes it from here
	if sink != nil {
		switch {
		case arch.Name == "gpt2", arch.Name == "internlm2", arch.GPTOSS != nil:
			return nil, ErrDirNoStream{Family: arch.Name}
		}
	}
	if arch.Name == "gpt2" {
		if lora != nil {
			return nil, fmt.Errorf("decoder: LoRA merge unsupported for the gpt2 (Conv1D/fused-QKV) layout")
		}
		return buildGPT2Weights(cfg, arch, st, quant, embedInt4, needCanonical, skipRow4, &h) // Conv1D layout + fused QKV need a dedicated path
	}
	if arch.Granite != nil {
		if lora != nil {
			return nil, fmt.Errorf("decoder: LoRA merge unsupported for the granitemoehybrid (Mamba-2 + fused-MoE) layout")
		}
		return buildGraniteWeights(cfg, arch, st, quant, embedInt4, needCanonical, skipRow4, sink, id, abort, &h) // per-layer mamba/attention + fused experts
	}
	if arch.Nemotron != nil {
		if lora != nil {
			return nil, fmt.Errorf("decoder: LoRA merge unsupported for the nemotron_h (single-op-block) layout")
		}
		return buildNemotronWeights(cfg, arch, st, quant, embedInt4, needCanonical, skipRow4, sink, id, abort, &h) // per-layer mamba | attention | mlp
	}
	if arch.Name == "phi3" {
		if lora != nil {
			return nil, fmt.Errorf("decoder: LoRA merge unsupported for the phi3 (fused qkv/gate_up) layout")
		}
		return buildPhi3Weights(cfg, arch, st, quant, embedInt4, needCanonical, skipRow4, sink, id, abort, &h) // split fused qkv_proj + gate_up_proj → generic forward
	}
	if arch.Name == "glm_ocr" {
		if lora != nil {
			return nil, fmt.Errorf("decoder: LoRA merge unsupported for the glm_ocr (fused gate_up) layout")
		}
		return buildGlmOcrWeights(cfg, arch, st, quant, embedInt4, needCanonical, skipRow4, sink, id, abort, &h) // split fused gate_up_proj; the MTP layer is never requested
	}
	if arch.Name == "internlm2" {
		return buildInternLM2Weights(cfg, arch, st, quant, embedInt4, needCanonical, skipRow4, &h) // renamed tensors + GROUPED fused wqkv
	}
	if arch.Name == "spark2_5" {
		if lora != nil {
			return nil, fmt.Errorf("decoder: LoRA merge unsupported for the spark2_5 (fused qkv) layout")
		}
		return buildSpark25Weights(cfg, arch, st, quant, embedInt4, needCanonical, skipRow4, sink, id, abort, &h) // split fused q_k_v_proj → generic forward
	}
	if arch.Llama4 != nil {
		if lora != nil {
			return nil, fmt.Errorf("decoder: LoRA merge unsupported for the llama4_text (iRoPE + fused-expert) layout")
		}
		return buildLlama4Weights(cfg, arch, st, quant, embedInt4, needCanonical, skipRow4, sink, id, abort, &h) // per-layer dense/MoE + transposed fused experts
	}
	if arch.GPTOSS != nil {
		// gpt-oss safetensors: MXFP4 experts as paired U8 *_blocks/*_scales tensors with
		// an INTERLEAVED gate_up, per-expert biases, and per-head attention sinks. The
		// GGUF path reads the same family through a different layout entirely (llama.cpp's
		// converter separates gate/up and re-packs the nibbles), so it gets its own loader
		// rather than a shared one with branches.
		return buildGptOssWeights(cfg, arch, st, quant, embedInt4, needCanonical, skipRow4, &h)
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
			if err := lora.ValidateTargets(cfg.NumLayers, s, tn); err != nil {
				return nil, err
			}
		}
	}

	// Pre-quantized safetensors checkpoints (GPTQ/AWQ/FP8/NVFP4/MLX);
	// resolve the params once. nil ⇒ a normal f32/bf16 checkpoint.
	qc, err := ParseCheckpointQuantConfig(cfg)
	if err != nil {
		return nil, err
	}

	// loadMatQ loads a matmul weight and quantizes it immediately (the streaming-quant memory win), for tensors that are not
	// LoRA targets (MoE experts, router); the LoRA-mergeable projections go through loadProj. With skipRow4, experts, router
	// and everything loadQwen35Attn and loadBailingKDA route through this (as mkQ) skip the arm64 row4 side-copy.
	loadMatQ := func(name string, rows, cols int) (linalg.WeightMat, error) {
		m, merr := LoadMat(st, name, rows, cols)
		if merr == nil {
			m = h.QuantizeLayerWM(m, quant, needCanonical, skipRow4)
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
		if data, derr = mergeLoRAMat(lora, name, data, fresh, out, in); derr != nil {
			return linalg.WeightMat{}, derr
		}
		m := linalg.WrapF32(data, out, in)
		if batched {
			m = h.QuantizeBatchedProjWM(m, quant, needCanonical, skipRow4)
		} else {
			// o_proj, down_proj and router: everything loadProj reaches that is not a batched Q/K/V/gate/up tensor.
			m = h.QuantizeLayerWM(m, quant, needCanonical, skipRow4)
		}
		return m, nil
	}

	// loadF32Merged reads an f32 tensor the way st.TensorF32 does and merges its LoRA delta, if it has one, into a writable
	// copy (TensorF32 may alias the read-only mmap). loadMatMerged is loadMatQ with that merge before quantizing. Both equal
	// loadMatQ / TensorF32 for a tensor with no delta, so a load without an adapter is unchanged. They serve the projections
	// loaded outside loadProj (qwen35's).
	loadF32Merged := func(name string, shape ...int) ([]float32, error) {
		data, derr := st.TensorF32(name, shape...)
		if derr != nil || lora == nil || !lora.Has(name) {
			return data, derr
		}
		if len(shape) != 2 {
			return nil, fmt.Errorf("decoder(lora): %q is rank %d; only a rank-2 weight can take a LoRA delta", name, len(shape))
		}
		data = slices.Clone(data)
		if derr = lora.Merge(name, data, shape[0], shape[1]); derr != nil {
			return nil, derr
		}
		return data, nil
	}
	loadMatMerged := func(name string, rows, cols int) (linalg.WeightMat, error) {
		if lora == nil || !lora.Has(name) {
			return loadMatQ(name, rows, cols)
		}
		data, derr := loadF32Merged(name, rows, cols)
		if derr != nil {
			return linalg.WeightMat{}, derr
		}
		m := linalg.WrapF32(data, rows, cols)
		return h.QuantizeLayerWM(m, quant, needCanonical, skipRow4), nil
	}

	// Input embedding + final norm. The embedding is the (tied or untied) LM
	// head, so it is logit-critical — quantize it with the embedding policy
	// (int8 even in int4 mode), not the projection mode.
	embedTable := func(name string, rows int) (linalg.WeightMat, error) {
		return loadEmbedTable(st, name, rows, hd, quant.EmbeddingWith(embedInt4), needCanonical, sink, &h)
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

	b := &stBuild{cfg: cfg, arch: arch, s: s, st: st, quant: quant, needCanonical: needCanonical, skipRow4: skipRow4, sink: sink, abort: abort, h: h, hd: hd, headDim: headDim, qDim: qDim, kvDim: kvDim, w: w, have: have, tn: tn, fusedExperts: fusedExperts, switchMLPExperts: switchMLPExperts, loadMatQ: loadMatQ, loadProj: loadProj, loadF32Merged: loadF32Merged, loadMatMerged: loadMatMerged, optNorm: optNorm, mp: mp, id: id}
	return b.finish()
}

// LoadMat loads + shape-validates a [rows, cols] matrix (via the aikit
// embed.SafetensorsFile.TensorF32 typed read — F32/BF16/F16 dispatch + shape
// check) and wraps it as a (f32) linalg.WeightMat ready for matmul/quantization.
func LoadMat(st *embed.SafetensorsFile, name string, rows, cols int) (linalg.WeightMat, error) {
	data, _, err := reconstructSafetensorsMat2D(st, nil, name, rows, cols)
	if err != nil {
		return linalg.WeightMat{}, err
	}
	return linalg.WrapF32(data, rows, cols), nil
}

// TensorName returns the HF safetensors key for a per-layer tensor, kept in one place so a schema change touches one
// function.
func TensorName(layer int, suffix string) string {
	return fmt.Sprintf("model.layers.%d.%s", layer, suffix)
}

// mergeLoRAMat merges a LoRA delta into data, cloning first when data may alias a read-only mmap.
func mergeLoRAMat(lora LoRAMerger, name string, data []float32, fresh bool, out, in int) ([]float32, error) {
	if lora == nil || !lora.Has(name) {
		return data, nil
	}
	if !fresh {
		data = slices.Clone(data)
	}
	if err := lora.Merge(name, data, out, in); err != nil {
		return nil, err
	}
	return data, nil
}
