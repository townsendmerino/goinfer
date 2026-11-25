package decoder

import (
	"fmt"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// stBuild is the state buildWeightsFromSafetensorsTo hands the per-layer loader: the load's inputs, the weight bundle it fills, the
// tensor-name resolution (prefixes, have, tn) and the readers that merge a LoRA delta and quantize on the way in.
type stBuild struct {
	cfg              *Config
	arch             *Architecture
	s                *tensorSchema
	st               *embed.SafetensorsFile
	quant            quantMode
	needCanonical    bool
	skipRow4         bool
	sink             *giwWriter
	abort            <-chan struct{}
	hd               int
	headDim          int
	qDim             int
	kvDim            int
	w                *Weights
	have             map[string]bool
	tn               func(i int, suf string) string
	fusedExperts     bool
	switchMLPExperts bool
	loadMatQ         func(name string, rows, cols int) (linalg.WeightMat, error)
	loadProj         func(name string, out, in int, batched bool) (linalg.WeightMat, error)
	loadF32Merged    func(name string, shape ...int) ([]float32, error)
	loadMatMerged    func(name string, rows, cols int) (linalg.WeightMat, error)
	optNorm          func(i int, suffix string) ([]float32, error)
	mp               func(n string) string
	id               string
}

// Layers load in parallel — each is independent over the read-only mmap, and
// the per-tensor dequant/re-quant is the cost.
// loadLayer loads layer i's tensors into w.Layers[i].
func (b *stBuild) loadLayer(i int) error {
	cfg, arch, s, st, quant, needCanonical, skipRow4, sink, hd, headDim, qDim, kvDim, w, have, tn, fusedExperts, switchMLPExperts, loadMatQ, loadProj, loadF32Merged, loadMatMerged, optNorm := b.cfg, b.arch, b.s, b.st, b.quant, b.needCanonical, b.skipRow4, b.sink, b.hd, b.headDim, b.qDim, b.kvDim, b.w, b.have, b.tn, b.fusedExperts, b.switchMLPExperts, b.loadMatQ, b.loadProj, b.loadF32Merged, b.loadMatMerged, b.optNorm
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
		if err = loadQwen35Attn(i, l, arch, hd, tn, loadMatMerged, loadF32Merged, quant, needCanonical, skipRow4); err != nil {
			return err
		}
	} else if arch.kda != nil && arch.isLinearLayer(i) {
		// Bailing Hybrid (Ling 3.0): KDA linear layers. Non-linear layers of the SAME
		// family fall through to the arch.mla branch below (bailingHybridArchitecture
		// sets both kda and mla; this check must come first so the else-if chain
		// dispatches KDA-vs-MLA per layer, not by family alone).
		if err = loadBailingKDA(st, i, l, arch, hd, tn, loadMatQ); err != nil {
			return err
		}
	} else if arch.mla != nil {
		if err = loadDeepseekAttn(st, i, l, arch, hd, tn); err != nil {
			return err
		}
		if mlaProjInt8(quant) {
			l.mla.quantizeProjections(arch)
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
		if l.QProj, err = loadProj(tn(i, s.QProj), aqDim, hd, true); err != nil {
			return err
		}
		if l.QNorm, err = st.TensorF32(tn(i, s.QNorm), ahd); err != nil {
			return err
		}
		// Cross-layer KV sharing (num_kv_shared_layers): the last N layers carry no k_proj/k_norm/v_proj tensors in the
		// checkpoint; they reuse an earlier layer's KV at forward time (runLayersFromEmbedPLE's kvSrc). The GGUF loader
		// (loadG4) does the same. l.KVShared is the per-layer "reuses an earlier layer's KV" flag (LayerWeights.KVShared), not
		// arch.gemma4.KVShared, which is attention_k_eq_v (V reuses K's projection on global layers): same field name, different
		// struct, different meaning (see arch.go).
		firstShared := arch.NumLayers - arch.gemma4.SharedKVLayers
		l.KVShared = i >= firstShared
		if !l.KVShared {
			if l.KProj, err = loadProj(tn(i, s.KProj), akvDim, hd, true); err != nil {
				return err
			}
			if arch.gemma4.KVShared && arch.isGlobalLayer(i) {
				l.VFromK = true // V = v_norm(k_proj output); no v_proj tensor
			} else if l.VProj, err = loadProj(tn(i, s.VProj), akvDim, hd, true); err != nil {
				return err
			}
			if l.KNorm, err = st.TensorF32(tn(i, s.KNorm), ahd); err != nil {
				return err
			}
		}
		// OProj is per-layer regardless of KVShared — a shared-KV layer still has its
		// own attention output projection over the (shared-K/V-derived) attention result.
		if l.OProj, err = loadProj(tn(i, s.OProj), hd, aqDim, false); err != nil {
			return err
		}
	} else {
		// Attention projections ([out, in] row-major). qDim is per layer: some families (Laguna) vary the query head count by
		// layer type, so q_proj differs between layers. headsAt collapses to NumHeads for every other family, leaving
		// lqDim == qDim.
		lqDim := arch.headsAt(i) * headDim
		if l.QProj, err = loadProj(tn(i, s.QProj), lqDim, hd, true); err != nil {
			return err
		}
		if l.KProj, err = loadProj(tn(i, s.KProj), kvDim, hd, true); err != nil {
			return err
		}
		if l.VProj, err = loadProj(tn(i, s.VProj), kvDim, hd, true); err != nil {
			return err
		}
		if l.OProj, err = loadProj(tn(i, s.OProj), hd, lqDim, false); err != nil {
			return err
		}
		// Laguna attention output gate. Its row count SELECTS the granularity —
		// arch.headsAt(i) rows is per-head, headsAt(i)*headDim is per-element — so
		// both valid shapes are accepted and anything else is a hard error rather
		// than a silently mis-shaped gate. See applyGateRow for why the checkpoint,
		// not config.gating, is authoritative here.
		if s.GProj != "" {
			perHead, perElem := arch.headsAt(i), lqDim
			rows := perHead
			if arch.laguna != nil && !arch.laguna.GatePerHead {
				rows = perElem
			}
			if l.GProj, err = loadProj(tn(i, s.GProj), rows, hd, false); err != nil {
				if rows == perHead {
					rows = perElem
				} else {
					rows = perHead
				}
				if l.GProj, err = loadProj(tn(i, s.GProj), rows, hd, false); err != nil {
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
		// Per-head attention sink logit (Flint: self_attn.softmax_offset, or
		// self_attn.sinks on older builds).
		if arch.hasAttnSinks() && s.AttnSinks != "" {
			sinkName := tn(i, s.AttnSinks)
			if !have[sinkName] && s.AttnSinksAlt != "" && have[tn(i, s.AttnSinksAlt)] {
				sinkName = tn(i, s.AttnSinksAlt)
			}
			if l.AttnSinks, err = st.TensorF32(sinkName, arch.headsAt(i)); err != nil {
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
	// Experts path below. The PLE branch's per-layer tensors load here too; the
	// model-level PLE inputs load once, before the layer loop.
	if arch.gemma4 != nil {
		// Per-layer FFN width: arch.gemma4.FFNPerLayer is seeded (safetensors
		// checkpoints) below, right before this loop starts, from each layer's own
		// on-disk tensor shape — see that seeding step's comment for why.
		ffn := arch.ffnAt(i)
		if l.GateProj, err = loadProj(tn(i, s.GateProj), ffn, hd, true); err != nil {
			return err
		}
		if l.UpProj, err = loadProj(tn(i, s.UpProj), ffn, hd, true); err != nil {
			return err
		}
		if l.DownProj, err = loadProj(tn(i, s.DownProj), hd, ffn, false); err != nil {
			return err
		}
		// layer_scalar (a [1] buffer). Absent ⇒ 1.0 (no scaling).
		if ls, lerr := st.TensorF32(tn(i, "layer_scalar"), 1); lerr == nil {
			l.LayerScalar = ls[0]
		} else {
			l.LayerScalar = 1
		}
		if pleDim := arch.gemma4.HiddenSizePerLayerInput; pleDim > 0 {
			if l.PLEGate, err = loadProj(tn(i, "per_layer_input_gate.weight"), pleDim, hd, false); err != nil {
				return err
			}
			if l.PLEProj, err = loadProj(tn(i, "per_layer_projection.weight"), hd, pleDim, false); err != nil {
				return err
			}
			if l.PostPLENorm, err = st.TensorF32(tn(i, "post_per_layer_input_norm.weight"), hd); err != nil {
				return err
			}
		}
		if arch.MoE != nil {
			if l.gemma4moe, err = loadGemma4MoE(st, i, cfg, arch, hd, l, quant, tn, needCanonical, skipRow4, sink); err != nil {
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
			if l.Experts, err = loadFusedExperts(st, tn(i, "mlp.experts.gate_up_proj"), tn(i, "mlp.experts.down_proj"), arch.MoE.NumExperts, expInter, hd, quant, needCanonical, skipRow4, sink); err != nil {
				return err
			}
		} else if switchMLPExperts {
			// MLX MoE: all experts in three stacked 3-D tensors under mlp.switch_mlp.{gate,up,down}_proj.
			if l.Experts, err = loadSwitchMLPExperts(st, tn(i, "mlp.switch_mlp"), arch.MoE.NumExperts, expInter, hd, quant, needCanonical, skipRow4, sink); err != nil {
				return err
			}
		} else {
			l.Experts = make([]expertWeights, arch.MoE.NumExperts)
			if err = forExperts(arch.MoE.NumExperts, sink, func(e int) error {
				ex := &l.Experts[e]
				var err error
				if ex.Gate, err = loadMatQ(tn(i, fmt.Sprintf(s.ExpertGate, e)), expInter, hd); err != nil {
					return err
				}
				if ex.Up, err = loadMatQ(tn(i, fmt.Sprintf(s.ExpertUp, e)), expInter, hd); err != nil {
					return err
				}
				ex.Down, err = loadMatQ(tn(i, fmt.Sprintf(s.ExpertDown, e)), hd, expInter)
				return err
			}); err != nil {
				return err
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
	if l.GateProj, err = loadProj(tn(i, s.GateProj), cfg.IntermediateDim, hd, true); err != nil {
		return err
	}
	if l.UpProj, err = loadProj(tn(i, s.UpProj), cfg.IntermediateDim, hd, true); err != nil {
		return err
	}
	if l.DownProj, err = loadProj(tn(i, s.DownProj), hd, cfg.IntermediateDim, false); err != nil {
		return err
	}
	return nil
}

// finish seeds what only some families need after the per-layer loader is built (Gemma 4's per-layer-embedding inputs and per-layer
// FFN widths, before the layers load) and loads the layers.
func (b *stBuild) finish() (*Weights, error) {
	cfg, arch, s, st, quant, needCanonical, sink, hd, w, tn, loadProj, mp, id := b.cfg, b.arch, b.s, b.st, b.quant, b.needCanonical, b.sink, b.hd, b.w, b.tn, b.loadProj, b.mp, b.id
	loadLayer := b.loadLayer
	var err error
	// Gemma 4 per-layer FFN width: a safetensors checkpoint can vary intermediate_size by layer, but config.json carries one
	// scalar (GGUF stores an explicit per-layer array: ggufIntArray -> cfg.FFNPerLayer). The forward calls arch.ffnAt(i) per
	// layer, so it is seeded here, sequentially, before the parallel loop above reads it: each layer's width comes from its
	// own GateProj tensor's on-disk shape (a header lookup; Tensor() does not decode data), not from a rule guessed from the
	// KV-shared boundary, which is correlated on one checkpoint but is not something the config asserts.
	if arch.gemma4 != nil {
		// Per-Layer-Embedding (PLE) model-level inputs: the same three tensors GGUF loads, at the same precisions. The token
		// table takes the embedding policy and streams a row at a time; the projection is a plain matmul weight; the norm is
		// taken as-is (safetensors norms carry the HF convention already, where GGUF's vnorm has to undo llama.cpp's baked +1).
		// Leaving them nil would crash runLayersFromEmbedPLE's rmsNorm on an E-model.
		if pleDim := arch.gemma4.HiddenSizePerLayerInput; pleDim > 0 {
			pleVocab := arch.gemma4.VocabSizePerLayerInput
			if pleVocab == 0 {
				pleVocab = cfg.VocabSize
			}
			// The forward indexes the table by the main-vocab token id (forward_gemma4.go), with no
			// HF-style "ids past vocab_size_per_layer_input read row 0" mask, so a smaller table would
			// read past its end. Every released E-model has the two equal.
			if pleVocab != cfg.VocabSize {
				return nil, fmt.Errorf("gemma4: vocab_size_per_layer_input %d != vocab_size %d is not supported", pleVocab, cfg.VocabSize)
			}
			pleTotal := cfg.NumLayers * pleDim
			name := mp("model.embed_tokens_per_layer.weight")
			t, terr := st.Tensor(name)
			if terr != nil {
				return nil, terr
			}
			if len(t.Shape) != 2 || t.Shape[0] != pleVocab || t.Shape[1] != pleTotal {
				return nil, fmt.Errorf("gemma4: %q shape %v, want [%d %d]", name, t.Shape, pleVocab, pleTotal)
			}
			readPLE := func(r int, dst []float32) error { // keeps no state: every row range shares it
				row, rerr := t.SubF32(r*pleTotal, pleTotal)
				copy(dst, row)
				return rerr
			}
			if w.PerLayerTokenEmbed, err = streamQuantizedEmbed(pleVocab, pleTotal, quant.gatherTable(), needCanonical, !repackForDecode(sink), 1, streamEmbedWorkers(sink), func() rowReader { return readPLE }); err != nil {
				return nil, err
			}
			if w.PerLayerModelProj, err = loadProj(mp("model.per_layer_model_projection.weight"), pleTotal, hd, false); err != nil {
				return nil, err
			}
			if w.PerLayerProjNorm, err = st.TensorF32(mp("model.per_layer_projection_norm.weight"), pleDim); err != nil {
				return nil, err
			}
		}
		ffnPerLayer := make([]int, cfg.NumLayers)
		varies := false
		for i := range ffnPerLayer {
			ffnPerLayer[i] = cfg.IntermediateDim
			if gt, terr := st.Tensor(tn(i, s.GateProj)); terr == nil && len(gt.Shape) == 2 {
				ffnPerLayer[i] = gt.Shape[0]
			}
			if ffnPerLayer[i] != cfg.IntermediateDim {
				varies = true
			}
		}
		// Record it only when it genuinely varies: FeatGemma4EModel (features.go) reads "FFNPerLayer is non-empty" as "this
		// checkpoint declared real per-layer FFN metadata", as GGUF's ggufIntArray does only for a checkpoint that carries the
		// key. Setting it unconditionally would make every safetensors gemma4 checkpoint, including a uniform-width dense or MoE
		// fixture, report FeatGemma4EModel.
		if varies {
			arch.gemma4.FFNPerLayer = ffnPerLayer
		}
	}
	return finishLayers(w, arch, loadLayer, sink, id, b.abort)
}
