package wbuild

import (
	"fmt"
	"github.com/townsendmerino/goinfer/decoder/config"
	"slices"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/families"
)

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
func buildGPT2Weights(cfg *config.Config, arch *darch.Architecture, st *embed.SafetensorsFile, quant QuantMode, embedInt4, needCanonical, skipRow4 bool, h *Hooks) (*darch.Weights, error) {
	hidden, inter, vocab := arch.HiddenDim, arch.IntermediateDim, arch.VocabSize
	w := &darch.Weights{Cfg: *cfg, Arch: arch, ST: st, Layers: make([]darch.LayerWeights, arch.NumLayers)}
	var err error

	// maybeQuant streams a matmul weight to the resident precision, freeing its f32 (see loadWeights). The Conv1D projections
	// are built after the transpose, so the quantization is applied here rather than in a loader closure. skipRow4 covers
	// every layer projection; GPT-2 does not route through buildWeightsFromSafetensors, so quantizeBatchedProjWM never
	// applies to it.
	maybeQuant := func(m linalg.WeightMat) linalg.WeightMat {
		return h.QuantizeLayerWM(m, quant, needCanonical, skipRow4)
	}

	// Token + learned position embeddings (wte doubles as the tied LM head).
	// wte is the tied head → logit-critical, so quantize with the embedding
	// policy (int8 even in int4 mode); wpe is a positional lookup table added to
	// the embedding, never matmul'd → stays f32.
	if w.Embed, err = LoadMat(st, "wte.weight", vocab, hidden); err != nil {
		return nil, err
	}
	w.Embed = h.QuantizeEmbedWM(w.Embed, quant.EmbeddingWith(embedInt4), needCanonical)
	if w.PosEmbed, err = LoadMat(st, "wpe.weight", arch.MaxPositions, hidden); err != nil {
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
// [E, hidden, inter] = down) are exactly the LoadFusedExperts layout, so the routed
// FFN reuses moeMLP unchanged once split; the ungated shared_mlp loads into
// SharedExpert. Embeddings/experts/attention quantize; the Mamba-2 mixer stays f32.
func buildGraniteWeights(cfg *config.Config, arch *darch.Architecture, st *embed.SafetensorsFile, quant QuantMode, embedInt4, needCanonical, skipRow4 bool, sink *StreamWriter, id string, abort <-chan struct{}, h *Hooks) (*darch.Weights, error) {
	hidden, vocab, inter := arch.HiddenDim, arch.VocabSize, arch.IntermediateDim
	g := arch.Granite
	w := &darch.Weights{Cfg: *cfg, Arch: arch, ST: st, Layers: make([]darch.LayerWeights, arch.NumLayers)}
	var err error
	// q is quantizeWM, or quantizeWMSkipRow4 when skipRow4, for every layer projection below. Embed/LMHead stay on plain
	// quantizeWM: they are read via .Row() on the host on every backend.
	q := func(m linalg.WeightMat) linalg.WeightMat {
		return h.QuantizeLayerWM(m, quant, needCanonical, skipRow4)
	}
	if w.Embed, err = loadEmbedTable(st, "model.embed_tokens.weight", vocab, hidden, quant.EmbeddingWith(embedInt4), needCanonical, sink, h); err != nil {
		return nil, err
	}
	if w.FinalNorm, err = st.TensorF32("model.norm.weight", hidden); err != nil {
		return nil, err
	}
	arch.TiedLMHead = true
	if head, herr := loadEmbedTable(st, "lm_head.weight", vocab, hidden, quant.EmbeddingWith(embedInt4), needCanonical, sink, h); herr == nil {
		w.LMHead = head
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
		if arch.IsMambaLayer(l) {
			mw := &archparts.Mamba2Weights{}
			if mw.InProj, e = st.TensorF32(tn("mamba.in_proj.weight"), projDim, hidden); e != nil {
				return e
			}
			if mw.ConvW, e = st.TensorF32(tn("mamba.conv1d.weight"), convDim, 1, g.DConv); e != nil {
				return e
			}
			if mw.ConvB, e = st.TensorF32(tn("mamba.conv1d.bias"), convDim); e != nil {
				return e
			}
			if mw.ALog, e = st.TensorF32(tn("mamba.A_log"), g.NHeads); e != nil {
				return e
			}
			if mw.D, e = st.TensorF32(tn("mamba.D"), g.NHeads); e != nil {
				return e
			}
			if mw.DtBias, e = st.TensorF32(tn("mamba.dt_bias"), g.NHeads); e != nil {
				return e
			}
			if mw.NormW, e = st.TensorF32(tn("mamba.norm.weight"), dInner); e != nil {
				return e
			}
			if mw.OutProj, e = st.TensorF32(tn("mamba.out_proj.weight"), hidden, dInner); e != nil {
				return e
			}
			lw.Mamba = mw
		} else {
			if lw.QProj, e = LoadMat(st, tn("self_attn.q_proj.weight"), qDim, hidden); e != nil {
				return e
			}
			if lw.KProj, e = LoadMat(st, tn("self_attn.k_proj.weight"), kvDim, hidden); e != nil {
				return e
			}
			if lw.VProj, e = LoadMat(st, tn("self_attn.v_proj.weight"), kvDim, hidden); e != nil {
				return e
			}
			if lw.OProj, e = LoadMat(st, tn("self_attn.o_proj.weight"), hidden, qDim); e != nil {
				return e
			}
			lw.QProj, lw.KProj = q(lw.QProj), q(lw.KProj)
			lw.VProj, lw.OProj = q(lw.VProj), q(lw.OProj)
		}
		// MoE FFN (every layer): router (f32) + fused routed experts + ungated shared.
		if lw.Router, e = LoadMat(st, tn("block_sparse_moe.router.layer.weight"), arch.MoE.NumExperts, hidden); e != nil {
			return e
		}
		if lw.Experts, e = LoadFusedExperts(st, tn("block_sparse_moe.input_linear.weight"), tn("block_sparse_moe.output_linear.weight"), arch.MoE.NumExperts, inter, hidden, quant, needCanonical, skipRow4, sink, h); e != nil {
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
		lw.SharedExpert.Gate = q(linalg.WrapF32(slices.Clone(gu[:half]), sInter, hidden))
		lw.SharedExpert.Up = q(linalg.WrapF32(slices.Clone(gu[half:]), sInter, hidden))
		lw.SharedExpert.Down = q(linalg.WrapF32(slices.Clone(dn), hidden, sInter))
		return nil
	}
	return finishLayers(w, arch, loadLayer, sink, id, abort)
}

// buildNemotronWeights loads a Nemotron-H (nemotron_h) checkpoint: a single-op-per-
// block stack where each layer (under a "mixer" prefix) is a Mamba-2 mixer (f32,
// parity-first, reusing the Granite conventions), a NoPE GQA attention, or a
// non-gated relu² MLP — keyed by arch.nemotron.blockKind. Plain RMSNorm per layer.
func buildNemotronWeights(cfg *config.Config, arch *darch.Architecture, st *embed.SafetensorsFile, quant QuantMode, embedInt4, needCanonical, skipRow4 bool, sink *StreamWriter, id string, abort <-chan struct{}, h *Hooks) (*darch.Weights, error) {
	hidden, vocab, inter := arch.HiddenDim, arch.VocabSize, arch.IntermediateDim
	np := arch.Nemotron
	w := &darch.Weights{Cfg: *cfg, Arch: arch, ST: st, Layers: make([]darch.LayerWeights, arch.NumLayers)}
	var err error
	// q is quantizeWM, or quantizeWMSkipRow4 when skipRow4, for every layer projection below. Embed/LMHead stay on plain
	// quantizeWM: they are read via .Row() on the host on every backend.
	q := func(m linalg.WeightMat) linalg.WeightMat {
		return h.QuantizeLayerWM(m, quant, needCanonical, skipRow4)
	}
	// The released NVIDIA checkpoints name the embedding backbone.embeddings.weight;
	// transformers' own NemotronH names it backbone.embedding.weight, which is the only
	// spelling the tiny fixture (built by instantiating the config) ever produced. Every
	// other tensor name agrees, so this one word is the whole delta — try both.
	if w.Embed, err = loadEmbedTable(st, "backbone.embedding.weight", vocab, hidden, quant.EmbeddingWith(embedInt4), needCanonical, sink, h); err != nil {
		if w.Embed, err = loadEmbedTable(st, "backbone.embeddings.weight", vocab, hidden, quant.EmbeddingWith(embedInt4), needCanonical, sink, h); err != nil {
			return nil, err
		}
	}
	if w.FinalNorm, err = st.TensorF32("backbone.norm_f.weight", hidden); err != nil {
		return nil, err
	}
	arch.TiedLMHead = true
	if head, herr := loadEmbedTable(st, "lm_head.weight", vocab, hidden, quant.EmbeddingWith(embedInt4), needCanonical, sink, h); herr == nil {
		w.LMHead = head
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
		switch np.BlockKind[l] {
		case archparts.NemoMamba:
			mw := &archparts.Mamba2Weights{}
			if mw.InProj, e = st.TensorF32(tn("mixer.in_proj.weight"), projDim, hidden); e != nil {
				return e
			}
			if mw.ConvW, e = st.TensorF32(tn("mixer.conv1d.weight"), convDim, 1, np.DConv); e != nil {
				return e
			}
			if mw.ConvB, e = st.TensorF32(tn("mixer.conv1d.bias"), convDim); e != nil {
				return e
			}
			if mw.ALog, e = st.TensorF32(tn("mixer.A_log"), np.NHeads); e != nil {
				return e
			}
			if mw.D, e = st.TensorF32(tn("mixer.D"), np.NHeads); e != nil {
				return e
			}
			if mw.DtBias, e = st.TensorF32(tn("mixer.dt_bias"), np.NHeads); e != nil {
				return e
			}
			if mw.NormW, e = st.TensorF32(tn("mixer.norm.weight"), dInner); e != nil {
				return e
			}
			if mw.OutProj, e = st.TensorF32(tn("mixer.out_proj.weight"), hidden, dInner); e != nil {
				return e
			}
			if h.MambaProjInt8(quant) {
				mw.QuantizeProjections(projDim, hidden, dInner)
			}
			lw.Mamba = mw
		case archparts.NemoAttn:
			if lw.QProj, e = LoadMat(st, tn("mixer.q_proj.weight"), qDim, hidden); e != nil {
				return e
			}
			if lw.KProj, e = LoadMat(st, tn("mixer.k_proj.weight"), kvDim, hidden); e != nil {
				return e
			}
			if lw.VProj, e = LoadMat(st, tn("mixer.v_proj.weight"), kvDim, hidden); e != nil {
				return e
			}
			if lw.OProj, e = LoadMat(st, tn("mixer.o_proj.weight"), hidden, qDim); e != nil {
				return e
			}
			lw.QProj, lw.KProj = q(lw.QProj), q(lw.KProj)
			lw.VProj, lw.OProj = q(lw.VProj), q(lw.OProj)
		case archparts.NemoMLP:
			if lw.UpProj, e = LoadMat(st, tn("mixer.up_proj.weight"), inter, hidden); e != nil {
				return e
			}
			if lw.DownProj, e = LoadMat(st, tn("mixer.down_proj.weight"), hidden, inter); e != nil {
				return e
			}
			lw.UpProj, lw.DownProj = q(lw.UpProj), q(lw.DownProj)
		case archparts.NemoMoE:
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
			if lw.Router, e = LoadMat(st, tn("mixer.gate.weight"), moe.NumExperts, hidden); e != nil {
				return e
			}
			if lw.RouterBias, e = st.TensorF32(tn("mixer.gate.e_score_correction_bias"), moe.NumExperts); e != nil {
				return e
			}
			lw.Experts = make([]archparts.ExpertWeights, moe.NumExperts)
			if e = ForExperts(moe.NumExperts, sink, func(ei int) error {
				ex := &lw.Experts[ei]
				var e error
				if ex.Up, e = LoadMat(st, tn(fmt.Sprintf("mixer.experts.%d.up_proj.weight", ei)), moe.IntermediateDim, hidden); e != nil {
					return e
				}
				if ex.Down, e = LoadMat(st, tn(fmt.Sprintf("mixer.experts.%d.down_proj.weight", ei)), hidden, moe.IntermediateDim); e != nil {
					return e
				}
				ex.Up, ex.Down = q(ex.Up), q(ex.Down)
				return nil
			}); e != nil {
				return e
			}
			if moe.SharedIntermediateDim > 0 {
				se := &lw.SharedExpert
				if se.Up, e = LoadMat(st, tn("mixer.shared_experts.up_proj.weight"), moe.SharedIntermediateDim, hidden); e != nil {
					return e
				}
				if se.Down, e = LoadMat(st, tn("mixer.shared_experts.down_proj.weight"), hidden, moe.SharedIntermediateDim); e != nil {
					return e
				}
				se.Up, se.Down = q(se.Up), q(se.Down)
			}
		}
		return nil
	}
	return finishLayers(w, arch, loadLayer, sink, id, abort)
}

// buildPhi3Weights loads a Phi-3 / Phi-4 (phi3) checkpoint. Phi-3 is the llama skeleton
// (RMSNorm no-offset, Pre2, SwiGLU, NeoX RoPE, no QK-norm, no bias, untied head) with two
// FUSED tensors that this loader splits into the standard LayerWeights fields so the
// generic runLayers handles the forward unchanged: self_attn.qkv_proj is q‖k‖v stacked by
// output rows (split at NumHeads*HeadDim, then +NumKVHeads*HeadDim), and mlp.gate_up_proj
// is gate‖up (split in half). The fused tensors load to f32, slice by rows, and quantize
// per the resident mode (the GPT-2 fused-QKV precedent).
func buildPhi3Weights(cfg *config.Config, arch *darch.Architecture, st *embed.SafetensorsFile, quant QuantMode, embedInt4, needCanonical, skipRow4 bool, sink *StreamWriter, id string, abort <-chan struct{}, h *Hooks) (*darch.Weights, error) {
	hidden, inter, vocab := arch.HiddenDim, arch.IntermediateDim, arch.VocabSize
	hd := arch.HeadDim
	qDim, kvDim := arch.NumHeads*hd, arch.NumKVHeads*hd
	w := &darch.Weights{Cfg: *cfg, Arch: arch, ST: st, Layers: make([]darch.LayerWeights, arch.NumLayers)}
	var err error
	// q is quantizeWM, or quantizeWMSkipRow4 when skipRow4, for every layer projection below. Embed/LMHead stay on plain
	// quantizeWM: they are read via .Row() on the host on every backend.
	q := func(m linalg.WeightMat) linalg.WeightMat {
		return h.QuantizeLayerWM(m, quant, needCanonical, skipRow4)
	}
	if w.Embed, err = loadEmbedTable(st, "model.embed_tokens.weight", vocab, hidden, quant.EmbeddingWith(embedInt4), needCanonical, sink, h); err != nil {
		return nil, err
	}
	if w.FinalNorm, err = st.TensorF32("model.norm.weight", hidden); err != nil {
		return nil, err
	}
	arch.TiedLMHead = true
	if head, herr := loadEmbedTable(st, "lm_head.weight", vocab, hidden, quant.EmbeddingWith(embedInt4), needCanonical, sink, h); herr == nil {
		w.LMHead = head
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
		l.QProj = q(linalg.WrapF32(qkv[0:qDim*hidden], qDim, hidden))
		l.KProj = q(linalg.WrapF32(qkv[qDim*hidden:(qDim+kvDim)*hidden], kvDim, hidden))
		l.VProj = q(linalg.WrapF32(qkv[(qDim+kvDim)*hidden:(qDim+2*kvDim)*hidden], kvDim, hidden))
		if l.OProj, e = LoadMat(st, p+"self_attn.o_proj.weight", hidden, qDim); e != nil {
			return e
		}
		l.OProj = q(l.OProj)
		// Fused gate_up_proj [2*inter, hidden] → gate ‖ up (SwiGLU: down(silu(gate)·up)).
		gu, gerr := st.TensorF32(p+"mlp.gate_up_proj.weight", 2*inter, hidden)
		if gerr != nil {
			return gerr
		}
		l.GateProj = q(linalg.WrapF32(gu[0:inter*hidden], inter, hidden))
		l.UpProj = q(linalg.WrapF32(gu[inter*hidden:2*inter*hidden], inter, hidden))
		if l.DownProj, e = LoadMat(st, p+"mlp.down_proj.weight", hidden, inter); e != nil {
			return e
		}
		l.DownProj = q(l.DownProj)
		return nil
	}
	return finishLayers(w, arch, loadLayer, sink, id, abort)
}

// buildGlmOcrWeights loads the GLM-OCR text decoder (model_type glm_ocr / glm_ocr_text; the vision
// wrapper's checkpoint holds it under model.language_model.* with a top-level lm_head.weight). It is
// the llama skeleton with FOUR norms per layer (Sandwich4; families.GlmOcrTensorSchema carries the names,
// which are not Gemma's), separate bias-free q/k/v/o at an explicit head_dim, and ONE fused tensor:
// mlp.gate_up_proj is [2*inter, hidden] with the GATE rows first (HF: chunk(2); up * silu(gate)).
//
// The multi-token-prediction layer sits at layers.<NumLayers> (eh_proj, enorm, hnorm, its own
// embed_tokens, shared_head, plus a full block): 218 M parameters inference never uses, and HF's
// _keys_to_ignore_on_load_unexpected drops it too. Nothing here requests a layer index >=
// NumLayers, so it is skipped by construction; the vision tower (model.visual.*) is never requested
// either. The head is untied on the real checkpoint but finalized from lm_head.weight presence, like
// phi3, so a tied re-save would still load.
func buildGlmOcrWeights(cfg *config.Config, arch *darch.Architecture, st *embed.SafetensorsFile, quant QuantMode, embedInt4, needCanonical, skipRow4 bool, sink *StreamWriter, id string, abort <-chan struct{}, h *Hooks) (*darch.Weights, error) {
	s := &families.GlmOcrTensorSchema
	hidden, inter, vocab := arch.HiddenDim, arch.IntermediateDim, arch.VocabSize
	hd := arch.HeadDim
	qDim, kvDim := arch.NumHeads*hd, arch.NumKVHeads*hd
	w := &darch.Weights{Cfg: *cfg, Arch: arch, ST: st, Layers: make([]darch.LayerWeights, arch.NumLayers)}
	var err error
	q := func(m linalg.WeightMat) linalg.WeightMat {
		return h.QuantizeLayerWM(m, quant, needCanonical, skipRow4)
	}
	if w.Embed, err = loadEmbedTable(st, s.Embed, vocab, hidden, quant.EmbeddingWith(embedInt4), needCanonical, sink, h); err != nil {
		return nil, err
	}
	if w.FinalNorm, err = st.TensorF32(s.FinalNorm, hidden); err != nil {
		return nil, err
	}
	arch.TiedLMHead = true
	if head, herr := loadEmbedTable(st, s.LMHead, vocab, hidden, quant.EmbeddingWith(embedInt4), needCanonical, sink, h); herr == nil {
		w.LMHead = head
		arch.TiedLMHead = false
	}

	loadLayer := func(i int) error {
		l := &w.Layers[i]
		p := fmt.Sprintf("%slayers.%d.", families.GlmOcrPrefix, i)
		var e error
		if l.PreAttnNorm, e = st.TensorF32(p+s.PreAttnNorm, hidden); e != nil {
			return e
		}
		if l.PostAttnNorm, e = st.TensorF32(p+s.PostAttnNorm, hidden); e != nil {
			return e
		}
		if l.PreMLPNorm, e = st.TensorF32(p+s.PreMLPNorm, hidden); e != nil {
			return e
		}
		if l.PostMLPNorm, e = st.TensorF32(p+s.PostMLPNorm, hidden); e != nil {
			return e
		}
		if l.QProj, e = LoadMat(st, p+s.QProj, qDim, hidden); e != nil {
			return e
		}
		l.QProj = q(l.QProj)
		if l.KProj, e = LoadMat(st, p+s.KProj, kvDim, hidden); e != nil {
			return e
		}
		l.KProj = q(l.KProj)
		if l.VProj, e = LoadMat(st, p+s.VProj, kvDim, hidden); e != nil {
			return e
		}
		l.VProj = q(l.VProj)
		if l.OProj, e = LoadMat(st, p+s.OProj, hidden, qDim); e != nil {
			return e
		}
		l.OProj = q(l.OProj)
		// Fused gate_up_proj [2*inter, hidden] -> gate ‖ up by output rows, GATE FIRST.
		gu, gerr := st.TensorF32(p+families.GlmOcrGateUp, 2*inter, hidden)
		if gerr != nil {
			return gerr
		}
		l.GateProj = q(linalg.WrapF32(gu[0:inter*hidden], inter, hidden))
		l.UpProj = q(linalg.WrapF32(gu[inter*hidden:2*inter*hidden], inter, hidden))
		if l.DownProj, e = LoadMat(st, p+s.DownProj, hidden, inter); e != nil {
			return e
		}
		l.DownProj = q(l.DownProj)
		return nil
	}
	return finishLayers(w, arch, loadLayer, sink, id, abort)
}

// buildSpark25Weights loads Spark-X2.5 (model_type spark2_5): fused q_k_v_proj splits by
// output rows exactly like buildPhi3Weights's qkv_proj, but gate_proj/up_proj/down_proj are
// SEPARATE tensors (not fused, unlike Phi-3's gate_up_proj) so they load through the ordinary
// loadMat path. Two tensor names differ from the usual Llama-family convention (verified against
// the real modeling_spark.py, not assumed): the embedding is model.embedding.weight (not
// embed_tokens) and the attention output projection is self_attn.out_proj.weight (not o_proj).
// GProj (the sigmoid attention-output gate) is always per-head here — no per-element variant
// like Laguna's, since headwise_attn_output_gate is a plain bool — and loads unquantized, same
// as Laguna's own GProj (see the generic loader's own comment on why: deliberately excluded from
// quantization). No bias tensors: spark25Architecture rejects attention_bias=true before this
// runs, so every released config's shape is the only one this function needs to handle.
func buildSpark25Weights(cfg *config.Config, arch *darch.Architecture, st *embed.SafetensorsFile, quant QuantMode, embedInt4, needCanonical, skipRow4 bool, sink *StreamWriter, id string, abort <-chan struct{}, h *Hooks) (*darch.Weights, error) {
	hidden, inter, vocab := arch.HiddenDim, arch.IntermediateDim, arch.VocabSize
	hd, nH := arch.HeadDim, arch.NumHeads
	qDim, kvDim := nH*hd, arch.NumKVHeads*hd
	w := &darch.Weights{Cfg: *cfg, Arch: arch, ST: st, Layers: make([]darch.LayerWeights, arch.NumLayers)}
	var err error
	q := func(m linalg.WeightMat) linalg.WeightMat {
		return h.QuantizeLayerWM(m, quant, needCanonical, skipRow4)
	}
	if w.Embed, err = loadEmbedTable(st, "model.embedding.weight", vocab, hidden, quant.EmbeddingWith(embedInt4), needCanonical, sink, h); err != nil {
		return nil, err
	}
	if w.FinalNorm, err = st.TensorF32("model.norm.weight", hidden); err != nil {
		return nil, err
	}
	arch.TiedLMHead = true
	if head, herr := loadEmbedTable(st, "lm_head.weight", vocab, hidden, quant.EmbeddingWith(embedInt4), needCanonical, sink, h); herr == nil {
		w.LMHead = head
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
		// Fused q_k_v_proj [qDim+2*kvDim, hidden] → Q ‖ K ‖ V by output rows, exactly the
		// order modeling_spark.py's Spark2_5Attention.forward splits: q, then k, then v.
		qkv, qerr := st.TensorF32(p+"self_attn.q_k_v_proj.weight", qDim+2*kvDim, hidden)
		if qerr != nil {
			return qerr
		}
		l.QProj = q(linalg.WrapF32(qkv[0:qDim*hidden], qDim, hidden))
		l.KProj = q(linalg.WrapF32(qkv[qDim*hidden:(qDim+kvDim)*hidden], kvDim, hidden))
		l.VProj = q(linalg.WrapF32(qkv[(qDim+kvDim)*hidden:(qDim+2*kvDim)*hidden], kvDim, hidden))
		if l.OProj, e = LoadMat(st, p+"self_attn.out_proj.weight", hidden, qDim); e != nil {
			return e
		}
		l.OProj = q(l.OProj)
		gw, gerr := st.TensorF32(p+"self_attn.g_proj.weight", nH, hidden)
		if gerr != nil {
			return gerr
		}
		l.GProj = linalg.WrapF32(gw, nH, hidden) // unquantized — see the doc comment
		if l.GateProj, e = LoadMat(st, p+"mlp.gate_proj.weight", inter, hidden); e != nil {
			return e
		}
		l.GateProj = q(l.GateProj)
		if l.UpProj, e = LoadMat(st, p+"mlp.up_proj.weight", inter, hidden); e != nil {
			return e
		}
		l.UpProj = q(l.UpProj)
		if l.DownProj, e = LoadMat(st, p+"mlp.down_proj.weight", hidden, inter); e != nil {
			return e
		}
		l.DownProj = q(l.DownProj)
		return nil
	}
	return finishLayers(w, arch, loadLayer, sink, id, abort)
}

// buildLlama4Weights loads a Llama 4 text decoder (llama4_text). Attention is separate
// q/k/v/o (no fusion, no bias; the L2 QK-norm is parameter-free, applied in the forward).
// The FFN is per-layer: dense layers (arch.llama4.isMoE[l] false) use feed_forward.{gate,up,
// down}_proj at intermediate_size_mlp; MoE layers use feed_forward.router + a SHARED expert
// (feed_forward.shared_expert.*) + batched routed experts. The routed experts are stored
// FUSED and TRANSPOSED for a bmm — gate_up_proj is [nE, hidden, 2*inter] and down_proj is
// [nE, inter, hidden] ([in, out] per expert) — so each expert is transposed to goinfer's
// [out, in] WeightMat and the gate‖up halves split out.
func buildLlama4Weights(cfg *config.Config, arch *darch.Architecture, st *embed.SafetensorsFile, quant QuantMode, embedInt4, needCanonical, skipRow4 bool, sink *StreamWriter, id string, abort <-chan struct{}, h *Hooks) (*darch.Weights, error) {
	hidden, vocab := arch.HiddenDim, arch.VocabSize
	hd := arch.HeadDim
	qDim, kvDim := arch.NumHeads*hd, arch.NumKVHeads*hd
	denseInter := arch.IntermediateDim   // intermediate_size_mlp
	expInter := arch.MoE.IntermediateDim // intermediate_size (routed + shared)
	nE := arch.MoE.NumExperts
	lp := arch.Llama4
	w := &darch.Weights{Cfg: *cfg, Arch: arch, ST: st, Layers: make([]darch.LayerWeights, arch.NumLayers)}
	var err error
	// q is quantizeWM, or quantizeWMSkipRow4 when skipRow4, for every layer projection below. Embed/LMHead stay on plain
	// quantizeWM: they are read via .Row() on the host on every backend.
	q := func(m linalg.WeightMat) linalg.WeightMat {
		return h.QuantizeLayerWM(m, quant, needCanonical, skipRow4)
	}
	if w.Embed, err = loadEmbedTable(st, "model.embed_tokens.weight", vocab, hidden, quant.EmbeddingWith(embedInt4), needCanonical, sink, h); err != nil {
		return nil, err
	}
	if w.FinalNorm, err = st.TensorF32("model.norm.weight", hidden); err != nil {
		return nil, err
	}
	arch.TiedLMHead = true
	if head, herr := loadEmbedTable(st, "lm_head.weight", vocab, hidden, quant.EmbeddingWith(embedInt4), needCanonical, sink, h); herr == nil {
		w.LMHead = head
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
		if l.QProj, e = LoadMat(st, p+"self_attn.q_proj.weight", qDim, hidden); e != nil {
			return e
		}
		if l.KProj, e = LoadMat(st, p+"self_attn.k_proj.weight", kvDim, hidden); e != nil {
			return e
		}
		if l.VProj, e = LoadMat(st, p+"self_attn.v_proj.weight", kvDim, hidden); e != nil {
			return e
		}
		if l.OProj, e = LoadMat(st, p+"self_attn.o_proj.weight", hidden, qDim); e != nil {
			return e
		}
		l.QProj, l.KProj = q(l.QProj), q(l.KProj)
		l.VProj, l.OProj = q(l.VProj), q(l.OProj)
		if !lp.IsMoE[i] {
			// Dense FFN at intermediate_size_mlp.
			if l.GateProj, e = LoadMat(st, p+"feed_forward.gate_proj.weight", denseInter, hidden); e != nil {
				return e
			}
			if l.UpProj, e = LoadMat(st, p+"feed_forward.up_proj.weight", denseInter, hidden); e != nil {
				return e
			}
			if l.DownProj, e = LoadMat(st, p+"feed_forward.down_proj.weight", hidden, denseInter); e != nil {
				return e
			}
			l.GateProj, l.UpProj, l.DownProj = q(l.GateProj), q(l.UpProj), q(l.DownProj)
			return nil
		}
		// MoE: router + ungated shared expert + batched fused routed experts.
		if l.Router, e = LoadMat(st, p+"feed_forward.router.weight", nE, hidden); e != nil {
			return e
		}
		if l.SharedExpert.Gate, e = LoadMat(st, p+"feed_forward.shared_expert.gate_proj.weight", expInter, hidden); e != nil {
			return e
		}
		if l.SharedExpert.Up, e = LoadMat(st, p+"feed_forward.shared_expert.up_proj.weight", expInter, hidden); e != nil {
			return e
		}
		if l.SharedExpert.Down, e = LoadMat(st, p+"feed_forward.shared_expert.down_proj.weight", hidden, expInter); e != nil {
			return e
		}
		l.SharedExpert.Gate = q(l.SharedExpert.Gate)
		l.SharedExpert.Up = q(l.SharedExpert.Up)
		l.SharedExpert.Down = q(l.SharedExpert.Down)
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
		l.Experts = make([]archparts.ExpertWeights, nE)
		if err := ForExperts(nE, sink, func(ex int) error {
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
			l.Experts[ex] = archparts.ExpertWeights{
				Gate: q(linalg.WrapF32(gate, expInter, hidden)),
				Up:   q(linalg.WrapF32(up, expInter, hidden)),
				Down: q(linalg.WrapF32(down, hidden, expInter)),
			}
			return nil
		}); err != nil {
			return err
		}
		return nil
	}
	return finishLayers(w, arch, loadLayer, sink, id, abort)
}
