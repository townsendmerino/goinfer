package decoder

import (
	"fmt"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// buildGptOssWeights loads gpt-oss (20b/120b) from the released SAFETENSORS checkpoint,
// whose experts are MXFP4 while everything else is BF16.
//
// TWO LAYOUT FACTS DRIVE THIS FILE, and both were established by diffing a dequantized
// expert against the same weight read through the already-T3-validated GGUF path — not
// from the format's documentation, which does not say either one:
//
//  1. MXFP4 nibbles are SEQUENTIAL here (byte j holds elements 2j and 2j+1), where GGML
//     packs j and j+16. Measured: cosine 1.000000 sequential vs 0.081 GGML.
//  2. gate_up_proj is INTERLEAVED, not concatenated: row 2k is gate row k and row 2k+1 is
//     UP row k. Measured the same way (st row 1 == gguf up row 0, cosine 1.000000).
//
// Either mistake yields correct shapes, finite values, plausible magnitudes and entirely
// wrong weights, so both are asserted by the parity gate rather than trusted.
//
// MEMORY. The experts never materialize as f32: gpt-oss-20b is ~76GB dequantized across
// all layers. Each output row is dequantized on demand straight into streamQuantized,
// which emits the quantized WeightMat row by row.
func buildGptOssWeights(cfg *Config, arch *Architecture, st *embed.SafetensorsFile, quant quantMode) (*Weights, error) {
	hidden, vocab := arch.HiddenDim, arch.VocabSize
	hd := arch.HeadDim
	qDim, kvDim := arch.NumHeads*hd, arch.NumKVHeads*hd
	nE := arch.MoE.NumExperts
	expInter := arch.MoE.IntermediateDim
	const blockElems = mxfp4BlockElems

	w := &Weights{Cfg: *cfg, arch: arch, st: st, Layers: make([]LayerWeights, arch.NumLayers)}
	var err error
	if w.Embed, err = loadMat(st, "model.embed_tokens.weight", vocab, hidden); err != nil {
		return nil, err
	}
	w.Embed = quantizeWM(w.Embed, quant.embedding())
	if w.FinalNorm, err = st.TensorF32("model.norm.weight", hidden); err != nil {
		return nil, err
	}
	if head, herr := loadMat(st, "lm_head.weight", vocab, hidden); herr == nil {
		w.LMHead = quantizeWM(head, quant.embedding())
		arch.TiedLMHead = false
	} else {
		arch.TiedLMHead = true // gpt-oss ships lm_head; tie only if a variant omits it
	}

	// mxfp4Pair reads one *_blocks / *_scales tensor pair and returns a row reader.
	// rows is the tensor's first dim after the expert axis; cols must be a whole number
	// of 32-element blocks.
	mxfp4Pair := func(prefix string, rows, cols int) (func(expert, row int, dst []float32) error, error) {
		bT, berr := st.Tensor(prefix + "_blocks")
		if berr != nil {
			return nil, fmt.Errorf("decoder(gptoss-st): %s_blocks: %w", prefix, berr)
		}
		sT, serr := st.Tensor(prefix + "_scales")
		if serr != nil {
			return nil, fmt.Errorf("decoder(gptoss-st): %s_scales: %w", prefix, serr)
		}
		blocks, berr := bT.Uint8s()
		if berr != nil {
			return nil, fmt.Errorf("decoder(gptoss-st): %s_blocks bytes: %w", prefix, berr)
		}
		scales, serr := sT.Uint8s()
		if serr != nil {
			return nil, fmt.Errorf("decoder(gptoss-st): %s_scales bytes: %w", prefix, serr)
		}
		if cols%blockElems != 0 {
			return nil, fmt.Errorf("decoder(gptoss-st): %s cols %d is not a multiple of the %d-element MXFP4 block", prefix, cols, blockElems)
		}
		nB := cols / blockElems
		// Shapes are checked against the ARCHITECTURE, not just against each other: a
		// blocks/scales pair can be self-consistent and still be the wrong tensor.
		wantBlocks := nE * rows * nB * 16
		wantScales := nE * rows * nB
		if len(blocks) != wantBlocks || len(scales) != wantScales {
			return nil, fmt.Errorf("decoder(gptoss-st): %s has %d block bytes / %d scale bytes, want %d/%d for [%d experts, %d rows, %d cols]",
				prefix, len(blocks), len(scales), wantBlocks, wantScales, nE, rows, cols)
		}
		return func(expert, row int, dst []float32) error {
			r := expert*rows + row
			return mxfp4DequantSplitInto(blocks[r*nB*16:(r+1)*nB*16], scales[r*nB:(r+1)*nB], nB, dst)
		}, nil
	}

	// bf16Rows reads a BF16 tensor as f32 once (biases and router are small). `dims` are
	// the tensor's SHAPE, not its element count — TensorF32 validates them dimension by
	// dimension, so passing a flattened product is rejected rather than silently accepted.
	bf16Rows := func(name string, dims ...int) ([]float32, error) {
		return st.TensorF32(name, dims...)
	}

	for i := range arch.NumLayers {
		l := &w.Layers[i]
		p := fmt.Sprintf("model.layers.%d.", i)
		if l.PreAttnNorm, err = st.TensorF32(p+"input_layernorm.weight", hidden); err != nil {
			return nil, err
		}
		if l.PreMLPNorm, err = st.TensorF32(p+"post_attention_layernorm.weight", hidden); err != nil {
			return nil, err
		}
		for _, m := range []struct {
			dst     *linalg.WeightMat
			bias    *[]float32
			name    string
			out, in int
		}{
			{&l.QProj, &l.QBias, "q_proj", qDim, hidden},
			{&l.KProj, &l.KBias, "k_proj", kvDim, hidden},
			{&l.VProj, &l.VBias, "v_proj", kvDim, hidden},
			{&l.OProj, &l.OBias, "o_proj", hidden, qDim},
		} {
			mat, merr := loadMat(st, p+"self_attn."+m.name+".weight", m.out, m.in)
			if merr != nil {
				return nil, merr
			}
			*m.dst = quantizeWM(mat, matmulQuant(quant, m.name))
			if *m.bias, err = bf16Rows(p+"self_attn."+m.name+".bias", m.out); err != nil {
				return nil, err
			}
		}
		// Per-head attention sinks — the primitive that makes this family CPU-only.
		if l.AttnSinks, err = bf16Rows(p+"self_attn.sinks", arch.NumHeads); err != nil {
			return nil, err
		}
		// Router: weight + a plain logit bias (gpt-oss adds it BEFORE top-k, unlike
		// DeepSeek/GLM's selection-only correction bias that shares the field).
		rmat, rerr := loadMat(st, p+"mlp.router.weight", nE, hidden)
		if rerr != nil {
			return nil, rerr
		}
		l.Router = rmat // router stays f32: top-k selection is discrete and quantizing it flips experts
		if l.RouterBias, err = bf16Rows(p+"mlp.router.bias", nE); err != nil {
			return nil, err
		}

		gateUp, guErr := mxfp4Pair(p+"mlp.experts.gate_up_proj", 2*expInter, hidden)
		if guErr != nil {
			return nil, guErr
		}
		down, dErr := mxfp4Pair(p+"mlp.experts.down_proj", hidden, expInter)
		if dErr != nil {
			return nil, dErr
		}
		guBias, guBErr := bf16Rows(p+"mlp.experts.gate_up_proj_bias", nE, 2*expInter)
		if guBErr != nil {
			return nil, guBErr
		}
		dBias, dBErr := bf16Rows(p+"mlp.experts.down_proj_bias", nE, hidden)
		if dBErr != nil {
			return nil, dBErr
		}

		l.Experts = make([]expertWeights, nE)
		for e := range nE {
			// INTERLEAVED: gate row k is tensor row 2k, up row k is 2k+1.
			gm, gerr := streamQuantized(expInter, hidden, matmulQuant(quant, "expert_gate"), func(r int, dst []float32) error {
				return gateUp(e, 2*r, dst)
			})
			if gerr != nil {
				return nil, fmt.Errorf("decoder(gptoss-st): layer %d expert %d gate: %w", i, e, gerr)
			}
			um, uerr := streamQuantized(expInter, hidden, matmulQuant(quant, "expert_up"), func(r int, dst []float32) error {
				return gateUp(e, 2*r+1, dst)
			})
			if uerr != nil {
				return nil, fmt.Errorf("decoder(gptoss-st): layer %d expert %d up: %w", i, e, uerr)
			}
			dm, derr := streamQuantized(hidden, expInter, matmulQuant(quant, "expert_down"), func(r int, dst []float32) error {
				return down(e, r, dst)
			})
			if derr != nil {
				return nil, fmt.Errorf("decoder(gptoss-st): layer %d expert %d down: %w", i, e, derr)
			}
			// The biases interleave exactly as the weights do.
			gb := make([]float32, expInter)
			ub := make([]float32, expInter)
			base := e * 2 * expInter
			for k := range expInter {
				gb[k] = guBias[base+2*k]
				ub[k] = guBias[base+2*k+1]
			}
			l.Experts[e] = expertWeights{
				Gate: gm, Up: um, Down: dm,
				GateBias: gb, UpBias: ub,
				DownBias: append([]float32(nil), dBias[e*hidden:(e+1)*hidden]...),
			}
		}
	}
	return w, nil
}
