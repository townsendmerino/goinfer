package decoder

import (
	"fmt"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// buildInternLM2Weights loads InternLM2 (model_type internlm2), which is a llama in every
// respect the forward cares about and a stranger in every respect the LOADER does.
//
// Two departures, both naming/layout rather than math:
//
//  1. EVERY TENSOR IS RENAMED. tok_embeddings / output / attention.wqkv / attention.wo /
//     attention_norm / feed_forward.w1,w2,w3 / ffn_norm, where llama says embed_tokens /
//     lm_head / self_attn.*_proj / input_layernorm / mlp.*_proj / post_attention_layernorm.
//     w1 is gate, w3 is up, w2 is down — the original llama naming, not an arbitrary one.
//
//  2. wqkv IS FUSED AND GROUPED, which is the part worth care. It is NOT phi3's
//     [Q ‖ K ‖ V] concatenation. HF's InternLM2Attention reshapes it to
//     (..., num_kv_heads, 2 + num_kv_groups, head_dim) and takes
//     query = [..., :num_kv_groups, :] — so the rows run:
//
//     group 0: q q q q k v | group 1: q q q q k v | …
//
//     with num_kv_groups = num_heads/num_kv_heads query heads per KV head. Reading it as a
//     plain concat would put a K row where the 5th query head belongs: correct shapes, finite
//     values, and attention computed against the wrong vectors — the failure mode this repo
//     keeps meeting. The de-interleave below is the only new code, and gathering group by
//     group yields head order directly (head = g*groups + j), so no permutation is needed
//     beyond the gather.
func buildInternLM2Weights(cfg *Config, arch *Architecture, st *embed.SafetensorsFile, quant quantMode) (*Weights, error) {
	hidden, inter, vocab := arch.HiddenDim, arch.IntermediateDim, arch.VocabSize
	hd := arch.HeadDim
	nH, nKV := arch.NumHeads, arch.NumKVHeads
	if nKV == 0 || nH%nKV != 0 {
		return nil, fmt.Errorf("decoder(internlm2): num_attention_heads=%d must be a multiple of num_key_value_heads=%d", nH, nKV)
	}
	groups := nH / nKV // query heads per KV head
	gs := 2 + groups   // rows per KV group: `groups` Q, then K, then V
	qDim, kvDim := nH*hd, nKV*hd

	w := &Weights{Cfg: *cfg, arch: arch, st: st, Layers: make([]LayerWeights, arch.NumLayers)}
	var err error
	if w.Embed, err = loadMat(st, "model.tok_embeddings.weight", vocab, hidden); err != nil {
		return nil, err
	}
	w.Embed = quantizeWM(w.Embed, quant.embedding())
	if w.FinalNorm, err = st.TensorF32("model.norm.weight", hidden); err != nil {
		return nil, err
	}
	head, herr := loadMat(st, "output.weight", vocab, hidden)
	if herr != nil {
		return nil, fmt.Errorf("decoder(internlm2): output.weight: %w", herr)
	}
	w.LMHead = quantizeWM(head, quant.embedding())
	arch.TiedLMHead = false

	for i := range arch.NumLayers {
		l := &w.Layers[i]
		p := fmt.Sprintf("model.layers.%d.", i)
		if l.PreAttnNorm, err = st.TensorF32(p+"attention_norm.weight", hidden); err != nil {
			return nil, err
		}
		if l.PreMLPNorm, err = st.TensorF32(p+"ffn_norm.weight", hidden); err != nil {
			return nil, err
		}

		// The grouped de-interleave. Read the fused tensor once as f32 and gather.
		qkv, qerr := st.TensorF32(p+"attention.wqkv.weight", nKV*gs*hd, hidden)
		if qerr != nil {
			return nil, fmt.Errorf("decoder(internlm2): wqkv: %w", qerr)
		}
		q := make([]float32, qDim*hidden)
		k := make([]float32, kvDim*hidden)
		v := make([]float32, kvDim*hidden)
		for g := range nKV {
			base := g * gs * hd * hidden
			copy(q[g*groups*hd*hidden:], qkv[base:base+groups*hd*hidden])
			copy(k[g*hd*hidden:], qkv[base+groups*hd*hidden:base+(groups+1)*hd*hidden])
			copy(v[g*hd*hidden:], qkv[base+(groups+1)*hd*hidden:base+gs*hd*hidden])
		}
		l.QProj = quantizeWM(linalg.WrapF32(q, qDim, hidden), matmulQuant(quant, "q"))
		l.KProj = quantizeWM(linalg.WrapF32(k, kvDim, hidden), matmulQuant(quant, "k"))
		l.VProj = quantizeWM(linalg.WrapF32(v, kvDim, hidden), matmulQuant(quant, "v"))

		om, oerr := loadMat(st, p+"attention.wo.weight", hidden, qDim)
		if oerr != nil {
			return nil, oerr
		}
		l.OProj = quantizeWM(om, matmulQuant(quant, "o"))

		// w1 = gate, w3 = up, w2 = down (llama's original names).
		for _, m := range []struct {
			dst     *linalg.WeightMat
			name    string
			out, in int
		}{
			{&l.GateProj, "feed_forward.w1.weight", inter, hidden},
			{&l.UpProj, "feed_forward.w3.weight", inter, hidden},
			{&l.DownProj, "feed_forward.w2.weight", hidden, inter},
		} {
			mm, merr := loadMat(st, p+m.name, m.out, m.in)
			if merr != nil {
				return nil, merr
			}
			*m.dst = quantizeWM(mm, matmulQuant(quant, m.name))
		}
	}
	return w, nil
}
