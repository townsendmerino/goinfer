package wbuild

import (
	"errors"
	"fmt"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
)

// Laguna (poolside). llama.cpp names its tensors the way every other MoE family here is named, so the only genuinely new
// one is blk.N.attn_gate.weight, the softplus output gate. Two shape details drive the rest:
//
//   - qDim is per layer (arch.headsAt(i)): full-attention and sliding layers project different head counts, so
//     attn_q/attn_output differ by layer.
//   - the gate's granularity is read from attn_gate's row count, exactly as the safetensors loader does, because neither
//     format carries a trustworthy declaration of it (XS.2's config says per-element and ships per-head).
//
// Experts are fused and stacked per projection (ffn_*_exps), the shared expert is *_shexp, and the router bias is
// exp_probs_b: all shapes goinfer already reads for GLM/DeepSeek/Granite. Layers below FirstKDense are plain dense FFNs.
func (b *ggufBuild) loadLaguna() (*darch.Weights, error) {
	arch, g, w, sink, abort, hidden, streamMat, mat, vnorm, stackedExperts := b.arch, b.g, b.w, b.sink, b.abort, b.hidden, b.streamMat, b.mat, b.vnorm, b.stackedExperts
	kvDim := arch.NumKVHeads * arch.HeadDim
	// exp_probs_b is 1-D; the shared `mat` helper wants 2-D, so read it as a flat
	// vector the same way the other MoE GGUF loaders do.
	flat := func(name string, n int) ([]float32, error) { // 1-D or [1,n]/[n,1] → n floats
		defer b.release(name)
		_, data, derr := g.Tensor(name)
		if derr != nil {
			return nil, derr
		}
		if len(data) != n {
			return nil, fmt.Errorf("decoder(gguf-laguna): %q has %d elems, want %d", name, len(data), n)
		}
		return data, nil
	}
	loadLaguna := func(i int) error {
		l := &w.Layers[i]
		p := fmt.Sprintf("blk.%d.", i)
		lqDim := arch.HeadsAt(i) * arch.HeadDim
		var e error
		if l.PreAttnNorm, e = vnorm(p+"attn_norm.weight", hidden); e != nil {
			return e
		}
		if l.PreMLPNorm, e = vnorm(p+"ffn_norm.weight", hidden); e != nil {
			return e
		}
		if l.QProj, e = mat(p+"attn_q.weight", lqDim, hidden); e != nil {
			return e
		}
		if l.KProj, e = mat(p+"attn_k.weight", kvDim, hidden); e != nil {
			return e
		}
		if l.VProj, e = mat(p+"attn_v.weight", kvDim, hidden); e != nil {
			return e
		}
		if l.OProj, e = mat(p+"attn_output.weight", hidden, lqDim); e != nil {
			return e
		}
		if l.QNorm, e = vnorm(p+"attn_q_norm.weight", arch.HeadDim); e != nil {
			return e
		}
		if l.KNorm, e = vnorm(p+"attn_k_norm.weight", arch.HeadDim); e != nil {
			return e
		}
		// The gate: per-head (headsAt rows) or per-element (headsAt*headDim). Try the
		// per-head shape first and fall back, so both are accepted and anything else
		// is a hard error rather than a silently mis-shaped gate.
		if l.GProj, e = mat(p+"attn_gate.weight", arch.HeadsAt(i), hidden); e != nil {
			if l.GProj, e = mat(p+"attn_gate.weight", lqDim, hidden); e != nil {
				return fmt.Errorf("decoder(gguf-laguna): layer %d attn_gate is neither per-head [%d,%d] nor per-element [%d,%d]: %w",
					i, arch.HeadsAt(i), hidden, lqDim, hidden, e)
			}
		}
		if i < arch.FirstKDense {
			// Dense prefix layer: a plain SwiGLU at the model's feed_forward_length.
			if l.GateProj, e = mat(p+"ffn_gate.weight", arch.IntermediateDim, hidden); e != nil {
				return e
			}
			if l.UpProj, e = mat(p+"ffn_up.weight", arch.IntermediateDim, hidden); e != nil {
				return e
			}
			if l.DownProj, e = mat(p+"ffn_down.weight", hidden, arch.IntermediateDim); e != nil {
				return e
			}
			return nil
		}
		moe := arch.MoE
		// Router stays f32 regardless of the ambient quant mode (matching qwen35/gptoss above): top-k selection is discrete, so
		// quantizing it flips which experts win rather than adding rounding noise.
		if l.Router, e = streamMat(p+"ffn_gate_inp.weight", moe.NumExperts, hidden, QuantNone, func(r int) int { return r * hidden }); e != nil {
			return e
		}
		// exp_probs_b is llama.cpp's name for e_score_correction_bias.
		if l.RouterBias, e = flat(p+"exp_probs_b.bias", moe.NumExperts); e != nil {
			return e
		}
		expInter := moe.IntermediateDim
		gate, ge := stackedExperts(p+"ffn_gate_exps.weight", expInter, hidden, moe.NumExperts)
		up, ue := stackedExperts(p+"ffn_up_exps.weight", expInter, hidden, moe.NumExperts)
		down, de := stackedExperts(p+"ffn_down_exps.weight", hidden, expInter, moe.NumExperts)
		if ge != nil || ue != nil || de != nil {
			return fmt.Errorf("decoder(gguf-laguna): experts layer %d: %w", i, errors.Join(ge, ue, de))
		}
		l.Experts = make([]archparts.ExpertWeights, moe.NumExperts)
		for ei := range moe.NumExperts {
			l.Experts[ei] = archparts.ExpertWeights{Gate: gate[ei], Up: up[ei], Down: down[ei]}
		}
		if sInter := moe.SharedIntermediateDim; sInter > 0 {
			if l.SharedExpert.Gate, e = mat(p+"ffn_gate_shexp.weight", sInter, hidden); e != nil {
				return e
			}
			if l.SharedExpert.Up, e = mat(p+"ffn_up_shexp.weight", sInter, hidden); e != nil {
				return e
			}
			if l.SharedExpert.Down, e = mat(p+"ffn_down_shexp.weight", hidden, sInter); e != nil {
				return e
			}
		}
		return nil
	}
	// Streaming (sink != nil): loadLaguna is self-contained per layer (every tensor it reads is blk.{i}.*-named;
	// decoder/gguf_streaming_shape_test.go proves this from source and keeps it true), so the loadQ35/loadGptOss streaming
	// shape applies unchanged.
	if sink != nil {
		if err := StreamLayers(w, arch, sink, loadLaguna); err != nil {
			return nil, err
		}
		return w, nil
	}
	if err := ParallelLayers(arch.NumLayers, abort, loadLaguna); err != nil {
		return nil, err
	}
	return w, nil
}
