package decoder

import (
	"errors"
	"fmt"
)

// gpt-oss: sparse MoE on every layer with per-head attention sinks, per-expert biases and a router-logit bias. q/k/v/o all
// carry biases; no QK-norm. Attention/embeddings/router quantize per policy; the (F32) sinks and biases load raw. The
// expert weights are MXFP4 (ggml type 39) in the real checkpoint; stackedExperts routes them through aikit's
// RowDequantizer.
func (b *ggufBuild) loadGptOss() (*Weights, error) {
	arch, g, w, sink, abort, hidden, hd, streamMat, mat, vec, vnorm, stackedExperts := b.arch, b.g, b.w, b.sink, b.abort, b.hidden, b.hd, b.streamMat, b.mat, b.vec, b.vnorm, b.stackedExperts
	nH, nKV := arch.NumHeads, arch.NumKVHeads
	qDim, kvDim := nH*hd, nKV*hd
	nExp := arch.MoE.NumExperts
	expInter := arch.MoE.IntermediateDim
	// stackedExpertBias slices a 2-D [out, nExpert] (fastest-first) f32 bias
	// tensor into one [out] slice per expert (expert e occupies [e*out, e*out+out)).
	stackedExpertBias := func(name string, out int) ([][]float32, error) {
		dims, data, derr := g.Tensor(name)
		if derr != nil {
			return nil, derr
		}
		if len(dims) != 2 || dims[0] != out || dims[1] != nExp {
			return nil, fmt.Errorf("decoder(gguf-gptoss): %q dims %v, want [out=%d, experts=%d]", name, dims, out, nExp)
		}
		res := make([][]float32, nExp)
		for e := range nExp {
			res[e] = append([]float32(nil), data[e*out:(e+1)*out]...)
		}
		return res, nil
	}
	loadGptOss := func(i int) error {
		l := &w.Layers[i]
		p := fmt.Sprintf("blk.%d.", i)
		var e error
		if l.PreAttnNorm, e = vnorm(p+"attn_norm.weight", hidden); e != nil {
			return e
		}
		if l.PreMLPNorm, e = vnorm(p+"post_attention_norm.weight", hidden); e != nil {
			return e
		}
		// Attention: q/k/v/o projections + biases. gpt-oss GGUF stores q/k in HF
		// order (no undo_permute), so a plain mat load — no permMat.
		if l.QProj, e = mat(p+"attn_q.weight", qDim, hidden); e != nil {
			return e
		}
		if l.KProj, e = mat(p+"attn_k.weight", kvDim, hidden); e != nil {
			return e
		}
		if l.VProj, e = mat(p+"attn_v.weight", kvDim, hidden); e != nil {
			return e
		}
		if l.OProj, e = mat(p+"attn_output.weight", hidden, qDim); e != nil {
			return e
		}
		if l.QBias, e = vec(p+"attn_q.bias", qDim); e != nil {
			return e
		}
		if l.KBias, e = vec(p+"attn_k.bias", kvDim); e != nil {
			return e
		}
		if l.VBias, e = vec(p+"attn_v.bias", kvDim); e != nil {
			return e
		}
		if l.OBias, e = vec(p+"attn_output.bias", hidden); e != nil {
			return e
		}
		if l.AttnSinks, e = vec(p+"attn_sinks.weight", nH); e != nil {
			return e
		}
		// MoE: router (+ logit bias) + stacked routed experts (+ per-expert biases). Router stays f32 regardless of the ambient
		// quant mode (matching gptoss_safetensors.go and qwen35's streamMat(..., quantNone, ...) below): top-k selection is
		// discrete, so quantizing it flips which experts win rather than adding rounding noise. Plain mat() would quantize it.
		if l.Router, e = streamMat(p+"ffn_gate_inp.weight", nExp, hidden, quantNone, func(r int) int { return r * hidden }); e != nil {
			return e
		}
		if l.RouterBias, e = vec(p+"ffn_gate_inp.bias", nExp); e != nil {
			return e
		}
		gate, ge := stackedExperts(p+"ffn_gate_exps.weight", expInter, hidden, nExp)
		up, ue := stackedExperts(p+"ffn_up_exps.weight", expInter, hidden, nExp)
		down, de := stackedExperts(p+"ffn_down_exps.weight", hidden, expInter, nExp)
		if ge != nil || ue != nil || de != nil {
			return fmt.Errorf("decoder(gguf-gptoss): experts layer %d: %w", i, errors.Join(ge, ue, de))
		}
		gb, gbe := stackedExpertBias(p+"ffn_gate_exps.bias", expInter)
		ub, ube := stackedExpertBias(p+"ffn_up_exps.bias", expInter)
		db, dbe := stackedExpertBias(p+"ffn_down_exps.bias", hidden)
		if gbe != nil || ube != nil || dbe != nil {
			return fmt.Errorf("decoder(gguf-gptoss): expert biases layer %d: %w", i, errors.Join(gbe, ube, dbe))
		}
		l.Experts = make([]expertWeights, nExp)
		for ei := range nExp {
			l.Experts[ei] = expertWeights{
				Gate: gate[ei], Up: up[ei], Down: down[ei],
				GateBias: gb[ei], UpBias: ub[ei], DownBias: db[ei],
			}
		}
		return nil
	}
	// Streaming (sink != nil): loadGptOss builds one layer's data independently of every other layer (every tensor it reads is
	// named blk.{i}.*, including the per-layer AttnSinks and RouterBias; stackedExperts and stackedExpertBias are per-tensor
	// RowDequantizer reads keyed by the same name), so the loadQ35 shape applies: streamLayers' build, write and release
	// bounds peak RSS to about one window. Non-streaming (sink == nil): all layers in parallel.
	if sink != nil {
		if err := streamLayers(w, arch, sink, loadGptOss); err != nil {
			return nil, err
		}
		return w, nil
	}
	if err := parallelLayers(arch.NumLayers, abort, loadGptOss); err != nil {
		return nil, err
	}
	return w, nil
}
