package decoder

import (
	"fmt"
	"math"
)

// Nemotron-H (nemotron_h): single-op-per-block hybrid. Each layer is one of
// mamba / attention (NoPE — no q/k permute) / non-gated mlp, under a single
// attn_norm. The ssm_* tensors reuse llama.cpp's conventions (ssm_a = −exp(A_log),
// conv [convDim,K], grouped ssm_norm). Mixer f32; attention/mlp/embeddings quantize.
func (b *ggufBuild) loadNemotron(npar *nemotronParams) (*Weights, error) {
	arch, g, w, quant, sink, abort, hidden, streamMat, mat, vnorm, stackedExperts := b.arch, b.g, b.w, b.quant, b.sink, b.abort, b.hidden, b.streamMat, b.mat, b.vnorm, b.stackedExperts
	dInner := npar.NHeads * npar.HeadDim
	convDim := dInner + 2*npar.NGroups*npar.DState
	projDim := 2*dInner + 2*npar.NGroups*npar.DState + npar.NHeads
	hd := arch.HeadDim
	qDim, kvDim := arch.NumHeads*hd, arch.NumKVHeads*hd
	inter := arch.IntermediateDim
	f32mat := func(name string, out, in int) ([]float32, error) {
		dims, data, derr := g.Tensor(name)
		if derr != nil {
			return nil, derr
		}
		if len(dims) != 2 || dims[0] != in || dims[1] != out {
			return nil, fmt.Errorf("decoder(gguf-nemotron): %q dims %v, want [in=%d, out=%d]", name, dims, in, out)
		}
		return data, nil
	}
	flat := func(name string, n int) ([]float32, error) {
		_, data, derr := g.Tensor(name)
		if derr != nil {
			return nil, derr
		}
		if len(data) != n {
			return nil, fmt.Errorf("decoder(gguf-nemotron): %q has %d elems, want %d", name, len(data), n)
		}
		return data, nil
	}
	loadNemo := func(i int) error {
		l := &w.Layers[i]
		p := fmt.Sprintf("blk.%d.", i)
		var e error
		if l.PreAttnNorm, e = vnorm(p+"attn_norm.weight", hidden); e != nil {
			return e
		}
		switch npar.blockKind[i] {
		case nemoMamba:
			mw := &mamba2Weights{}
			if mw.inProj, e = f32mat(p+"ssm_in.weight", projDim, hidden); e != nil {
				return e
			}
			if mw.convW, e = f32mat(p+"ssm_conv1d.weight", convDim, npar.DConv); e != nil {
				return e
			}
			if mw.convB, e = flat(p+"ssm_conv1d.bias", convDim); e != nil {
				return e
			}
			if mw.d, e = flat(p+"ssm_d", npar.NHeads); e != nil {
				return e
			}
			if mw.dtBias, e = flat(p+"ssm_dt.bias", npar.NHeads); e != nil {
				return e
			}
			if mw.normW, e = flat(p+"ssm_norm.weight", dInner); e != nil {
				return e
			}
			if mw.outProj, e = f32mat(p+"ssm_out.weight", hidden, dInner); e != nil {
				return e
			}
			a, aerr := flat(p+"ssm_a", npar.NHeads)
			if aerr != nil {
				return aerr
			}
			mw.aLog = make([]float32, npar.NHeads)
			for h := range a {
				mw.aLog[h] = float32(math.Log(-float64(a[h])))
			}
			if mambaProjInt8(quant) {
				mw.quantizeProjections(projDim, hidden, dInner)
			}
			l.mamba = mw
		case nemoAttn:
			// NoPE → no q/k permute.
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
		case nemoMLP:
			if l.UpProj, e = mat(p+"ffn_up.weight", inter, hidden); e != nil {
				return e
			}
			if l.DownProj, e = mat(p+"ffn_down.weight", hidden, inter); e != nil {
				return e
			}
		case nemoMoE:
			// Nemotron 3 Nano's MoE FFN. The tensor names are the GGUF's own, not llama.cpp's conversion-script source mapping (which
			// names the source safetensors tensor, not the output GGUF tensor). Experts are fused per projection (one 3-D
			// [in,out,nExpert] tensor each), unlike the safetensors path's one tensor per expert, so stackedExperts, the helper other
			// GGUF MoE families use for this shape, applies. exp_probs_b.bias is llama.cpp's name for e_score_correction_bias.
			moe := arch.MoE
			// Router stays f32; see the Laguna site above.
			if l.Router, e = streamMat(p+"ffn_gate_inp.weight", moe.NumExperts, hidden, quantNone, func(r int) int { return r * hidden }); e != nil {
				return e
			}
			if l.RouterBias, e = flat(p+"exp_probs_b.bias", moe.NumExperts); e != nil {
				return e
			}
			upExp, uerr := stackedExperts(p+"ffn_up_exps.weight", moe.IntermediateDim, hidden, moe.NumExperts)
			if uerr != nil {
				return uerr
			}
			downExp, derr := stackedExperts(p+"ffn_down_exps.weight", hidden, moe.IntermediateDim, moe.NumExperts)
			if derr != nil {
				return derr
			}
			l.Experts = make([]expertWeights, moe.NumExperts)
			for ei := range l.Experts {
				l.Experts[ei].Up, l.Experts[ei].Down = upExp[ei], downExp[ei]
			}
			if moe.SharedIntermediateDim > 0 {
				if l.SharedExpert.Up, e = mat(p+"ffn_up_shexp.weight", moe.SharedIntermediateDim, hidden); e != nil {
					return e
				}
				if l.SharedExpert.Down, e = mat(p+"ffn_down_shexp.weight", hidden, moe.SharedIntermediateDim); e != nil {
					return e
				}
			}
		}
		return nil
	}
	// Streaming (sink != nil): loadNemo is self-contained per layer (every tensor it reads is blk.{i}.*-named;
	// decoder/gguf_streaming_shape_test.go proves this from source and keeps it true), so the loadQ35/loadGptOss streaming
	// shape applies unchanged.
	if sink != nil {
		if err := streamLayers(w, arch, sink, loadNemo); err != nil {
			return nil, err
		}
		return w, nil
	}
	if err := parallelLayers(arch.NumLayers, abort, loadNemo); err != nil {
		return nil, err
	}
	return w, nil
}
