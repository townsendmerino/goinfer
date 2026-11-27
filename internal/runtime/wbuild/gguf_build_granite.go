package wbuild

import (
	"errors"
	"fmt"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"math"
)

// Granite-4.0-H (granitehybrid): per-layer Mamba-2 mixer or GQA attention, MoE on every layer. The Mamba-2 ssm_*
// tensors use llama.cpp's conventions (shared with qwen35): ssm_a stores -exp(A_log) directly (reversed to A_log via
// log(-a) so the shared mamba2Step works), conv1d is [convDim, K], ssm_norm raw. Mixer stays f32 (parity-first);
// experts/attention/embeddings quantize. NEOX rope, so no q/k permute.
func (b *ggufBuild) loadGranite(gp *archparts.GraniteParams) (*darch.Weights, error) {
	arch, g, w, sink, abort, hidden, streamMat, mat, vnorm, stackedExperts, permMat := b.arch, b.g, b.w, b.sink, b.abort, b.hidden, b.streamMat, b.mat, b.vnorm, b.stackedExperts, b.permMat
	dInner := gp.NHeads * gp.HeadDim
	convDim := dInner + 2*gp.NGroups*gp.DState
	projDim := 2*dInner + 2*gp.NGroups*gp.DState + gp.NHeads
	hd := arch.HeadDim
	qDim, kvDim := arch.NumHeads*hd, arch.NumKVHeads*hd
	f32mat := func(name string, out, in int) ([]float32, error) {
		defer b.release(name)
		dims, data, derr := g.Tensor(name)
		if derr != nil {
			return nil, derr
		}
		if len(dims) != 2 || dims[0] != in || dims[1] != out {
			return nil, fmt.Errorf("decoder(gguf-granite): %q dims %v, want [in=%d, out=%d]", name, dims, in, out)
		}
		return data, nil
	}
	flat := func(name string, n int) ([]float32, error) { // 1-D or [1,n]/[n,1] tensor → n floats
		defer b.release(name)
		_, data, derr := g.Tensor(name)
		if derr != nil {
			return nil, derr
		}
		if len(data) != n {
			return nil, fmt.Errorf("decoder(gguf-granite): %q has %d elems, want %d", name, len(data), n)
		}
		return data, nil
	}
	loadGranite := func(i int) error {
		l := &w.Layers[i]
		p := fmt.Sprintf("blk.%d.", i)
		var e error
		if l.PreAttnNorm, e = vnorm(p+"attn_norm.weight", hidden); e != nil {
			return e
		}
		if l.PreMLPNorm, e = vnorm(p+"ffn_norm.weight", hidden); e != nil {
			return e
		}
		if arch.IsMambaLayer(i) {
			mw := &archparts.Mamba2Weights{}
			if mw.InProj, e = f32mat(p+"ssm_in.weight", projDim, hidden); e != nil {
				return e
			}
			if mw.ConvW, e = f32mat(p+"ssm_conv1d.weight", convDim, gp.DConv); e != nil {
				return e
			}
			if mw.ConvB, e = flat(p+"ssm_conv1d.bias", convDim); e != nil {
				return e
			}
			if mw.D, e = flat(p+"ssm_d", gp.NHeads); e != nil {
				return e
			}
			if mw.DtBias, e = flat(p+"ssm_dt.bias", gp.NHeads); e != nil {
				return e
			}
			if mw.NormW, e = flat(p+"ssm_norm.weight", dInner); e != nil {
				return e
			}
			if mw.OutProj, e = f32mat(p+"ssm_out.weight", hidden, dInner); e != nil {
				return e
			}
			// ssm_a is −exp(A_log); reverse to A_log so mamba2Step's −exp(aLog) recovers it.
			a, aerr := flat(p+"ssm_a", gp.NHeads)
			if aerr != nil {
				return aerr
			}
			mw.ALog = make([]float32, gp.NHeads)
			for h := range a {
				mw.ALog[h] = float32(math.Log(-float64(a[h])))
			}
			l.Mamba = mw
		} else {
			// llama.cpp converts GraniteHybrid attention with undo_permute=True, so
			// q/k are RoPE-permuted in the GGUF (like Llama) and must be un-permuted.
			if l.QProj, e = permMat(p+"attn_q.weight", qDim, hidden, arch.NumHeads); e != nil {
				return e
			}
			if l.KProj, e = permMat(p+"attn_k.weight", kvDim, hidden, arch.NumKVHeads); e != nil {
				return e
			}
			if l.VProj, e = mat(p+"attn_v.weight", kvDim, hidden); e != nil {
				return e
			}
			if l.OProj, e = mat(p+"attn_output.weight", hidden, qDim); e != nil {
				return e
			}
		}
		// MoE on every layer: router + stacked routed experts + ungated shared.
		expInter := arch.MoE.IntermediateDim
		// Router stays f32; see the Laguna site above.
		if l.Router, e = streamMat(p+"ffn_gate_inp.weight", arch.MoE.NumExperts, hidden, QuantNone, func(r int) int { return r * hidden }); e != nil {
			return e
		}
		gate, ge := stackedExperts(p+"ffn_gate_exps.weight", expInter, hidden, arch.MoE.NumExperts)
		up, ue := stackedExperts(p+"ffn_up_exps.weight", expInter, hidden, arch.MoE.NumExperts)
		down, de := stackedExperts(p+"ffn_down_exps.weight", hidden, expInter, arch.MoE.NumExperts)
		if ge != nil || ue != nil || de != nil {
			return fmt.Errorf("decoder(gguf-granite): experts layer %d: %w", i, errors.Join(ge, ue, de))
		}
		l.Experts = make([]archparts.ExpertWeights, arch.MoE.NumExperts)
		for ei := range arch.MoE.NumExperts {
			l.Experts[ei] = archparts.ExpertWeights{Gate: gate[ei], Up: up[ei], Down: down[ei]}
		}
		sInter := arch.MoE.SharedIntermediateDim
		if l.SharedExpert.Gate, e = mat(p+"ffn_gate_shexp.weight", sInter, hidden); e != nil {
			return e
		}
		if l.SharedExpert.Up, e = mat(p+"ffn_up_shexp.weight", sInter, hidden); e != nil {
			return e
		}
		if l.SharedExpert.Down, e = mat(p+"ffn_down_shexp.weight", hidden, sInter); e != nil {
			return e
		}
		return nil
	}
	// Streaming (sink != nil): loadGranite is self-contained per layer (every tensor it reads is blk.{i}.*-named;
	// decoder/gguf_streaming_shape_test.go proves this from source and keeps it true), so the loadQ35/loadGptOss streaming
	// shape applies unchanged.
	if sink != nil {
		if err := StreamLayers(w, arch, sink, loadGranite); err != nil {
			return nil, err
		}
		return w, nil
	}
	if err := ParallelLayers(arch.NumLayers, abort, loadGranite); err != nil {
		return nil, err
	}
	return w, nil
}
