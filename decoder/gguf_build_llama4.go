package decoder

import (
	"errors"
	"fmt"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
)

func (b *ggufBuild) loadLlama4(lp *archparts.Llama4Params) (*darch.Weights, error) {
	arch, w, sink, abort, hidden, hd, streamMat, mat, vnorm, stackedExperts := b.arch, b.w, b.sink, b.abort, b.hidden, b.hd, b.streamMat, b.mat, b.vnorm, b.stackedExperts
	var err error
	// Llama 4: separate q/k/v/o (no q/k-norm tensors — the L2 QK-norm is
	// parameter-free, applied in the forward), per-layer dense/MoE. MoE = router
	// (NO exp_probs_b — top-1 sigmoid, no bias) + stacked per-expert SwiGLU
	// (ffn_{gate,up,down}_exps, the standard [in,out,nE] layout) + an ungated shared
	// expert. NEOX/interleaved rope ⇒ no q/k permute.
	qDim, kvDim := arch.NumHeads*hd, arch.NumKVHeads*hd
	expInter, denseInter, nE := arch.MoE.IntermediateDim, arch.IntermediateDim, arch.MoE.NumExperts
	loadL4 := func(i int) error {
		l := &w.Layers[i]
		p := fmt.Sprintf("blk.%d.", i)
		var e error
		if l.PreAttnNorm, e = vnorm(p+"attn_norm.weight", hidden); e != nil {
			return e
		}
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
		if l.PreMLPNorm, e = vnorm(p+"ffn_norm.weight", hidden); e != nil {
			return e
		}
		if !lp.IsMoE[i] {
			if l.GateProj, e = mat(p+"ffn_gate.weight", denseInter, hidden); e != nil {
				return e
			}
			if l.UpProj, e = mat(p+"ffn_up.weight", denseInter, hidden); e != nil {
				return e
			}
			if l.DownProj, e = mat(p+"ffn_down.weight", hidden, denseInter); e != nil {
				return e
			}
			return nil
		}
		// Router stays f32; see the Laguna site above.
		if l.Router, e = streamMat(p+"ffn_gate_inp.weight", nE, hidden, quantNone, func(r int) int { return r * hidden }); e != nil {
			return e
		}
		gate, ge := stackedExperts(p+"ffn_gate_exps.weight", expInter, hidden, nE)
		up, ue := stackedExperts(p+"ffn_up_exps.weight", expInter, hidden, nE)
		down, de := stackedExperts(p+"ffn_down_exps.weight", hidden, expInter, nE)
		if ge != nil || ue != nil || de != nil {
			return fmt.Errorf("decoder(gguf-llama4): experts layer %d: %w", i, errors.Join(ge, ue, de))
		}
		l.Experts = make([]archparts.ExpertWeights, nE)
		for x := range l.Experts {
			l.Experts[x] = archparts.ExpertWeights{Gate: gate[x], Up: up[x], Down: down[x]}
		}
		if l.SharedExpert.Gate, e = mat(p+"ffn_gate_shexp.weight", expInter, hidden); e != nil {
			return e
		}
		if l.SharedExpert.Up, e = mat(p+"ffn_up_shexp.weight", expInter, hidden); e != nil {
			return e
		}
		if l.SharedExpert.Down, e = mat(p+"ffn_down_shexp.weight", hidden, expInter); e != nil {
			return e
		}
		return nil
	}
	// Streaming (sink != nil): loadL4 is self-contained per layer (every tensor it reads is blk.{i}.*-named;
	// decoder/gguf_streaming_shape_test.go proves this from source and keeps it true), so the loadQ35/loadGptOss streaming
	// shape applies unchanged.
	if sink != nil {
		if err := streamLayers(w, arch, sink, loadL4); err != nil {
			return nil, err
		}
		return w, nil
	}
	if err = parallelLayers(arch.NumLayers, abort, loadL4); err != nil {
		return nil, err
	}
	return w, nil
}
