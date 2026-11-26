package decoder

import (
	"errors"
	"fmt"

	"github.com/townsendmerino/aikit/linalg"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
)

// loadInkling2 loads an Inkling 2 GGUF (llama.cpp's inkling2 / inkling2_loop tensor names). It loads each physical layer
// once and gives every visit (Architecture.NumLayers) a copy of its physical layer's LayerWeights, which share the tensors.
// The fused expert gate_up is split into Gate (its first half of rows) and Up at load. It has no streaming (.giw) path.
func (b *ggufBuild) loadInkling2(p *archparts.Inkling2Params) (*darch.Weights, error) {
	arch, g, w, hidden, hd, mat, vec, vnorm, stackedExperts, streamMat := b.arch, b.g, b.w, b.hidden, b.hd, b.mat, b.vec, b.vnorm, b.stackedExperts, b.streamMat
	if b.sink != nil {
		return nil, fmt.Errorf("decoder(gguf-inkling2): no .giw build for this family yet; load the GGUF directly (-direct-load)")
	}
	nH, nKV, K := arch.NumHeads, arch.NumKVHeads, p.ConvK
	qDim, kvDim := nH*hd, nKV*hd
	moe := arch.MoE
	f32 := func(name string, n int) ([]float32, error) { // any-rank F32 tensor of n elements
		_, data, err := g.Tensor(name)
		if err != nil {
			return nil, err
		}
		if len(data) != n {
			return nil, fmt.Errorf("decoder(gguf-inkling2): %q has %d elements, want %d", name, len(data), n)
		}
		return data, nil
	}
	head := &archparts.Inkling2HeadWeights{}
	var err error
	if head.EmbedNorm, err = vnorm("token_embd_norm.weight", hidden); err != nil {
		return nil, err
	}
	if head.OutputResQuery, err = vec("output_res_query.weight", hidden); err != nil {
		return nil, err
	}
	arch.Inkling2Head = head

	phys := make([]darch.LayerWeights, p.PhysLayers)
	loadLayer := func(i int) error {
		l := &phys[i]
		x := &archparts.Inkling2LayerWeights{}
		l.Inkling = x
		pf := fmt.Sprintf("blk.%d.", i)
		swa := p.SWA[i]
		extent := p.RelExtent
		if swa {
			extent = arch.SlidingWindow
		}
		var e error
		load := func(dst *linalg.WeightMat, name string, out, in int) {
			if e == nil {
				*dst, e = mat(pf+name, out, in)
			}
		}
		load(&l.QProj, "attn_q.weight", qDim, hidden)
		load(&l.KProj, "attn_k.weight", kvDim, hidden)
		load(&l.VProj, "attn_v.weight", kvDim, hidden)
		load(&l.OProj, "attn_output.weight", hidden, qDim)
		load(&l.GProj, "attn_gate.weight", qDim, hidden)
		load(&x.SinkProj, "attn_sink_proj.weight", nH, hidden)
		load(&x.Rel, "attn_rel.weight", nH*p.DRel, hidden)
		if !swa {
			load(&x.KConvProj, "attn_k_conv_proj.weight", nKV*K, hidden)
			load(&x.VConvProj, "attn_v_conv_proj.weight", nKV*K, hidden)
		}
		if e != nil {
			return e
		}
		vecs := []f32Tensor{
			{&x.QClip, "attn_q_clip.weight", qDim},
			{&x.RelClip, "attn_rel_clip.weight", nH * p.DRel},
			{&x.RelBias, "attn_rel_bias.weight", extent * p.DRel},
			{&x.KConv, "attn_k_conv.weight", kvDim * K},
			{&x.VConv, "attn_v_conv.weight", kvDim * K},
			{&x.AttnConv, "attn_post_conv.weight", hidden * K},
			{&x.FFConv, "ffn_post_conv.weight", hidden * K},
			{&x.FFNResQuery, "ffn_res_query.weight", hidden},
			{&l.PreMLPNorm, "ffn_norm.weight", hidden},
			{&l.RouterBias, "exp_probs_b.bias", moe.NumExperts},
		}
		if swa {
			vecs = append(vecs, f32Tensor{&l.PreAttnNorm, "attn_norm.weight", hidden})
		} else {
			vecs = append(vecs, []f32Tensor{{&l.PreAttnNorm, "attn_q_inp_norm.weight", hidden}, {&x.KInpNorm, "attn_k_inp_norm.weight", hidden}, {&x.VInpNorm, "attn_v_inp_norm.weight", hidden}}...)
		}
		if i > 0 {
			nq := 1
			if !swa {
				nq = 3
			}
			vecs = append(vecs, f32Tensor{&x.AttnResQuery, "attn_res_query.weight", nq * hidden})
		}
		if p.LoopPasses > 1 && i >= p.LoopStart && i < p.LoopEnd {
			vecs = append(vecs, f32Tensor{&x.MsgQuery, "loop_msg_query.weight", 4 * hidden})
			for c, kind := range []string{"q", "k", "v", "ffn"} {
				ct := &x.Ctrl[c]
				vecs = append(vecs, []f32Tensor{
					{&ct.Down, "loop_" + kind + "_ctrl_down.weight", p.GateRank * 2 * hidden},
					{&ct.Gain, "loop_" + kind + "_ctrl_gain.weight", hidden * p.GateRank},
					{&ct.Forget, "loop_" + kind + "_ctrl_forget.weight", hidden * p.GateRank},
				}...)
			}
		}
		for _, v := range vecs {
			if *v.dst, e = f32(pf+v.name, v.n); e != nil {
				return e
			}
		}
		// The router stays f32: top-k selection is discrete, so quantizing it flips which experts win.
		if l.Router, e = streamMat(pf+"ffn_gate_inp.weight", moe.NumExperts, hidden, quantNone, func(r int) int { return r * hidden }); e != nil {
			return e
		}
		inter := moe.IntermediateDim
		gate, up, ge := b.splitGateUpExperts(pf+"ffn_gate_up_exps.weight", inter, hidden, moe.NumExperts)
		down, de := stackedExperts(pf+"ffn_down_exps.weight", hidden, inter, moe.NumExperts)
		if ge != nil || de != nil {
			return fmt.Errorf("decoder(gguf-inkling2): experts of layer %d: %w", i, errors.Join(ge, de))
		}
		l.Experts = make([]archparts.ExpertWeights, moe.NumExperts)
		for ei := range moe.NumExperts {
			l.Experts[ei] = archparts.ExpertWeights{Gate: gate[ei], Up: up[ei], Down: down[ei]}
		}
		return nil
	}
	if err := parallelLayers(p.PhysLayers, b.abort, loadLayer); err != nil {
		return nil, err
	}
	for v, vis := range p.Visits() {
		w.Layers[v] = phys[vis.Phys]
	}
	return w, nil
}

// splitGateUpExperts loads a fused [in, 2*inter, nExpert] expert tensor whose first inter rows of each expert are the gate
// and the next inter the up projection, as one Gate and one Up WeightMat per expert.
func (b *ggufBuild) splitGateUpExperts(name string, inter, in, nExpert int) (gate, up []linalg.WeightMat, err error) {
	dims, into, err := ggufScaledRowDequantizer(b.g, name)
	if err != nil {
		return nil, nil, err
	}
	if len(dims) != 3 || dims[0] != in || dims[1] != 2*inter || dims[2] != nExpert {
		return nil, nil, fmt.Errorf("decoder(gguf): %q dims %v, want [in=%d, out=%d, experts=%d]", name, dims, in, 2*inter, nExpert)
	}
	gate, up = make([]linalg.WeightMat, nExpert), make([]linalg.WeightMat, nExpert)
	mode := matmulQuant(b.quant, name)
	one := func(e int) error {
		for half, dst := range []*linalg.WeightMat{&gate[e], &up[e]} {
			base := (e*2 + half) * inter
			m, err := streamQuantizedLayer(inter, in, mode, b.needCanonical, b.skipRow4, func(r int, row []float32) error {
				return ggufRowFiltered(name, into, (base+r)*in, row)
			})
			if err != nil {
				return err
			}
			*dst = m
		}
		return nil
	}
	return gate, up, forExperts(nExpert, b.sink, one)
}

// f32Tensor names an F32 GGUF tensor of n elements and where it loads.
type f32Tensor struct {
	dst  *[]float32
	name string
	n    int
}

// paddedVocabRows reports whether a table of stored rows is an Inkling 2 embedding or head padded past the vocabulary of out
// rows: the loader then reads its first out rows.
func paddedVocabRows(arch *darch.Architecture, stored, out int) bool {
	return arch.Inkling2 != nil && stored == arch.Inkling2.PaddedVocab && out < stored
}
