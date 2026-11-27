package inkling2

import (
	"fmt"

	"github.com/townsendmerino/aikit/linalg"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
)

// MatTensor is a GGUF matrix of a layer, [Out, In], and the WeightMat it loads into.
type MatTensor struct {
	Dst     *linalg.WeightMat
	Name    string
	Out, In int
}

// VecTensor is an F32 GGUF tensor of N elements, of any rank, and the slice it loads into.
type VecTensor struct {
	Dst  *[]float32
	Name string
	N    int
}

// LayerTensors lists physical layer i's tensors under llama.cpp's names (conversion/inkling2.py), prefix included, and where
// each loads in lw, which it gives an Inkling2LayerWeights. The router and the experts are the caller's.
func LayerTensors(a *darch.Architecture, lw *darch.LayerWeights, i int) ([]MatTensor, []VecTensor) {
	p := a.Inkling2
	x := &archparts.Inkling2LayerWeights{}
	lw.Inkling = x
	pf := fmt.Sprintf("blk.%d.", i)
	hidden, nH, nKV, K := a.HiddenDim, a.NumHeads, a.NumKVHeads, p.ConvK
	qDim, kvDim := nH*a.HeadDim, nKV*a.HeadDim
	swa := p.SWA[i]
	extent := p.RelExtent
	if swa {
		extent = a.SlidingWindow
	}
	mats := []MatTensor{
		{&lw.QProj, pf + "attn_q.weight", qDim, hidden},
		{&lw.KProj, pf + "attn_k.weight", kvDim, hidden},
		{&lw.VProj, pf + "attn_v.weight", kvDim, hidden},
		{&lw.OProj, pf + "attn_output.weight", hidden, qDim},
		{&lw.GProj, pf + "attn_gate.weight", qDim, hidden},
		{&x.SinkProj, pf + "attn_sink_proj.weight", nH, hidden},
		{&x.Rel, pf + "attn_rel.weight", nH * p.DRel, hidden},
	}
	vecs := []VecTensor{
		{&x.QClip, pf + "attn_q_clip.weight", qDim},
		{&x.RelClip, pf + "attn_rel_clip.weight", nH * p.DRel},
		{&x.RelBias, pf + "attn_rel_bias.weight", extent * p.DRel},
		{&x.KConv, pf + "attn_k_conv.weight", kvDim * K},
		{&x.VConv, pf + "attn_v_conv.weight", kvDim * K},
		{&x.AttnConv, pf + "attn_post_conv.weight", hidden * K},
		{&x.FFConv, pf + "ffn_post_conv.weight", hidden * K},
		{&x.FFNResQuery, pf + "ffn_res_query.weight", hidden},
		{&lw.PreMLPNorm, pf + "ffn_norm.weight", hidden},
		{&lw.RouterBias, pf + "exp_probs_b.bias", a.MoE.NumExperts},
	}
	nq := 1
	if swa {
		vecs = append(vecs, VecTensor{&lw.PreAttnNorm, pf + "attn_norm.weight", hidden})
	} else {
		nq = 3
		mats = append(mats, MatTensor{&x.KConvProj, pf + "attn_k_conv_proj.weight", nKV * K, hidden},
			MatTensor{&x.VConvProj, pf + "attn_v_conv_proj.weight", nKV * K, hidden})
		vecs = append(vecs, VecTensor{&lw.PreAttnNorm, pf + "attn_q_inp_norm.weight", hidden},
			VecTensor{&x.KInpNorm, pf + "attn_k_inp_norm.weight", hidden}, VecTensor{&x.VInpNorm, pf + "attn_v_inp_norm.weight", hidden})
	}
	if i > 0 {
		vecs = append(vecs, VecTensor{&x.AttnResQuery, pf + "attn_res_query.weight", nq * hidden})
	}
	if p.LoopPasses > 1 && i >= p.LoopStart && i < p.LoopEnd {
		vecs = append(vecs, VecTensor{&x.MsgQuery, pf + "loop_msg_query.weight", 4 * hidden})
		for c, kind := range []string{"q", "k", "v", "ffn"} {
			ct := &x.Ctrl[c]
			vecs = append(vecs, VecTensor{&ct.Down, pf + "loop_" + kind + "_ctrl_down.weight", p.GateRank * 2 * hidden},
				VecTensor{&ct.Gain, pf + "loop_" + kind + "_ctrl_gain.weight", hidden * p.GateRank},
				VecTensor{&ct.Forget, pf + "loop_" + kind + "_ctrl_forget.weight", hidden * p.GateRank})
		}
	}
	return mats, vecs
}
