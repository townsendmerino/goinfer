package decoder

import (
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/inkling2"
	"github.com/townsendmerino/goinfer/internal/runtime/normact"
)

// inkling2Token is one position's state across its Inkling 2 layer visits: the attention-residual blocks completed so far
// (raw and unit-RMS-normalized), and on a loop model the prelude's output and the previous pass's normalized bank, which the
// later passes' messages read.
type inkling2Token struct {
	normed   [][]float32 // UnitRMS of each completed block, oldest first
	prelude  []float32   // the partial block after the prelude (the loop's restart point)
	nPrelude int         // len(normed) after the prelude
	bank     [][]float32 // the previous pass's blocks after its first, and its output, normalized
}

// inkling2ConvState is one visit's four short-convolution windows: K, V, the attention output and the FFN output.
type inkling2ConvState struct{ k, v, attn, ffn inkling2.ConvWindow }

func newInkling2ConvState() *inkling2ConvState { return new(inkling2ConvState) }

// reset empties the windows.
func (s *inkling2ConvState) reset() {
	s.k.Reset()
	s.v.Reset()
	s.attn.Reset()
	s.ffn.Reset()
}

// unitRMSCopy returns a new UnitRMS of x.
func unitRMSCopy(x []float32) []float32 {
	n := make([]float32, len(x))
	inkling2.UnitRMS(n, x, 1e-6)
	return n
}

// inkling2Visit runs one layer visit on h ([hidden], the partial block, updated in place): the depth mixes that form its
// inputs, the messages on a later pass, the attention and its convolutions, the block boundary, and the SITU MoE.
func (m *Model) inkling2Visit(o *op, h []float32, lw *darch.LayerWeights, cache *KVCache, l, pos int, t *inkling2Token) error {
	a := m.w.Arch
	p := a.Inkling2
	ip := &o.Ink
	x := lw.Inkling
	hidden, nH, nKV, hd := a.HiddenDim, a.NumHeads, a.NumKVHeads, a.HeadDim
	be := m.be
	if p.LoopPasses > 1 && ip.Phys == p.LoopStart {
		if ip.Pass == 0 {
			t.prelude, t.nPrelude = append(t.prelude[:0], h...), len(t.normed)
		} else {
			t.bank = append(append([][]float32(nil), t.normed[t.nPrelude+1:]...), unitRMSCopy(h))
			copy(h, t.prelude)
			t.normed = t.normed[:t.nPrelude]
		}
	}
	states := append(append(make([][]float32, 0, len(t.normed)+1), t.normed...), unitRMSCopy(h))

	// The attention inputs: one depth mix on a sliding layer, three (q, k, v) on a full one; the raw partial block on layer 0.
	var in [3][]float32
	if q := x.AttnResQuery; q != nil {
		nq := len(q) / hidden
		mixed := make([]float32, nq*hidden)
		inkling2.Mix(mixed, q, nq, states, hidden)
		for j := range in {
			s := 0
			if nq == 3 {
				s = j
			}
			in[j] = append([]float32(nil), mixed[s*hidden:(s+1)*hidden]...)
		}
	} else {
		for j := range in {
			in[j] = append([]float32(nil), h...)
		}
	}
	var msgs []float32
	if ip.Pass > 0 {
		msgs = make([]float32, 4*hidden)
		inkling2.Mix(msgs, x.MsgQuery, 4, t.bank, hidden)
		for j := range in {
			c := &x.Ctrl[j]
			inkling2.Control(in[j], msgs[j*hidden:(j+1)*hidden], c.Down, c.Gain, c.Forget, p.GateRank, 1e-6)
		}
	}
	norms := [3][]float32{lw.PreAttnNorm, lw.PreAttnNorm, lw.PreAttnNorm}
	if !ip.SWA {
		norms[1], norms[2] = x.KInpNorm, x.VInpNorm
	}
	for j := range in {
		normact.RMS(in[j], norms[j], 1, hidden, ip.Eps, false)
	}
	xq, xk, xv := in[0], in[1], in[2]

	q, k, v := make([]float32, nH*hd), make([]float32, nKV*hd), make([]float32, nKV*hd)
	rel, sinks := make([]float32, nH*p.DRel), make([]float32, nH)
	matmul(be, &lw.QProj, xq, q, 1)
	matmul(be, &lw.KProj, xk, k, 1)
	matmul(be, &lw.VProj, xv, v, 1)
	matmul(be, &x.Rel, xq, rel, 1)
	matmul(be, &x.SinkProj, xq, sinks, 1)
	var dk, dv []float32
	if !ip.SWA {
		dk, dv = make([]float32, nKV*ip.ConvK), make([]float32, nKV*ip.ConvK)
		matmul(be, &x.KConvProj, xk, dk, 1)
		matmul(be, &x.VConvProj, xv, dv, 1)
	}
	conv := cache.ink[l]
	inkling2.Clip(q, x.QClip, hd, ip.QKClip)
	inkling2.Conv(k, x.KConv, ip.ConvK, dk, &conv.k)
	inkling2.Clip(k, nil, hd, ip.QKClip)
	inkling2.Conv(v, x.VConv, ip.ConvK, dv, &conv.v)
	for g := range nKV {
		inkling2.UnitRMS(v[g*hd:(g+1)*hd], v[g*hd:(g+1)*hd], ip.Eps)
	}
	inkling2.Clip(rel, x.RelClip, p.DRel, ip.RelClip)
	if !ip.SWA {
		tau := float32(inkling2.LogScale(pos, ip.LogFloor, ip.LogAlpha))
		for i := range q {
			q[i] *= tau
		}
		for i := range rel {
			rel[i] *= tau
		}
	}

	cache.Append(l, k, v)
	kvDim := nKV * hd
	keys, vals, stride := cache.layerRows(l, kvDim)
	nKeys := cache.storedRows(l, kvDim)
	ctx := make([]float32, nH*hd)
	inkling2.Attend(inkling2.AttendParams{Heads: nH, KVHeads: nKV, HeadDim: hd, DRel: p.DRel, Extent: ip.Extent, Window: ip.Window, Scale: a.AttnScale},
		ctx, q, keys, vals, stride, nKeys, 0, pos, rel, x.RelBias, sinks)
	gate := make([]float32, nH*hd)
	matmul(be, &lw.GProj, xq, gate, 1)
	for i := range ctx {
		ctx[i] *= inkling2.Sigmoid(gate[i])
	}
	attn := make([]float32, hidden)
	matmul(be, &lw.OProj, ctx, attn, 1)
	inkling2.Conv(attn, x.AttnConv, ip.ConvK, nil, &conv.attn)

	// A block's first layer starts a new partial block; its input joins the completed blocks.
	res := attn
	if ip.Boundary {
		t.normed = append(t.normed, states[len(states)-1])
	} else {
		for i := range res {
			res[i] += h[i]
		}
	}
	ffIn := make([]float32, hidden)
	inkling2.Mix(ffIn, x.FFNResQuery, 1, append(append(make([][]float32, 0, len(t.normed)+1), t.normed...), unitRMSCopy(res)), hidden)
	if msgs != nil {
		c := &x.Ctrl[3]
		inkling2.Control(ffIn, msgs[3*hidden:], c.Down, c.Gain, c.Forget, p.GateRank, 1e-6)
	}
	normact.RMS(ffIn, lw.PreMLPNorm, 1, hidden, ip.Eps, false)
	ff, err := m.inkling2MoE(ffIn, lw, &o.FFN.MoE, ip, cache.scr)
	if err != nil {
		return err
	}
	inkling2.Conv(ff, x.FFConv, ip.ConvK, nil, &conv.ffn)
	for i := range h {
		h[i] = res[i] + ff[i]
	}
	return nil
}

// inkling2MoE is the routed MoE of an Inkling 2 layer: sigmoid router scores, the correction bias for selection only, the
// top-k weights normalized and scaled, and SITU experts.
func (m *Model) inkling2MoE(x []float32, lw *darch.LayerWeights, moe *MoEConfig, ip *inkling2Params, scr *decodeScratch) ([]float32, error) {
	be := m.be
	logits := make([]float32, moe.NumExperts)
	matmul(be, &lw.Router, x, logits, 1)
	idx, wts := routeExperts(logits, lw.RouterBias, moe.TopK, moe.RouterSigmoid, moe.NormTopKProb, moe.RoutedScale, 0, 0)
	if moeSelTrace != nil {
		moeSelTrace = append(moeSelTrace, append([]int(nil), idx...))
		moeWtsTrace = append(moeWtsTrace, append([]float32(nil), wts...))
	}
	hidden, inter := len(x), moe.IntermediateDim
	out, expOut := make([]float32, hidden), make([]float32, hidden)
	gate, up := make([]float32, inter), make([]float32, inter)
	hq := new(w4a8Act)
	if len(idx) > 0 {
		hq.prepare(be, &lw.Experts[idx[0]].Gate, x, 1)
	}
	for j, e := range idx {
		ex := &lw.Experts[e]
		matmulPre(be, &ex.Gate, hq, x, gate, 1)
		matmulPre(be, &ex.Up, hq, x, up, 1)
		inkling2.Situ(gate, up, ip.SituBeta, ip.SituUp)
		matmul(be, &ex.Down, gate, expOut, 1)
		addScaled(out, expOut, wts[j])
	}
	return out, nil
}

// inkling2Final is the head stage's input: the final depth mix of the completed blocks and the last partial block, in place
// on h.
func (m *Model) inkling2Final(h []float32, t *inkling2Token) {
	hidden := m.w.Arch.HiddenDim
	states := append(append(make([][]float32, 0, len(t.normed)+1), t.normed...), unitRMSCopy(h))
	inkling2.Mix(h, m.w.Arch.Inkling2Head.OutputResQuery, 1, states, hidden)
}
