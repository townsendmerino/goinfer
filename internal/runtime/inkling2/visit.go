package inkling2

import (
	"math"
	"sort"

	"github.com/townsendmerino/aikit/linalg"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/normact"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// Host is what a visit needs from the engine that owns the weights and the cache: a matrix-vector product on a layer's
// weights, and the visit's KV store, which appends this position's key and value and returns every stored row (positions
// 0..n-1 at stride).
type Host interface {
	MatVec(w *linalg.WeightMat, x, dst []float32)
	AppendKV(visit int, k, v []float32) (keys, vals []float32, stride, n int)
}

// Token is one position's state across its layer visits: the attention-residual blocks completed so far (unit-RMS
// normalized), and on a loop model the prelude's output and the previous pass's normalized bank, which the later passes'
// messages read. The zero value starts a position.
type Token struct {
	normed   [][]float32
	prelude  []float32
	nPrelude int
	bank     [][]float32
}

// ConvState is one visit's four short-convolution windows: K, V, the attention output and the FFN output.
type ConvState struct{ K, V, Attn, FFN [][]float32 }

// Reset empties the windows.
func (s *ConvState) Reset() { s.K, s.V, s.Attn, s.FFN = s.K[:0], s.V[:0], s.Attn[:0], s.FFN[:0] }

func unitRMSCopy(x []float32) []float32 {
	n := make([]float32, len(x))
	UnitRMS(n, x, 1e-6)
	return n
}

// with returns states with one more normalized state appended, in a new slice.
func with(states [][]float32, x []float32) [][]float32 {
	return append(append(make([][]float32, 0, len(states)+1), states...), unitRMSCopy(x))
}

// Visit runs layer visit l (the program's opInkling2 o at position pos) on h ([hidden], the partial block, updated in place):
// the depth mixes that form its inputs, the messages on a later pass, the attention and its convolutions, the block
// boundary, and the SITU MoE. lw is the visit's physical layer's weights.
func Visit(host Host, a *darch.Architecture, lw *darch.LayerWeights, o *program.Op, conv *ConvState, t *Token, l, pos int, h []float32) {
	p, ip, x := a.Inkling2, &o.Ink, lw.Inkling
	hidden, nH, nKV, hd := a.HiddenDim, a.NumHeads, a.NumKVHeads, a.HeadDim
	if p.LoopPasses > 1 && ip.Phys == p.LoopStart {
		if ip.Pass == 0 {
			t.prelude, t.nPrelude = append(t.prelude[:0], h...), len(t.normed)
		} else {
			t.bank = with(t.normed[t.nPrelude+1:], h)
			copy(h, t.prelude)
			t.normed = t.normed[:t.nPrelude]
		}
	}
	states := with(t.normed, h)

	// The attention inputs: one depth mix on a sliding layer, three (q, k, v) on a full one; the raw partial block on layer 0.
	var in [3][]float32
	if q := x.AttnResQuery; q != nil {
		nq := len(q) / hidden
		mixed := make([]float32, nq*hidden)
		Mix(mixed, q, nq, states, hidden)
		for j := range in {
			s := min(j, nq-1)
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
		Mix(msgs, x.MsgQuery, 4, t.bank, hidden)
		for j := range in {
			c := &x.Ctrl[j]
			Control(in[j], msgs[j*hidden:(j+1)*hidden], c.Down, c.Gain, c.Forget, p.GateRank, 1e-6)
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
	host.MatVec(&lw.QProj, xq, q)
	host.MatVec(&lw.KProj, xk, k)
	host.MatVec(&lw.VProj, xv, v)
	host.MatVec(&x.Rel, xq, rel)
	host.MatVec(&x.SinkProj, xq, sinks)
	var dk, dv []float32
	if !ip.SWA {
		dk, dv = make([]float32, nKV*ip.ConvK), make([]float32, nKV*ip.ConvK)
		host.MatVec(&x.KConvProj, xk, dk)
		host.MatVec(&x.VConvProj, xv, dv)
	}
	Clip(q, x.QClip, hd, ip.QKClip)
	Conv(k, x.KConv, ip.ConvK, dk, &conv.K)
	Clip(k, nil, hd, ip.QKClip)
	Conv(v, x.VConv, ip.ConvK, dv, &conv.V)
	for g := range nKV {
		UnitRMS(v[g*hd:(g+1)*hd], v[g*hd:(g+1)*hd], ip.Eps)
	}
	Clip(rel, x.RelClip, p.DRel, ip.RelClip)
	if !ip.SWA {
		tau := float32(LogScale(pos, ip.LogFloor, ip.LogAlpha))
		for i := range q {
			q[i] *= tau
		}
		for i := range rel {
			rel[i] *= tau
		}
	}

	keys, vals, stride, nKeys := host.AppendKV(l, k, v)
	ctx := make([]float32, nH*hd)
	Attend(AttendParams{Heads: nH, KVHeads: nKV, HeadDim: hd, DRel: p.DRel, Extent: ip.Extent, Window: ip.Window, Scale: a.AttnScale},
		ctx, q, keys, vals, stride, nKeys, 0, pos, rel, x.RelBias, sinks)
	gate := make([]float32, nH*hd)
	host.MatVec(&lw.GProj, xq, gate)
	for i := range ctx {
		ctx[i] *= Sigmoid(gate[i])
	}
	res := make([]float32, hidden)
	host.MatVec(&lw.OProj, ctx, res)
	Conv(res, x.AttnConv, ip.ConvK, nil, &conv.Attn)

	// A block's first layer starts a new partial block; its input joins the completed blocks.
	if ip.Boundary {
		t.normed = append(t.normed, states[len(states)-1])
	} else {
		for i := range res {
			res[i] += h[i]
		}
	}
	ffIn := make([]float32, hidden)
	Mix(ffIn, x.FFNResQuery, 1, with(t.normed, res), hidden)
	if msgs != nil {
		c := &x.Ctrl[3]
		Control(ffIn, msgs[3*hidden:], c.Down, c.Gain, c.Forget, p.GateRank, 1e-6)
	}
	normact.RMS(ffIn, lw.PreMLPNorm, 1, hidden, ip.Eps, false)
	ff := moe(host, lw, &o.FFN.MoE, ip, ffIn)
	Conv(ff, x.FFConv, ip.ConvK, nil, &conv.FFN)
	for i := range h {
		h[i] = res[i] + ff[i]
	}
}

// Route picks the top-k experts from router logits: sigmoid scores, the correction bias added for the selection only, the
// chosen scores normalized to sum 1 and multiplied by scale.
func Route(logits, bias []float32, k int, scale float64) (idx []int, wts []float32) {
	scores := make([]float64, len(logits))
	order := make([]int, len(logits))
	sel := make([]float64, len(logits))
	for i, l := range logits {
		scores[i] = 1 / (1 + math.Exp(-float64(l)))
		sel[i] = scores[i]
		if bias != nil {
			sel[i] += float64(bias[i])
		}
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return sel[order[a]] > sel[order[b]] })
	idx = order[:k]
	var sum float64
	for _, e := range idx {
		sum += scores[e]
	}
	wts = make([]float32, k)
	for j, e := range idx {
		wts[j] = float32(scores[e] / (sum + 1e-20) * scale)
	}
	return idx, wts
}

// moe is the routed MoE of a layer on x: the router in f32, Route, and SITU experts.
func moe(host Host, lw *darch.LayerWeights, cfg *program.MoEConfig, ip *program.Inkling2Params, x []float32) []float32 {
	logits := make([]float32, cfg.NumExperts)
	host.MatVec(&lw.Router, x, logits)
	idx, wts := Route(logits, lw.RouterBias, cfg.TopK, cfg.RoutedScale)
	out, eo := make([]float32, len(x)), make([]float32, len(x))
	gate, up := make([]float32, cfg.IntermediateDim), make([]float32, cfg.IntermediateDim)
	for j, e := range idx {
		ex := &lw.Experts[e]
		host.MatVec(&ex.Gate, x, gate)
		host.MatVec(&ex.Up, x, up)
		Situ(gate, up, ip.SituBeta, ip.SituUp)
		host.MatVec(&ex.Down, gate, eo)
		for i := range out {
			out[i] += wts[j] * eo[i]
		}
	}
	return out
}

// NewToken starts a position on a, or returns nil when a is not an Inkling 2 model.
func NewToken(a *darch.Architecture) *Token {
	if a.Inkling2 == nil {
		return nil
	}
	return new(Token)
}

// Final is the head stage's input: the final depth mix of the completed blocks and the last partial block, in place on h.
// A nil t (not an Inkling 2 model) leaves h as it is.
func Final(a *darch.Architecture, t *Token, h []float32) {
	if t == nil {
		return
	}
	Mix(h, a.Inkling2Head.OutputResQuery, 1, with(t.normed, h), a.HiddenDim)
}
