package audio

import (
	"fmt"
	"math"

	"github.com/townsendmerino/goinfer/internal/vision"
)

// Gemma4AudioProgram is Gemma 4 audio's conformer block (spec §2.3): a half-step feed-forward (RMSNorm, the up linear,
// SiLU, the down linear, RMSNorm, residual_weight times it added); attention (RMSNorm, the q, k and v linears, local
// causal attention, the post linear, RMSNorm, the residual add); the light conv (RMSNorm, the start linear to
// 2·hidden, the GLU, the causal depthwise conv, RMSNorm, SiLU, the end linear, the residual add); the second half-step
// feed-forward; a final RMSNorm of the residual. Every linear clamps where the checkpoint has bounds.
func Gemma4AudioProgram(c Gemma4AudioConfig) vision.TowerBlock {
	eps := c.RMSNormEps
	norm := func(in, out vision.TowerReg, w vision.TowerWeight) vision.TowerOp {
		return vision.TowerOp{Kind: vision.TowerNorm, In: in, Out: out, W: w, Form: vision.TowerRMSNorm, Eps: eps}
	}
	lin := func(in, out vision.TowerReg, w vision.TowerWeight) vision.TowerOp {
		return vision.TowerOp{Kind: vision.TowerLinear, In: in, Out: out, W: w}
	}
	x, xn, mid, ctx := vision.TowerX, vision.TowerXN, vision.TowerMid, vision.TowerCtx
	ff := func(pre, up, down, post vision.TowerWeight) []vision.TowerOp {
		return []vision.TowerOp{norm(x, xn, pre), lin(xn, mid, up), {Kind: vision.TowerSiLU, Out: mid}, lin(mid, xn, down),
			norm(xn, xn, post), {Kind: vision.TowerAddScaled, In: xn, Out: x, Scale: c.ResidualWeight}}
	}
	p := vision.TowerBlock(ff(vision.TowerNormFF1Pre, vision.TowerWFF1Up, vision.TowerWFF1Down, vision.TowerNormFF1Post))
	p = append(p,
		norm(x, xn, vision.TowerNorm1),
		lin(xn, vision.TowerQ, vision.TowerWQ),
		lin(xn, vision.TowerK, vision.TowerWK),
		lin(xn, vision.TowerV, vision.TowerWV),
		vision.TowerOp{Kind: vision.TowerLocalAttention, Out: ctx, Window: c.Window(), Scale: c.LogitCap},
		lin(ctx, xn, vision.TowerWO),
		norm(xn, xn, vision.TowerNormPostAttn),
		vision.TowerOp{Kind: vision.TowerAdd, In: xn, Out: x},
		norm(x, xn, vision.TowerNormConvPre),
		lin(xn, vision.TowerWide, vision.TowerWConvStart),
		vision.TowerOp{Kind: vision.TowerGLU, In: vision.TowerWide, Out: ctx},
		vision.TowerOp{Kind: vision.TowerDWConv, In: ctx, Out: xn, W: vision.TowerConvW},
		norm(xn, xn, vision.TowerNormConv),
		vision.TowerOp{Kind: vision.TowerSiLU, Out: xn},
		lin(xn, ctx, vision.TowerWConvEnd),
		vision.TowerOp{Kind: vision.TowerAdd, In: ctx, Out: x},
	)
	p = append(p, ff(vision.TowerNormFF2Pre, vision.TowerWFF2Up, vision.TowerWFF2Down, vision.TowerNormFF2Post)...)
	p = append(p, norm(x, x, vision.TowerNormOut))
	return vision.PlantedProgram("gemma4-audio", p)
}

// gemma4Regs are the CPU walker's registers: x is the residual stream [n, hidden]; mid is [n, 4·hidden]; wide is
// [n, 2·hidden]; the others are [n, hidden].
type gemma4Regs struct {
	x, xn, mid, q, k, v, ctx, wide []float32
}

func newGemma4Regs(h []float32, n, H int) *gemma4Regs {
	b := func(w int) []float32 { return make([]float32, n*w) }
	return &gemma4Regs{x: h, xn: b(H), mid: b(4 * H), q: b(H), k: b(H), v: b(H), ctx: b(H), wide: b(2 * H)}
}

func (r *gemma4Regs) reg(n vision.TowerReg) []float32 {
	switch n {
	case vision.TowerX:
		return r.x
	case vision.TowerXN:
		return r.xn
	case vision.TowerQ:
		return r.q
	case vision.TowerK:
		return r.k
	case vision.TowerV:
		return r.v
	case vision.TowerCtx:
		return r.ctx
	case vision.TowerMid:
		return r.mid
	case vision.TowerWide:
		return r.wide
	}
	return nil
}

// walkBlock runs one block's program over r.x [n, hidden] in place, each op through the function the hand-ordered block
// called: rmsNorm (a norm into another register copies its input first), Gemma4Proj.apply with its clamps, silu, the
// GLU and causal depthwise-conv loops, attendLocal, and the residual adds. An op whose registers do not fit its kind
// (vision.TowerOp.CheckRegs), whose weight the layer lacks or does not fit its registers (CheckShape), or that does not
// fit this tower (an RMSNorm only, a window of at least one key and a positive softcap, a kernel of the configured
// size, at least one row) is an error before it runs.
func (e *Gemma4AudioEncoder) walkBlock(prog vision.TowerBlock, ly *Gemma4AudioLayer, r *gemma4Regs, n int) error {
	c := e.Cfg
	H := c.HiddenSize
	inter := 4 * H // the feed-forwards' width (LoadGemma4AudioEncoder)
	if n <= 0 {
		return fmt.Errorf("%d rows", n)
	}
	view := func(reg vision.TowerReg) ([]float32, error) {
		w := vision.TowerRegWidth(reg, H, inter)
		b := r.reg(reg)
		if w == 0 || len(b) < n*w {
			return nil, fmt.Errorf("register %d missing or under %d values", reg, n*w)
		}
		return b[:n*w], nil
	}
	for i, op := range prog {
		if err := op.CheckRegs(); err != nil {
			return fmt.Errorf("op %d: %w", i, err)
		}
		var in, out []float32
		var err error
		if op.ReadsIn() {
			if in, err = view(op.In); err != nil {
				return fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
			}
		}
		if out, err = view(op.Out); err != nil {
			return fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
		}
		switch op.Kind {
		case vision.TowerNorm:
			w, ok := ly.towerNorm(op.W)
			if !ok || op.Form != vision.TowerRMSNorm || len(w) != H {
				return fmt.Errorf("op %d: norm of weight %d, form %d", i, op.W, op.Form)
			}
			if op.In != op.Out {
				copy(out, in)
			}
			rmsNorm(out, w, n, H, op.Eps)
		case vision.TowerLinear:
			p := ly.towerProj(op.W)
			if p == nil {
				return fmt.Errorf("op %d: linear of weight %d", i, op.W)
			}
			if err := op.CheckShape(p.Out, p.In, H, inter); err != nil {
				return fmt.Errorf("op %d: %w", i, err)
			}
			copy(out, p.apply(in, n))
		case vision.TowerSiLU:
			for j, v := range out {
				out[j] = silu(v)
			}
		case vision.TowerGLU:
			for t := range n {
				for j := range H {
					a, b := in[t*2*H+j], in[t*2*H+H+j]
					out[t*H+j] = a * float32(1/(1+math.Exp(float64(-b))))
				}
			}
		case vision.TowerDWConv:
			k := c.ConvKernelSize
			if k < 1 || len(ly.ConvW) != H*k {
				return fmt.Errorf("op %d: depthwise kernel of %d values, want %d·%d", i, len(ly.ConvW), H, k)
			}
			for t := range n {
				for j := range H {
					var s float32
					for kk := range k {
						ti := t - (k - 1) + kk // causal: left pad k-1
						if ti >= 0 {
							s += ly.ConvW[j*k+kk] * in[ti*H+j]
						}
					}
					out[t*H+j] = s
				}
			}
		case vision.TowerLocalAttention:
			if op.Window < 1 || op.Scale <= 0 || len(ly.RelK) < op.Window*H || len(ly.PerDimScale) != H/c.NumAttentionHeads {
				return fmt.Errorf("op %d: local attention over %d keys, softcap %g, %d relative-key values, %d per-dimension scales",
					i, op.Window, op.Scale, len(ly.RelK), len(ly.PerDimScale))
			}
			q, err1 := view(vision.TowerQ)
			k, err2 := view(vision.TowerK)
			v, err3 := view(vision.TowerV)
			if err1 != nil || err2 != nil || err3 != nil {
				return fmt.Errorf("op %d (%s): %v %v %v", i, op.Kind, err1, err2, err3)
			}
			e.attendLocal(ly, q, k, v, out, n, op.Window, float32(op.Scale))
		case vision.TowerAddScaled:
			rw := float32(op.Scale)
			for j, v := range in {
				out[j] += rw * v
			}
		case vision.TowerAdd:
			for j, v := range in {
				out[j] += v
			}
		default:
			return fmt.Errorf("op %d: kind %s not implemented on the CPU audio tower", i, op.Kind)
		}
	}
	return nil
}

func (ly *Gemma4AudioLayer) towerNorm(w vision.TowerWeight) ([]float32, bool) {
	switch w {
	case vision.TowerNormFF1Pre:
		return ly.FF1.PreNorm, true
	case vision.TowerNormFF1Post:
		return ly.FF1.PostNorm, true
	case vision.TowerNormFF2Pre:
		return ly.FF2.PreNorm, true
	case vision.TowerNormFF2Post:
		return ly.FF2.PostNorm, true
	case vision.TowerNorm1:
		return ly.NormPreAttn, true
	case vision.TowerNormPostAttn:
		return ly.NormPostAttn, true
	case vision.TowerNormConvPre:
		return ly.ConvPreNorm, true
	case vision.TowerNormConv:
		return ly.ConvNorm, true
	case vision.TowerNormOut:
		return ly.NormOut, true
	}
	return nil, false
}

func (ly *Gemma4AudioLayer) towerProj(w vision.TowerWeight) *Gemma4Proj {
	switch w {
	case vision.TowerWFF1Up:
		return &ly.FF1.Up
	case vision.TowerWFF1Down:
		return &ly.FF1.Down
	case vision.TowerWFF2Up:
		return &ly.FF2.Up
	case vision.TowerWFF2Down:
		return &ly.FF2.Down
	case vision.TowerWQ:
		return &ly.Q
	case vision.TowerWK:
		return &ly.K
	case vision.TowerWV:
		return &ly.V
	case vision.TowerWO:
		return &ly.Post
	case vision.TowerWConvStart:
		return &ly.ConvStart
	case vision.TowerWConvEnd:
		return &ly.ConvEnd
	}
	return nil
}
