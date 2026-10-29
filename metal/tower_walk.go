//go:build darwin

package metal

import (
	"fmt"

	"github.com/townsendmerino/goinfer/internal/vision"
)

// reg is a grid tower's activation buffer for a program register (vision.TowerReg): x is the residual stream, mid and
// mid2 are f1 and f2 ([rows, intermediate]), the rest are [rows, hidden].
func (a *gridVAccel) reg(n vision.TowerReg) (Buffer, error) {
	switch n {
	case vision.TowerX:
		return a.x, nil
	case vision.TowerXN:
		return a.xn, nil
	case vision.TowerQ:
		return a.q, nil
	case vision.TowerK:
		return a.k, nil
	case vision.TowerV:
		return a.v, nil
	case vision.TowerCtx:
		return a.ctx, nil
	case vision.TowerMid:
		return a.f1, nil
	case vision.TowerMid2:
		return a.f2, nil
	}
	return Buffer{}, fmt.Errorf("register %d unknown", n)
}

// regWidth is a register's row width: intermediate for mid and mid2, hidden for the other registers, 0 for none.
func (a *gridVAccel) regWidth(n vision.TowerReg) int {
	switch n {
	case 0:
		return 0
	case vision.TowerMid, vision.TowerMid2:
		return a.inter
	}
	return a.hidden
}

// readsIn reports whether an op of kind k reads its In register; the others run in place on Out, or read q, k and v.
func readsIn(k vision.TowerOpKind) bool {
	switch k {
	case vision.TowerNorm, vision.TowerLinear, vision.TowerLinearAdd, vision.TowerLinearQKV, vision.TowerSiLUMul:
		return true
	}
	return false
}

// walkBlock queues one block's program over rows, each op through the tower's existing operation: a norm through
// layerNorm or rmsEps at the op's epsilon; a per-head norm through rmsEps over rows·heads rows of head_dim (dropped under
// the noQKNorm plant); a linear through linear; the fused qkv as the three split linears, q then k then v; RoPE through
// the rope kernel on q or k; attention as the q prescale at the op's scale (dropped under the noScale plant) and
// attendSegments over sg; a linear-add as linear into the o scratch then the residual add; GELU-tanh and the SiLU product
// in place. An op routed to registers its operation does not use (a norm is hidden-wide in and out; a per-head norm and
// RoPE run on q or k; attention reads q, k and v and writes ctx; a linear's registers must match its weight's shape; a
// linear-add's sum is hidden-wide; the SiLU product reads another intermediate-wide register) is an error before anything
// is queued.
// cur is run's current command buffer, which a flush replaces; mark is run's profiling class boundary.
func (a *gridVAccel) walkBlock(prog vision.TowerBlock, b *gvBlock, rows int, sg []int, cur func() *Encoder, mark func(string), flush func() error) error {
	for i, op := range prog {
		var in, out Buffer
		var err error
		if readsIn(op.Kind) {
			if in, err = a.reg(op.In); err != nil {
				return fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
			}
		} else if op.Kind != vision.TowerAttention && op.In != 0 && op.In != op.Out { // attention checks its own
			return fmt.Errorf("op %d (%s): runs in place on register %d, not from register %d", i, op.Kind, op.Out, op.In)
		}
		if op.Kind != vision.TowerLinearQKV { // it writes q, k and v
			if out, err = a.reg(op.Out); err != nil {
				return fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
			}
		} else if op.Out != 0 {
			return fmt.Errorf("op %d: the fused qkv writes q, k and v, not register %d", i, op.Out)
		}
		switch op.Kind {
		case vision.TowerNorm:
			w, bias, ok := b.towerNorm(op.W)
			if !ok || (op.Form != vision.TowerLayerNorm && op.Form != vision.TowerRMSNorm) {
				return fmt.Errorf("op %d: norm of weight %d, form %d", i, op.W, op.Form)
			}
			if a.regWidth(op.In) != a.hidden || a.regWidth(op.Out) != a.hidden {
				return fmt.Errorf("op %d: the norm is %d wide, registers %d → %d are %d → %d wide", i, a.hidden, op.In, op.Out,
					a.regWidth(op.In), a.regWidth(op.Out))
			}
			mark("norm")
			if op.Form == vision.TowerRMSNorm {
				a.rmsEps(cur(), in, w, out, rows, a.hidden, float32(op.Eps))
				break
			}
			a.layerNorm(cur(), in, w, bias, out, rows, float32(op.Eps))
		case vision.TowerHeadNorm:
			w, ok := b.towerHeadNorm(op.W)
			if !ok || op.Form != vision.TowerRMSNorm {
				return fmt.Errorf("op %d: per-head norm of weight %d, form %d", i, op.W, op.Form)
			}
			if op.Out != vision.TowerQ && op.Out != vision.TowerK {
				return fmt.Errorf("op %d: the per-head norm runs on q or k, not register %d", i, op.Out)
			}
			if a.planted.noQKNorm { // G-S2c (4) drops it
				break
			}
			mark("norm")
			a.rmsEps(cur(), out, w, out, rows*a.heads, a.hd, float32(op.Eps))
		case vision.TowerLinearQKV:
			if op.W != vision.TowerWQKV {
				return fmt.Errorf("op %d: fused qkv of weight %d", i, op.W)
			}
			for _, p := range []*gvProj{&b.q, &b.k, &b.v} {
				if p.in != a.regWidth(op.In) || p.out != a.hidden {
					return fmt.Errorf("op %d: a split qkv weight is [%d, %d], register %d is %d wide, q, k and v are %d", i, p.out,
						p.in, op.In, a.regWidth(op.In), a.hidden)
				}
			}
			mark("gemm")
			a.linear(cur(), in, &b.q, a.q, rows)
			a.linear(cur(), in, &b.k, a.k, rows)
			a.linear(cur(), in, &b.v, a.v, rows)
		case vision.TowerRoPE:
			if op.Out != vision.TowerQ && op.Out != vision.TowerK {
				return fmt.Errorf("op %d: RoPE runs on q or k, not register %d", i, op.Out)
			}
			mark("elementwise")
			nr := rows * a.heads * a.hd / 2
			cur().Dispatch(a.p.rope, nr, 64, out, a.ropeCos, a.ropeSin, a.u32(uint32(a.heads)), a.u32(uint32(a.hd)), a.u32(uint32(nr)))
		case vision.TowerSiLUMul:
			if op.In == op.Out || a.regWidth(op.In) != a.inter || a.regWidth(op.Out) != a.inter {
				return fmt.Errorf("op %d: the SiLU product is %d wide from another register, registers %d → %d are %d → %d wide", i,
					a.inter, op.In, op.Out, a.regWidth(op.In), a.regWidth(op.Out))
			}
			mark("elementwise")
			n := rows * a.inter
			cur().Dispatch(a.vit.SiLUMul, n, 256, out, in, a.u32(uint32(n)))
		case vision.TowerLinear, vision.TowerLinearAdd:
			p, ok := b.towerLinear(op.W)
			if !ok {
				return fmt.Errorf("op %d: linear of weight %d", i, op.W)
			}
			if p.in != a.regWidth(op.In) || p.out != a.regWidth(op.Out) {
				return fmt.Errorf("op %d (%s): weight %d is [%d, %d], registers %d → %d are %d → %d wide", i, op.Kind, op.W,
					p.out, p.in, op.In, op.Out, a.regWidth(op.In), a.regWidth(op.Out))
			}
			if op.Kind == vision.TowerLinearAdd && p.out != a.hidden {
				return fmt.Errorf("op %d: a linear-add's sum goes through the %d-wide o scratch, and weight %d is %d wide", i,
					a.hidden, op.W, p.out)
			}
			mark("gemm")
			if op.Kind == vision.TowerLinear {
				a.linear(cur(), in, p, out, rows)
				break
			}
			a.linear(cur(), in, p, a.o, rows)
			mark("elementwise")
			n := rows * a.hidden
			cur().Dispatch(a.p.add, n, 256, out, a.o, a.u32(uint32(n)))
		case vision.TowerAttention:
			if op.In != 0 || op.Out != vision.TowerCtx {
				return fmt.Errorf("op %d: attention reads q, k and v and writes ctx, not register %d → %d", i, op.In, op.Out)
			}
			mark("elementwise")
			if !a.planted.noScale { // G-S2c (1), G-S3a (1) drop it
				n := rows * a.hidden
				cur().Dispatch(a.scale, n, 256, a.q, a.u32(uint32(n)), a.f32(float32(op.Scale)))
			}
			mark("attention")
			if err := a.attendSegments(cur, sg, flush); err != nil {
				return err
			}
		case vision.TowerGELUTanh:
			n := rows * a.regWidth(op.Out)
			mark("elementwise")
			cur().Dispatch(a.vit.GELUTanh, n, 256, out, a.u32(uint32(n)))
		default:
			return fmt.Errorf("op %d: kind %s not implemented on Metal", i, op.Kind)
		}
	}
	return nil
}

func (b *gvBlock) towerNorm(w vision.TowerWeight) (weight, bias Buffer, ok bool) {
	switch w {
	case vision.TowerNorm1:
		return b.norm1w, b.norm1b, true
	case vision.TowerNorm2:
		return b.norm2w, b.norm2b, true
	}
	return Buffer{}, Buffer{}, false
}

func (b *gvBlock) towerLinear(w vision.TowerWeight) (*gvProj, bool) {
	switch w {
	case vision.TowerWQ:
		return &b.q, true
	case vision.TowerWK:
		return &b.k, true
	case vision.TowerWV:
		return &b.v, true
	case vision.TowerWO:
		return &b.proj, true
	case vision.TowerWFC1:
		return &b.fc1, true
	case vision.TowerWFC2:
		return &b.fc2, true
	case vision.TowerWGate:
		return &b.gate, true
	case vision.TowerWUp:
		return &b.up, true
	case vision.TowerWDown:
		return &b.down, true
	}
	return nil, false
}

func (b *gvBlock) towerHeadNorm(w vision.TowerWeight) (Buffer, bool) {
	switch w {
	case vision.TowerQNorm:
		return b.qNorm, true
	case vision.TowerKNorm:
		return b.kNorm, true
	}
	return Buffer{}, false
}
