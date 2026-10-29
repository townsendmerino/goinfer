//go:build darwin

package metal

import (
	"fmt"

	"github.com/townsendmerino/goinfer/internal/vision"
)

// reg is a grid tower's activation buffer for a program register (vision.TowerReg): x is the residual stream, mid is f1
// ([rows, intermediate]), the rest are [rows, hidden].
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
	}
	return Buffer{}, fmt.Errorf("register %d unknown", n)
}

// regWidth is a register's row width: intermediate for mid, hidden for the other registers, 0 for none.
func (a *gridVAccel) regWidth(n vision.TowerReg) int {
	switch n {
	case 0:
		return 0
	case vision.TowerMid:
		return a.inter
	}
	return a.hidden
}

// walkBlock queues one block's program over rows, each op through the tower's existing operation: norm through
// layerNorm at the op's epsilon, linear through linear, attention as the q prescale at the op's scale (dropped under the
// noScale plant) and attendSegments over sg, linear-add as linear into the o scratch then the residual add, GELU-tanh in
// place. An op routed to registers its operation does not use (a norm is hidden-wide in and out; attention reads q, k
// and v and writes ctx; a linear's registers must match its weight's shape; a linear-add's sum is hidden-wide) is an
// error before anything is queued.
// cur is run's current command buffer, which a flush replaces; mark is run's profiling class boundary.
func (a *gridVAccel) walkBlock(prog vision.TowerBlock, b *gvBlock, rows int, sg []int, cur func() *Encoder, mark func(string), flush func() error) error {
	for i, op := range prog {
		var in Buffer
		if op.Kind != vision.TowerAttention && op.Kind != vision.TowerGELUTanh { // they read no In register
			var err error
			if in, err = a.reg(op.In); err != nil {
				return fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
			}
		}
		out, err := a.reg(op.Out)
		if err != nil {
			return fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
		}
		switch op.Kind {
		case vision.TowerNorm:
			w, bias, ok := b.towerNorm(op.W)
			if !ok || op.Form != vision.TowerLayerNorm {
				return fmt.Errorf("op %d: norm of weight %d, form %d", i, op.W, op.Form)
			}
			if a.regWidth(op.In) != a.hidden || a.regWidth(op.Out) != a.hidden {
				return fmt.Errorf("op %d: the norm is %d wide, registers %d → %d are %d → %d wide", i, a.hidden, op.In, op.Out,
					a.regWidth(op.In), a.regWidth(op.Out))
			}
			mark("norm")
			a.layerNorm(cur(), in, w, bias, out, rows, float32(op.Eps))
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
			if op.In != 0 && op.In != op.Out {
				return fmt.Errorf("op %d: GELU-tanh runs in place on register %d, not from register %d", i, op.Out, op.In)
			}
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
	}
	return nil, false
}
