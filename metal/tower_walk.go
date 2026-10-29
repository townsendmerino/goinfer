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

// walkBlock queues one block's program over rows, each op through the tower's existing operation: norm through
// layerNorm at the op's epsilon, linear through linear, attention as the q prescale at the op's scale (dropped under the
// noScale plant) and attendSegments over sg, linear-add as linear into the o scratch then the residual add, GELU-tanh in
// place. cur is run's current command buffer, which a flush replaces; mark is run's profiling class boundary.
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
			mark("norm")
			a.layerNorm(cur(), in, w, bias, out, rows, float32(op.Eps))
		case vision.TowerLinear, vision.TowerLinearAdd:
			p, ok := b.towerLinear(op.W)
			if !ok {
				return fmt.Errorf("op %d: linear of weight %d", i, op.W)
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
			n := rows * a.hidden
			if op.Out == vision.TowerMid {
				n = rows * a.inter
			}
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
