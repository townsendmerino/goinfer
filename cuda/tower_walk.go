//go:build cuda

package cuda

import (
	"fmt"

	"github.com/townsendmerino/goinfer/internal/vision"
)

// towerRegs are a tower walker's activation buffers, by vision.TowerReg: x is the residual stream, mid is [rows,
// intermediate], tmp is a linear-add's product; the rest are [rows, hidden].
type towerRegs struct {
	x, xn, q, k, v, ctx, mid, tmp Buffer
}

func (r *towerRegs) reg(n vision.TowerReg) (Buffer, error) {
	switch n {
	case vision.TowerX:
		return r.x, nil
	case vision.TowerXN:
		return r.xn, nil
	case vision.TowerQ:
		return r.q, nil
	case vision.TowerK:
		return r.k, nil
	case vision.TowerV:
		return r.v, nil
	case vision.TowerCtx:
		return r.ctx, nil
	case vision.TowerMid:
		return r.mid, nil
	}
	return Buffer{}, fmt.Errorf("register %d unknown", n)
}

// walkBlock queues one block's program over rows, each op through the tower's existing operation (towerOps.layerNorm,
// gridTower.linear and linearAdd, towerOps.attention, towerOps.geluTanh). scale, when non-zero, replaces the attention
// op's own: the planted-defect seam (gridDefect.noScale).
func (a *gridTower) walkBlock(prog vision.TowerBlock, b *gridBlock, r *towerRegs, rows int, scale float32) error {
	for i, op := range prog {
		var in Buffer
		if op.Kind != vision.TowerAttention && op.Kind != vision.TowerGELUTanh { // they read no In register
			var err error
			if in, err = r.reg(op.In); err != nil {
				return fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
			}
		}
		out, err := r.reg(op.Out)
		if err != nil {
			return fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
		}
		switch op.Kind {
		case vision.TowerNorm:
			w, bias, ok := b.towerNorm(op.W)
			if !ok || op.Form != vision.TowerLayerNorm {
				return fmt.Errorf("op %d: norm of weight %d, form %d", i, op.W, op.Form)
			}
			a.ops.layerNorm(in, w, bias, out, rows, a.hidden, float32(op.Eps))
		case vision.TowerLinear, vision.TowerLinearAdd:
			p, ok := b.towerLinear(op.W)
			if !ok {
				return fmt.Errorf("op %d: linear of weight %d", i, op.W)
			}
			if op.Kind == vision.TowerLinear {
				a.linear(in, p, out, rows)
			} else {
				a.linearAdd(in, p, out, r.tmp, rows)
			}
		case vision.TowerAttention:
			s := float32(op.Scale)
			if scale != 0 {
				s = scale
			}
			if err := a.ops.attention(r.q, r.k, r.v, out, rows, a.heads, a.hd, s); err != nil {
				return err
			}
		case vision.TowerGELUTanh:
			width := a.hidden
			if op.Out == vision.TowerMid {
				width = a.inter
			}
			a.ops.geluTanh(out, rows*width)
		default:
			return fmt.Errorf("op %d: kind %s not implemented on CUDA", i, op.Kind)
		}
	}
	return nil
}

func (b *gridBlock) towerNorm(w vision.TowerWeight) (weight, bias Buffer, ok bool) {
	switch w {
	case vision.TowerNorm1:
		return b.norm1w, b.norm1b, true
	case vision.TowerNorm2:
		return b.norm2w, b.norm2b, true
	}
	return Buffer{}, Buffer{}, false
}

func (b *gridBlock) towerLinear(w vision.TowerWeight) (*gridProj, bool) {
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
