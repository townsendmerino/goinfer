//go:build cuda

package cuda

import (
	"fmt"

	gpu "github.com/townsendmerino/goinfer/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/internal/vision"
)

// towerRegs are a tower walker's activation buffers, by vision.TowerReg: x is the residual stream, mid and mid2 are
// [rows, intermediate], tmp is a linear-add's product; the rest are [rows, hidden].
type towerRegs struct {
	x, xn, q, k, v, ctx, mid, mid2, tmp Buffer
}

// towerWalkEnv is one forward's inputs to walkBlock, beside the program.
type towerWalkEnv struct {
	scale      float32 // when non-zero, replaces the attention op's own (gridDefect.noScale)
	noHeadNorm bool    // the per-head norms skipped (gridDefect.noQKNorm)
	cs, sn     Buffer  // RoPE tables [rows, head_dim]; zero for a tower with no RoPE
	segs       []int   // attention segments as cumulative row offsets; nil is one segment over every row
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
	case vision.TowerMid2:
		return r.mid2, nil
	}
	return Buffer{}, fmt.Errorf("register %d unknown", n)
}

// walkBlock queues one block's program over rows, each op through the tower's existing operation (towerOps.layerNorm and
// rms, gridTower.linear and linearAdd, the fused qkv as the three split linears, towerOps.ropeHalfTo, towerOps.attention
// per segment, towerOps.geluTanh, the SiLUMul kernel).
func (a *gridTower) walkBlock(prog vision.TowerBlock, b *gridBlock, r *towerRegs, rows int, env *towerWalkEnv) error {
	o := a.ops
	width := func(n vision.TowerReg) int {
		if n == vision.TowerMid || n == vision.TowerMid2 {
			return a.inter
		}
		return a.hidden
	}
	for i, op := range prog {
		var in, out Buffer
		var err error
		if op.ReadsIn() {
			if in, err = r.reg(op.In); err != nil {
				return fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
			}
		}
		if op.Kind != vision.TowerLinearQKV {
			if out, err = r.reg(op.Out); err != nil {
				return fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
			}
		}
		switch op.Kind {
		case vision.TowerNorm, vision.TowerHeadNorm:
			w, bias, ok := b.towerNorm(op.W)
			if !ok {
				return fmt.Errorf("op %d: norm of weight %d", i, op.W)
			}
			switch {
			case op.Kind == vision.TowerNorm && op.Form == vision.TowerLayerNorm:
				o.layerNorm(in, w, bias, out, rows, a.hidden, float32(op.Eps))
			case op.Kind == vision.TowerNorm && op.Form == vision.TowerRMSNorm:
				o.rms(in, w, out, rows, a.hidden, float32(op.Eps))
			case op.Kind == vision.TowerHeadNorm && op.Form == vision.TowerRMSNorm:
				if !env.noHeadNorm {
					o.rms(out, w, out, rows*a.heads, a.hd, float32(op.Eps))
				}
			default:
				return fmt.Errorf("op %d (%s): norm form %d not implemented on CUDA", i, op.Kind, op.Form)
			}
		case vision.TowerLinearQKV:
			if op.W != vision.TowerWQKV {
				return fmt.Errorf("op %d: fused qkv of weight %d", i, op.W)
			}
			a.linear(in, &b.q, r.q, rows)
			a.linear(in, &b.k, r.k, rows)
			a.linear(in, &b.v, r.v, rows)
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
		case vision.TowerRoPE:
			if env.cs == (Buffer{}) || env.sn == (Buffer{}) {
				return fmt.Errorf("op %d (%s): no RoPE tables", i, op.Kind)
			}
			o.ropeHalfTo(out, env.cs, env.sn, rows, a.heads, a.hd)
		case vision.TowerAttention:
			s := float32(op.Scale)
			if env.scale != 0 {
				s = env.scale
			}
			if env.segs == nil {
				if err := o.attention(r.q, r.k, r.v, out, rows, a.heads, a.hd, s); err != nil {
					return err
				}
				continue
			}
			for si := 1; si < len(env.segs); si++ {
				off, T := env.segs[si-1]*a.hidden*4, env.segs[si]-env.segs[si-1]
				if err := o.attention(r.q.At(off), r.k.At(off), r.v.At(off), out.At(off), T, a.heads, a.hd, s); err != nil {
					return err
				}
			}
		case vision.TowerGELUTanh:
			o.geluTanh(out, rows*width(op.Out))
		case vision.TowerSiLUMul:
			n := rows * width(op.Out)
			o.cls(clsElem)
			o.launch(o.vit.SiLUMul, gpu.Grid1D(n, 256), Arg(out), Arg(in), i32(int32(n)))
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
	case vision.TowerQNorm:
		return b.qNorm, Buffer{}, true
	case vision.TowerKNorm:
		return b.kNorm, Buffer{}, true
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
	case vision.TowerWGate:
		return &b.gate, true
	case vision.TowerWUp:
		return &b.up, true
	case vision.TowerWDown:
		return &b.down, true
	}
	return nil, false
}
