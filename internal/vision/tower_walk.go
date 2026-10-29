package vision

import (
	"fmt"

	"github.com/townsendmerino/aikit/linalg"
)

// towerWeights is one CPU tower block's weights, by the names a program uses.
type towerWeights interface {
	towerNorm(TowerWeight) (weight, bias []float32, ok bool)
	towerLinear(TowerWeight) (*linalg.WeightMat, []float32, bool)
}

// towerRegs are the CPU walker's activation buffers, each at least rows × its width: x is the residual stream, mid and
// mid2 are [rows, intermediate], qkv is [rows, 3·hidden] for the fused qkv's product, tmp is [rows, hidden] for a
// linear-add's product, and the rest are [rows, hidden].
type towerRegs struct {
	x, xn, q, k, v, ctx, mid, mid2, qkv, tmp []float32
}

// towerEnv is what a CPU tower gives the walker: its shape, its own matmul entry point and attention schedule, and the
// forward's RoPE tables ([rows, head_dim], nil for a tower with none).
type towerEnv struct {
	rows, hidden, heads, inter int
	matmul                     func(w *linalg.WeightMat, a, dst []float32, M int)
	attend                     func(q, k, v, out []float32, scale float64) error
	cos, sin                   []float32
}

func (r *towerRegs) reg(n TowerReg) []float32 {
	switch n {
	case TowerX:
		return r.x
	case TowerXN:
		return r.xn
	case TowerQ:
		return r.q
	case TowerK:
		return r.k
	case TowerV:
		return r.v
	case TowerCtx:
		return r.ctx
	case TowerMid:
		return r.mid
	case TowerMid2:
		return r.mid2
	}
	return nil
}

// walkTowerBlock runs one block's program over r.x in place, each op through the CPU's own operation (layerNormInto,
// rmsNormEpsInto, the tower's matmul and addBias, applyRotaryVision, the tower's attention, geluTanh, the SiLU product,
// addResidual). A linear-add writes its product to r.tmp and adds it to its output register, the two steps the
// hand-written loops took.
func walkTowerBlock(prog TowerBlock, wts towerWeights, r *towerRegs, env *towerEnv) error {
	rows, hidden := env.rows, env.hidden
	hd := hidden / env.heads
	width := func(n TowerReg) int {
		if n == TowerMid || n == TowerMid2 {
			return env.inter
		}
		return hidden
	}
	view := func(n TowerReg) ([]float32, error) {
		b := r.reg(n)
		if b == nil || len(b) < rows*width(n) {
			return nil, fmt.Errorf("register %d missing or under %d values", n, rows*width(n))
		}
		return b[:rows*width(n)], nil
	}
	for i, op := range prog {
		var in, out []float32
		var err error
		if err := op.CheckRegs(); err != nil {
			return fmt.Errorf("op %d: %w", i, err)
		}
		if op.ReadsIn() {
			if in, err = view(op.In); err != nil {
				return fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
			}
		}
		if op.Kind != TowerLinearQKV {
			if out, err = view(op.Out); err != nil {
				return fmt.Errorf("op %d (%s): %w", i, op.Kind, err)
			}
		}
		switch op.Kind {
		case TowerNorm, TowerHeadNorm:
			w, b, ok := wts.towerNorm(op.W)
			if !ok {
				return fmt.Errorf("op %d: norm of weight %d", i, op.W)
			}
			switch {
			case op.Kind == TowerNorm && op.Form == TowerLayerNorm:
				layerNormInto(out, in, w, b, rows, hidden, op.Eps)
			case op.Kind == TowerNorm && op.Form == TowerRMSNorm:
				rmsNormEpsInto(out, in, w, rows, hidden, op.Eps)
			case op.Kind == TowerHeadNorm && op.Form == TowerRMSNorm:
				rmsNormEpsInto(out, out, w, rows*env.heads, hd, op.Eps)
			default:
				return fmt.Errorf("op %d (%s): norm form %d not implemented on the CPU", i, op.Kind, op.Form)
			}
		case TowerLinear, TowerLinearAdd, TowerLinearQKV:
			w, b, ok := wts.towerLinear(op.W)
			if !ok {
				return fmt.Errorf("op %d: linear of weight %d", i, op.W)
			}
			n := w.Rows()
			dst := out
			switch op.Kind {
			case TowerLinearAdd:
				dst = r.tmp
			case TowerLinearQKV:
				dst = r.qkv
			}
			if len(in) < rows*w.Cols() || len(dst) < rows*n || (op.Kind != TowerLinearQKV && len(out) < rows*n) {
				return fmt.Errorf("op %d (%s): weight %d is [%d,%d], registers too small", i, op.Kind, op.W, n, w.Cols())
			}
			dst = dst[:rows*n]
			env.matmul(w, in, dst, rows)
			addBias(dst, b, rows, n)
			switch op.Kind {
			case TowerLinearAdd:
				addResidual(out[:len(dst)], dst)
			case TowerLinearQKV: // rows are [3, heads, head_dim]: q, k and v are the three contiguous thirds
				if n != 3*hidden || len(r.q) < rows*hidden || len(r.k) < rows*hidden || len(r.v) < rows*hidden {
					return fmt.Errorf("op %d (%s): weight %d has %d rows, want %d", i, op.Kind, op.W, n, 3*hidden)
				}
				for p := range rows {
					base := p * 3 * hidden
					copy(r.q[p*hidden:(p+1)*hidden], dst[base:base+hidden])
					copy(r.k[p*hidden:(p+1)*hidden], dst[base+hidden:base+2*hidden])
					copy(r.v[p*hidden:(p+1)*hidden], dst[base+2*hidden:base+3*hidden])
				}
			}
		case TowerRoPE:
			if len(env.cos) < rows*hd || len(env.sin) < rows*hd {
				return fmt.Errorf("op %d (%s): no RoPE tables for %d rows", i, op.Kind, rows)
			}
			for p := range rows {
				co, si := env.cos[p*hd:p*hd+hd], env.sin[p*hd:p*hd+hd]
				for head := range env.heads {
					off := p*hidden + head*hd
					applyRotaryVision(out[off:off+hd], co, si)
				}
			}
		case TowerAttention:
			q, err1 := view(TowerQ)
			k, err2 := view(TowerK)
			v, err3 := view(TowerV)
			if err1 != nil || err2 != nil || err3 != nil {
				return fmt.Errorf("op %d (%s): %v %v %v", i, op.Kind, err1, err2, err3)
			}
			if err := env.attend(q, k, v, out, op.Scale); err != nil {
				return err
			}
		case TowerGELUTanh:
			geluTanh(out)
		case TowerSiLUMul:
			if len(in) != len(out) {
				return fmt.Errorf("op %d (%s): registers %d and %d differ in width", i, op.Kind, op.In, op.Out)
			}
			siluMulInto(out, in)
		default:
			return fmt.Errorf("op %d: kind %s not implemented on the CPU", i, op.Kind)
		}
	}
	return nil
}

// siluMulInto is g = silu(g) · u in place, elementwise.
func siluMulInto(g, u []float32) {
	if len(g) == 0 {
		return
	}
	_ = u[len(g)-1]
	parallelChunks(len(g), func(lo, hi int) {
		gc := g[lo:hi]
		linalg.SiLUContractInto(gc, gc)
		uc := u[lo:hi]
		for i := range gc {
			gc[i] *= uc[i]
		}
	})
}
