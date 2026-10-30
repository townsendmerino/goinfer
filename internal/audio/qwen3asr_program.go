package audio

import (
	"fmt"
	"math"

	"github.com/townsendmerino/goinfer/internal/vision"
)

// QwenASRProgram is Qwen3-ASR's encoder layer, pre-LayerNorm (eps 1e-5): LayerNorm, the q, k and v linears, attention
// at 1/sqrt(head_dim) within each window, o added to the residual, LayerNorm, fc1, exact GELU, fc2 added to the
// residual. Every linear has a bias.
func QwenASRProgram(c QwenASRConfig) vision.TowerBlock {
	const eps = 1e-5 // layerNorm's
	p := vision.TowerBlock{
		{Kind: vision.TowerNorm, In: vision.TowerX, Out: vision.TowerXN, W: vision.TowerNorm1, Form: vision.TowerLayerNorm, Eps: eps},
		{Kind: vision.TowerLinear, In: vision.TowerXN, Out: vision.TowerQ, W: vision.TowerWQ},
		{Kind: vision.TowerLinear, In: vision.TowerXN, Out: vision.TowerK, W: vision.TowerWK},
		{Kind: vision.TowerLinear, In: vision.TowerXN, Out: vision.TowerV, W: vision.TowerWV},
		{Kind: vision.TowerAttention, Out: vision.TowerCtx, Scale: 1 / math.Sqrt(float64(c.DModel/c.Heads))},
		{Kind: vision.TowerLinearAdd, In: vision.TowerCtx, Out: vision.TowerX, W: vision.TowerWO},
		{Kind: vision.TowerNorm, In: vision.TowerX, Out: vision.TowerXN, W: vision.TowerNorm2, Form: vision.TowerLayerNorm, Eps: eps},
		{Kind: vision.TowerLinear, In: vision.TowerXN, Out: vision.TowerMid, W: vision.TowerWFC1},
		{Kind: vision.TowerGELU, Out: vision.TowerMid},
		{Kind: vision.TowerLinearAdd, In: vision.TowerMid, Out: vision.TowerX, W: vision.TowerWFC2},
	}
	return vision.PlantedProgram("qwen3-asr", p)
}

// qwenASRRegs are the walker's registers: x is the residual stream [n, d_model]; mid is [n, ffn]; the others are
// [n, d_model].
type qwenASRRegs struct {
	x, xn, q, k, v, ctx, mid []float32
}

func (r *qwenASRRegs) reg(n vision.TowerReg) []float32 {
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
	}
	return nil
}

// walkBlock runs one layer's program over r.x [n, d_model] in place, attention within consecutive windows of window
// positions, each op through the function the hand-ordered layer called: layerNormEps (into another register, after
// copying its input), linear, attendWindowed, e.gelu (with its planted defect), and the residual adds. An op whose
// registers do not fit its kind (vision.TowerOp.CheckRegs), whose weight the layer lacks or does not fit its registers
// (CheckShape), or that does not fit this tower (a LayerNorm only) is an error before it runs.
func (e *QwenASREncoder) walkBlock(prog vision.TowerBlock, ly *qwenASRLayer, r *qwenASRRegs, n, window int) error {
	c := e.Cfg
	D, F := c.DModel, c.FFN
	view := func(reg vision.TowerReg) ([]float32, error) {
		w := vision.TowerRegWidth(reg, D, F)
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
			w, b, ok := ly.towerNorm(op.W)
			if !ok || op.Form != vision.TowerLayerNorm {
				return fmt.Errorf("op %d: norm of weight %d, form %d", i, op.W, op.Form)
			}
			if op.In != op.Out {
				copy(out, in)
			}
			layerNormEps(out, w, b, n, D, op.Eps)
		case vision.TowerLinear, vision.TowerLinearAdd:
			w, b, rows, cols, ok := ly.towerLinear(op.W, D, F)
			if !ok || len(w) != rows*cols || len(b) != rows {
				return fmt.Errorf("op %d: linear of weight %d", i, op.W)
			}
			if err := op.CheckShape(rows, cols, D, F); err != nil {
				return fmt.Errorf("op %d: %w", i, err)
			}
			y := linear(in, w, b, n, cols, rows)
			if op.Kind == vision.TowerLinear {
				copy(out, y)
				break
			}
			for j := range out {
				out[j] += y[j]
			}
		case vision.TowerAttention:
			q, err1 := view(vision.TowerQ)
			k, err2 := view(vision.TowerK)
			v, err3 := view(vision.TowerV)
			if err1 != nil || err2 != nil || err3 != nil {
				return fmt.Errorf("op %d (%s): %v %v %v", i, op.Kind, err1, err2, err3)
			}
			e.attendWindowed(q, k, v, out, n, window, float32(op.Scale))
		case vision.TowerGELU:
			for j, v := range out {
				out[j] = e.gelu(v)
			}
		default:
			return fmt.Errorf("op %d: kind %s not implemented on the Qwen3-ASR encoder", i, op.Kind)
		}
	}
	return nil
}

func (ly *qwenASRLayer) towerNorm(w vision.TowerWeight) (weight, bias []float32, ok bool) {
	switch w {
	case vision.TowerNorm1:
		return ly.ln1W, ly.ln1B, true
	case vision.TowerNorm2:
		return ly.ln2W, ly.ln2B, true
	}
	return nil, nil, false
}

// towerLinear is a linear weight [rows, cols] and its bias, by name; d is d_model and f the FFN width.
func (ly *qwenASRLayer) towerLinear(w vision.TowerWeight, d, f int) (weight, bias []float32, rows, cols int, ok bool) {
	switch w {
	case vision.TowerWQ:
		return ly.qW, ly.qB, d, d, true
	case vision.TowerWK:
		return ly.kW, ly.kB, d, d, true
	case vision.TowerWV:
		return ly.vW, ly.vB, d, d, true
	case vision.TowerWO:
		return ly.oW, ly.oB, d, d, true
	case vision.TowerWFC1:
		return ly.fc1W, ly.fc1B, f, d, true
	case vision.TowerWFC2:
		return ly.fc2W, ly.fc2B, d, f, true
	}
	return nil, nil, 0, 0, false
}
