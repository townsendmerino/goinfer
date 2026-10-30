package audio

import (
	"fmt"
	"math"

	"github.com/townsendmerino/goinfer/internal/vision"
)

// WhisperProgram is the Whisper encoder's pre-norm block, which Voxtral's tower shares: LayerNorm (eps 1e-5), the q, k
// and v linears (k has no bias), attention at 1/sqrt(head_dim) over the whole clip, o added to the residual, LayerNorm,
// fc1, the exact GELU, fc2 added to the residual.
func WhisperProgram(c WhisperEncoderConfig) vision.TowerBlock {
	p := vision.TowerBlock{
		{Kind: vision.TowerNorm, In: vision.TowerX, Out: vision.TowerXN, W: vision.TowerNorm1, Form: vision.TowerLayerNorm, Eps: whisperLayerNormEps},
		{Kind: vision.TowerLinear, In: vision.TowerXN, Out: vision.TowerQ, W: vision.TowerWQ},
		{Kind: vision.TowerLinear, In: vision.TowerXN, Out: vision.TowerK, W: vision.TowerWK},
		{Kind: vision.TowerLinear, In: vision.TowerXN, Out: vision.TowerV, W: vision.TowerWV},
		{Kind: vision.TowerAttention, Out: vision.TowerCtx, Scale: 1 / math.Sqrt(float64(c.DModel/c.Heads))},
		{Kind: vision.TowerLinearAdd, In: vision.TowerCtx, Out: vision.TowerX, W: vision.TowerWO},
		{Kind: vision.TowerNorm, In: vision.TowerX, Out: vision.TowerXN, W: vision.TowerNorm2, Form: vision.TowerLayerNorm, Eps: whisperLayerNormEps},
		{Kind: vision.TowerLinear, In: vision.TowerXN, Out: vision.TowerMid, W: vision.TowerWFC1},
		{Kind: vision.TowerGELUErf, Out: vision.TowerMid},
		{Kind: vision.TowerLinearAdd, In: vision.TowerMid, Out: vision.TowerX, W: vision.TowerWFC2},
	}
	return vision.PlantedProgram("whisper", p)
}

// whisperPostNormProgram is the block the post-norm defect runs (SetDefectForTest): each sublayer's LayerNorm after its
// residual add. The residual moves to xn after the attention's norm and back to x after the MLP's, so no norm runs in
// place.
func whisperPostNormProgram(c WhisperEncoderConfig) vision.TowerBlock {
	return WhisperPostNormProgramForTest(c)
}

// WhisperPostNormProgramForTest is whisperPostNormProgram, for a device tower's same planted defect.
func WhisperPostNormProgramForTest(c WhisperEncoderConfig) vision.TowerBlock {
	return vision.TowerBlock{
		{Kind: vision.TowerLinear, In: vision.TowerX, Out: vision.TowerQ, W: vision.TowerWQ},
		{Kind: vision.TowerLinear, In: vision.TowerX, Out: vision.TowerK, W: vision.TowerWK},
		{Kind: vision.TowerLinear, In: vision.TowerX, Out: vision.TowerV, W: vision.TowerWV},
		{Kind: vision.TowerAttention, Out: vision.TowerCtx, Scale: 1 / math.Sqrt(float64(c.DModel/c.Heads))},
		{Kind: vision.TowerLinearAdd, In: vision.TowerCtx, Out: vision.TowerX, W: vision.TowerWO},
		{Kind: vision.TowerNorm, In: vision.TowerX, Out: vision.TowerXN, W: vision.TowerNorm1, Form: vision.TowerLayerNorm, Eps: whisperLayerNormEps},
		{Kind: vision.TowerLinear, In: vision.TowerXN, Out: vision.TowerMid, W: vision.TowerWFC1},
		{Kind: vision.TowerGELUErf, Out: vision.TowerMid},
		{Kind: vision.TowerLinearAdd, In: vision.TowerMid, Out: vision.TowerXN, W: vision.TowerWFC2},
		{Kind: vision.TowerNorm, In: vision.TowerXN, Out: vision.TowerX, W: vision.TowerNorm2, Form: vision.TowerLayerNorm, Eps: whisperLayerNormEps},
	}
}

// walkBlock runs one block's program over h [n][D] in place, each op through the encoder's own operation (layerNormEps on
// a copy of its input, linear, attend, geluInto, the residual add). x is h: an op writing x writes into it. Every other
// register is a slice the op that writes it allocates; one read before it is written is zeros. An op that does not fit its kind (vision.TowerOp.CheckRegs), its
// weight (vision.TowerOp.CheckShape) or this encoder is an error before anything is computed for it. The planted defects
// act at their ops: no scale at attention, the tanh GELU in geluInto, a window for attention.
func (e *WhisperEncoder) walkBlock(prog vision.TowerBlock, ly *whisperLayer, h []float32, n, window int) error {
	c := e.Cfg
	D, F := c.DModel, c.FFN
	regs := map[vision.TowerReg][]float32{vision.TowerX: h}
	set := func(r vision.TowerReg, v []float32) { // x is h, written in place; the others take the op's own slice
		if r == vision.TowerX {
			copy(h, v)
			return
		}
		regs[r] = v
	}
	reg := func(r vision.TowerReg) []float32 {
		if b := regs[r]; b != nil {
			return b
		}
		b := make([]float32, n*vision.TowerRegWidth(r, D, F))
		regs[r] = b
		return b
	}
	for i, op := range prog {
		if err := op.CheckRegs(); err != nil {
			return fmt.Errorf("op %d: %w", i, err)
		}
		if err := ly.checkOp(op, D, F); err != nil {
			return fmt.Errorf("op %d: %w", i, err)
		}
		var in []float32
		if op.ReadsIn() {
			in = reg(op.In)
		}
		switch op.Kind {
		case vision.TowerNorm:
			w, b, _ := ly.towerNorm(op.W)
			x := append([]float32(nil), in...)
			layerNormEps(x, w, b, n, D, op.Eps)
			set(op.Out, x)
		case vision.TowerLinear, vision.TowerLinearAdd:
			w, b, out, in2, _ := ly.towerLinear(op.W, D, F)
			y := linear(in, w, b, n, in2, out)
			if op.Kind == vision.TowerLinear {
				set(op.Out, y)
				continue
			}
			dst := reg(op.Out)
			for j := range dst {
				dst[j] += y[j]
			}
		case vision.TowerAttention:
			scale := float32(op.Scale)
			if e.defect == whisperEncDefectNoScale {
				scale = 1
			}
			set(op.Out, e.attend(reg(vision.TowerQ), reg(vision.TowerK), reg(vision.TowerV), n, window, scale))
		case vision.TowerGELUErf:
			e.geluInto(reg(op.Out))
		}
	}
	return nil
}

// checkOp refuses an op this encoder does not take: a kind it does not implement, a norm that is not a LayerNorm of a
// layer's norm with weight and bias of width D, and a linear whose weight or bias is missing, the wrong length, or does
// not fit its registers (vision.TowerOp.CheckShape).
func (ly *whisperLayer) checkOp(op vision.TowerOp, D, F int) error {
	switch op.Kind {
	case vision.TowerNorm:
		w, b, ok := ly.towerNorm(op.W)
		if !ok || op.Form != vision.TowerLayerNorm {
			return fmt.Errorf("norm of weight %d, form %d", op.W, op.Form)
		}
		if len(w) != D || len(b) != D {
			return fmt.Errorf("norm of weight %d has %d weights and %d biases, not %d", op.W, len(w), len(b), D)
		}
	case vision.TowerLinear, vision.TowerLinearAdd:
		w, b, out, in, ok := ly.towerLinear(op.W, D, F)
		if !ok {
			return fmt.Errorf("linear of weight %d", op.W)
		}
		if len(w) != out*in || len(b) != out {
			return fmt.Errorf("linear of weight %d holds %d values and %d biases, not [%d, %d] and %d", op.W, len(w), len(b), out, in, out)
		}
		return op.CheckShape(out, in, D, F)
	case vision.TowerAttention, vision.TowerGELUErf:
	default:
		return fmt.Errorf("kind %s not implemented on the Whisper encoder", op.Kind)
	}
	return nil
}

func (ly *whisperLayer) towerNorm(w vision.TowerWeight) (weight, bias []float32, ok bool) {
	switch w {
	case vision.TowerNorm1:
		return ly.ln1W, ly.ln1B, true
	case vision.TowerNorm2:
		return ly.ln2W, ly.ln2B, true
	}
	return nil, nil, false
}

// towerLinear is a projection's weight, bias and [out, in]. k has no bias, so it adds zeros, as the encoder always has.
func (ly *whisperLayer) towerLinear(w vision.TowerWeight, D, F int) (weight, bias []float32, out, in int, ok bool) {
	switch w {
	case vision.TowerWQ:
		return ly.qW, ly.qB, D, D, true
	case vision.TowerWK:
		return ly.kW, make([]float32, D), D, D, true
	case vision.TowerWV:
		return ly.vW, ly.vB, D, D, true
	case vision.TowerWO:
		return ly.oW, ly.oB, D, D, true
	case vision.TowerWFC1:
		return ly.fc1W, ly.fc1B, F, D, true
	case vision.TowerWFC2:
		return ly.fc2W, ly.fc2B, D, F, true
	}
	return nil, nil, 0, 0, false
}
