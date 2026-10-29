package vision

import (
	"fmt"
	"math"
)

// A tower block as a program (docs/tasks/task-subsystems-2026-10.md §8, "SB9 step M"): one description of a block's
// operations, which the CPU and every device backend walk over their own operation layers. Every block of a tower runs
// the same program over its own weights. A walker implements each kind exactly as its hand-written loop did, so a tower's
// output is bit-identical to its golden (internal/towerid); a fusion is its own kind (TowerLinearAdd), never folded in by
// a walker.

// TowerOpKind is one kind of tower operation. The set is closed: a walker returns an error for a kind it does not
// implement.
type TowerOpKind uint8

const (
	TowerNorm      TowerOpKind = iota + 1 // In → Out: the norm Form with Eps, weight (and bias) W
	TowerLinear                           // In → Out: In · Wᵀ + bias
	TowerAttention                        // Q, K, V → Out: bidirectional attention at Scale over each segment
	TowerLinearAdd                        // Out += In · Wᵀ + bias: the residual add is part of the operation
	TowerGELUTanh                         // Out in place: tanh-approximated GELU
)

// TowerReg names one of a walker's activation buffers. TowerX is the residual stream, [rows, hidden]; TowerMid is
// [rows, intermediate]; the others are [rows, hidden].
type TowerReg uint8

const (
	TowerX TowerReg = iota + 1
	TowerXN
	TowerQ
	TowerK
	TowerV
	TowerCtx
	TowerMid
)

// TowerWeight names one weight of a block: a norm's weight and bias, or a projection's matrix and bias.
type TowerWeight uint8

const (
	TowerNorm1 TowerWeight = iota + 1
	TowerNorm2
	TowerWQ
	TowerWK
	TowerWV
	TowerWO
	TowerWFC1
	TowerWFC2
)

// TowerNormForm is a norm's formula.
type TowerNormForm uint8

const (
	TowerLayerNorm TowerNormForm = iota + 1 // mean and variance, weight and bias
)

// TowerOp is one operation. Fields a kind does not read are zero.
type TowerOp struct {
	Kind    TowerOpKind
	In, Out TowerReg
	W       TowerWeight
	Form    TowerNormForm // TowerNorm
	Eps     float64       // TowerNorm
	Scale   float64       // TowerAttention
}

// TowerBlock is one block's operations, in order.
type TowerBlock []TowerOp

// TowerProgramPlantForTest, when set, rewrites every program SiglipProgram returns: the seam the program's plants (G-SB9M-2)
// use on every backend, which all read their program from here. Nil in production and in every other test.
var TowerProgramPlantForTest func(TowerBlock) TowerBlock

// SiglipProgram is SigLIP's pre-norm block: LayerNorm, the q, k and v linears, attention at 1/sqrt(head_dim), the output
// projection added to the residual, LayerNorm, fc1, GELU-tanh, fc2 added to the residual.
func SiglipProgram(c EncoderConfig) TowerBlock {
	hd := c.HiddenSize / c.NumAttentionHeads
	p := TowerBlock{
		{Kind: TowerNorm, In: TowerX, Out: TowerXN, W: TowerNorm1, Form: TowerLayerNorm, Eps: c.LayerNormEps},
		{Kind: TowerLinear, In: TowerXN, Out: TowerQ, W: TowerWQ},
		{Kind: TowerLinear, In: TowerXN, Out: TowerK, W: TowerWK},
		{Kind: TowerLinear, In: TowerXN, Out: TowerV, W: TowerWV},
		{Kind: TowerAttention, Out: TowerCtx, Scale: 1 / math.Sqrt(float64(hd))},
		{Kind: TowerLinearAdd, In: TowerCtx, Out: TowerX, W: TowerWO},
		{Kind: TowerNorm, In: TowerX, Out: TowerXN, W: TowerNorm2, Form: TowerLayerNorm, Eps: c.LayerNormEps},
		{Kind: TowerLinear, In: TowerXN, Out: TowerMid, W: TowerWFC1},
		{Kind: TowerGELUTanh, Out: TowerMid},
		{Kind: TowerLinearAdd, In: TowerMid, Out: TowerX, W: TowerWFC2},
	}
	if TowerProgramPlantForTest != nil {
		p = TowerProgramPlantForTest(p)
	}
	return p
}

func (k TowerOpKind) String() string {
	switch k {
	case TowerNorm:
		return "norm"
	case TowerLinear:
		return "linear"
	case TowerAttention:
		return "attention"
	case TowerLinearAdd:
		return "linear-add"
	case TowerGELUTanh:
		return "gelu-tanh"
	}
	return fmt.Sprintf("TowerOpKind(%d)", uint8(k))
}
