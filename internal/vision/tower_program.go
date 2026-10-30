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
	TowerHeadNorm                         // Out in place: the norm Form with Eps and weight W over each head's head_dim values
	TowerLinearQKV                        // In → Q, K, V: the fused [3·hidden, hidden] W, rows q then k then v, with its bias
	TowerRoPE                             // Out in place: NeoX rotate-half RoPE from the forward's [rows, head_dim] tables
	TowerSiLUMul                          // Out in place: silu(Out) · In
	TowerAdd                              // Out += In
	TowerGELUMul                          // Out in place: gelu_tanh(Out) · In
	TowerRoPEAxial                        // Out in place: 2-D axial RoPE, rotate-half within each half of every head
)

// TowerReg names one of a walker's activation buffers. TowerX is the residual stream, [rows, hidden]; TowerMid and
// TowerMid2 are [rows, intermediate]; the others are [rows, hidden].
type TowerReg uint8

const (
	TowerX TowerReg = iota + 1
	TowerXN
	TowerQ
	TowerK
	TowerV
	TowerCtx
	TowerMid
	TowerMid2
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
	TowerWQKV
	TowerQNorm
	TowerKNorm
	TowerWGate
	TowerWUp
	TowerWDown
	TowerNormPostAttn // the norm of the attention's output projection, before its residual add
	TowerNormPostMLP  // the norm of the MLP's output, before its residual add
)

// TowerNormForm is a norm's formula.
type TowerNormForm uint8

const (
	TowerLayerNorm       TowerNormForm = iota + 1 // mean and variance, weight and bias
	TowerRMSNorm                                  // root mean square, weight only
	TowerRMSNormUnscaled                          // root mean square, no weight (W is unset)
)

// TowerOp is one operation. Fields a kind does not read are zero.
type TowerOp struct {
	Kind    TowerOpKind
	In, Out TowerReg
	W       TowerWeight
	Form    TowerNormForm // TowerNorm, TowerHeadNorm
	Eps     float64       // TowerNorm, TowerHeadNorm
	Scale   float64       // TowerAttention
}

// TowerBlock is one block's operations, in order.
type TowerBlock []TowerOp

// TowerProgramPlantForTest, when set, rewrites every program a constructor below returns, given the program's name
// (siglip, qwen3, glm-ocr, gemma4, qwen2.5): the seam the programs' plants (G-SB9M-2) use on every backend, which all
// read their program from here. Nil in production and in every other test.
var TowerProgramPlantForTest func(program string, p TowerBlock) TowerBlock

func planted(program string, p TowerBlock) TowerBlock {
	if TowerProgramPlantForTest != nil {
		return TowerProgramPlantForTest(program, p)
	}
	return p
}

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
	return planted("siglip", p)
}

// Qwen3Program is the Qwen3-VL tower's block: LayerNorm, the fused qkv, RoPE on q and k, attention at 1/sqrt(head_dim)
// within each frame, proj added to the residual, LayerNorm, fc1, GELU-tanh, fc2 added to the residual.
func Qwen3Program(c Qwen3EncoderConfig) TowerBlock {
	p := TowerBlock{
		{Kind: TowerNorm, In: TowerX, Out: TowerXN, W: TowerNorm1, Form: TowerLayerNorm, Eps: qwen3LNEps},
		{Kind: TowerLinearQKV, In: TowerXN, W: TowerWQKV},
		{Kind: TowerRoPE, Out: TowerQ},
		{Kind: TowerRoPE, Out: TowerK},
		{Kind: TowerAttention, Out: TowerCtx, Scale: 1 / math.Sqrt(float64(c.HiddenSize/c.NumHeads))},
		{Kind: TowerLinearAdd, In: TowerCtx, Out: TowerX, W: TowerWO},
		{Kind: TowerNorm, In: TowerX, Out: TowerXN, W: TowerNorm2, Form: TowerLayerNorm, Eps: qwen3LNEps},
		{Kind: TowerLinear, In: TowerXN, Out: TowerMid, W: TowerWFC1},
		{Kind: TowerGELUTanh, Out: TowerMid},
		{Kind: TowerLinearAdd, In: TowerMid, Out: TowerX, W: TowerWFC2},
	}
	return planted("qwen3", p)
}

// GlmOcrProgram is GLM-OCR's tower block: RMSNorm, the fused qkv, the per-head RMSNorm of q and k, RoPE on q and k,
// attention at 1/sqrt(head_dim) within each frame, proj added to the residual, RMSNorm, the gate and up linears,
// silu(gate) · up, down added to the residual.
func GlmOcrProgram(c GlmOcrEncoderConfig) TowerBlock {
	eps := c.RMSNormEps
	p := TowerBlock{
		{Kind: TowerNorm, In: TowerX, Out: TowerXN, W: TowerNorm1, Form: TowerRMSNorm, Eps: eps},
		{Kind: TowerLinearQKV, In: TowerXN, W: TowerWQKV},
		{Kind: TowerHeadNorm, Out: TowerQ, W: TowerQNorm, Form: TowerRMSNorm, Eps: eps},
		{Kind: TowerHeadNorm, Out: TowerK, W: TowerKNorm, Form: TowerRMSNorm, Eps: eps},
		{Kind: TowerRoPE, Out: TowerQ},
		{Kind: TowerRoPE, Out: TowerK},
		{Kind: TowerAttention, Out: TowerCtx, Scale: 1 / math.Sqrt(float64(c.HiddenSize/c.NumHeads))},
		{Kind: TowerLinearAdd, In: TowerCtx, Out: TowerX, W: TowerWO},
		{Kind: TowerNorm, In: TowerX, Out: TowerXN, W: TowerNorm2, Form: TowerRMSNorm, Eps: eps},
		{Kind: TowerLinear, In: TowerXN, Out: TowerMid, W: TowerWGate},
		{Kind: TowerLinear, In: TowerXN, Out: TowerMid2, W: TowerWUp},
		{Kind: TowerSiLUMul, In: TowerMid2, Out: TowerMid},
		{Kind: TowerLinearAdd, In: TowerMid, Out: TowerX, W: TowerWDown},
	}
	return planted("glm-ocr", p)
}

// Gemma4Program is Gemma 4 vision's sandwich-norm block: RMSNorm; the q, k and v linears (clipped: the tower's weights
// carry clamp bounds); a per-head RMSNorm on q and k and an unscaled one on v; axial RoPE on q and k; attention at scale
// 1.0; the o linear, an RMSNorm of its output and its residual add; RMSNorm; the gate and up linears, the GELU-tanh
// product and the down linear; an RMSNorm of its output and its residual add.
func Gemma4Program(c Gemma4EncoderConfig) TowerBlock {
	eps := c.RMSNormEps
	p := TowerBlock{
		{Kind: TowerNorm, In: TowerX, Out: TowerXN, W: TowerNorm1, Form: TowerRMSNorm, Eps: eps},
		{Kind: TowerLinear, In: TowerXN, Out: TowerQ, W: TowerWQ},
		{Kind: TowerLinear, In: TowerXN, Out: TowerK, W: TowerWK},
		{Kind: TowerLinear, In: TowerXN, Out: TowerV, W: TowerWV},
		{Kind: TowerHeadNorm, Out: TowerQ, W: TowerQNorm, Form: TowerRMSNorm, Eps: eps},
		{Kind: TowerHeadNorm, Out: TowerK, W: TowerKNorm, Form: TowerRMSNorm, Eps: eps},
		{Kind: TowerHeadNorm, Out: TowerV, Form: TowerRMSNormUnscaled, Eps: eps},
		{Kind: TowerRoPEAxial, Out: TowerQ},
		{Kind: TowerRoPEAxial, Out: TowerK},
		{Kind: TowerAttention, Out: TowerCtx, Scale: 1},
		{Kind: TowerLinear, In: TowerCtx, Out: TowerXN, W: TowerWO},
		{Kind: TowerNorm, In: TowerXN, Out: TowerXN, W: TowerNormPostAttn, Form: TowerRMSNorm, Eps: eps},
		{Kind: TowerAdd, In: TowerXN, Out: TowerX},
		{Kind: TowerNorm, In: TowerX, Out: TowerXN, W: TowerNorm2, Form: TowerRMSNorm, Eps: eps},
		{Kind: TowerLinear, In: TowerXN, Out: TowerMid, W: TowerWGate},
		{Kind: TowerLinear, In: TowerXN, Out: TowerMid2, W: TowerWUp},
		{Kind: TowerGELUMul, In: TowerMid2, Out: TowerMid},
		{Kind: TowerLinear, In: TowerMid, Out: TowerXN, W: TowerWDown},
		{Kind: TowerNorm, In: TowerXN, Out: TowerXN, W: TowerNormPostMLP, Form: TowerRMSNorm, Eps: eps},
		{Kind: TowerAdd, In: TowerXN, Out: TowerX},
	}
	return planted("gemma4", p)
}

// CheckRegs reports an op whose registers do not fit its kind: a norm from and into a hidden-wide register; the SiLU
// and GELU products between the two intermediate-wide ones; a residual add between two different hidden-wide ones;
// attention writing ctx; an in-place op (GELU-tanh, RoPE, axial RoPE, the per-head norm) naming no In; the fused qkv
// naming no Out. It also reports a norm reading a weight of the wrong width: a norm reads a hidden-wide norm weight, a
// per-head norm q's or k's, and the unscaled per-head norm none. A linear's widths follow its weight, so each walker
// checks those (CheckShape).
func (op TowerOp) CheckRegs() error {
	known := func(r TowerReg) bool { return r >= TowerX && r <= TowerMid2 }
	hiddenWide := func(r TowerReg) bool { return known(r) && r != TowerMid && r != TowerMid2 }
	ok := true
	switch op.Kind {
	case TowerNorm:
		ok = hiddenWide(op.In) && hiddenWide(op.Out)
	case TowerHeadNorm, TowerRoPE, TowerRoPEAxial:
		ok = op.In == 0 && hiddenWide(op.Out)
	case TowerGELUTanh:
		ok = op.In == 0 && known(op.Out)
	case TowerSiLUMul, TowerGELUMul:
		ok = (op.In == TowerMid || op.In == TowerMid2) && (op.Out == TowerMid || op.Out == TowerMid2) && op.In != op.Out
	case TowerAdd:
		ok = hiddenWide(op.In) && hiddenWide(op.Out) && op.In != op.Out
	case TowerAttention:
		ok = op.In == 0 && op.Out == TowerCtx
	case TowerLinearQKV:
		ok = known(op.In) && op.Out == 0
	case TowerLinear, TowerLinearAdd:
		ok = known(op.In) && known(op.Out)
	default:
		return fmt.Errorf("kind %s unknown", op.Kind)
	}
	if !ok {
		return fmt.Errorf("%s does not take registers %d → %d", op.Kind, op.In, op.Out)
	}
	switch hiddenNormW := op.W == TowerNorm1 || op.W == TowerNorm2 || op.W == TowerNormPostAttn || op.W == TowerNormPostMLP; {
	case op.Kind == TowerNorm:
		ok = hiddenNormW
	case op.Kind == TowerHeadNorm && op.Form == TowerRMSNormUnscaled:
		ok = op.W == 0
	case op.Kind == TowerHeadNorm:
		ok = op.W == TowerQNorm || op.W == TowerKNorm
	}
	if !ok {
		return fmt.Errorf("%s (form %d) does not take weight %d", op.Kind, op.Form, op.W)
	}
	return nil
}

// TowerRegWidth is a register's row width: intermediate for TowerMid and TowerMid2, hidden for the others, and 0 for no
// register or an unknown one.
func TowerRegWidth(r TowerReg, hidden, inter int) int {
	switch {
	case r == TowerMid || r == TowerMid2:
		return inter
	case r >= TowerX && r <= TowerMid2:
		return hidden
	}
	return 0
}

// CheckShape reports a linear op whose weight [out, in] does not fit its registers: in must be the In register's width
// and out the Out register's, or three times hidden for the fused qkv. A register of no width fails. Other kinds take no
// weight matrix and pass.
func (op TowerOp) CheckShape(out, in, hidden, inter int) error {
	var wantOut int
	switch op.Kind {
	case TowerLinear, TowerLinearAdd:
		wantOut = TowerRegWidth(op.Out, hidden, inter)
	case TowerLinearQKV:
		wantOut = 3 * hidden
	default:
		return nil
	}
	if wantIn := TowerRegWidth(op.In, hidden, inter); wantOut == 0 || wantIn == 0 || out != wantOut || in != wantIn {
		return fmt.Errorf("%s: weight %d is [%d, %d], registers %d → %d take [%d, %d]", op.Kind, op.W, out, in, op.In, op.Out, wantOut, wantIn)
	}
	return nil
}

// Qwen25Program is Qwen2.5-VL's tower block: RMSNorm (eps 1e-6), the fused qkv, RoPE on q and k, attention at
// 1/sqrt(head_dim) within each of the block's segments (its windows, or the whole frame in a full-attention block),
// proj added to the residual, RMSNorm, the gate and up linears, silu(gate) · up, down added to the residual.
func Qwen25Program(c QwenEncoderConfig) TowerBlock {
	const eps = 1e-6 // rmsNormRows'
	p := TowerBlock{
		{Kind: TowerNorm, In: TowerX, Out: TowerXN, W: TowerNorm1, Form: TowerRMSNorm, Eps: eps},
		{Kind: TowerLinearQKV, In: TowerXN, W: TowerWQKV},
		{Kind: TowerRoPE, Out: TowerQ},
		{Kind: TowerRoPE, Out: TowerK},
		{Kind: TowerAttention, Out: TowerCtx, Scale: 1 / math.Sqrt(float64(c.HiddenSize/c.NumHeads))},
		{Kind: TowerLinearAdd, In: TowerCtx, Out: TowerX, W: TowerWO},
		{Kind: TowerNorm, In: TowerX, Out: TowerXN, W: TowerNorm2, Form: TowerRMSNorm, Eps: eps},
		{Kind: TowerLinear, In: TowerXN, Out: TowerMid, W: TowerWGate},
		{Kind: TowerLinear, In: TowerXN, Out: TowerMid2, W: TowerWUp},
		{Kind: TowerSiLUMul, In: TowerMid2, Out: TowerMid},
		{Kind: TowerLinearAdd, In: TowerMid, Out: TowerX, W: TowerWDown},
	}
	return planted("qwen2.5", p)
}

// ReadsIn reports whether the op reads its In register.
func (op TowerOp) ReadsIn() bool {
	switch op.Kind {
	case TowerNorm, TowerLinear, TowerLinearAdd, TowerLinearQKV, TowerSiLUMul, TowerAdd, TowerGELUMul:
		return true
	}
	return false
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
	case TowerHeadNorm:
		return "head-norm"
	case TowerLinearQKV:
		return "linear-qkv"
	case TowerRoPE:
		return "rope"
	case TowerSiLUMul:
		return "silu-mul"
	case TowerAdd:
		return "add"
	case TowerGELUMul:
		return "gelu-mul"
	case TowerRoPEAxial:
		return "rope-axial"
	}
	return fmt.Sprintf("TowerOpKind(%d)", uint8(k))
}
