package decoder

import "fmt"

// A program is a model's forward written as data: an embedding stage, one list of operations per layer, and a head stage, built
// once at load from the resolved Architecture and the loaded weights (buildProgram). The CPU walks it (Model.walkToken), and its
// logits are bit-identical to the loops it replaces. Design, steps and gates: docs/tasks/task-forward-architecture-2026-10.md §4.8.
//
// Every implementation reads its parameters from the operation, never from the Architecture, so a parameter changed in the program
// changes the forward (gate L). Shapes (head counts, head and hidden widths) are not parameters: the weights fix them, and the
// implementations read them from the Architecture.
//
// A family with its own layer loop (ownForwards) has no layers yet; each moves when its step lands. Its embedding and head
// stages are built like any other family's.
type program struct {
	embed  embedParams
	layers [][]op // nil for a family that still runs its own loop
	head   headParams
}

// opKind is one kind of operation. The set is closed: a walker returns an error for a kind it does not implement.
type opKind uint8

const (
	opNorm      opKind = iota + 1 // in → out, the normalisation (or a copy, for post-only placement)
	opProjQKV                     // in → the stream's q, k, v
	opQKNorm                      // q, k in place
	opRope                        // q, k in place
	opAttnTemp                    // q in place
	opAttention                   // stores this position's k, v; q → the stream's attention context
	opOutGate                     // context in place, gated by a projection of in
	opProjO                       // context → out
	opFFN                         // in → out
	opResidual                    // the residual += regA (and regB)
)

// opReg names a per-stream buffer an operation reads or writes.
type opReg uint8

const (
	regH opReg = iota // the residual stream
	regX              // a sublayer's input
	regA              // a sublayer's output
	regB              // the parallel block's second output
)

// op is one operation: its kind, its operands, and the parameters of its kind (the other kinds' fields are zero).
type op struct {
	kind    opKind
	in, out opReg
	norm    normParams
	proj    projParams // opProjQKV and opProjO
	qkNorm  qkNormParams
	rope    ropeParams
	temp    attnTempParams
	attn    attentionParams
	gate    outGateParams
	ffn     ffnParams
	res     residualParams
}

// normSlot selects which of a layer's norm weights an opNorm applies.
type normSlot uint8

const (
	slotPreAttn normSlot = iota + 1
	slotPreMLP
	slotPostAttn
	slotPostMLP
	slotFinal
)

// normForm is the normalisation an opNorm computes. normCopy is post-only placement's input: the raw residual, unnormed.
type normForm uint8

const (
	normRMS normForm = iota + 1
	normLayer
	normCopy
)

// The parameter structs' fields are exported so gate L can set each one by reflection.
type normParams struct {
	Slot   normSlot
	Form   normForm
	Eps    float64
	AddOne bool // RMSNorm's (1+w)
	Bias   bool // LayerNorm's bias; RMSNorm has none
}

type projParams struct{ Bias bool }

type qkNormParams struct {
	Whole  bool // one statistic over all heads, not one per head
	Eps    float64
	AddOne bool
}

type ropeParams struct {
	InvFreq          []float64 // its length is the rotated half-width
	Mscale           float64
	Interleave       bool
	MRopeSection     []int
	MRopeInterleaved bool
}

type attnTempParams struct{ Beta, OrigMaxPos float64 }

// attnOrder is the CPU's reduction order for attention. Families compute attention in different orders today and OP3 moves no
// number, so each keeps its own; a GPU backend ignores it.
type attnOrder uint8

const (
	attnBatchedAcc64 attnOrder = iota + 1 // attendBatchedHeads with float64 accumulation
)

type attentionParams struct {
	Window  int // 0 attends every stored key; else the most recent Window positions
	Scale   float64
	Softcap float64 // 0 = none
	Order   attnOrder
}

type outGateParams struct{ Sigmoid bool } // sigmoid, else softplus

type ffnForm uint8

const (
	ffnGated ffnForm = iota + 1
	ffnNonGated
	ffnMoE
)

type ffnParams struct {
	Form ffnForm
	Act  ActKind
	MoE  MoEConfig // ffnMoE only
}

type residualParams struct{ Two bool } // adds regA and regB, the parallel block's single residual

type embedParams struct {
	Scale      float64 // 0 or 1 = none
	LearnedPos bool
}

type headParams struct {
	Norm       normParams
	Tied       bool    // the embedding doubles as the head
	Softcap    float64 // 0 = none
	LogitScale float64 // logits are divided by it; 0 or 1 = none
}

// buildProgram builds a's program. experts reports whether layer l carries MoE experts: a MoE family's leading dense layers
// (first_k_dense_replace) do not, and only the weights say so.
func buildProgram(a *Architecture, experts func(l int) bool) *program {
	p := &program{
		embed: embedParams{Scale: a.EmbedScale, LearnedPos: a.LearnedPosEmbed},
		head: headParams{
			Norm: normParams{Slot: slotFinal, Form: a.normForm(), Eps: a.NormEps, AddOne: a.RMSAddOne, Bias: true},
			Tied: a.TiedLMHead, Softcap: a.FinalLogitSoftcap, LogitScale: a.LogitScale,
		},
	}
	if _, own := a.ownForward(); own {
		return p
	}
	p.layers = make([][]op, a.NumLayers)
	for l := range a.NumLayers {
		p.layers[l] = a.layerOps(l, experts != nil && experts(l))
	}
	return p
}

func (a *Architecture) normForm() normForm {
	if a.Norm == NormLayer {
		return normLayer
	}
	return normRMS
}

// layerOps is one generic layer: the placement decides the order of norms, sublayers and residual adds.
func (a *Architecture) layerOps(l int, hasExperts bool) []op {
	placement := a.normPlacementAt(l)
	postOnly := placement == NormPostOnly
	postNorm := placement == NormSandwich4 || postOnly
	input := func(slot normSlot) op {
		if postOnly {
			return op{kind: opNorm, in: regH, out: regX, norm: normParams{Slot: slot, Form: normCopy}}
		}
		return op{kind: opNorm, in: regH, out: regX, norm: normParams{Slot: slot, Form: a.normForm(), Eps: a.NormEps, AddOne: a.RMSAddOne, Bias: true}}
	}
	post := func(slot normSlot) op {
		return op{kind: opNorm, in: regA, out: regA, norm: normParams{Slot: slot, Form: a.normForm(), Eps: a.NormEps, AddOne: a.RMSAddOne}}
	}
	ffn := func(out opReg) op {
		f := ffnParams{Form: ffnGated, Act: a.Act}
		switch {
		case a.MoE != nil && hasExperts:
			f.Form, f.MoE = ffnMoE, *a.MoE
		case a.NonGatedMLP:
			f.Form = ffnNonGated
		}
		return op{kind: opFFN, in: regX, out: out, ffn: f}
	}

	ops := []op{input(slotPreAttn)}
	ops = append(ops, a.attentionOps(l)...)
	if placement == NormParallel {
		// One input norm feeds both sublayers, and their outputs meet in one residual add.
		return append(ops, ffn(regB), op{kind: opResidual, res: residualParams{Two: true}})
	}
	if postNorm {
		ops = append(ops, post(slotPostAttn))
	}
	ops = append(ops, op{kind: opResidual}, input(slotPreMLP), ffn(regA))
	if postNorm {
		ops = append(ops, post(slotPostMLP))
	}
	return append(ops, op{kind: opResidual})
}

// attentionOps is the generic attention sublayer, regX → regA.
func (a *Architecture) attentionOps(l int) []op {
	ops := []op{{kind: opProjQKV, in: regX, proj: projParams{Bias: a.QKVBias}}}
	if a.QKNorm {
		ops = append(ops, op{kind: opQKNorm, qkNorm: qkNormParams{Whole: a.QKNormWhole, Eps: a.NormEps, AddOne: a.RMSAddOne}})
	}
	if !a.LearnedPosEmbed && !a.isNoPELayer(l) {
		ops = append(ops, op{kind: opRope, rope: ropeParams{
			InvFreq: a.ropeInvFreq(l), Mscale: a.ropeMscale(l), Interleave: a.ropeInterleave,
			MRopeSection: a.MRopeSection, MRopeInterleaved: a.MRopeInterleaved,
		}})
	}
	if a.AttnTempBeta != 0 {
		ops = append(ops, op{kind: opAttnTemp, temp: attnTempParams{Beta: a.AttnTempBeta, OrigMaxPos: a.AttnTempOrigMaxPos}})
	}
	window := 0
	if !a.isGlobalLayer(l) {
		window = a.SlidingWindow
	}
	ops = append(ops, op{kind: opAttention, attn: attentionParams{Window: window, Scale: a.AttnScale, Softcap: a.AttnLogitSoftcap, Order: attnBatchedAcc64}})
	if a.hasAttnOutputGate() {
		ops = append(ops, op{kind: opOutGate, in: regX, gate: outGateParams{Sigmoid: a.AttnGate == GateSigmoid}})
	}
	return append(ops, op{kind: opProjO, out: regA, proj: projParams{Bias: a.OutBias}})
}

func (k opKind) String() string {
	names := [...]string{opNorm: "norm", opProjQKV: "proj-qkv", opQKNorm: "qk-norm", opRope: "rope", opAttnTemp: "attn-temp",
		opAttention: "attention", opOutGate: "out-gate", opProjO: "proj-o", opFFN: "ffn", opResidual: "residual"}
	if int(k) < len(names) && names[k] != "" {
		return names[k]
	}
	return fmt.Sprintf("op(%d)", uint8(k))
}

// start is the first key position a query at pos attends.
func (p attentionParams) start(pos int) int {
	if p.Window <= 0 {
		return 0
	}
	return max(pos-p.Window+1, 0)
}

// attnParamsFor is the attention parameters the loops that do not walk a program read from the Architecture: the cache's window
// on a local layer (the window the cache was built with), every key on a global one.
func (a *Architecture) attnParamsFor(cache *KVCache, global bool) attentionParams {
	p := attentionParams{Scale: a.AttnScale, Softcap: a.AttnLogitSoftcap, Order: attnBatchedAcc64}
	if !global && cache.window > 0 {
		p.Window = cache.window
	}
	return p
}
