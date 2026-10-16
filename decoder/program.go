package decoder

import (
	"fmt"
	"slices"
)

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
	opMLA                         // in → out, the whole Multi-head Latent Attention sublayer (mlaAttentionWith)
	opKDA                         // in → out, the whole Kimi Delta Attention mixer (kdaMixerStep), on the layer's recurrent state
	opShortConv                   // in → out, LFM2's gated short convolution (shortConvStep), on the layer's conv window
	opMamba2                      // in → out, the Mamba-2 selective-scan mixer (mamba2Step), on the layer's SSM state
)

// recurrent reports whether the operation mutates per-layer state in place, which no positional rewind restores.
func (k opKind) recurrent() bool { return k == opKDA || k == opShortConv || k == opMamba2 }

// genericKinds are the operations of the generic layer: the ones the batched and multi-sequence walkers implement, LoRA's
// deltas reach (q/k/v, o and the gated FFN), and the layer pager's spans cover.
var genericKinds = []opKind{opNorm, opProjQKV, opQKNorm, opRope, opAttnTemp, opAttention, opOutGate, opProjO, opFFN, opResidual}

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
	mla     mlaOpParams
	kda     kdaOpParams
	mamba   mamba2OpParams
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

// qkNormForm is the QK-norm an opQKNorm computes.
type qkNormForm uint8

const (
	qkRMS qkNormForm = iota + 1 // weighted RMSNorm (rmsNorm), four-way sums
	qkL2                        // Llama 4: parameter-free RMS over each head, sequential float64 (l2NormHeads)
)

type qkNormParams struct {
	Form   qkNormForm
	Whole  bool // one statistic over all heads, not one per head (qkRMS)
	Eps    float64
	AddOne bool
}

// ropeKernel is the rotation an opRope computes.
type ropeKernel uint8

const (
	ropeAtKernel  ropeKernel = iota + 1 // ropeAt: NeoX or GPT-J pairing, m-RoPE
	ropeMLAKernel                       // mlaRope: GPT-J pairing by de-interleaving, then NeoX (Llama 4, MLA)
)

type ropeParams struct {
	Kernel           ropeKernel
	InvFreq          []float64 // its length is the rotated half-width
	Mscale           float64
	Interleave       bool
	MRopeSection     []int
	MRopeInterleaved bool
}

// attnTempParams scales the query by 1 + Beta·log1p(floor((pos+PosOffset)/OrigMaxPos)).
type attnTempParams struct {
	Beta, OrigMaxPos float64
	PosOffset        int
}

// attnOrder is the CPU's reduction order for attention. Families compute attention in different orders today and OP3 moves no
// number, so each keeps its own; a GPU backend ignores it.
type attnOrder uint8

const (
	attnBatchedAcc64 attnOrder = iota + 1 // attendBatchedHeads with float64 accumulation
	attnSinksF64                          // gpt-oss: inline float64 dots and softmax with a learned sink per head (attendSinksToken)
	attnQuery                             // attendQuery: float64 dots, float32 accumulation of the values
	attnMLA                               // the family attends through opMLA, over a latent cache
)

type attentionParams struct {
	Window  int // 0 attends every stored key; else the most recent Window positions
	Chunk   int // 0 none; else a query attends only within its own block of Chunk positions (Llama 4's RoPE layers)
	Scale   float64
	Softcap float64 // 0 = none
	Order   attnOrder
}

type outGateParams struct{ Sigmoid bool } // sigmoid, else softplus

// mlaOpParams are an opMLA's: the rotation of the rope-carrying dims, the latent and q-LoRA norms' epsilon, the score scale, and
// whether the optional sigmoid output gate is per head. The ranks and head widths are shapes, read from the Architecture.
type mlaOpParams struct {
	InvFreq    []float64
	Mscale     float64
	Interleave bool
	Eps        float64
	Scale      float64
	HeadWise   bool
}

// kdaOpParams are an opKDA's: the gated output norm's epsilon and the decay gate's lower bound. The head and conv sizes, and
// whether the gates carry a LoRA, are shapes, read from the Architecture; the CPU implements the safe (bounded) decay gate only,
// and the adapter refuses a checkpoint without it.
type kdaOpParams struct {
	Eps        float64
	LowerBound float64
}

// mlaOpParamsAt is layer l's opMLA parameters.
func (a *Architecture) mlaOpParamsAt(l int) mlaOpParams {
	return mlaOpParams{InvFreq: a.ropeInvFreq(l), Mscale: a.ropeMscale(l), Interleave: a.mla.ropeInterleave, Eps: a.NormEps,
		Scale: a.AttnScale, HeadWise: a.mla.GateGranularity == "head_wise"}
}

type ffnForm uint8

const (
	ffnGated ffnForm = iota + 1
	ffnNonGated
	ffnMoE
	ffnGptOss    // gpt-oss: a router with a logit bias, softmax over the top k, clamped SwiGLU experts with biases (gptOssMoEWith)
	ffnLlama4MoE // Llama 4: top-k by logit, each expert's INPUT scaled by sigmoid(logit), plus an ungated shared expert (llama4MoEWith)
)

type ffnParams struct {
	Form         ffnForm
	Act          ActKind
	MoE          MoEConfig // ffnMoE and ffnGptOss
	Alpha, Limit float64   // ffnGptOss: the gate's sigmoid gain and the clamp bound
}

// residualParams: Two adds regA and regB, the parallel block's single residual; Scale, when not 0, multiplies what is added
// (Granite's residual_multiplier).
type residualParams struct {
	Two   bool
	Scale float64
}

// mamba2OpParams are an opMamba2's: the gated norm's epsilon and its group count (1 normalises the whole inner width). The
// heads, state and conv sizes are shapes, read from the Architecture.
type mamba2OpParams struct {
	Eps        float64
	NormGroups int
}

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
		embed: embedParams{Scale: a.embedScale(), LearnedPos: a.LearnedPosEmbed},
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

// embedScale is the factor the embedding stage multiplies by: Granite's embedding_multiplier (1 under GOINFER_SSM_NOMUL), else
// EmbedScale.
func (a *Architecture) embedScale() float64 {
	if a.granite != nil {
		if ssmNoMul {
			return 1
		}
		return float64(a.granite.EmbMul)
	}
	return a.EmbedScale
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
		case a.gptoss != nil:
			f.Form, f.MoE, f.Alpha, f.Limit = ffnGptOss, *a.MoE, a.gptoss.SwigluAlpha, a.gptoss.SwigluLimit
		case a.MoE != nil && hasExperts:
			f.Form, f.MoE = ffnMoE, *a.MoE
		case a.NonGatedMLP:
			f.Form = ffnNonGated
		}
		return op{kind: opFFN, in: regX, out: out, ffn: f}
	}

	if a.llama4 != nil {
		return a.llama4LayerOps(l, input, ffn)
	}
	if a.granite != nil {
		return a.graniteLayerOps(l, input, ffn)
	}
	if a.lfm2 != nil && a.isConvLayer(l) {
		return []op{input(slotPreAttn), {kind: opShortConv, in: regX, out: regA}, {kind: opResidual}, input(slotPreMLP), ffn(regA), {kind: opResidual}}
	}
	if a.mla != nil {
		mixer := op{kind: opMLA, in: regX, out: regA, mla: a.mlaOpParamsAt(l)}
		if a.kda != nil && a.isLinearLayer(l) { // Bailing Hybrid: KDA on the linear layers, MLA on the rest
			mixer = op{kind: opKDA, in: regX, out: regA, kda: kdaOpParams{Eps: a.NormEps, LowerBound: a.kda.LowerBound}}
		}
		return []op{input(slotPreAttn), mixer, {kind: opResidual}, input(slotPreMLP), ffn(regA), {kind: opResidual}}
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
		ops = append(ops, op{kind: opQKNorm, qkNorm: qkNormParams{Form: qkRMS, Whole: a.QKNormWhole, Eps: a.NormEps, AddOne: a.RMSAddOne}})
	}
	if !a.LearnedPosEmbed && !a.isNoPELayer(l) {
		ops = append(ops, op{kind: opRope, rope: ropeParams{
			Kernel: ropeAtKernel, InvFreq: a.ropeInvFreq(l), Mscale: a.ropeMscale(l), Interleave: a.ropeInterleave,
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
	ops = append(ops, op{kind: opAttention, attn: attentionParams{Window: window, Scale: a.AttnScale, Softcap: a.AttnLogitSoftcap, Order: a.attnOrder()}})
	if a.hasAttnOutputGate() {
		ops = append(ops, op{kind: opOutGate, in: regX, gate: outGateParams{Sigmoid: a.AttnGate == GateSigmoid}})
	}
	return append(ops, op{kind: opProjO, out: regA, proj: projParams{Bias: a.OutBias}})
}

func (k opKind) String() string {
	names := [...]string{opNorm: "norm", opProjQKV: "proj-qkv", opQKNorm: "qk-norm", opRope: "rope", opAttnTemp: "attn-temp",
		opAttention: "attention", opOutGate: "out-gate", opProjO: "proj-o", opFFN: "ffn", opResidual: "residual", opMLA: "mla", opKDA: "kda", opShortConv: "short-conv", opMamba2: "mamba2"}
	if int(k) < len(names) && names[k] != "" {
		return names[k]
	}
	return fmt.Sprintf("op(%d)", uint8(k))
}

// recurrent reports whether a layer carries state mutated in place per token (a conv window, an SSM state, a linear-attention
// state), which no positional rewind restores: from the own-loop table for a family still listed there, else from its
// operations. Speculative rollback and the hybrid image prefill read it before any cache exists.
func (a *Architecture) recurrent() bool {
	if f, own := a.ownForward(); own {
		return f.Recurrent
	}
	for l := range a.NumLayers {
		for _, o := range a.layerOps(l, false) {
			if o.kind.recurrent() {
				return true
			}
		}
	}
	return false
}

// genericKindsOnly reports whether every layer of a is made of genericKinds, so the generic machinery around the layers applies:
// LoRA, and the dense layer pager (with the fit guard's retry that mirrors it). False for a family still in the own-loop table.
func (a *Architecture) genericKindsOnly() bool {
	if _, own := a.ownForward(); own {
		return false
	}
	for l := range a.NumLayers {
		for _, o := range a.layerOps(l, false) {
			if !slices.Contains(genericKinds, o.kind) {
				return false
			}
		}
	}
	return true
}

// attnOrder is the attention kernel this family's CPU forward computes in.
func (a *Architecture) attnOrder() attnOrder {
	switch {
	case a.gptoss != nil:
		return attnSinksF64
	case a.llama4 != nil:
		return attnQuery
	case a.mla != nil:
		return attnMLA
	case a.lfm2 != nil, a.granite != nil:
		return attnQuery
	}
	return attnBatchedAcc64
}

// graniteLayerOps is a Granite-4.0-H layer: pre-norm, a Mamba-2 mixer or GQA attention, each sublayer's output scaled by
// residual_multiplier before its add, and a MoE FFN. The isolation seams GOINFER_SSM_NOMUL (multipliers 1) and
// GOINFER_SSM_SKIPFFN (no FFN) shape the program, as they shaped the loop it replaced.
func (a *Architecture) graniteLayerOps(l int, input func(normSlot) op, ffn func(opReg) op) []op {
	resid := op{kind: opResidual, res: residualParams{Scale: float64(a.granite.ResidMul)}}
	if ssmNoMul {
		resid.res.Scale = 1
	}
	ops := []op{input(slotPreAttn)}
	if a.isMambaLayer(l) {
		ops = append(ops, op{kind: opMamba2, in: regX, out: regA, mamba: mamba2OpParams{Eps: a.NormEps, NormGroups: 1}})
	} else {
		ops = append(ops, a.attentionOps(l)...)
	}
	ops = append(ops, resid)
	if ssmSkipFFN {
		return ops
	}
	return append(ops, input(slotPreMLP), ffn(regA), resid)
}

// mamba2Shapes is the Mamba-2 geometry of a Granite or Nemotron-H layer, with gated-norm group count g.
func (a *Architecture) mamba2Shapes(g int) mamba2Params {
	switch {
	case a.granite != nil:
		p := a.granite
		return mamba2Params{NHeads: p.NHeads, HeadDim: p.HeadDim, DState: p.DState, NGroups: p.NGroups, DConv: p.DConv, Hidden: a.HiddenDim, NormGroups: g}
	case a.nemotron != nil:
		p := a.nemotron
		return mamba2Params{NHeads: p.NHeads, HeadDim: p.HeadDim, DState: p.DState, NGroups: p.NGroups, DConv: p.DConv, Hidden: a.HiddenDim, NormGroups: g}
	}
	return mamba2Params{}
}

// llama4LayerOps is a Llama 4 layer: pre-norm, and attention that varies by layer. A RoPE layer rotates q and k (mlaRope's
// interleaved form, no mscale), then normalises them without weights, and attends within its chunk; a NoPE layer scales the
// query by the attention temperature instead. The FFN is dense or Llama 4's MoE (moe_layers).
func (a *Architecture) llama4LayerOps(l int, input func(normSlot) op, ffn func(opReg) op) []op {
	lp := a.llama4
	ops := []op{input(slotPreAttn), {kind: opProjQKV, in: regX}}
	chunk := 0
	if lp.useRope[l] {
		ops = append(ops, op{kind: opRope, rope: ropeParams{Kernel: ropeMLAKernel, InvFreq: a.ropeInvFreq(l), Mscale: 1, Interleave: true}})
		if lp.useQKNorm {
			ops = append(ops, op{kind: opQKNorm, qkNorm: qkNormParams{Form: qkL2, Eps: a.NormEps}})
		}
		chunk = lp.chunkSize
	} else if lp.attnTemp {
		ops = append(ops, op{kind: opAttnTemp, temp: attnTempParams{Beta: lp.attnScale, OrigMaxPos: lp.floorScale, PosOffset: 1}})
	}
	ops = append(ops,
		op{kind: opAttention, attn: attentionParams{Chunk: chunk, Scale: a.AttnScale, Softcap: a.AttnLogitSoftcap, Order: attnQuery}},
		op{kind: opProjO, out: regA},
		op{kind: opResidual}, input(slotPreMLP))
	f := ffn(regA)
	if lp.isMoE[l] {
		f.ffn = ffnParams{Form: ffnLlama4MoE, Act: a.Act, MoE: *a.MoE}
	}
	return append(ops, f, op{kind: opResidual})
}

// start is the first key position a query at pos attends: past the window, and inside the query's chunk.
func (p attentionParams) start(pos int) int {
	s := 0
	if p.Window > 0 {
		s = max(pos-p.Window+1, 0)
	}
	if p.Chunk > 0 {
		s = max(s, (pos/p.Chunk)*p.Chunk)
	}
	return s
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

// attnParamsQueryFor is attnParamsFor for attendQuery's callers, with Llama 4's chunk on its RoPE layers.
func (a *Architecture) attnParamsQueryFor(cache *KVCache, layer int, global bool) attentionParams {
	p := a.attnParamsFor(cache, global)
	p.Order = attnQuery
	if l4 := a.llama4; l4 != nil && l4.chunkSize > 0 && layer < len(l4.useRope) && l4.useRope[layer] {
		p.Chunk = l4.chunkSize
	}
	return p
}
