// Package program is a model's forward written as data, shared by decoder (which builds it from an Architecture and walks it
// on the CPU) and the GPU backends (which read it at build time). A model's Program is built once and shared by every walker and
// backend, so nothing modifies one after decoder builds it, including its slices (Layers, a rotation's InvFreq). Design and steps:
// docs/tasks/task-forward-architecture-2026-10.md §4.8 and §4.9.
package program

import (
	"fmt"
	"math"
	"slices"
)

// Program is a model's forward written as data: an embedding stage, one list of operations per layer, and a head stage, built
// once at load by decoder from the resolved Architecture and the loaded weights. Decoder's CPU walkers run it, and the GPU
// backends read it at build time.
//
// Every implementation reads its parameters from the operation, never from the Architecture, so a parameter changed in the program
// changes the forward (gate L). Shapes (head counts, head and hidden widths) are not parameters: the weights fix them, and the
// implementations read them from the Architecture. The fp tags are the field names decoder's program fingerprint hashes.
type Program struct {
	Embed  EmbedParams `fp:"embed"`
	Layers [][]Op      `fp:"layers"`
	Head   HeadParams  `fp:"head"`
}

// OpKind is one kind of operation. The set is closed: a walker returns an error for a kind it does not implement.
type OpKind uint8

const (
	OpNorm       OpKind = iota + 1 // in → out, the normalisation (or a copy, for post-only placement)
	OpProjQKV                      // in → the stream's q, k, v
	OpQKNorm                       // q, k in place
	OpRope                         // q, k in place
	OpAttnTemp                     // q in place
	OpAttention                    // stores this position's k, v; q → the stream's attention context
	OpOutGate                      // context in place, gated by a projection of in
	OpProjO                        // context → out
	OpFFN                          // in → out
	OpResidual                     // the residual += regA (and regB)
	OpMLA                          // in → out, the whole Multi-head Latent Attention sublayer (mlaAttentionWith)
	OpKDA                          // in → out, the whole Kimi Delta Attention mixer (kdaMixerStep), on the layer's recurrent state
	OpShortConv                    // in → out, LFM2's gated short convolution (shortConvStep), on the layer's conv window
	OpMamba2                       // in → out, the Mamba-2 selective-scan mixer (mamba2Step), on the layer's SSM state
	OpDeltaNet                     // in → out, the Gated DeltaNet mixer (gatedDeltaNetStep), on the layer's recurrent state
	OpGatedAttn                    // in → out, qwen3.5's gated attention sublayer: [query ‖ gate] q_proj, QK-norm, rope, attendQuery, ×sigmoid(gate), o_proj
	OpGemma4Attn                   // in → out, Gemma 4's attention sublayer (gemma4AttnToken): per-layer widths, shared KV, scale-less v-norm
	OpGemma4MoE                    // the residual in place: Gemma 4's parallel dense ‖ MoE block, which returns the whole layer output
	OpPLE                          // the residual in place: Gemma 4's per-layer-embedding branch
	OpLayerScale                   // the residual in place: × the layer's learned output scalar, when it has one
)

// Recurrent reports whether the operation mutates per-layer state in place, which no positional rewind restores.
func (k OpKind) Recurrent() bool {
	return k == OpKDA || k == OpShortConv || k == OpMamba2 || k == OpDeltaNet
}

// GenericKinds are the operations of the generic layer: the ones the batched and multi-sequence walkers implement, LoRA's
// deltas reach (q/k/v, o and the gated FFN), and the layer pager's spans cover.
var GenericKinds = []OpKind{OpNorm, OpProjQKV, OpQKNorm, OpRope, OpAttnTemp, OpAttention, OpOutGate, OpProjO, OpFFN, OpResidual}

// Reg names a per-stream buffer an operation reads or writes.
type Reg uint8

const (
	RegH Reg = iota // the residual stream
	RegX            // a sublayer's input
	RegA            // a sublayer's output
	RegB            // the parallel block's second output
)

// Op is one operation: its kind, its operands, and the parameters of its kind (the other kinds' fields are zero).
type Op struct {
	Kind   OpKind           `fp:"kind"`
	In     Reg              `fp:"in"`
	Out    Reg              `fp:"out"`
	Norm   NormParams       `fp:"norm"`
	Proj   ProjParams       `fp:"proj"` // opProjQKV and opProjO
	QKNorm QKNormParams     `fp:"qkNorm"`
	Rope   RopeParams       `fp:"rope"`
	Temp   AttnTempParams   `fp:"temp"`
	Attn   AttentionParams  `fp:"attn"`
	Gate   OutGateParams    `fp:"gate"`
	FFN    FFNParams        `fp:"ffn"`
	Res    ResidualParams   `fp:"res"`
	MLA    MLAOpParams      `fp:"mla"`
	KDA    KDAOpParams      `fp:"kda"`
	Mamba  Mamba2OpParams   `fp:"mamba"`
	Delta  DeltaNetOpParams `fp:"delta"`
	Gated  GatedAttnParams  `fp:"gated"`
	G4     Gemma4AttnParams `fp:"g4"`
	Norms  NormEpsParams    `fp:"norms"` // opGemma4MoE and opPLE
}

// NormSlot selects which of a layer's norm weights an opNorm applies.
type NormSlot uint8

const (
	SlotPreAttn NormSlot = iota + 1
	SlotPreMLP
	SlotPostAttn
	SlotPostMLP
	SlotFinal
)

// NormForm is the normalisation an opNorm computes. normCopy is post-only placement's input: the raw residual, unnormed.
type NormForm uint8

const (
	NormRMS NormForm = iota + 1
	NormLayer
	NormCopy
)

// The parameter structs' fields are exported so gate L can set each one by reflection.
type NormParams struct {
	Slot   NormSlot
	Form   NormForm
	Eps    float64
	AddOne bool // RMSNorm's (1+w)
	Bias   bool // LayerNorm's bias; RMSNorm has none
}

type ProjParams struct{ Bias bool }

// QKNormForm is the QK-norm an opQKNorm computes.
type QKNormForm uint8

const (
	QKRMS QKNormForm = iota + 1 // weighted RMSNorm (rmsNorm), four-way sums
	QKL2                        // Llama 4: parameter-free RMS over each head, sequential float64 (l2NormHeads)
)

type QKNormParams struct {
	Form   QKNormForm
	Whole  bool // one statistic over all heads, not one per head (qkRMS)
	Eps    float64
	AddOne bool
}

// RopeKernel is the rotation an opRope computes.
type RopeKernel uint8

const (
	RopeAtKernel  RopeKernel = iota + 1 // ropeAt: NeoX or GPT-J pairing, m-RoPE
	RopeMLAKernel                       // mlaRope: GPT-J pairing by de-interleaving, then NeoX (Llama 4, MLA)
)

type RopeParams struct {
	Kernel           RopeKernel
	InvFreq          []float64 // its length is the rotated half-width
	Mscale           float64
	Interleave       bool
	MRopeSection     []int
	MRopeInterleaved bool
}

// AttnTempParams scales the query by 1 + Beta·log1p(floor((pos+PosOffset)/OrigMaxPos)).
type AttnTempParams struct {
	Beta, OrigMaxPos float64
	PosOffset        int
}

// AttnOrder is the CPU's reduction order for attention. Families compute attention in different orders today and OP3 moves no
// number, so each keeps its own; a GPU backend ignores it.
type AttnOrder uint8

const (
	AttnBatchedAcc64 AttnOrder = iota + 1 // attendBatchedHeads with float64 accumulation
	AttnSinksF64                          // attendSinksToken: inline float64 dots and float32 AV with a learned sink per head; gpt-oss's order before KS-O4, kept as its fidelity gate's exact arm
	AttnQuery                             // attendQuery: float64 dots, float32 accumulation of the values
	AttnMLA                               // the family attends through opMLA, over a latent cache
	AttnGemma4                            // the family attends through opGemma4Attn (gemma4Attend: float32 matmuls)
)

type AttentionParams struct {
	Window  int // 0 attends every stored key; else the most recent Window positions
	Chunk   int // 0 none; else a query attends only within its own block of Chunk positions (Llama 4's RoPE layers)
	Scale   float64
	Softcap float64 // 0 = none
	Order   AttnOrder
	// Sinks is the layer's learned per-head sink logit (gpt-oss, LayerWeights.AttnSinks): it joins every softmax's max-shift and
	// denominator and carries no value. The walker sets it at the call from the layer's weights; nil for every other family.
	Sinks []float32
}

type OutGateParams struct{ Sigmoid bool } // sigmoid, else softplus

// MLAOpParams are an opMLA's: the rotation of the rope-carrying dims, the latent and q-LoRA norms' epsilon, the score scale, and
// whether the optional sigmoid output gate is per head. The ranks and head widths are shapes, read from the Architecture.
type MLAOpParams struct {
	InvFreq    []float64
	Mscale     float64
	Interleave bool
	Eps        float64
	Scale      float64
	HeadWise   bool
}

// KDAOpParams are an opKDA's: the gated output norm's epsilon and the decay gate's lower bound. The head and conv sizes, and
// whether the gates carry a LoRA, are shapes, read from the Architecture; the CPU implements the safe (bounded) decay gate only,
// and the adapter refuses a checkpoint without it.
type KDAOpParams struct {
	Eps        float64
	LowerBound float64
}

type FFNForm uint8

const (
	FfnGated FFNForm = iota + 1
	FfnNonGated
	FfnMoE
	FfnGptOss      // gpt-oss: a router with a logit bias, softmax over the top k, clamped SwiGLU experts with biases (gptOssMoEWith)
	FfnLlama4MoE   // Llama 4: top-k by logit, each expert's INPUT scaled by sigmoid(logit), plus an ungated shared expert (llama4MoEWith)
	FfnNemotronMoE // Nemotron 3 Nano: routeExperts over non-gated relu² experts, plus an ungated shared expert (nemotronMoEWith)
	FfnGemma4Dense // Gemma 4's dense GeGLU at the layer's own width, in its own call form (gemma4DenseFFN)
)

type FFNParams struct {
	Form         FFNForm
	Act          ActKind
	MoE          MoEConfig // ffnMoE and ffnGptOss
	Alpha, Limit float64   // ffnGptOss: the gate's sigmoid gain and the clamp bound
}

// ResidualParams: Two adds regA and regB, the parallel block's single residual; Scale, when not 0, multiplies what is added
// (Granite's residual_multiplier).
type ResidualParams struct {
	Two   bool
	Scale float64
}

// DeltaNetOpParams are an opDeltaNet's: the norm epsilon, the output gated norm's own epsilon (0 = Eps), and whether the
// write gate's range is widened to [0, 2). The head counts and widths, and the checkpoint's tensor layout, are shapes.
type DeltaNetOpParams struct {
	Eps, ONormEps float64
	NegEigval     bool
}

// GatedAttnParams are an opGatedAttn's: its QK-norm, rotation and attention, each as the generic operation's parameters.
type GatedAttnParams struct {
	QKNorm QKNormParams
	Rope   RopeParams
	Attn   AttentionParams
}

// Gemma4AttnParams are an opGemma4Attn's: the q/k norms' and the scale-less v-norm's epsilon and the q/k norms' (1+w), the
// layer's rope table and scale (the local full-rotary table, or the global proportional one), the window (0 on a global layer),
// the score scale, and the layer whose K/V it attends with whether it stores its own (a KV-shared tail layer reads an earlier
// layer's). The head width and KV-head count are per-layer shapes, read from the Architecture.
type Gemma4AttnParams struct {
	Eps     float64
	AddOne  bool
	InvFreq []float64
	Mscale  float64
	Window  int
	Scale   float64
	KVSrc   int
	OwnsKV  bool
}

// NormEpsParams are the norm epsilon and (1+w) of an operation with norms inside it (opGemma4MoE, opPLE).
type NormEpsParams struct {
	Eps    float64
	AddOne bool
}

// Mamba2OpParams are an opMamba2's: the gated norm's epsilon and its group count (1 normalises the whole inner width). The
// heads, state and conv sizes are shapes, read from the Architecture.
type Mamba2OpParams struct {
	Eps        float64
	NormGroups int
}

type EmbedParams struct {
	Scale      float64 // 0 or 1 = none
	LearnedPos bool
	PLE        bool // Gemma 4: per-layer embedding inputs from the token id (gemma4PLEInputs), read by the layers' opPLE
}

type HeadParams struct {
	Norm       NormParams
	Tied       bool    // the embedding doubles as the head
	Softcap    float64 // 0 = none
	LogitScale float64 // logits are divided by it; 0 or 1 = none
}

func (k OpKind) String() string {
	names := [...]string{OpNorm: "norm", OpProjQKV: "proj-qkv", OpQKNorm: "qk-norm", OpRope: "rope", OpAttnTemp: "attn-temp",
		OpAttention: "attention", OpOutGate: "out-gate", OpProjO: "proj-o", OpFFN: "ffn", OpResidual: "residual", OpMLA: "mla", OpKDA: "kda", OpShortConv: "short-conv", OpMamba2: "mamba2", OpDeltaNet: "deltanet", OpGatedAttn: "gated-attn", OpGemma4Attn: "gemma4-attn", OpGemma4MoE: "gemma4-moe", OpPLE: "ple",
		OpLayerScale: "layer-scale"}
	if int(k) < len(names) && names[k] != "" {
		return names[k]
	}
	return fmt.Sprintf("op(%d)", uint8(k))
}

// SubCaptureUncovered reports whether a layer of p has an operation that changes the residual in place without a captured
// contribution: Gemma 4's PLE branch (E-models) and its parallel dense ‖ MoE block (26B). Decoder's ForwardSubCapture refuses
// such a model, since its contributions do not account for the layer (docs/tasks/task-forward-architecture-2026-10.md §4.8, the
// seams PR).
func (p *Program) SubCaptureUncovered() bool {
	for _, ops := range p.Layers {
		for i := range ops {
			if ops[i].Kind == OpPLE || ops[i].Kind == OpGemma4MoE {
				return true
			}
		}
	}
	return false
}

// Start is the first key position a query at pos attends: past the window, and inside the query's chunk.
func (p AttentionParams) Start(pos int) int {
	s := 0
	if p.Window > 0 {
		s = max(pos-p.Window+1, 0)
	}
	if p.Chunk > 0 {
		s = max(s, (pos/p.Chunk)*p.Chunk)
	}
	return s
}

// Scale is the query's temperature at pos.
func (p *AttnTempParams) Scale(pos int) float32 {
	return float32(1 + p.Beta*math.Log1p(math.Floor(float64(pos+p.PosOffset)/p.OrigMaxPos)))
}

// AnySoftcap reports whether any attention operation caps its scores.
func (p *Program) AnySoftcap() bool {
	for _, ops := range p.Layers {
		for i := range ops {
			if ops[i].Kind == OpAttention && ops[i].Attn.Softcap != 0 {
				return true
			}
		}
	}
	return false
}

// MultiStepEligible is the program-level scope of the multi-sequence walker: the batched attention kernel on every layer, every
// layer pre-norm placed (no post-sublayer norm, no copy for post-only placement, no parallel block), a gated dense FFN, no output
// gate, and no learned positions.
func (p *Program) MultiStepEligible() error {
	if p.Layers == nil {
		return fmt.Errorf("no layer program (an own loop)")
	}
	if !p.AttnOrderIs(AttnBatchedAcc64) || !p.KindsIn(GenericKinds...) {
		return fmt.Errorf("an attention kernel or an operation this walker does not run")
	}
	if p.Embed.LearnedPos {
		return fmt.Errorf("learned positions")
	}
	for l, ops := range p.Layers {
		for i := range ops {
			o := &ops[i]
			switch {
			case o.Kind == OpNorm && (o.Norm.Slot == SlotPostAttn || o.Norm.Slot == SlotPostMLP || o.Norm.Form == NormCopy):
				return fmt.Errorf("layer %d: a post-sublayer or post-only norm", l)
			case o.Kind == OpResidual && o.Res.Two:
				return fmt.Errorf("layer %d: a parallel block", l)
			case o.Kind == OpFFN && o.FFN.Form != FfnGated:
				return fmt.Errorf("layer %d: not a gated dense FFN", l)
			case o.Kind == OpOutGate:
				return fmt.Errorf("layer %d: an attention output gate", l)
			}
		}
	}
	return nil
}

// BatchEligible reports whether the batched walker runs prog: a layer program with no learned positions, the batched attention
// kernel on every layer, and a gated or MoE FFN.
func (p *Program) BatchEligible() bool {
	if p.Layers == nil || p.Embed.LearnedPos || !p.AttnOrderIs(AttnBatchedAcc64) || !p.KindsIn(GenericKinds...) {
		return false
	}
	for _, ops := range p.Layers {
		for i := range ops {
			if ops[i].Kind == OpFFN && ops[i].FFN.Form != FfnGated && ops[i].FFN.Form != FfnMoE {
				return false
			}
		}
	}
	return true
}

// AnyOp reports whether any operation in p satisfies f.
func (p *Program) AnyOp(f func(o *Op) bool) bool {
	for _, ops := range p.Layers {
		for i := range ops {
			if f(&ops[i]) {
				return true
			}
		}
	}
	return false
}

// KindsIn reports whether every operation in p is one of kinds.
func (p *Program) KindsIn(kinds ...OpKind) bool {
	for _, ops := range p.Layers {
		for i := range ops {
			if !slices.Contains(kinds, ops[i].Kind) {
				return false
			}
		}
	}
	return true
}

// AttnOrderIs reports whether every attention operation in p computes in order o.
func (p *Program) AttnOrderIs(o AttnOrder) bool {
	for _, ops := range p.Layers {
		for i := range ops {
			if ops[i].Kind == OpAttention && ops[i].Attn.Order != o {
				return false
			}
		}
	}
	return true
}

// MoEConfig describes a sparse mixture-of-experts FFN.
// A router scores all experts, the top-k run as gated MLPs, and their outputs
// combine weighted by the (renormalized) router probabilities. Mixtral:
// NumExperts=8, TopK=2, NormTopKProb=true.
type MoEConfig struct {
	NumExperts   int  // experts per layer (E)
	TopK         int  // experts evaluated per token (k)
	NormTopKProb bool // renormalize the top-k router weights to sum to 1 (Mixtral)
	// IntermediateDim is the per-expert FFN width. Mixtral's experts use the model's intermediate_size; Mellum gives them a
	// narrower moe_intermediate_size, so the expert width is tracked here, not read from arch.IntermediateDim.
	IntermediateDim int

	// SharedIntermediateDim is the FFN width of the always-on shared expert
	// (Qwen-MoE / Qwen2-MoE: shared_expert_intermediate_size). 0 means no shared
	// expert (Mixtral/Mellum). When set, every token additionally runs a gated
	// SwiGLU shared expert scaled by sigmoid(shared_gate·h), added to the routed
	// sum — unless SharedUngated (GLM/DeepSeek add it with no gate).
	SharedIntermediateDim int

	// DeepSeek-style routing (GLM-4.5/4.6). Defaults (false / 0) reproduce the
	// Mixtral/Qwen2-MoE softmax-topk path exactly.
	RouterSigmoid bool    // score experts with per-expert sigmoid(logit) instead of softmax; top-k weights are the chosen sigmoid scores (then NormTopKProb). e_score_correction_bias (LayerWeights.RouterBias) shifts the top-k SELECTION only.
	RoutedScale   float64 // routed_scaling_factor applied to the top-k weights (0 or 1 = no-op).
	SharedUngated bool    // GLM/DeepSeek add the shared expert with NO sigmoid gate (out += shared(h)); else the Qwen2-MoE sigmoid(SharedGate·h) gate.

	// Group-limited routing (DeepSeek-V3 noaux_tc). Experts are partitioned into
	// NGroup contiguous groups; each group is scored by its top-2 selection scores
	// summed, the top TopkGroup groups are kept, and the per-token top-k is taken
	// only among experts in those groups. NGroup ≤ 1 (GLM, V2-Lite) ⇒ no grouping,
	// the plain global top-k.
	NGroup    int
	TopkGroup int

	// Gemma 4 26B-A4B router extras (Gemma4TextRouter; false for every other MoE family). Selection is still softmax over all
	// experts, top-k, renorm (NormTopKProb is unconditionally true), so routeExperts applies; these two flags add the parts it
	// lacks (docs/completed/task-gemma4-moe.md).
	RouterPreNorm  bool // before the router projection, the hidden state passes a WEIGHTLESS RMSNorm, a learned [hidden] scale (LayerWeights.RouterScale), and a hidden^-0.5 constant — not a bare Linear on the raw hidden state.
	PerExpertScale bool // the renormalized top-k weights are multiplied by a learned per-expert scale (LayerWeights.PerExpertScale), indexed by the chosen experts.
}

// ActKind selects the MLP activation. GeluTanh = Gemma's GeGLU; SiLU = the
// SwiGLU used by Llama/Mistral/Qwen. Gelu/ReLU2/non-gated MLPs are later G's.
type ActKind int

const (
	ActGeluTanh ActKind = iota
	ActSiLU
	ActReLU2 // ReLU-squared (relu(x)²) — Nemotron-H's non-gated MLP
	// ActGelu is the EXACT erf GELU — HF's "gelu", a different function from
	// "gelu_new"/"gelu_pytorch_tanh" (ActGeluTanh) rather than a spelling of it.
	// Appended deliberately: GatedActResident passes the ordinal straight to resident
	// kernels that know 0 (GELU-tanh) and 1 (SiLU), so the existing values cannot be
	// renumbered. Gated and non-gated archs both use it, and no resident kernel
	// implements it: residentFeatures derives FeatGatedGELUErf or FeatNonGatedGELUErf,
	// which no backend declares.
	ActGelu
)

// String renders the activation for the capability matrix / logs. The gated vs
// non-gated distinction is rendered by the matrix from Architecture.NonGatedMLP.
func (a ActKind) String() string {
	switch a {
	case ActGeluTanh:
		return "GELU-tanh"
	case ActSiLU:
		return "SiLU"
	case ActReLU2:
		return "ReLU²"
	case ActGelu:
		return "GELU"
	default:
		return "unknown"
	}
}
