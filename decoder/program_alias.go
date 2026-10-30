// The program's types live in internal/program, which the GPU backends share; decoder builds the program and keeps these
// names for its own code.

package decoder

import "github.com/townsendmerino/goinfer/internal/program"

type (
	layerProgram     = program.Program
	opKind           = program.OpKind
	opReg            = program.Reg
	op               = program.Op
	normSlot         = program.NormSlot
	normForm         = program.NormForm
	normParams       = program.NormParams
	projParams       = program.ProjParams
	qkNormParams     = program.QKNormParams
	ropeParams       = program.RopeParams
	attnTempParams   = program.AttnTempParams
	attnOrder        = program.AttnOrder
	attentionParams  = program.AttentionParams
	outGateParams    = program.OutGateParams
	mlaOpParams      = program.MLAOpParams
	kdaOpParams      = program.KDAOpParams
	ffnParams        = program.FFNParams
	residualParams   = program.ResidualParams
	deltaNetOpParams = program.DeltaNetOpParams
	gatedAttnParams  = program.GatedAttnParams
	gemma4AttnParams = program.Gemma4AttnParams
	normEpsParams    = program.NormEpsParams
	mamba2OpParams   = program.Mamba2OpParams
	embedParams      = program.EmbedParams
	headParams       = program.HeadParams
)

const (
	opNorm           = program.OpNorm
	opProjQKV        = program.OpProjQKV
	opQKNorm         = program.OpQKNorm
	opRope           = program.OpRope
	opAttnTemp       = program.OpAttnTemp
	opAttention      = program.OpAttention
	opOutGate        = program.OpOutGate
	opProjO          = program.OpProjO
	opFFN            = program.OpFFN
	opResidual       = program.OpResidual
	opMLA            = program.OpMLA
	opKDA            = program.OpKDA
	opShortConv      = program.OpShortConv
	opMamba2         = program.OpMamba2
	opDeltaNet       = program.OpDeltaNet
	opGatedAttn      = program.OpGatedAttn
	opGemma4Attn     = program.OpGemma4Attn
	opGemma4MoE      = program.OpGemma4MoE
	opPLE            = program.OpPLE
	opLayerScale     = program.OpLayerScale
	regH             = program.RegH
	regX             = program.RegX
	regA             = program.RegA
	regB             = program.RegB
	slotPreAttn      = program.SlotPreAttn
	slotPreMLP       = program.SlotPreMLP
	slotPostAttn     = program.SlotPostAttn
	slotPostMLP      = program.SlotPostMLP
	slotFinal        = program.SlotFinal
	normRMS          = program.NormRMS
	normLayer        = program.NormLayer
	normCopy         = program.NormCopy
	qkRMS            = program.QKRMS
	qkL2             = program.QKL2
	ropeAtKernel     = program.RopeAtKernel
	ropeMLAKernel    = program.RopeMLAKernel
	attnBatchedAcc64 = program.AttnBatchedAcc64
	attnSinksF64     = program.AttnSinksF64
	attnQuery        = program.AttnQuery
	attnMLA          = program.AttnMLA
	attnGemma4       = program.AttnGemma4
	ffnGated         = program.FfnGated
	ffnNonGated      = program.FfnNonGated
	ffnMoE           = program.FfnMoE
	ffnGptOss        = program.FfnGptOss
	ffnLlama4MoE     = program.FfnLlama4MoE
	ffnNemotronMoE   = program.FfnNemotronMoE
	ffnGemma4Dense   = program.FfnGemma4Dense
)

// genericKinds are the operations of the generic layer (program.GenericKinds).
var genericKinds = program.GenericKinds

// Program returns m's program. The GPU backends read it at build time. It is read-only by contract, slices included: every
// walker and backend shares the one copy. A Model assembled by a test without modelFromOptions has none, and each call builds a
// fresh one.
func (m *Model) Program() *program.Program { return m.program() }
