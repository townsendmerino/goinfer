package decoder

import (
	darch "github.com/townsendmerino/goinfer/decoder/arch"
)

// Descriptor, weight and load-profile types live in decoder/arch so packages below
// decoder and the backend modules can name them without importing decoder.
type (
	Architecture       = darch.Architecture
	NormKind           = darch.NormKind
	NormPlacement      = darch.NormPlacement
	GateKind           = darch.GateKind
	ResidentFeature    = darch.ResidentFeature
	KVGeometry         = darch.KVGeometry
	KVGroup            = darch.KVGroup
	KVState            = darch.KVState
	KVRetention        = darch.KVRetention
	KVClass            = darch.KVClass
	KVClassMap         = darch.KVClassMap
	KVSequenceState    = darch.KVSequenceState
	KVWindowStepStatus = darch.KVWindowStepStatus
	KVFacts            = darch.KVFacts
	LayerWeights       = darch.LayerWeights
	Weights            = darch.Weights
	LoadPhase          = darch.LoadPhase
	LoadProfile        = darch.LoadProfile
)

const (
	NormRMS                    = darch.NormRMS
	NormLayer                  = darch.NormLayer
	NormPre2                   = darch.NormPre2
	NormSandwich4              = darch.NormSandwich4
	NormParallel               = darch.NormParallel
	NormPostOnly               = darch.NormPostOnly
	GateNone                   = darch.GateNone
	GateSoftplus               = darch.GateSoftplus
	GateSigmoid                = darch.GateSigmoid
	ActGeluTanh                = darch.ActGeluTanh
	ActSiLU                    = darch.ActSiLU
	ActReLU2                   = darch.ActReLU2
	ActGelu                    = darch.ActGelu
	FeatQKNorm                 = darch.FeatQKNorm
	FeatSlidingWindow          = darch.FeatSlidingWindow
	FeatPartialRotary          = darch.FeatPartialRotary
	FeatPerLayerRoPE           = darch.FeatPerLayerRoPE
	FeatRopeMscale             = darch.FeatRopeMscale
	FeatRMSAddOne              = darch.FeatRMSAddOne
	FeatEmbedScale             = darch.FeatEmbedScale
	FeatAttnLogitSoftcap       = darch.FeatAttnLogitSoftcap
	FeatFinalLogitSoftcap      = darch.FeatFinalLogitSoftcap
	FeatSandwichNorm           = darch.FeatSandwichNorm
	FeatLayerNorm              = darch.FeatLayerNorm
	FeatParallelBlock          = darch.FeatParallelBlock
	FeatNoPE                   = darch.FeatNoPE
	FeatGatedGELU              = darch.FeatGatedGELU
	FeatNonGatedMLP            = darch.FeatNonGatedMLP
	FeatLearnedPos             = darch.FeatLearnedPos
	FeatOutBias                = darch.FeatOutBias
	FeatLogitScale             = darch.FeatLogitScale
	FeatMoE                    = darch.FeatMoE
	FeatMoEGatedShared         = darch.FeatMoEGatedShared
	FeatMLA                    = darch.FeatMLA
	FeatSSM                    = darch.FeatSSM
	FeatDeltaNet               = darch.FeatDeltaNet
	FeatAttnSink               = darch.FeatAttnSink
	FeatAttnOutputGateSoftplus = darch.FeatAttnOutputGateSoftplus
	FeatAttnOutputGateSigmoid  = darch.FeatAttnOutputGateSigmoid
	FeatPerLayerQueryHeads     = darch.FeatPerLayerQueryHeads
	FeatShortConv              = darch.FeatShortConv
	FeatGemma4EModel           = darch.FeatGemma4EModel
	FeatAttnTemp               = darch.FeatAttnTemp
	FeatPostOnlyNorm           = darch.FeatPostOnlyNorm
	FeatQKNormWhole            = darch.FeatQKNormWhole
	FeatKDA                    = darch.FeatKDA
	FeatPairwiseRoPE           = darch.FeatPairwiseRoPE
	FeatPairwiseMRoPE          = darch.FeatPairwiseMRoPE
	FeatGatedGELUErf           = darch.FeatGatedGELUErf
	FeatNonGatedGELUErf        = darch.FeatNonGatedGELUErf
	FeatNonGatedGELUTanh       = darch.FeatNonGatedGELUTanh
	FeatNonGatedSiLU           = darch.FeatNonGatedSiLU
	FeatNonGatedReLU2          = darch.FeatNonGatedReLU2
	FeatNonGatedActUnknown     = darch.FeatNonGatedActUnknown
	FeatGatedActUnknown        = darch.FeatGatedActUnknown
	KVRetainFull               = darch.KVRetainFull
	KVRetainWindow             = darch.KVRetainWindow
	KVRetainChunk              = darch.KVRetainChunk
)
