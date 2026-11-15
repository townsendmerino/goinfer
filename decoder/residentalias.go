package decoder

import "github.com/townsendmerino/goinfer/decoder/resident"

// The resident contract a backend implements lives in decoder/resident, a leaf over internal/cost and internal/adapter, so a
// caller can name it without importing the loader. These are its names in decoder.
type (
	BlockDrafterWeights       = resident.BlockDrafterWeights
	DrafterGeometry           = resident.DrafterGeometry
	DrafterLayerWeights       = resident.DrafterLayerWeights
	ExpertCacheStats          = resident.ExpertCacheStats
	KVHostBudget              = resident.KVHostBudget
	KVPoolDetail              = resident.KVPoolDetail
	PrefillTailExact          = resident.PrefillTailExact
	ResidentBatchDraw         = resident.BatchDraw
	ResidentBatchOut          = resident.BatchOut
	ResidentBatchSeq          = resident.BatchSeq
	ResidentBlockDrafter      = resident.BlockDrafter
	ResidentCaps              = resident.Caps
	ResidentDecodeSubsystem   = resident.DecodeSubsystem
	ResidentDrafterHost       = resident.DrafterHost
	ResidentForward           = resident.Forward
	ResidentKVSnapshot        = resident.KVSnapshot
	ResidentKVSubsystem       = resident.KVSubsystem
	ResidentPrefillSubsystem  = resident.PrefillSubsystem
	ResidentRaggedOut         = resident.RaggedOut
	ResidentRaggedPrefill     = resident.RaggedPrefill
	ResidentRaggedSeq         = resident.RaggedSeq
	ResidentRuntimeSubsystem  = resident.RuntimeSubsystem
	ResidentSubsystemProvider = resident.SubsystemProvider
	ResidentSubsystems        = resident.Subsystems
	TopKRow                   = resident.TopKRow
)
