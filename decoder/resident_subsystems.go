package decoder

import (
	"context"
	"errors"

	"github.com/townsendmerino/goinfer/internal/cost"
)

var errSubsystemUnavailable = errors.New("decoder: resident subsystem capability not implemented")

// Compile-time checks that the NoopResident* embeddable defaults cover every contract method.
var (
	_ ResidentDecodeSubsystem  = NoopResidentDecode{}
	_ ResidentPrefillSubsystem = NoopResidentPrefill{}
	_ ResidentKVSubsystem      = NoopResidentKV{}
	_ ResidentRuntimeSubsystem = NoopResidentRuntime{}
)

func noopLeave() {}

var noopResidentSubsystems = ResidentSubsystems{
	Decode:  NoopResidentDecode{},
	Prefill: NoopResidentPrefill{},
	KV:      NoopResidentKV{},
	Runtime: NoopResidentRuntime{},
}

// NoopResidentDecode provides safe absent defaults for ResidentDecodeSubsystem. Backends embed it in their decode
// subsystem wrapper (never on the resident struct itself) and override the capabilities they implement.
type NoopResidentDecode struct{}

func (NoopResidentDecode) GreedyAvailable() bool { return false }
func (NoopResidentDecode) ForwardArgmax([]float32, int) (int, error) {
	return 0, errSubsystemUnavailable
}
func (NoopResidentDecode) TopKAvailable() bool { return false }
func (NoopResidentDecode) ForwardTopK([]float32, int, int, float64, bool) (TopKRow, error) {
	return TopKRow{}, errSubsystemUnavailable
}
func (NoopResidentDecode) SampleAvailable() bool { return false }
func (NoopResidentDecode) ForwardSample([]float32, int, float64, uint64, uint64) (int, error) {
	return 0, errSubsystemUnavailable
}
func (NoopResidentDecode) MRoPEDecodeAvailable() bool { return false }
func (NoopResidentDecode) ForwardMRoPE([]float32, int, int) ([]float32, error) {
	return nil, errSubsystemUnavailable
}
func (NoopResidentDecode) NoLogitsAvailable() bool              { return false }
func (NoopResidentDecode) ForwardNoLogits([]float32, int) error { return errSubsystemUnavailable }
func (NoopResidentDecode) GreedyChainAvailable() bool           { return false }
func (NoopResidentDecode) GreedyChainStart(int, int) error      { return errSubsystemUnavailable }
func (NoopResidentDecode) GreedyChainNext() (int, error)        { return 0, errSubsystemUnavailable }
func (NoopResidentDecode) GreedyChainStop()                     {}
func (NoopResidentDecode) SampleChainAvailable() bool           { return false }
func (NoopResidentDecode) SampleChainStart(int, int, float64, uint64, uint64) error {
	return errSubsystemUnavailable
}
func (NoopResidentDecode) SampleChainNext(uint64, uint64) (int, error) {
	return 0, errSubsystemUnavailable
}
func (NoopResidentDecode) BatchStepperAvailable() bool  { return false }
func (NoopResidentDecode) BatchStepRange() (lo, hi int) { return 0, 0 }
func (NoopResidentDecode) StepBatch([]ResidentBatchSeq) ([]ResidentBatchOut, error) {
	return nil, errSubsystemUnavailable
}
func (NoopResidentDecode) RaggedStepperAvailable() bool { return false }
func (NoopResidentDecode) StepRagged([]ResidentRaggedSeq) ([]ResidentRaggedOut, error) {
	return nil, errSubsystemUnavailable
}

// NoopResidentPrefill provides safe absent defaults for ResidentPrefillSubsystem.
type NoopResidentPrefill struct{}

func (NoopResidentPrefill) PrefillAvailable() bool { return false }
func (NoopResidentPrefill) PrefillLast(context.Context, [][]float32, int) ([]float32, error) {
	return nil, errSubsystemUnavailable
}
func (NoopResidentPrefill) PrefillRowsAvailable() bool { return false }
func (NoopResidentPrefill) PrefillLastRows(context.Context, int, int, func(int, int) [][]float32) ([]float32, error) {
	return nil, errSubsystemUnavailable
}
func (NoopResidentPrefill) PrefillMarkedAvailable() bool { return false }
func (NoopResidentPrefill) PrefillMarkAlign() int        { return 0 }
func (NoopResidentPrefill) PrefillLastMarked(context.Context, [][]float32, int, int, func(int)) ([]float32, error) {
	return nil, errSubsystemUnavailable
}
func (NoopResidentPrefill) PrefillPathInfo() (bool, string, bool)  { return false, "", false }
func (NoopResidentPrefill) PrefillTailExact() bool                 { return false }
func (NoopResidentPrefill) PrefillBlocksAvailable() bool           { return false }
func (NoopResidentPrefill) PrefillBlockRows() int                  { return 0 }
func (NoopResidentPrefill) PrefillChunksAvailable() bool           { return false }
func (NoopResidentPrefill) PrefillChunkRows() int                  { return 0 }
func (NoopResidentPrefill) PrefillScratchBytes(int) int64          { return 0 }
func (NoopResidentPrefill) PrefillSpanScratchBytes(int, int) int64 { return 0 }
func (NoopResidentPrefill) HiddenLastAvailable() bool              { return false }
func (NoopResidentPrefill) HiddenLast(context.Context, [][]float32, int) ([]float32, error) {
	return nil, errSubsystemUnavailable
}
func (NoopResidentPrefill) ResidualAllAvailable() bool { return false }
func (NoopResidentPrefill) ResidualAll(context.Context, [][]float32, int) ([][]float32, error) {
	return nil, errSubsystemUnavailable
}
func (NoopResidentPrefill) ImagePrefillAvailable() bool { return false }
func (NoopResidentPrefill) PrefillImageLast(context.Context, [][]float32, int, int, int) ([]float32, error) {
	return nil, errSubsystemUnavailable
}
func (NoopResidentPrefill) ImageBlocksPrefillAvailable() bool { return false }
func (NoopResidentPrefill) PrefillImageBlocksLast(context.Context, [][]float32, int, [][2]int) ([]float32, error) {
	return nil, errSubsystemUnavailable
}
func (NoopResidentPrefill) SpanPrefillAvailable() bool { return false }
func (NoopResidentPrefill) PrefillSpansLast(context.Context, [][]float32, int, [][2]int) ([]float32, error) {
	return nil, errSubsystemUnavailable
}
func (NoopResidentPrefill) HybridMRoPEPrefill() bool    { return false }
func (NoopResidentPrefill) MRoPEPrefillAvailable() bool { return false }
func (NoopResidentPrefill) PrefillMRoPELast(context.Context, [][]float32, int, [][3]int) ([]float32, error) {
	return nil, errSubsystemUnavailable
}
func (NoopResidentPrefill) MRoPEDeepstackPrefillAvailable() bool { return false }
func (NoopResidentPrefill) PrefillMRoPEDeepstackLast(context.Context, [][]float32, int, [][3]int, [][]float32, int, int) ([]float32, error) {
	return nil, errSubsystemUnavailable
}

// NoopResidentKV provides safe absent defaults for ResidentKVSubsystem.
type NoopResidentKV struct{}

func (NoopResidentKV) KVSlotterAvailable() bool             { return false }
func (NoopResidentKV) KVSlots() int                         { return 1 }
func (NoopResidentKV) UseKVSlot(int) error                  { return nil }
func (NoopResidentKV) SetKVSlotBusy(int, bool)              {}
func (NoopResidentKV) KVPool() KVPoolDetail                 { return KVPoolDetail{} }
func (NoopResidentKV) ReusableKV(prefix int) int            { return prefix }
func (NoopResidentKV) ReusableKVSlotAvailable() bool        { return false }
func (NoopResidentKV) ReusableKVSlot(_ int, prefix int) int { return prefix }
func (NoopResidentKV) KVUploadPassAvailable() bool          { return false }
func (NoopResidentKV) BeginKVUpload() error                 { return errSubsystemUnavailable }
func (NoopResidentKV) EndKVUpload(bool) error               { return errSubsystemUnavailable }
func (NoopResidentKV) KVCopierAvailable() bool              { return false }
func (NoopResidentKV) CopyKV(int, int, int, int) error      { return errSubsystemUnavailable }
func (NoopResidentKV) KVSnapshotterAvailable() bool         { return false }
func (NoopResidentKV) SnapshotKV(int, int) (ResidentKVSnapshot, error) {
	return nil, errSubsystemUnavailable
}
func (NoopResidentKV) RestoreKV(ResidentKVSnapshot, int, int) error { return errSubsystemUnavailable }
func (NoopResidentKV) KVSnapshotBytes(int) int64                    { return 0 }
func (NoopResidentKV) KVSnapshotRollback() int                      { return 0 }
func (NoopResidentKV) KVSnapshotRecurrent() bool                    { return false }
func (NoopResidentKV) KVDemoterAvailable() bool                     { return false }
func (NoopResidentKV) SetKVHostBudget(KVHostBudget)                 {}
func (NoopResidentKV) PromoteKV(int, int) (int, error)              { return 0, errSubsystemUnavailable }
func (NoopResidentKV) CopyDemotedKV(int, int, int, int) (int, bool, error) {
	return 0, false, errSubsystemUnavailable
}
func (NoopResidentKV) ReserveKVHost(int64) bool          { return false }
func (NoopResidentKV) RecurrentCheckpointBytes() int64   { return 0 }
func (NoopResidentKV) SaveRecurrentCheckpoint() error    { return errSubsystemUnavailable }
func (NoopResidentKV) RestoreRecurrentCheckpoint() error { return errSubsystemUnavailable }
func (NoopResidentKV) RecurrentCheckpointHeld(int) bool  { return false }

// NoopResidentRuntime provides safe absent defaults for ResidentRuntimeSubsystem.
type NoopResidentRuntime struct{}

func (NoopResidentRuntime) ResidentCaps() ResidentCaps { return ResidentCaps{} }
func (NoopResidentRuntime) CostAvailable() bool        { return false }
func (NoopResidentRuntime) CostSpan(_ *cost.Counter, _ cost.Phase, work func() (int, error)) error {
	_, err := work()
	return err
}
func (NoopResidentRuntime) AdapterAvailable() bool                  { return false }
func (NoopResidentRuntime) SetAdapter([]ResidentAdapterLayer) error { return errSubsystemUnavailable }
func (NoopResidentRuntime) ExpertCacheAvailable() bool              { return false }
func (NoopResidentRuntime) ExpertCacheStats() (ExpertCacheStats, bool) {
	return ExpertCacheStats{}, false
}
func (NoopResidentRuntime) ResizeExpertCache(int64) (int, error) { return 0, errSubsystemUnavailable }
func (NoopResidentRuntime) VerifyPathInfo() (bool, string, bool) { return false, "", false }
func (NoopResidentRuntime) VerifyCost() []float64                { return nil }
func (NoopResidentRuntime) DecodeVerifyDivergence() error        { return nil }
func (NoopResidentRuntime) EnterExactAttention() func()          { return noopLeave }
func (NoopResidentRuntime) SeedArgmaxAvailable() bool            { return false }
func (NoopResidentRuntime) PrefillSeedArgmax([][]float32, int) (int, error) {
	return 0, errSubsystemUnavailable
}
func (NoopResidentRuntime) LastNArgmaxAvailable() bool { return false }
func (NoopResidentRuntime) PrefillLastNArgmax([][]float32, int) ([]int, error) {
	return nil, errSubsystemUnavailable
}
func (NoopResidentRuntime) DrafterHostAvailable() bool { return false }
func (NoopResidentRuntime) CanStreamDrafterRows() bool { return false }
func (NoopResidentRuntime) AttachBlockDrafter(BlockDrafterWeights) (ResidentBlockDrafter, error) {
	return nil, errSubsystemUnavailable
}
func (NoopResidentRuntime) SetBatchedCapture([]int) error  { return errSubsystemUnavailable }
func (NoopResidentRuntime) BatchedCapture() [][]float32    { return nil }
func (NoopResidentRuntime) RecurrentSpecAvailable() bool   { return false }
func (NoopResidentRuntime) PrepareRecurrentSpec(int) error { return errSubsystemUnavailable }
func (NoopResidentRuntime) RecurrentSpecRows() int         { return 0 }
func (NoopResidentRuntime) VerifyLastNArgmax([][]float32, int) ([]int, error) {
	return nil, errSubsystemUnavailable
}
func (NoopResidentRuntime) FoldVerified(int) error { return errSubsystemUnavailable }
func (NoopResidentRuntime) FlushFold() error       { return errSubsystemUnavailable }
func (NoopResidentRuntime) DropFold()              {}

// SubsystemsOf resolves r's four resident subsystems: a ResidentSubsystemProvider returns its own pre-built wrappers
// (zero allocation); nil, and any value that is not a provider, has no capability and gets the no-op subsystems. r is a resident:
// a subsystem is not a provider, so passing one is that silent no-capability answer, and under goinfer_testhooks it panics.
func SubsystemsOf(r any) ResidentSubsystems {
	if p, ok := r.(ResidentSubsystemProvider); ok {
		return p.ResidentSubsystems()
	}
	checkSubsystemsOfArg(r)
	return noopResidentSubsystems
}

// drafterHostOf is the resident r itself as a drafter host. A BlockSpec keeps the resident, not its Runtime subsystem, because step and
// SubsystemsOf resolve a resident; ok is false where r does not implement it.
func drafterHostOf(r ResidentForward) (host ResidentDrafterHost, ok bool) {
	host, ok = r.(ResidentDrafterHost)
	return host, ok
}

// subsystems returns m.resident's four subsystems.
func (m *Model) subsystems() ResidentSubsystems {
	if m == nil || m.resident == nil {
		return noopResidentSubsystems
	}
	return SubsystemsOf(m.resident)
}

func (m *Model) resDecode() ResidentDecodeSubsystem   { return m.subsystems().Decode }
func (m *Model) resPrefill() ResidentPrefillSubsystem { return m.subsystems().Prefill }
func (m *Model) resKV() ResidentKVSubsystem           { return m.subsystems().KV }
func (m *Model) resRuntime() ResidentRuntimeSubsystem { return m.subsystems().Runtime }
