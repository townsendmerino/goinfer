package residency

import (
	"context"
	"errors"

	"github.com/townsendmerino/goinfer/decoder/resident"
	"github.com/townsendmerino/goinfer/internal/cost"
	"github.com/townsendmerino/goinfer/internal/runtime/adapter"
)

var errSubsystemUnavailable = errors.New("decoder: resident subsystem capability not implemented")

// Compile-time checks that the NoopResident* embeddable defaults cover every contract method.
var (
	_ resident.DecodeSubsystem  = NoopDecode{}
	_ resident.PrefillSubsystem = NoopPrefill{}
	_ resident.KVSubsystem      = NoopKV{}
	_ resident.RuntimeSubsystem = NoopRuntime{}
)

func noopLeave() {}

var NoopSubsystems = resident.Subsystems{
	Decode:  NoopDecode{},
	Prefill: NoopPrefill{},
	KV:      NoopKV{},
	Runtime: NoopRuntime{},
}

// NoopDecode provides safe absent defaults for resident.DecodeSubsystem. Backends embed it in their decode
// subsystem wrapper (never on the resident struct itself) and override the capabilities they implement.
type NoopDecode struct{}

func (NoopDecode) GreedyAvailable() bool { return false }
func (NoopDecode) ForwardArgmax([]float32, int) (int, error) {
	return 0, errSubsystemUnavailable
}
func (NoopDecode) TopKAvailable() bool { return false }
func (NoopDecode) ForwardTopK([]float32, int, int, float64, bool) (resident.TopKRow, error) {
	return resident.TopKRow{}, errSubsystemUnavailable
}
func (NoopDecode) SampleAvailable() bool { return false }
func (NoopDecode) ForwardSample([]float32, int, float64, uint64, uint64) (int, error) {
	return 0, errSubsystemUnavailable
}
func (NoopDecode) MRoPEDecodeAvailable() bool { return false }
func (NoopDecode) ForwardMRoPE([]float32, int, int) ([]float32, error) {
	return nil, errSubsystemUnavailable
}
func (NoopDecode) NoLogitsAvailable() bool              { return false }
func (NoopDecode) ForwardNoLogits([]float32, int) error { return errSubsystemUnavailable }
func (NoopDecode) GreedyChainAvailable() bool           { return false }
func (NoopDecode) GreedyChainStart(int, int) error      { return errSubsystemUnavailable }
func (NoopDecode) GreedyChainNext() (int, error)        { return 0, errSubsystemUnavailable }
func (NoopDecode) GreedyChainStop()                     {}
func (NoopDecode) SampleChainAvailable() bool           { return false }
func (NoopDecode) SampleChainStart(int, int, float64, uint64, uint64) error {
	return errSubsystemUnavailable
}
func (NoopDecode) SampleChainNext(uint64, uint64) (int, error) {
	return 0, errSubsystemUnavailable
}
func (NoopDecode) BatchStepperAvailable() bool  { return false }
func (NoopDecode) BatchStepRange() (lo, hi int) { return 0, 0 }
func (NoopDecode) StepBatch([]resident.BatchSeq) ([]resident.BatchOut, error) {
	return nil, errSubsystemUnavailable
}
func (NoopDecode) RaggedStepperAvailable() bool { return false }
func (NoopDecode) StepRagged([]resident.RaggedSeq) ([]resident.RaggedOut, error) {
	return nil, errSubsystemUnavailable
}

// NoopPrefill provides safe absent defaults for resident.PrefillSubsystem.
type NoopPrefill struct{}

func (NoopPrefill) PrefillAvailable() bool { return false }
func (NoopPrefill) PrefillLast(context.Context, [][]float32, int) ([]float32, error) {
	return nil, errSubsystemUnavailable
}
func (NoopPrefill) PrefillRowsAvailable() bool { return false }
func (NoopPrefill) PrefillLastRows(context.Context, int, int, func(int, int) [][]float32) ([]float32, error) {
	return nil, errSubsystemUnavailable
}
func (NoopPrefill) PrefillMarkedAvailable() bool { return false }
func (NoopPrefill) PrefillMarkAlign() int        { return 0 }
func (NoopPrefill) PrefillLastMarked(context.Context, [][]float32, int, int, func(int)) ([]float32, error) {
	return nil, errSubsystemUnavailable
}
func (NoopPrefill) PrefillPathInfo() (bool, string, bool)  { return false, "", false }
func (NoopPrefill) PrefillTailExact() bool                 { return false }
func (NoopPrefill) PrefillBlocksAvailable() bool           { return false }
func (NoopPrefill) PrefillBlockRows() int                  { return 0 }
func (NoopPrefill) PrefillChunksAvailable() bool           { return false }
func (NoopPrefill) PrefillChunkRows() int                  { return 0 }
func (NoopPrefill) PrefillScratchBytes(int) int64          { return 0 }
func (NoopPrefill) PrefillSpanScratchBytes(int, int) int64 { return 0 }
func (NoopPrefill) HiddenLastAvailable() bool              { return false }
func (NoopPrefill) HiddenLast(context.Context, [][]float32, int) ([]float32, error) {
	return nil, errSubsystemUnavailable
}
func (NoopPrefill) ResidualAllAvailable() bool { return false }
func (NoopPrefill) ResidualAll(context.Context, [][]float32, int) ([][]float32, error) {
	return nil, errSubsystemUnavailable
}
func (NoopPrefill) ImagePrefillAvailable() bool { return false }
func (NoopPrefill) PrefillImageLast(context.Context, [][]float32, int, int, int) ([]float32, error) {
	return nil, errSubsystemUnavailable
}
func (NoopPrefill) ImageBlocksPrefillAvailable() bool { return false }
func (NoopPrefill) PrefillImageBlocksLast(context.Context, [][]float32, int, [][2]int) ([]float32, error) {
	return nil, errSubsystemUnavailable
}
func (NoopPrefill) SpanPrefillAvailable() bool { return false }
func (NoopPrefill) PrefillSpansLast(context.Context, [][]float32, int, [][2]int) ([]float32, error) {
	return nil, errSubsystemUnavailable
}
func (NoopPrefill) HybridMRoPEPrefill() bool    { return false }
func (NoopPrefill) MRoPEPrefillAvailable() bool { return false }
func (NoopPrefill) PrefillMRoPELast(context.Context, [][]float32, int, [][3]int) ([]float32, error) {
	return nil, errSubsystemUnavailable
}
func (NoopPrefill) MRoPEDeepstackPrefillAvailable() bool { return false }
func (NoopPrefill) PrefillMRoPEDeepstackLast(context.Context, [][]float32, int, [][3]int, [][]float32, int, int) ([]float32, error) {
	return nil, errSubsystemUnavailable
}

// NoopKV provides safe absent defaults for resident.KVSubsystem.
type NoopKV struct{}

func (NoopKV) KVSlotterAvailable() bool             { return false }
func (NoopKV) KVSlots() int                         { return 1 }
func (NoopKV) UseKVSlot(int) error                  { return nil }
func (NoopKV) SetKVSlotBusy(int, bool)              {}
func (NoopKV) KVPool() resident.KVPoolDetail        { return resident.KVPoolDetail{} }
func (NoopKV) ReusableKV(prefix int) int            { return prefix }
func (NoopKV) ReusableKVSlotAvailable() bool        { return false }
func (NoopKV) ReusableKVSlot(_ int, prefix int) int { return prefix }
func (NoopKV) KVUploadPassAvailable() bool          { return false }
func (NoopKV) BeginKVUpload() error                 { return errSubsystemUnavailable }
func (NoopKV) EndKVUpload(bool) error               { return errSubsystemUnavailable }
func (NoopKV) KVCopierAvailable() bool              { return false }
func (NoopKV) CopyKV(int, int, int, int) error      { return errSubsystemUnavailable }
func (NoopKV) KVSnapshotterAvailable() bool         { return false }
func (NoopKV) SnapshotKV(int, int) (resident.KVSnapshot, error) {
	return nil, errSubsystemUnavailable
}
func (NoopKV) RestoreKV(resident.KVSnapshot, int, int) error { return errSubsystemUnavailable }
func (NoopKV) KVSnapshotBytes(int) int64                     { return 0 }
func (NoopKV) KVSnapshotRollback() int                       { return 0 }
func (NoopKV) KVSnapshotRecurrent() bool                     { return false }
func (NoopKV) KVDemoterAvailable() bool                      { return false }
func (NoopKV) SetKVHostBudget(resident.KVHostBudget)         {}
func (NoopKV) PromoteKV(int, int) (int, error)               { return 0, errSubsystemUnavailable }
func (NoopKV) CopyDemotedKV(int, int, int, int) (int, bool, error) {
	return 0, false, errSubsystemUnavailable
}
func (NoopKV) ReserveKVHost(int64) bool          { return false }
func (NoopKV) RecurrentCheckpointBytes() int64   { return 0 }
func (NoopKV) SaveRecurrentCheckpoint() error    { return errSubsystemUnavailable }
func (NoopKV) RestoreRecurrentCheckpoint() error { return errSubsystemUnavailable }
func (NoopKV) RecurrentCheckpointHeld(int) bool  { return false }

// NoopRuntime provides safe absent defaults for resident.RuntimeSubsystem.
type NoopRuntime struct{}

func (NoopRuntime) ResidentCaps() resident.Caps { return resident.Caps{} }
func (NoopRuntime) CostAvailable() bool         { return false }
func (NoopRuntime) CostSpan(_ *cost.Counter, _ cost.Phase, work func() (int, error)) error {
	_, err := work()
	return err
}
func (NoopRuntime) AdapterAvailable() bool                          { return false }
func (NoopRuntime) SetAdapter([]adapter.ResidentAdapterLayer) error { return errSubsystemUnavailable }
func (NoopRuntime) ExpertCacheAvailable() bool                      { return false }
func (NoopRuntime) ExpertCacheStats() (resident.ExpertCacheStats, bool) {
	return resident.ExpertCacheStats{}, false
}
func (NoopRuntime) ResizeExpertCache(int64) (int, error) { return 0, errSubsystemUnavailable }
func (NoopRuntime) VerifyPathInfo() (bool, string, bool) { return false, "", false }
func (NoopRuntime) VerifyCost() []float64                { return nil }
func (NoopRuntime) DecodeVerifyDivergence() error        { return nil }
func (NoopRuntime) EnterExactAttention() func()          { return noopLeave }
func (NoopRuntime) SeedArgmaxAvailable() bool            { return false }
func (NoopRuntime) PrefillSeedArgmax([][]float32, int) (int, error) {
	return 0, errSubsystemUnavailable
}
func (NoopRuntime) LastNArgmaxAvailable() bool { return false }
func (NoopRuntime) PrefillLastNArgmax([][]float32, int) ([]int, error) {
	return nil, errSubsystemUnavailable
}
func (NoopRuntime) DrafterHostAvailable() bool { return false }
func (NoopRuntime) CanStreamDrafterRows() bool { return false }
func (NoopRuntime) AttachBlockDrafter(resident.BlockDrafterWeights) (resident.BlockDrafter, error) {
	return nil, errSubsystemUnavailable
}
func (NoopRuntime) SetBatchedCapture([]int) error  { return errSubsystemUnavailable }
func (NoopRuntime) BatchedCapture() [][]float32    { return nil }
func (NoopRuntime) RecurrentSpecAvailable() bool   { return false }
func (NoopRuntime) PrepareRecurrentSpec(int) error { return errSubsystemUnavailable }
func (NoopRuntime) RecurrentSpecRows() int         { return 0 }
func (NoopRuntime) VerifyLastNArgmax([][]float32, int) ([]int, error) {
	return nil, errSubsystemUnavailable
}
func (NoopRuntime) FoldVerified(int) error { return errSubsystemUnavailable }
func (NoopRuntime) FlushFold() error       { return errSubsystemUnavailable }
func (NoopRuntime) DropFold()              {}

// SubsystemsOf resolves r's four resident subsystems: a resident.SubsystemProvider returns its own pre-built wrappers
// (zero allocation); nil, and any value that is not a provider, has no capability and gets the no-op subsystems. r is a resident:
// a subsystem is not a provider, so passing one is that silent no-capability answer, and under goinfer_testhooks it panics.
func ResolveSubsystems(r any) resident.Subsystems {
	if p, ok := r.(resident.SubsystemProvider); ok {
		return p.ResidentSubsystems()
	}
	return NoopSubsystems
}

// drafterHostOf is the resident r itself as a drafter host. A BlockSpec keeps the resident, not its Runtime subsystem, because step and
// SubsystemsOf resolve a resident; ok is false where r does not implement it.
func DrafterHostOf(r resident.Forward) (host resident.DrafterHost, ok bool) {
	host, ok = r.(resident.DrafterHost)
	return host, ok
}
