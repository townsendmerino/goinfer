package decoder

import (
	"context"
	"errors"
	"reflect"
	"sync/atomic"
	"unsafe"

	"github.com/townsendmerino/goinfer/internal/cost"
)

var errSubsystemUnavailable = errors.New("decoder: resident subsystem capability not implemented")

// ResidentSubsystemProvider is implemented by GPU resident runners (cuda, metal, webgpu) to expose their capabilities
// through four cohesive subsystem interfaces instead of ad-hoc per-method type assertions. Backends pin their subsystem
// wrappers to ResidentDecodeSubsystem, ResidentPrefillSubsystem, ResidentKVSubsystem and ResidentRuntimeSubsystem at
// compile time, so signature drift fails the build instead of silently disarming an optional fast path at runtime.
//
// In-package test doubles that implement only ResidentForward plus one or two legacy micro-interfaces need not implement
// ResidentSubsystemProvider: SubsystemsOf bridges them automatically.
type ResidentSubsystemProvider interface {
	ResidentSubsystems() ResidentSubsystems
}

// ResidentSubsystems groups a resident runner's capabilities by domain: single-token and multi-sequence decode, batched
// prompt prefill, KV slot and snapshot management, and runtime control / speculation / telemetry.
type ResidentSubsystems struct {
	Decode  ResidentDecodeSubsystem
	Prefill ResidentPrefillSubsystem
	KV      ResidentKVSubsystem
	Runtime ResidentRuntimeSubsystem
}

// ResidentDecodeSubsystem covers single-token decode fast paths (argmax, top-K, device Gumbel sample, m-RoPE, KV-only),
// device-chained greedy and sampled decode, and multi-sequence batched / ragged decode steps.
type ResidentDecodeSubsystem interface {
	GreedyAvailable() bool
	ForwardArgmax(embedding []float32, pos int) (int, error)

	TopKAvailable() bool
	ForwardTopK(embedding []float32, pos, k int, temperature float64, wantZ bool) (TopKRow, error)

	SampleAvailable() bool
	ForwardSample(embedding []float32, pos int, temperature float64, seed, draw uint64) (int, error)

	MRoPEDecodeAvailable() bool
	ForwardMRoPE(embedding []float32, pos, ropePos int) (logits []float32, err error)

	NoLogitsAvailable() bool
	ForwardNoLogits(embedding []float32, pos int) error

	GreedyChainAvailable() bool
	GreedyChainStart(id, pos int) error
	GreedyChainNext() (int, error)
	GreedyChainStop()

	SampleChainAvailable() bool
	SampleChainStart(id, pos int, temperature float64, seed, draw uint64) error
	SampleChainNext(seed, draw uint64) (int, error)

	BatchStepperAvailable() bool
	BatchStepRange() (lo, hi int)
	StepBatch(seqs []ResidentBatchSeq) ([]ResidentBatchOut, error)

	RaggedStepperAvailable() bool
	StepRagged(seqs []ResidentRaggedSeq) ([]ResidentRaggedOut, error)
}

// ResidentPrefillSubsystem covers batched prompt ingestion, row-chunked and boundary-marked prefill, block and chunk
// scratch pricing, hidden / residual extraction, and multimodal / m-RoPE / span prefill.
type ResidentPrefillSubsystem interface {
	PrefillAvailable() bool
	PrefillLast(ctx context.Context, embeddings [][]float32, startPos int) (logits []float32, err error)

	PrefillRowsAvailable() bool
	PrefillLastRows(ctx context.Context, n, startPos int, rows func(lo, hi int) [][]float32) ([]float32, error)

	PrefillMarkedAvailable() bool
	PrefillMarkAlign() int
	PrefillLastMarked(ctx context.Context, embeddings [][]float32, startPos, mark int, at func(pos int)) ([]float32, error)

	PrefillPathInfo() (batched bool, reason string, ok bool)
	PrefillTailExact() bool

	PrefillBlocksAvailable() bool
	PrefillBlockRows() int
	PrefillChunksAvailable() bool
	PrefillChunkRows() int
	PrefillScratchBytes(rows int) int64
	PrefillSpanScratchBytes(rows, keys int) int64

	HiddenLastAvailable() bool
	HiddenLast(ctx context.Context, embeddings [][]float32, startPos int) (hidden []float32, err error)

	ResidualAllAvailable() bool
	ResidualAll(ctx context.Context, embeddings [][]float32, startPos int) (residuals [][]float32, err error)

	ImagePrefillAvailable() bool
	PrefillImageLast(ctx context.Context, embeddings [][]float32, startPos, imgStart, imgEnd int) (logits []float32, err error)

	ImageBlocksPrefillAvailable() bool
	PrefillImageBlocksLast(ctx context.Context, embeddings [][]float32, startPos int, blocks [][2]int) (logits []float32, err error)

	SpanPrefillAvailable() bool
	PrefillSpansLast(ctx context.Context, embeddings [][]float32, startPos int, spans [][2]int) (logits []float32, err error)

	HybridMRoPEPrefill() bool

	MRoPEPrefillAvailable() bool
	PrefillMRoPELast(ctx context.Context, embeddings [][]float32, startPos int, mropePos [][3]int) (logits []float32, err error)

	MRoPEDeepstackPrefillAvailable() bool
	PrefillMRoPEDeepstackLast(ctx context.Context, embeddings [][]float32, startPos int, mropePos [][3]int, deep [][]float32, imgStart, imgLen int) (logits []float32, err error)
}

// ResidentKVSubsystem covers multi-slot KV binding, shared block-pool telemetry, slot activity, cross-slot KV copy,
// windowed reuse queries, atomic upload passes, host prefix snapshots, host KV demotion, and recurrent checkpoints.
type ResidentKVSubsystem interface {
	KVSlotterAvailable() bool
	KVSlots() int
	UseKVSlot(i int) error

	SetKVSlotBusy(slot int, busy bool)
	KVPool() KVPoolDetail

	ReusableKV(prefix int) int
	ReusableKVSlotAvailable() bool
	ReusableKVSlot(slot, prefix int) int

	KVUploadPassAvailable() bool
	BeginKVUpload() error
	EndKVUpload(failed bool) error

	KVCopierAvailable() bool
	CopyKV(src, dst, start, end int) error

	KVSnapshotterAvailable() bool
	SnapshotKV(slot, end int) (ResidentKVSnapshot, error)
	RestoreKV(snap ResidentKVSnapshot, slot, end int) error
	KVSnapshotBytes(end int) int64
	KVSnapshotRollback() int
	KVSnapshotRecurrent() bool

	KVDemoterAvailable() bool
	SetKVHostBudget(b KVHostBudget)
	PromoteKV(slot, rows int) (blocks int, err error)
	CopyDemotedKV(src, dst, start, end int) (blocks int, held bool, err error)
	ReserveKVHost(n int64) bool

	RecurrentCheckpointBytes() int64
	SaveRecurrentCheckpoint() error
	RestoreRecurrentCheckpoint() error
	RecurrentCheckpointHeld(slot int) bool
}

// ResidentRuntimeSubsystem covers load-time capabilities, device cost spans, compute-time LoRA adapters, MoE expert
// cache management, speculative verify path / cost / divergence guards, batched argmax verify, block-drafting host, and
// recurrent speculative fold.
type ResidentRuntimeSubsystem interface {
	ResidentCaps() ResidentCaps

	CostAvailable() bool
	CostSpan(c *cost.Counter, p cost.Phase, work func() (tokens int, err error)) error

	AdapterAvailable() bool
	SetAdapter(layers []ResidentAdapterLayer) error

	ExpertCacheAvailable() bool
	ExpertCacheStats() (s ExpertCacheStats, ok bool)
	ResizeExpertCache(reserve int64) (slots int, err error)

	VerifyPathInfo() (batched bool, reason string, ok bool)
	VerifyCost() []float64
	DecodeVerifyDivergence() error
	EnterExactAttention() (leave func())

	SeedArgmaxAvailable() bool
	PrefillSeedArgmax(embeddings [][]float32, startPos int) (int, error)

	LastNArgmaxAvailable() bool
	PrefillLastNArgmax(embeddings [][]float32, startPos int) ([]int, error)

	DrafterHostAvailable() bool
	CanStreamDrafterRows() bool
	AttachBlockDrafter(w BlockDrafterWeights) (ResidentBlockDrafter, error)
	SetBatchedCapture(taps []int) error
	BatchedCapture() [][]float32

	RecurrentSpecAvailable() bool
	PrepareRecurrentSpec(rows int) error
	RecurrentSpecRows() int
	VerifyLastNArgmax(embeddings [][]float32, startPos int) ([]int, error)
	FoldVerified(n int) error
	FlushFold() error
	DropFold()
}

// Compile-time checks that the four subsystem interfaces satisfy the legacy composite interfaces callers pass around,
// and that the NoopResident* embeddable defaults cover every method.
var (
	_ ResidentDecodeSubsystem  = NoopResidentDecode{}
	_ ResidentPrefillSubsystem = NoopResidentPrefill{}
	_ ResidentKVSubsystem      = NoopResidentKV{}
	_ ResidentRuntimeSubsystem = NoopResidentRuntime{}

	_ ResidentSampleChain         = (ResidentDecodeSubsystem)(nil)
	_ ResidentPrefillBlocks       = (ResidentPrefillSubsystem)(nil)
	_ ResidentKVSnapshotter       = (ResidentKVSubsystem)(nil)
	_ ResidentRecurrentCheckpoint = (ResidentKVSubsystem)(nil)
	_ ResidentReusableKVSlot      = (ResidentKVSubsystem)(nil)
	_ ResidentAdapter             = (ResidentRuntimeSubsystem)(nil)
	_ ResidentCost                = (ResidentRuntimeSubsystem)(nil)
	_ ResidentDrafterHost         = (ResidentRuntimeSubsystem)(nil)
	_ ResidentRecurrentSpec       = (ResidentRuntimeSubsystem)(nil)
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

func (NoopResidentDecode) GreedyAvailable() bool                       { return false }
func (NoopResidentDecode) ForwardArgmax([]float32, int) (int, error)   { return 0, errSubsystemUnavailable }
func (NoopResidentDecode) TopKAvailable() bool                         { return false }
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
func (NoopResidentDecode) NoLogitsAvailable() bool             { return false }
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
func (NoopResidentDecode) BatchStepperAvailable() bool    { return false }
func (NoopResidentDecode) BatchStepRange() (lo, hi int)   { return 0, 0 }
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
func (NoopResidentPrefill) PrefillPathInfo() (bool, string, bool) { return false, "", false }
func (NoopResidentPrefill) PrefillTailExact() bool                { return false }
func (NoopResidentPrefill) PrefillBlocksAvailable() bool          { return false }
func (NoopResidentPrefill) PrefillBlockRows() int                 { return 0 }
func (NoopResidentPrefill) PrefillChunksAvailable() bool          { return false }
func (NoopResidentPrefill) PrefillChunkRows() int                 { return 0 }
func (NoopResidentPrefill) PrefillScratchBytes(int) int64         { return 0 }
func (NoopResidentPrefill) PrefillSpanScratchBytes(int, int) int64 { return 0 }
func (NoopResidentPrefill) HiddenLastAvailable() bool             { return false }
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
func (NoopResidentPrefill) HybridMRoPEPrefill() bool     { return false }
func (NoopResidentPrefill) MRoPEPrefillAvailable() bool  { return false }
func (NoopResidentPrefill) PrefillMRoPELast(context.Context, [][]float32, int, [][3]int) ([]float32, error) {
	return nil, errSubsystemUnavailable
}
func (NoopResidentPrefill) MRoPEDeepstackPrefillAvailable() bool { return false }
func (NoopResidentPrefill) PrefillMRoPEDeepstackLast(context.Context, [][]float32, int, [][3]int, [][]float32, int, int) ([]float32, error) {
	return nil, errSubsystemUnavailable
}

// NoopResidentKV provides safe absent defaults for ResidentKVSubsystem.
type NoopResidentKV struct{}

func (NoopResidentKV) KVSlotterAvailable() bool                     { return false }
func (NoopResidentKV) KVSlots() int                                 { return 1 }
func (NoopResidentKV) UseKVSlot(int) error                          { return nil }
func (NoopResidentKV) SetKVSlotBusy(int, bool)                      {}
func (NoopResidentKV) KVPool() KVPoolDetail                         { return KVPoolDetail{} }
func (NoopResidentKV) ReusableKV(prefix int) int                    { return prefix }
func (NoopResidentKV) ReusableKVSlotAvailable() bool                { return false }
func (NoopResidentKV) ReusableKVSlot(_ int, prefix int) int         { return prefix }
func (NoopResidentKV) KVUploadPassAvailable() bool                  { return false }
func (NoopResidentKV) BeginKVUpload() error                         { return errSubsystemUnavailable }
func (NoopResidentKV) EndKVUpload(bool) error                       { return errSubsystemUnavailable }
func (NoopResidentKV) KVCopierAvailable() bool                      { return false }
func (NoopResidentKV) CopyKV(int, int, int, int) error              { return errSubsystemUnavailable }
func (NoopResidentKV) KVSnapshotterAvailable() bool                 { return false }
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
func (NoopResidentRuntime) AdapterAvailable() bool                          { return false }
func (NoopResidentRuntime) SetAdapter([]ResidentAdapterLayer) error         { return errSubsystemUnavailable }
func (NoopResidentRuntime) ExpertCacheAvailable() bool                      { return false }
func (NoopResidentRuntime) ExpertCacheStats() (ExpertCacheStats, bool)      { return ExpertCacheStats{}, false }
func (NoopResidentRuntime) ResizeExpertCache(int64) (int, error)            { return 0, errSubsystemUnavailable }
func (NoopResidentRuntime) VerifyPathInfo() (bool, string, bool)            { return false, "", false }
func (NoopResidentRuntime) VerifyCost() []float64                           { return nil }
func (NoopResidentRuntime) DecodeVerifyDivergence() error                   { return nil }
func (NoopResidentRuntime) EnterExactAttention() func()                     { return noopLeave }
func (NoopResidentRuntime) SeedArgmaxAvailable() bool                       { return false }
func (NoopResidentRuntime) PrefillSeedArgmax([][]float32, int) (int, error) { return 0, errSubsystemUnavailable }
func (NoopResidentRuntime) LastNArgmaxAvailable() bool                      { return false }
func (NoopResidentRuntime) PrefillLastNArgmax([][]float32, int) ([]int, error) {
	return nil, errSubsystemUnavailable
}
func (NoopResidentRuntime) DrafterHostAvailable() bool { return false }
func (NoopResidentRuntime) CanStreamDrafterRows() bool { return false }
func (NoopResidentRuntime) AttachBlockDrafter(BlockDrafterWeights) (ResidentBlockDrafter, error) {
	return nil, errSubsystemUnavailable
}
func (NoopResidentRuntime) SetBatchedCapture([]int) error                   { return errSubsystemUnavailable }
func (NoopResidentRuntime) BatchedCapture() [][]float32                     { return nil }
func (NoopResidentRuntime) RecurrentSpecAvailable() bool                    { return false }
func (NoopResidentRuntime) PrepareRecurrentSpec(int) error                  { return errSubsystemUnavailable }
func (NoopResidentRuntime) RecurrentSpecRows() int                          { return 0 }
func (NoopResidentRuntime) VerifyLastNArgmax([][]float32, int) ([]int, error) {
	return nil, errSubsystemUnavailable
}
func (NoopResidentRuntime) FoldVerified(int) error { return errSubsystemUnavailable }
func (NoopResidentRuntime) FlushFold() error       { return errSubsystemUnavailable }
func (NoopResidentRuntime) DropFold()              {}

// SubsystemsOf resolves r's four resident subsystems. Production GPU backends implement ResidentSubsystemProvider and
// return pre-built zero-allocation wrappers directly; nil returns no-op subsystems; an existing subsystem value wraps
// itself; and legacy / test stubs are bridged through the single compatibility adapter in this file.
func SubsystemsOf(r any) ResidentSubsystems {
	if r == nil {
		return noopResidentSubsystems
	}
	if p, ok := r.(ResidentSubsystemProvider); ok {
		return p.ResidentSubsystems()
	}
	switch s := r.(type) {
	case ResidentRuntimeSubsystem:
		out := noopResidentSubsystems
		out.Runtime = s
		return out
	case ResidentDecodeSubsystem:
		out := noopResidentSubsystems
		out.Decode = s
		return out
	case ResidentPrefillSubsystem:
		out := noopResidentSubsystems
		out.Prefill = s
		return out
	case ResidentKVSubsystem:
		out := noopResidentSubsystems
		out.KV = s
		return out
	}
	return ResidentSubsystems{
		Decode:  &legacyDecodeSub{r: r},
		Prefill: &legacyPrefillSub{r: r},
		KV:      &legacyKVSub{r: r},
		Runtime: &legacyRuntimeSub{r: r},
	}
}

type cachedResidentSubsystems struct {
	res any
	sub ResidentSubsystems
}

// sameResident compares two ResidentForward values without panicking when a test assigns an uncomparable value struct
// (for example, one holding a slice) to Model.resident.
func sameResident(a, b any) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	ta := reflect.TypeOf(a)
	if ta != reflect.TypeOf(b) || !ta.Comparable() {
		return false
	}
	return a == b
}

// subsystems returns m.resident's four subsystems with zero steady-state allocations on the decode hot path.
func (m *Model) subsystems() ResidentSubsystems {
	if m == nil || m.resident == nil {
		return noopResidentSubsystems
	}
	r := m.resident
	if p, ok := r.(ResidentSubsystemProvider); ok {
		return p.ResidentSubsystems()
	}
	if c := (*cachedResidentSubsystems)(atomic.LoadPointer(&m.resSubP)); c != nil && sameResident(c.res, r) {
		return c.sub
	}
	sub := SubsystemsOf(r)
	if t := reflect.TypeOf(r); t != nil && t.Comparable() {
		atomic.StorePointer(&m.resSubP, unsafe.Pointer(&cachedResidentSubsystems{res: r, sub: sub}))
	}
	return sub
}

func (m *Model) resDecode() ResidentDecodeSubsystem   { return m.subsystems().Decode }
func (m *Model) resPrefill() ResidentPrefillSubsystem { return m.subsystems().Prefill }
func (m *Model) resKV() ResidentKVSubsystem           { return m.subsystems().KV }
func (m *Model) resRuntime() ResidentRuntimeSubsystem { return m.subsystems().Runtime }

// --- Legacy compatibility adapters (sole place legacy micro-interfaces are asserted) ---

type legacyDecodeSub struct{ r any }

func (d *legacyDecodeSub) GreedyAvailable() bool {
	_, ok := d.r.(ResidentGreedy)
	return ok
}

func (d *legacyDecodeSub) ForwardArgmax(embedding []float32, pos int) (int, error) {
	return d.r.(ResidentGreedy).ForwardArgmax(embedding, pos)
}

func (d *legacyDecodeSub) TopKAvailable() bool {
	tk, ok := d.r.(ResidentTopK)
	return ok && tk.TopKAvailable()
}

func (d *legacyDecodeSub) ForwardTopK(embedding []float32, pos, k int, temperature float64, wantZ bool) (TopKRow, error) {
	return d.r.(ResidentTopK).ForwardTopK(embedding, pos, k, temperature, wantZ)
}

func (d *legacyDecodeSub) SampleAvailable() bool {
	sm, ok := d.r.(ResidentSample)
	return ok && sm.SampleAvailable()
}

func (d *legacyDecodeSub) ForwardSample(embedding []float32, pos int, temperature float64, seed, draw uint64) (int, error) {
	return d.r.(ResidentSample).ForwardSample(embedding, pos, temperature, seed, draw)
}

func (d *legacyDecodeSub) MRoPEDecodeAvailable() bool {
	_, ok := d.r.(ResidentMRoPE)
	return ok
}

func (d *legacyDecodeSub) ForwardMRoPE(embedding []float32, pos, ropePos int) ([]float32, error) {
	return d.r.(ResidentMRoPE).ForwardMRoPE(embedding, pos, ropePos)
}

func (d *legacyDecodeSub) NoLogitsAvailable() bool {
	_, ok := d.r.(ResidentPrefillKV)
	return ok
}

func (d *legacyDecodeSub) ForwardNoLogits(embedding []float32, pos int) error {
	return d.r.(ResidentPrefillKV).ForwardNoLogits(embedding, pos)
}

func (d *legacyDecodeSub) GreedyChainAvailable() bool {
	gc, ok := d.r.(ResidentGreedyChain)
	return ok && gc.GreedyChainAvailable()
}

func (d *legacyDecodeSub) GreedyChainStart(id, pos int) error {
	return d.r.(ResidentGreedyChain).GreedyChainStart(id, pos)
}

func (d *legacyDecodeSub) GreedyChainNext() (int, error) {
	return d.r.(ResidentGreedyChain).GreedyChainNext()
}

func (d *legacyDecodeSub) GreedyChainStop() {
	if gc, ok := d.r.(ResidentGreedyChain); ok {
		gc.GreedyChainStop()
	}
}

func (d *legacyDecodeSub) SampleChainAvailable() bool {
	sc, ok := d.r.(ResidentSampleChain)
	return ok && sc.SampleChainAvailable()
}

func (d *legacyDecodeSub) SampleChainStart(id, pos int, temperature float64, seed, draw uint64) error {
	return d.r.(ResidentSampleChain).SampleChainStart(id, pos, temperature, seed, draw)
}

func (d *legacyDecodeSub) SampleChainNext(seed, draw uint64) (int, error) {
	return d.r.(ResidentSampleChain).SampleChainNext(seed, draw)
}

func (d *legacyDecodeSub) BatchStepperAvailable() bool {
	_, ok := d.r.(ResidentBatchStepper)
	return ok
}

func (d *legacyDecodeSub) BatchStepRange() (lo, hi int) {
	if st, ok := d.r.(ResidentBatchStepper); ok {
		return st.BatchStepRange()
	}
	return 0, 0
}

func (d *legacyDecodeSub) StepBatch(seqs []ResidentBatchSeq) ([]ResidentBatchOut, error) {
	return d.r.(ResidentBatchStepper).StepBatch(seqs)
}

func (d *legacyDecodeSub) RaggedStepperAvailable() bool {
	_, ok := d.r.(ResidentRaggedStepper)
	return ok
}

func (d *legacyDecodeSub) StepRagged(seqs []ResidentRaggedSeq) ([]ResidentRaggedOut, error) {
	return d.r.(ResidentRaggedStepper).StepRagged(seqs)
}

type legacyPrefillSub struct{ r any }

func (p *legacyPrefillSub) PrefillAvailable() bool {
	_, ok := p.r.(Prefiller)
	return ok
}

func (p *legacyPrefillSub) PrefillLast(ctx context.Context, embeddings [][]float32, startPos int) ([]float32, error) {
	return p.r.(Prefiller).PrefillLast(ctx, embeddings, startPos)
}

func (p *legacyPrefillSub) PrefillRowsAvailable() bool {
	_, ok := p.r.(ResidentPrefillRows)
	return ok
}

func (p *legacyPrefillSub) PrefillLastRows(ctx context.Context, n, startPos int, rows func(lo, hi int) [][]float32) ([]float32, error) {
	return p.r.(ResidentPrefillRows).PrefillLastRows(ctx, n, startPos, rows)
}

func (p *legacyPrefillSub) PrefillMarkedAvailable() bool {
	_, ok := p.r.(ResidentPrefillMarked)
	return ok
}

func (p *legacyPrefillSub) PrefillMarkAlign() int {
	if mp, ok := p.r.(ResidentPrefillMarked); ok {
		return mp.PrefillMarkAlign()
	}
	return 0
}

func (p *legacyPrefillSub) PrefillLastMarked(ctx context.Context, embeddings [][]float32, startPos, mark int, at func(pos int)) ([]float32, error) {
	return p.r.(ResidentPrefillMarked).PrefillLastMarked(ctx, embeddings, startPos, mark, at)
}

func (p *legacyPrefillSub) PrefillPathInfo() (bool, string, bool) {
	if rep, ok := p.r.(PrefillPathReporter); ok {
		b, reason := rep.PrefillPath()
		return b, reason, true
	}
	return false, "", false
}

func (p *legacyPrefillSub) PrefillTailExact() bool {
	te, ok := p.r.(PrefillTailExact)
	return ok && te.PrefillTailExact()
}

func (p *legacyPrefillSub) PrefillBlocksAvailable() bool {
	_, ok := p.r.(ResidentPrefillBlocks)
	return ok
}

func (p *legacyPrefillSub) PrefillBlockRows() int {
	if pb, ok := p.r.(ResidentPrefillBlocks); ok {
		return pb.PrefillBlockRows()
	}
	return 0
}

func (p *legacyPrefillSub) PrefillChunksAvailable() bool {
	_, ok := p.r.(ResidentPrefillChunks)
	return ok
}

func (p *legacyPrefillSub) PrefillChunkRows() int {
	if pc, ok := p.r.(ResidentPrefillChunks); ok {
		return pc.PrefillChunkRows()
	}
	return 0
}

func (p *legacyPrefillSub) PrefillScratchBytes(rows int) int64 {
	if pb, ok := p.r.(ResidentPrefillBlocks); ok {
		return pb.PrefillScratchBytes(rows)
	}
	if pc, ok := p.r.(ResidentPrefillChunks); ok {
		return pc.PrefillScratchBytes(rows)
	}
	return 0
}

func (p *legacyPrefillSub) PrefillSpanScratchBytes(rows, keys int) int64 {
	if ss, ok := p.r.(ResidentPrefillSpanScratch); ok {
		return ss.PrefillSpanScratchBytes(rows, keys)
	}
	return 0
}

func (p *legacyPrefillSub) HiddenLastAvailable() bool {
	_, ok := p.r.(ResidentHiddenLast)
	return ok
}

func (p *legacyPrefillSub) HiddenLast(ctx context.Context, embeddings [][]float32, startPos int) ([]float32, error) {
	return p.r.(ResidentHiddenLast).HiddenLast(ctx, embeddings, startPos)
}

func (p *legacyPrefillSub) ResidualAllAvailable() bool {
	_, ok := p.r.(ResidentResidualAll)
	return ok
}

func (p *legacyPrefillSub) ResidualAll(ctx context.Context, embeddings [][]float32, startPos int) ([][]float32, error) {
	return p.r.(ResidentResidualAll).ResidualAll(ctx, embeddings, startPos)
}

func (p *legacyPrefillSub) ImagePrefillAvailable() bool {
	_, ok := p.r.(ResidentImagePrefill)
	return ok
}

func (p *legacyPrefillSub) PrefillImageLast(ctx context.Context, embeddings [][]float32, startPos, imgStart, imgEnd int) ([]float32, error) {
	return p.r.(ResidentImagePrefill).PrefillImageLast(ctx, embeddings, startPos, imgStart, imgEnd)
}

func (p *legacyPrefillSub) ImageBlocksPrefillAvailable() bool {
	_, ok := p.r.(ResidentImageBlocksPrefill)
	return ok
}

func (p *legacyPrefillSub) PrefillImageBlocksLast(ctx context.Context, embeddings [][]float32, startPos int, blocks [][2]int) ([]float32, error) {
	return p.r.(ResidentImageBlocksPrefill).PrefillImageBlocksLast(ctx, embeddings, startPos, blocks)
}

func (p *legacyPrefillSub) SpanPrefillAvailable() bool {
	_, ok := p.r.(ResidentSpanPrefill)
	return ok
}

func (p *legacyPrefillSub) PrefillSpansLast(ctx context.Context, embeddings [][]float32, startPos int, spans [][2]int) ([]float32, error) {
	return p.r.(ResidentSpanPrefill).PrefillSpansLast(ctx, embeddings, startPos, spans)
}

func (p *legacyPrefillSub) HybridMRoPEPrefill() bool {
	hp, ok := p.r.(ResidentHybridMRoPEPrefill)
	return ok && hp.HybridMRoPEPrefill()
}

func (p *legacyPrefillSub) MRoPEPrefillAvailable() bool {
	_, ok := p.r.(ResidentMRoPEPrefill)
	return ok
}

func (p *legacyPrefillSub) PrefillMRoPELast(ctx context.Context, embeddings [][]float32, startPos int, mropePos [][3]int) ([]float32, error) {
	return p.r.(ResidentMRoPEPrefill).PrefillMRoPELast(ctx, embeddings, startPos, mropePos)
}

func (p *legacyPrefillSub) MRoPEDeepstackPrefillAvailable() bool {
	_, ok := p.r.(ResidentMRoPEDeepstackPrefill)
	return ok
}

func (p *legacyPrefillSub) PrefillMRoPEDeepstackLast(ctx context.Context, embeddings [][]float32, startPos int, mropePos [][3]int, deep [][]float32, imgStart, imgLen int) ([]float32, error) {
	return p.r.(ResidentMRoPEDeepstackPrefill).PrefillMRoPEDeepstackLast(ctx, embeddings, startPos, mropePos, deep, imgStart, imgLen)
}

type legacyKVSub struct{ r any }

func (k *legacyKVSub) KVSlotterAvailable() bool {
	_, ok := k.r.(ResidentKVSlotter)
	return ok
}

func (k *legacyKVSub) KVSlots() int {
	if sl, ok := k.r.(ResidentKVSlotter); ok {
		return sl.KVSlots()
	}
	return 1
}

func (k *legacyKVSub) UseKVSlot(i int) error {
	if sl, ok := k.r.(ResidentKVSlotter); ok {
		return sl.UseKVSlot(i)
	}
	return nil
}

func (k *legacyKVSub) SetKVSlotBusy(slot int, busy bool) {
	if act, ok := k.r.(ResidentKVSlotActivity); ok {
		act.SetKVSlotBusy(slot, busy)
	}
}

func (k *legacyKVSub) KVPool() KVPoolDetail {
	if p, ok := k.r.(ResidentKVPoolInfo); ok {
		return p.KVPool()
	}
	return KVPoolDetail{}
}

func (k *legacyKVSub) ReusableKV(prefix int) int {
	if rk, ok := k.r.(ResidentReusableKV); ok {
		return rk.ReusableKV(prefix)
	}
	return prefix
}

func (k *legacyKVSub) ReusableKVSlotAvailable() bool {
	_, ok := k.r.(ResidentReusableKVSlot)
	return ok
}

func (k *legacyKVSub) ReusableKVSlot(slot, prefix int) int {
	if rk, ok := k.r.(ResidentReusableKVSlot); ok {
		return rk.ReusableKVSlot(slot, prefix)
	}
	return prefix
}

func (k *legacyKVSub) KVUploadPassAvailable() bool {
	_, ok := k.r.(ResidentKVUploadPass)
	return ok
}

func (k *legacyKVSub) BeginKVUpload() error {
	return k.r.(ResidentKVUploadPass).BeginKVUpload()
}

func (k *legacyKVSub) EndKVUpload(failed bool) error {
	return k.r.(ResidentKVUploadPass).EndKVUpload(failed)
}

func (k *legacyKVSub) KVCopierAvailable() bool {
	_, ok := k.r.(ResidentKVCopier)
	return ok
}

func (k *legacyKVSub) CopyKV(src, dst, start, end int) error {
	return k.r.(ResidentKVCopier).CopyKV(src, dst, start, end)
}

func (k *legacyKVSub) KVSnapshotterAvailable() bool {
	_, ok := k.r.(ResidentKVSnapshotter)
	return ok
}

func (k *legacyKVSub) SnapshotKV(slot, end int) (ResidentKVSnapshot, error) {
	return k.r.(ResidentKVSnapshotter).SnapshotKV(slot, end)
}

func (k *legacyKVSub) RestoreKV(snap ResidentKVSnapshot, slot, end int) error {
	return k.r.(ResidentKVSnapshotter).RestoreKV(snap, slot, end)
}

func (k *legacyKVSub) KVSnapshotBytes(end int) int64 {
	if sn, ok := k.r.(ResidentKVSnapshotter); ok {
		return sn.KVSnapshotBytes(end)
	}
	return 0
}

func (k *legacyKVSub) KVSnapshotRollback() int {
	if sn, ok := k.r.(ResidentKVSnapshotter); ok {
		return sn.KVSnapshotRollback()
	}
	return 0
}

func (k *legacyKVSub) KVSnapshotRecurrent() bool {
	sn, ok := k.r.(ResidentKVSnapshotter)
	return ok && sn.KVSnapshotRecurrent()
}

func (k *legacyKVSub) KVDemoterAvailable() bool {
	_, ok := k.r.(ResidentKVDemoter)
	return ok
}

func (k *legacyKVSub) SetKVHostBudget(b KVHostBudget) {
	if d, ok := k.r.(ResidentKVDemoter); ok {
		d.SetKVHostBudget(b)
	}
}

func (k *legacyKVSub) PromoteKV(slot, rows int) (int, error) {
	return k.r.(ResidentKVDemoter).PromoteKV(slot, rows)
}

func (k *legacyKVSub) CopyDemotedKV(src, dst, start, end int) (int, bool, error) {
	return k.r.(ResidentKVDemoter).CopyDemotedKV(src, dst, start, end)
}

func (k *legacyKVSub) ReserveKVHost(n int64) bool {
	if d, ok := k.r.(ResidentKVDemoter); ok {
		return d.ReserveKVHost(n)
	}
	return false
}

func (k *legacyKVSub) RecurrentCheckpointBytes() int64 {
	if rc, ok := k.r.(ResidentRecurrentCheckpoint); ok {
		return rc.RecurrentCheckpointBytes()
	}
	return 0
}

func (k *legacyKVSub) SaveRecurrentCheckpoint() error {
	return k.r.(ResidentRecurrentCheckpoint).SaveRecurrentCheckpoint()
}

func (k *legacyKVSub) RestoreRecurrentCheckpoint() error {
	return k.r.(ResidentRecurrentCheckpoint).RestoreRecurrentCheckpoint()
}

func (k *legacyKVSub) RecurrentCheckpointHeld(slot int) bool {
	rc, ok := k.r.(ResidentRecurrentCheckpoint)
	return ok && rc.RecurrentCheckpointHeld(slot)
}

type legacyLastNArgmax interface {
	PrefillLastNArgmax(embeddings [][]float32, startPos int) ([]int, error)
}

type legacyRuntimeSub struct{ r any }

func (rt *legacyRuntimeSub) ResidentCaps() ResidentCaps {
	if c, ok := rt.r.(ResidentCapsReporter); ok {
		return c.ResidentCaps()
	}
	return ResidentCaps{}
}

func (rt *legacyRuntimeSub) CostAvailable() bool {
	_, ok := rt.r.(ResidentCost)
	return ok
}

func (rt *legacyRuntimeSub) CostSpan(c *cost.Counter, p cost.Phase, work func() (int, error)) error {
	return rt.r.(ResidentCost).CostSpan(c, p, work)
}

func (rt *legacyRuntimeSub) AdapterAvailable() bool {
	_, ok := rt.r.(ResidentAdapter)
	return ok
}

func (rt *legacyRuntimeSub) SetAdapter(layers []ResidentAdapterLayer) error {
	return rt.r.(ResidentAdapter).SetAdapter(layers)
}

func (rt *legacyRuntimeSub) ExpertCacheAvailable() bool {
	_, ok := rt.r.(ResidentExpertCache)
	return ok
}

func (rt *legacyRuntimeSub) ExpertCacheStats() (ExpertCacheStats, bool) {
	if ec, ok := rt.r.(ResidentExpertCache); ok {
		return ec.ExpertCacheStats()
	}
	return ExpertCacheStats{}, false
}

func (rt *legacyRuntimeSub) ResizeExpertCache(reserve int64) (int, error) {
	return rt.r.(ResidentExpertCache).ResizeExpertCache(reserve)
}

func (rt *legacyRuntimeSub) VerifyPathInfo() (bool, string, bool) {
	if vp, ok := rt.r.(VerifyPathReporter); ok {
		b, reason := vp.VerifyPath()
		return b, reason, true
	}
	return false, "", false
}

func (rt *legacyRuntimeSub) VerifyCost() []float64 {
	if cr, ok := rt.r.(VerifyCostReporter); ok {
		return cr.VerifyCost()
	}
	return nil
}

func (rt *legacyRuntimeSub) DecodeVerifyDivergence() error {
	if dv, ok := rt.r.(DecodeVerifyDiverger); ok {
		return dv.DecodeVerifyDivergence()
	}
	return nil
}

func (rt *legacyRuntimeSub) EnterExactAttention() func() {
	if sc, ok := rt.r.(ExactAttentionScoper); ok {
		return sc.EnterExactAttention()
	}
	return noopLeave
}

func (rt *legacyRuntimeSub) SeedArgmaxAvailable() bool {
	_, ok := rt.r.(ResidentSeedArgmax)
	return ok
}

func (rt *legacyRuntimeSub) PrefillSeedArgmax(embeddings [][]float32, startPos int) (int, error) {
	return rt.r.(ResidentSeedArgmax).PrefillSeedArgmax(embeddings, startPos)
}

func (rt *legacyRuntimeSub) LastNArgmaxAvailable() bool {
	_, ok := rt.r.(legacyLastNArgmax)
	return ok
}

func (rt *legacyRuntimeSub) PrefillLastNArgmax(embeddings [][]float32, startPos int) ([]int, error) {
	return rt.r.(legacyLastNArgmax).PrefillLastNArgmax(embeddings, startPos)
}

func (rt *legacyRuntimeSub) DrafterHostAvailable() bool {
	_, ok := rt.r.(ResidentDrafterHost)
	return ok
}

func (rt *legacyRuntimeSub) CanStreamDrafterRows() bool {
	_, ok := rt.r.(StreamingDrafterHost)
	return ok
}

func (rt *legacyRuntimeSub) AttachBlockDrafter(w BlockDrafterWeights) (ResidentBlockDrafter, error) {
	return rt.r.(ResidentDrafterHost).AttachBlockDrafter(w)
}

func (rt *legacyRuntimeSub) SetBatchedCapture(taps []int) error {
	return rt.r.(ResidentDrafterHost).SetBatchedCapture(taps)
}

func (rt *legacyRuntimeSub) BatchedCapture() [][]float32 {
	if dh, ok := rt.r.(ResidentDrafterHost); ok {
		return dh.BatchedCapture()
	}
	return nil
}

func (rt *legacyRuntimeSub) RecurrentSpecAvailable() bool {
	_, ok := rt.r.(ResidentRecurrentSpec)
	return ok
}

func (rt *legacyRuntimeSub) PrepareRecurrentSpec(rows int) error {
	return rt.r.(ResidentRecurrentSpec).PrepareRecurrentSpec(rows)
}

func (rt *legacyRuntimeSub) RecurrentSpecRows() int {
	if rs, ok := rt.r.(ResidentRecurrentSpec); ok {
		return rs.RecurrentSpecRows()
	}
	return 0
}

func (rt *legacyRuntimeSub) VerifyLastNArgmax(embeddings [][]float32, startPos int) ([]int, error) {
	return rt.r.(ResidentRecurrentSpec).VerifyLastNArgmax(embeddings, startPos)
}

func (rt *legacyRuntimeSub) FoldVerified(n int) error {
	return rt.r.(ResidentRecurrentSpec).FoldVerified(n)
}

func (rt *legacyRuntimeSub) FlushFold() error {
	return rt.r.(ResidentRecurrentSpec).FlushFold()
}

func (rt *legacyRuntimeSub) DropFold() {
	if rs, ok := rt.r.(ResidentRecurrentSpec); ok {
		rs.DropFold()
	}
}
