package serveapp

import (
	"context"

	"github.com/townsendmerino/goinfer/decoder"
)

// modelHandle is a loaded model as the serving shell calls it (docs/tasks/task-subsystems-2026-10.md, SB7, "the model
// handle"): every method of *decoder.Model the shell uses, so a test can serve a fake instead of a checkpoint. A new
// call from the shell to the model adds its method here. Assign one only through handleOf, so a nil model stays a nil
// handle (an interface holding a typed nil is not nil, and the shell tests model == nil).
type modelHandle interface {
	Close() error
	AdmitPrefillMemoryShare(int, int, bool, int) error
	BlockSpecCapable() bool
	Config() *decoder.Config
	CPUBatchActive() bool
	CPUBatchKVBypass() string
	CPUBatchStats() decoder.ResidentBatchStats
	CPUConcurrentSafe() bool
	DecodePath() string
	EffectiveBackend() string
	EnableCPUBatch(int) bool
	EnableResidentConcurrency(int) int
	ExpertCacheStats() (decoder.ExpertCacheStats, bool)
	ExtraResidentBytes() int64
	FitBudgetSummary() (int, int64, int64, int64, bool)
	FitDisabled() bool
	Generate(context.Context, []int, int, decoder.SamplingParams) (<-chan int, *decoder.Generation)
	GenerateAudio(context.Context, []int, int, int, func() ([]float32, error), int, decoder.SamplingParams) (<-chan int, *decoder.Generation)
	GenerateGemma4VLSpans(context.Context, []int, []decoder.ImageSpan, func() ([]float32, error), int, decoder.SamplingParams) (<-chan int, *decoder.Generation)
	GenerateNgramSpeculativeAdaptive(context.Context, []int, int, decoder.Drafter, *decoder.AdaptiveDepth, decoder.SamplingParams) (<-chan int, *decoder.Generation, error)
	GenerateQwenVLDeepstackSpans(context.Context, []int, []decoder.ImageSpan, func() ([]float32, [][]float32, error), [][3]int, int, int, int, decoder.SamplingParams) (<-chan int, *decoder.Generation)
	GenerateVLCausalSpans(context.Context, []int, []decoder.ImageSpan, func() ([]float32, error), int, decoder.SamplingParams) (<-chan int, *decoder.Generation)
	GenerateVLSpans(context.Context, []int, []decoder.ImageSpan, func() ([]float32, error), int, decoder.SamplingParams) (<-chan int, *decoder.Generation)
	HeadTable() string
	KeepsRecurrentSnapshots() bool
	Knob(string) (string, bool)
	KVDelivered() string
	KVPrecisionReports() []decoder.KVPrecisionReport
	KVSnapshotsSupported() bool
	KVSnapshotStats() (decoder.KVSnapshotStats, bool)
	LoadAdapter(string, string) error
	LoadProfile() *decoder.LoadProfile
	NewBlockSpec(decoder.BlockDrafterWeights, []int) (*decoder.BlockSpec, error)
	PrefillPath() (bool, string)
	Quant() string
	ResidentActive() bool
	ResidentBatchStats() decoder.ResidentBatchStats
	ResidentConcurrency() int
	ResidentContextCap() int
	ResidentContextKVBytes(int) int64
	ResidentContextNote() string
	ResidentDecline() string
	ResidentKVPool() (int, int)
	ResidentKVSlots() int
	ResidentWeightBytes() int64
	SetSpecAdaptive(bool)
}

var _ modelHandle = (*decoder.Model)(nil)

// handleOf is m as a handle: nil for a nil m.
func handleOf(m *decoder.Model) modelHandle {
	if m == nil {
		return nil
	}
	return m
}

// blockDrafter is an attached pretrained block drafter as the shell calls it. Assign one only through drafterOf, as
// modelHandle through handleOf.
type blockDrafter interface {
	GenerateStream(ctx context.Context, prompt []int, maxTokens int, sp decoder.SamplingParams) (<-chan int, *decoder.Generation, error)
}

var _ blockDrafter = (*decoder.BlockSpec)(nil)

// drafterOf is b as a blockDrafter: nil for a nil b.
func drafterOf(b *decoder.BlockSpec) blockDrafter {
	if b == nil {
		return nil
	}
	return b
}

// asDecoder is the *decoder.Model behind h, for the calls that take the concrete model (the session LRU, the decision
// heads); nil for a nil or fake handle.
func asDecoder(h modelHandle) *decoder.Model {
	m, _ := h.(*decoder.Model)
	return m
}
