package engine

import (
	"context"
	"reflect"

	"github.com/townsendmerino/goinfer/decoder"
)

// Model is a loaded model as the serving shell calls it: every method of *decoder.Model the shell uses, so a test can
// serve a fake instead of a checkpoint. A new call from the shell to the model adds its method here. Make one only
// through HandleOf, so a nil model stays a nil Model (an interface holding a typed nil is not nil).
type Model interface {
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

var _ Model = (*decoder.Model)(nil)

// HandleOf is m as a Model: nil for a nil m.
func HandleOf(m *decoder.Model) Model {
	if m == nil {
		return nil
	}
	return m
}

// Call is the Model to call a method on: h, or a nil *decoder.Model when h is nil. A nil Model has no methods to call,
// while decoder's methods answer for a nil *Model where they always have (no load profile), so a holder with no model
// answers as before. Compare h itself with nil; methods decoder does not answer for nil stay behind such a check.
func Call(h Model) Model {
	if h == nil {
		return (*decoder.Model)(nil)
	}
	return h
}

// IsNil reports whether h is nil or holds a nil pointer (a typed nil, which h == nil does not catch).
func IsNil(h Model) bool {
	if h == nil {
		return true
	}
	v := reflect.ValueOf(h)
	return v.Kind() == reflect.Pointer && v.IsNil()
}

// AsDecoder is the *decoder.Model behind h, for calls that take the concrete model; nil for a nil or fake Model.
func AsDecoder(h Model) *decoder.Model {
	m, _ := h.(*decoder.Model)
	return m
}

// BlockDrafter is an attached pretrained block drafter as the shell calls it. Make one only through DrafterOf.
type BlockDrafter interface {
	GenerateStream(ctx context.Context, prompt []int, maxTokens int, sp decoder.SamplingParams) (<-chan int, *decoder.Generation, error)
}

var _ BlockDrafter = (*decoder.BlockSpec)(nil)

// DrafterOf is b as a BlockDrafter: nil for a nil b.
func DrafterOf(b *decoder.BlockSpec) BlockDrafter {
	if b == nil {
		return nil
	}
	return b
}
