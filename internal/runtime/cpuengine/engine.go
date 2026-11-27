package cpuengine

import (
	"errors"
	"sync/atomic"
	"unsafe"

	"github.com/townsendmerino/aikit/linalg"
	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

// ErrNotImplemented mirrors decoder's unimplemented sentinel for unloaded-weight guards.
var ErrNotImplemented = errors.New("not implemented")

// DefaultDecodeParallelThreshold is the per-row work threshold below which single-token decode stays serial.
const DefaultDecodeParallelThreshold = 2048

// Backend abstracts the weight Matmul primitive the CPU forward calls (matching decoder.Backend).
type Backend interface {
	MatmulBT(a, b, dst []float32, M, K, N int)
}

// QuantBackend is the optional W8A8 Matmul extension (matching decoder.QuantBackend).
type QuantBackend interface {
	MatmulW8A8(a []float32, bQ []int8, bScales []float32, dst []float32, M, K, N int) bool
}

// QuantBackend4 is the optional W4A8 Matmul extension (matching decoder.QuantBackend4).
type QuantBackend4 interface {
	MatmulW4A8(a []float32, bQ4 []byte, bScales16 []uint16, group int, dst []float32, M, K, N int) bool
}

// QuantBatchBackend is the optional shared-activation W8A8 batch extension (matching decoder.QuantBatchBackend).
type QuantBatchBackend interface {
	MatmulW8A8Batch(a []float32, M, K int, ops []linalg.BatchOp) bool
}

// QuantBatchBackend4 is the optional shared-activation W4A8 batch extension (matching decoder.QuantBatchBackend4).
type QuantBatchBackend4 interface {
	MatmulW4A8Batch(a []float32, M, K, group int, ops []linalg.BatchOp) bool
}

// LayerPager is the narrow per-layer weight pager interface.
type LayerPager interface {
	EnterLayer(l int)
	FinishLayers()
}

// ExpertPager is the narrow MoE expert pager interface.
type ExpertPager interface {
	Lock()
	Unlock()
	Touch(key unsafe.Pointer)
}

// KnobSource provides per-model operator knob readings without importing decoder.
type KnobSource interface {
	AttnGrouped() bool
	FusedAttention() bool
	MLANaive() bool
	MoEExpertMajor() bool
	CPUFastAttention() bool
	BatchedPrefillDisabled() bool
	PositiveInt(name string) (int, bool)
}

// Knobs names the knob keys cpuengine reads via KnobSource.PositiveInt.
const (
	KnobAttnRowTile    = "GOINFER_ATTN_ROW_TILE"
	KnobPrefillWorkers = "GOINFER_PREFILL_ATTN_WORKERS"
)

func resolveKnobs(k KnobSource) KnobSource {
	if k != nil {
		return k
	}
	if h := ActiveHooks(); h != nil {
		return h.DefaultKnobs
	}
	return nil
}

func knobAttnGrouped(k KnobSource) bool {
	if k = resolveKnobs(k); k == nil {
		return true
	}
	return k.AttnGrouped()
}

func knobFusedAttention(k KnobSource) bool {
	if k = resolveKnobs(k); k == nil {
		return true
	}
	return k.FusedAttention()
}

func knobMLANaive(k KnobSource) bool {
	if k = resolveKnobs(k); k == nil {
		return false
	}
	return k.MLANaive()
}

func knobMoEExpertMajor(k KnobSource) bool {
	if k = resolveKnobs(k); k == nil {
		return true
	}
	return k.MoEExpertMajor()
}

func knobCPUFastAttention(k KnobSource) bool {
	if k = resolveKnobs(k); k == nil {
		return true
	}
	return k.CPUFastAttention()
}

func knobPositiveInt(k KnobSource, name string) (int, bool) {
	if k = resolveKnobs(k); k != nil {
		return k.PositiveInt(name)
	}
	return 0, false
}

// SeamPlant identifies a planted hidden-state seam defect for decoder/seams_test.go.
type SeamPlant uint8

const (
	PlantNone SeamPlant = iota
	PlantHiddenNoNorm
	PlantCaptureEarly
	PlantSubPrePostNorm
	PlantCostDoubleCount
	PlantSkipPLECapture
	PlantCtxBeforeAttend
	PlantCtxNarrow
	PlantMLPPreAfterNorm
)

// WalkBatchPlant identifies a planted batched-walker defect for tests.
type WalkBatchPlant uint8

const (
	PlantBatchNone WalkBatchPlant = iota
	PlantBatchDropConvWindow
	PlantBatchNoAttnGate
	PlantBatchImagePLEToken
	PlantBatchNoBidirectional
)

// AttnScoreStats records capped attention-score statistics when armed by tests.
type AttnScoreStats struct {
	Total   atomic.Int64
	Over15  atomic.Int64
	MaxBits atomic.Uint64
}

// DecodeTimingStats mirrors decoder's per-component decode timing counters (ns).
type DecodeTimingStats struct {
	Attn, MLP, Head, GU, Act, ActDown, QKV, O, AttnAll *int64
}

func (s *DecodeTimingStats) add(dst *int64, d int64) {
	if s != nil && dst != nil {
		atomic.AddInt64(dst, d)
	}
}

// Hooks carries diagnostic flags and test seams injected by decoder so cpuengine holds zero *ForTest globals
// and zero package-init env reads. Its zero value is safe; decoder syncs live hook values before each top-level call.
type Hooks struct {
	KVRoundF16        bool
	KVF16NewRowsF32   bool
	KVF16RingOffset   bool
	ConvWinSlice      bool
	Int4FloatAct      bool
	CPUFusedGateUp    bool
	DeltaNetTiming    bool
	DecodeTiming      bool
	W4A8BatchEnabled  bool
	W4A8PreOff        bool
	W4A8PreCalls      *atomic.Int64
	AttnScoreStats    *atomic.Pointer[AttnScoreStats]
	SeamPlant         SeamPlant
	WalkBatchPlant    WalkBatchPlant
	SSMStopEnabled    bool
	SSMStopLayer      int
	G4TraceHidden     func(layer int, h []float32)
	G4TraceQKV        func(layer int, q, k, v []float32)
	ReleaseHostMemory func()
	DecodeTimingStats *DecodeTimingStats
	DefaultKnobs      KnobSource
}

var defaultHooks = Hooks{
	CPUFusedGateUp:   FusedGateUpDefault,
	W4A8BatchEnabled: W4A8BatchDefault,
}

var activeHooksPtr atomic.Pointer[Hooks]

// SyncHooks publishes h as the active Hooks snapshot for free kernel functions.
func SyncHooks(h *Hooks) { activeHooksPtr.Store(h) }

func activeHooks() *Hooks {
	if h := activeHooksPtr.Load(); h != nil {
		return h
	}
	return &defaultHooks
}

// ImageSpan names a contiguous multimodal image range [Pos, Pos+Len) in a prompt.
type ImageSpan struct {
	Pos, Len int
	Hash     uint64
}

func imageSpanRow(spans []ImageSpan, pos int) (int, bool) {
	off := 0
	for _, s := range spans {
		if pos >= s.Pos && pos < s.Pos+s.Len {
			return off + pos - s.Pos, true
		}
		off += s.Len
	}
	return 0, false
}

// Engine holds the immutable weights, compiled layer program, backend, knobs, pagers, and hooks
// needed to execute CPU forward passes without depending on decoder.Model.
type Engine struct {
	W                 *darch.Weights
	Prog              *program.Program
	Be                Backend
	Knobs             KnobSource
	ActQuantGroup     int
	ExactPrefill      bool
	Pager             ExpertPager
	LayerPager        LayerPager
	Hooks             *Hooks
	BuildProgramCalls *atomic.Int64
}

// Program returns e's compiled layer program, building one on demand if e.Prog is nil.
func (e *Engine) Program() *program.Program {
	if e.Prog != nil {
		return e.Prog
	}
	return BuildProgram(e.W.Arch, e.W.HasExperts, e.BuildProgramCalls)
}

// BuildProgram compiles a's per-layer operation program.
func BuildProgram(a *darch.Architecture, experts func(l int) bool, calls *atomic.Int64) *program.Program {
	if calls != nil {
		calls.Add(1)
	}
	p := &program.Program{
		Embed: program.EmbedParams{
			Scale:      a.EffectiveEmbedScale(),
			LearnedPos: a.LearnedPosEmbed,
			PLE:        a.Gemma4 != nil && a.Gemma4.HiddenSizePerLayerInput > 0,
		},
		Head: program.HeadParams{
			Norm:       program.NormParams{Slot: program.SlotFinal, Form: a.NormForm(), Eps: a.NormEps, AddOne: a.RMSAddOne, Bias: true},
			Tied:       a.TiedLMHead,
			Softcap:    a.FinalLogitSoftcap,
			LogitScale: a.LogitScale,
		},
	}
	p.Layers = make([][]program.Op, a.NumLayers)
	for l := range a.NumLayers {
		p.Layers[l] = a.LayerOps(l, experts != nil && experts(l))
	}
	return p
}

// ActiveHooks returns the currently published Hooks snapshot.
func ActiveHooks() *Hooks { return activeHooks() }

var inexactRewindCheck func(*Cache) bool

// SetInexactRewindCheck registers decoder's cache-state grid check for partial Cache.TruncateTo rewinds.
func SetInexactRewindCheck(fn func(*Cache) bool) { inexactRewindCheck = fn }
