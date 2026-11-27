package wbuild

import (
	"context"
	"errors"
	"fmt"
	"io"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/giwcodec"
)

// LoRAMerger is the narrow subset of a LoRA adapter that safetensors weight loading queries and merges into base tensors.
type LoRAMerger interface {
	CheckAllMerged() error
	ValidateTargets(numLayers int, s *archparts.TensorSchema, name func(layer int, suffix string) string) error
	Has(name string) bool
	Merge(name string, data []float32, out, in int) error
}

// ErrDirNoStream is BuildWeightsFromSafetensorsTo's answer, given a sink, for a family whose dedicated builder does not
// stream yet; StreamTranscodeDir's caller then takes the resident transcode instead.
type ErrDirNoStream struct{ Family string }

func (e ErrDirNoStream) Error() string {
	return fmt.Sprintf("decoder: the %s safetensors loader does not stream layer by layer", e.Family)
}

// Hooks carries optional diagnostic and test seams injected by decoder so internal/runtime/wbuild holds zero mutable globals.
// Its zero value is the production default.
type Hooks struct {
	FakeQuantScheme        string
	FakeQuantF32Act        bool
	FakeQuantExpertsOnly   bool
	FakeQuantPerRow        bool
	DisableStreamTables    bool
	StreamTableBlockBytes  int
	StreamDirSwap          bool
	Q4KWiden56             bool
	FusedExpertsTransp     bool
	MambaKeepF32           bool
	MLAInt8                bool
	MLAProjF32             map[string]bool
	DisableRow4Repack      bool
	Row4RepackCalls        *atomic.Int64
	GGUFRowFilter          func(name string, row []float32)
	GIWF32Kind1            bool
	GIWIslandsF32          bool
	SplitHalfRepackEnabled bool
	SplitHalfRepacked      *atomic.Int64
	SplitHalfSkipped       *atomic.Int64
	SplitHalfBytes         *atomic.Int64
}

func (h *Hooks) streamTableBlockBytes() int {
	if h != nil && h.StreamTableBlockBytes > 0 {
		return h.StreamTableBlockBytes
	}
	return 32 << 20
}

func (h *Hooks) streamTablesOn() bool {
	return h == nil || !h.DisableStreamTables
}

// MambaProjInt8 reports whether Mamba-2 projections quantize to int8 under q.
func (h *Hooks) MambaProjInt8(q QuantMode) bool {
	switch q {
	case QuantInt8, QuantInt8I8, QuantInt4, QuantInt4Mix, QuantQ4K:
		return h == nil || !h.MambaKeepF32
	}
	return false
}

// MLAProjInt8 reports whether MLA projections quantize to int8 under q.
func (h *Hooks) MLAProjInt8(q QuantMode) bool {
	switch q {
	case QuantInt8, QuantInt8I8, QuantInt4, QuantInt4Mix, QuantQ4K:
		return h != nil && h.MLAInt8
	}
	return false
}

func (h *Hooks) mlaProjF32() map[string]bool {
	if h == nil {
		return nil
	}
	return h.MLAProjF32
}

func (h *Hooks) fakeQuantScheme() string {
	if h == nil {
		return ""
	}
	return h.FakeQuantScheme
}

func (h *Hooks) fakeQuantExpertsOnly() bool {
	return h != nil && h.FakeQuantExpertsOnly
}

func (h *Hooks) fakeQuantPerRow() bool {
	return h != nil && h.FakeQuantPerRow
}

func (h *Hooks) fakeQuantF32Act() bool {
	return h != nil && h.FakeQuantF32Act
}

func (h *Hooks) row4RepackCalls() *atomic.Int64 {
	if h == nil {
		return nil
	}
	return h.Row4RepackCalls
}

func (h *Hooks) q4kWiden56() bool {
	return h != nil && h.Q4KWiden56
}

func (h *Hooks) fusedExpertsTransposed() bool {
	return h != nil && h.FusedExpertsTransp
}

func (h *Hooks) ggufRowFilter() func(name string, row []float32) {
	if h == nil {
		return nil
	}
	return h.GGUFRowFilter
}

func (h *Hooks) streamDirSwap() bool {
	return h != nil && h.StreamDirSwap
}

type streamHooksKey struct{}

// StreamTestHooks holds optional test seams attached to a streaming transcode context and copied onto StreamWriter.
type StreamTestHooks struct {
	LayerWindow    int
	ExpertsInOrder bool
	ExpertFanouts  *atomic.Int64
	EmbedWorkers   int
}

// WithStreamTestHooks attaches streaming test hooks to ctx.
func WithStreamTestHooks(ctx context.Context, h StreamTestHooks) context.Context {
	return context.WithValue(ctx, streamHooksKey{}, h)
}

// StreamTestHooksOf reads any streaming test hooks attached to ctx.
func StreamTestHooksOf(ctx context.Context) StreamTestHooks {
	if ctx == nil {
		return StreamTestHooks{}
	}
	h, _ := ctx.Value(streamHooksKey{}).(StreamTestHooks)
	return h
}

// StreamWriter wraps giwcodec.Writer with layer-streaming and streamed-table state.
type StreamWriter struct {
	giwcodec.Writer
	Tables      map[*linalg.WeightMat]*StreamedTable
	LayerBudget int64
	TestHooks   StreamTestHooks
	Hooks       Hooks
}

// NewStreamWriter constructs a StreamWriter targeting sink.
func NewStreamWriter(sink io.Writer, target giwcodec.Target, layerBudget int64, th StreamTestHooks, h Hooks) *StreamWriter {
	w := &StreamWriter{
		Writer:      giwcodec.Writer{Sink: sink, Target: target, F32Kind1: h.GIWF32Kind1, IslandsF32: h.GIWIslandsF32},
		LayerBudget: layerBudget,
		TestHooks:   th,
		Hooks:       h,
	}
	w.InitHeadTable()
	return w
}

// InitHeadTable binds Writer.HeadTableFn to w so embedded copies route streamed tables through w.
func (w *StreamWriter) InitHeadTable() {
	w.Writer.HeadTableFn = func(_ *giwcodec.Writer, dst *linalg.WeightMat) bool {
		if t := w.Tables[dst]; t != nil {
			w.StreamTable(t)
			return true
		}
		return false
	}
}

// ErrLoadAborted is parallelLayers' sentinel when abort closes mid-build, the swap tripwire's load-time consumer.
// decoder.Load does not know why abort closed (a swap watch it neither owns nor
// imports); the caller that armed the watch checks errors.Is(err, ErrLoadAborted) and wraps it with its own reason and
// pricing detail (see Options.LoadAbort).
var ErrLoadAborted = errors.New("decoder: load aborted")

// ParallelLayers runs fn over the n layer indices across a worker pool, so the per-tensor dequant and re-quant
// (independent per layer: distinct WeightMat slots, read-only source) fans out across cores. The first error stops
// further work and is returned. Transient memory scales with the worker count (each in-flight layer briefly holds its
// dequantized f32); GOMAXPROCS workers is the right trade on a machine that can hold the model.
//
// abort, if non-nil, is checked between layers (at grab, never inside a layer in flight): once closed, no new fn(i)
// starts, in-flight ones finish, and the call returns ErrLoadAborted, unless an fn(i) returned a real build error, which
// always wins over a concurrent abort. A nil abort blocks forever in a select, so it is a zero-cost no-op for callers
// that pass none.
func ParallelLayers(n int, abort <-chan struct{}, fn func(i int) error) error {
	if n <= 1 {
		if n == 1 {
			select {
			case <-abort:
				return ErrLoadAborted
			default:
			}
			return fn(0)
		}
		return nil
	}
	workers := min(runtime.GOMAXPROCS(0), n)
	var (
		next     int
		mu       sync.Mutex
		firstErr error
		aborted  bool
		wg       sync.WaitGroup
	)
	grab := func() (int, bool) {
		mu.Lock()
		defer mu.Unlock()
		if next >= n || firstErr != nil {
			return 0, false
		}
		select {
		case <-abort:
			aborted = true
			return 0, false
		default:
		}
		i := next
		next++
		return i, true
	}
	for range workers {
		wg.Go(func() {
			for {
				i, ok := grab()
				if !ok {
					return
				}
				if err := fn(i); err != nil {
					mu.Lock()
					if firstErr == nil {
						firstErr = err
					}
					mu.Unlock()
					return
				}
			}
		})
	}
	wg.Wait()
	if firstErr != nil {
		return firstErr // a real build error always wins over a concurrent abort
	}
	if aborted {
		return ErrLoadAborted
	}
	return nil
}
