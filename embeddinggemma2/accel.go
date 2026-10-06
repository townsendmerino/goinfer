package embeddinggemma2

import (
	"fmt"
	"sort"
	"sync"
)

// Accelerator runs the encoder's forward on another device (Phase M of docs/tasks/task-embeddinggemma2.md: Metal).
// It returns what Model.forward returns: the projected last hidden state, [len(ids), Dim()] row-major, and, with
// keepLayers, every layer's input and the last layer's output for per-layer differencing. Pooling, normalisation and
// tokenization stay in this package, shared with the CPU path.
type Accelerator interface {
	Name() string
	Forward(ids []int, keepLayers bool) (last []float32, layers [][]float32, err error)
	// ForwardEmbeds is Forward from T prepared input rows x [T, hidden] (Model.EmbedTokens with any image rows
	// spliced in); x is not modified.
	ForwardEmbeds(x []float32, T int, keepLayers bool) (last []float32, layers [][]float32, err error)
	Close() error
}

var (
	accelMu sync.Mutex
	accels  = map[string]func(*Model) (Accelerator, error){}
)

// RegisterAccelerator makes an accelerator available by name. A backend module registers itself from init (the root
// module cannot import it), as decoder.RegisterBackend does for the decoder's backends.
func RegisterAccelerator(name string, factory func(*Model) (Accelerator, error)) {
	accelMu.Lock()
	defer accelMu.Unlock()
	accels[name] = factory
}

// Accelerators lists the registered accelerator names, sorted.
func Accelerators() []string {
	accelMu.Lock()
	defer accelMu.Unlock()
	var n []string
	for k := range accels {
		n = append(n, k)
	}
	sort.Strings(n)
	return n
}

// NewAccelerator builds the named accelerator over this model's weights.
func (m *Model) NewAccelerator(name string) (Accelerator, error) {
	accelMu.Lock()
	f, ok := accels[name]
	accelMu.Unlock()
	if !ok {
		return nil, fmt.Errorf("embeddinggemma2: no %q accelerator in this binary (registered: %v)", name, Accelerators())
	}
	return f(m)
}

// ForwardCPU is the CPU forward an accelerator is compared with: the projected last hidden state and, with
// keepLayers, every layer's input and the last layer's output.
func (m *Model) ForwardCPU(ids []int, keepLayers bool) ([]float32, [][]float32, error) {
	return m.forward(ids, keepLayers)
}

// LayerWeights is one encoder layer's weights as an accelerator uploads them: row-major [out, in] as nn.Linear
// stores them, float32, read-only.
type LayerWeights struct {
	Full             bool // full attention; otherwise a symmetric window of radius Config.SlidingWindow
	HeadDim, KVHeads int
	RopeTheta        float64
	InNorm           []float32
	Q, K, V, O       []float32
	QNorm, KNorm     []float32
	PostAttnNorm     []float32
	PreFFNorm        []float32
	Gate, Up, Down   []float32
	PostFFNorm       []float32
	PLEGate, PLEProj []float32
	PLEPostNorm      []float32
	LayerScalar      float32
}

// Weights is the whole text encoder as an accelerator uploads it. The slices alias the model's own; do not write
// them.
type Weights struct {
	Config     Config
	Embed      []float32 // [vocab, hidden]
	PLEProj    []float32 // [layers*ple, hidden]
	PLENorm    []float32 // [ple]
	Layers     []LayerWeights
	FinalNorm  []float32
	Projection []float32 // [embedding_dim, hidden]
}

// Weights returns the model's weights for an accelerator.
func (m *Model) Weights() Weights {
	w := Weights{Config: m.cfg, Embed: m.embed, PLEProj: m.pleProj, PLENorm: m.pleNorm, FinalNorm: m.finalNorm,
		Projection: m.projection, Layers: make([]LayerWeights, len(m.layers))}
	for i := range m.layers {
		l := &m.layers[i]
		w.Layers[i] = LayerWeights{Full: l.full, HeadDim: l.headDim, KVHeads: l.kvH, RopeTheta: l.theta, InNorm: l.inNorm,
			Q: l.q, K: l.k, V: l.v, O: l.o, QNorm: l.qNorm, KNorm: l.kNorm, PostAttnNorm: l.postAttnNorm,
			PreFFNorm: l.preFFNorm, Gate: l.gate, Up: l.up, Down: l.down, PostFFNorm: l.postFFNorm, PLEGate: l.pleGate,
			PLEProj: l.pleProj, PLEPostNorm: l.plePostNorm, LayerScalar: l.scalar}
	}
	return w
}

// RopeTables returns the cos and sin tables the CPU forward uses for T positions at head dim hd and base theta,
// [T, hd/2] each, so an accelerator rotates with the same float32 values.
func RopeTables(T, hd int, theta float64) (cos, sin []float32) { return ropeTables(T, hd, theta) }

// UseAccelerator makes the encoder run its forward on the named accelerator; Embed and the aikit methods then use it,
// and the CPU forward is no longer called. The vision tower, loaded or not yet, runs on a tower accelerator of the same
// name when one is registered (RegisterVisionAccelerator). It returns the accelerator, or an error leaving the CPU path
// in place.
func (e *Encoder) UseAccelerator(name string) (Accelerator, error) {
	a, err := e.m.NewAccelerator(name)
	if err != nil {
		return nil, err
	}
	e.accel = a
	e.bindVisionAccel() // a tower already loaded moves too
	return a, nil
}

// Accelerator is the accelerator the encoder runs on, or nil on the CPU path.
func (e *Encoder) Accelerator() Accelerator { return e.accel }

// embedIDs is the sentence embedding of ids on whichever path the encoder runs.
func (e *Encoder) embedIDs(ids []int) ([]float32, error) {
	if e.accel == nil {
		return e.m.Embed(ids)
	}
	h, _, err := e.accel.Forward(ids, false)
	if err != nil {
		return nil, err
	}
	return poolNormalize(h, len(ids), e.m.cfg.EmbeddingDim), nil
}
