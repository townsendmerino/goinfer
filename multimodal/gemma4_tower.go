package multimodal

import (
	"fmt"
	"sort"
	"sync"

	"github.com/townsendmerino/aikit/vision"
)

// Gemma 4 vision tower accelerators (docs/multimodal.md, "Finishing this doc", F2). A backend module (goinfer/metal)
// registers a device tower built from aikit's export (vision.Gemma4Encoder.Weights) here, and every Gemma 4 tower
// consumer can use it: Gemma 4's own image input in serve and EmbeddingGemma 2's image embeddings. The tower runs the
// patch embed and the encoder layers; the pool and projection after them stay aikit's (Gemma4Encoder.FinishHidden).

// Gemma4TowerAccelerator runs the Gemma 4 vision tower up to its pool on another device.
type Gemma4TowerAccelerator interface {
	Name() string
	// Hidden is the last encoder layer's output [len(pos), hidden] for patches [len(pos), 3*16*16] in [0, 1] at their
	// (x, y) positions; patches is not modified.
	Hidden(patches []float32, pos [][2]int) ([]float32, error)
	Close() error
}

var (
	g4TowerMu sync.Mutex
	g4Towers  = map[string]func(*vision.Gemma4Encoder) (Gemma4TowerAccelerator, error){}
)

// RegisterGemma4Tower makes a Gemma 4 tower accelerator available by name, from a backend module's init.
func RegisterGemma4Tower(name string, factory func(*vision.Gemma4Encoder) (Gemma4TowerAccelerator, error)) {
	g4TowerMu.Lock()
	defer g4TowerMu.Unlock()
	g4Towers[name] = factory
}

// UnregisterGemma4Tower removes a registration (tests that register a fake clean up with it).
func UnregisterGemma4Tower(name string) {
	g4TowerMu.Lock()
	defer g4TowerMu.Unlock()
	delete(g4Towers, name)
}

// Gemma4Towers lists the registered tower accelerator names, sorted.
func Gemma4Towers() []string {
	g4TowerMu.Lock()
	defer g4TowerMu.Unlock()
	var n []string
	for k := range g4Towers {
		n = append(n, k)
	}
	sort.Strings(n)
	return n
}

// NewGemma4Tower builds the named tower accelerator over a loaded tower (float32, LoadGemma4Encoder quant=false).
func NewGemma4Tower(name string, enc *vision.Gemma4Encoder) (Gemma4TowerAccelerator, error) {
	g4TowerMu.Lock()
	f, ok := g4Towers[name]
	g4TowerMu.Unlock()
	if !ok {
		return nil, fmt.Errorf("multimodal: no %q Gemma 4 tower in this binary (registered: %v)", name, Gemma4Towers())
	}
	return f(enc)
}

// Gemma4TowerFeatures runs the tower on acc (nil: aikit's CPU Forward) and aikit's tail: the soft tokens.
func Gemma4TowerFeatures(enc *vision.Gemma4Encoder, acc Gemma4TowerAccelerator, patches []float32, pos [][2]int) ([]float32, error) {
	if acc == nil {
		return enc.Forward(patches, pos)
	}
	h, err := acc.Hidden(patches, pos)
	if err != nil {
		return nil, err
	}
	return enc.FinishHidden(h, pos)
}
