package multimodal

import (
	"fmt"
	"sort"
	"sync"

	"github.com/townsendmerino/aikit/vision"
)

// Device towers for the grid-based vision encoders, Qwen3.5+ and GLM-OCR (S2 of docs/tasks/task-multimodal-support-2026-10.md),
// on the pattern of the Gemma 4 registry (gemma4_tower.go): a backend module registers a tower built from aikit's export
// (the encoder's Weights, RopeTables, VisionSegments) here, the tower runs the patch embed and the blocks on its
// device, and the tail (Qwen3's merger; GLM-OCR's post-norm, downsample and merger) stays aikit's FinishHidden.

// GridTowerAccelerator runs a grid-based tower up to its last block on another device.
type GridTowerAccelerator interface {
	Name() string
	// Hidden is the last block's output [n_patches, hidden] for pre-patchified pixel values [n_patches, C·T·P·P] in the
	// tower's patch order, with per-image grids (t, h, w in patches); pixels is not modified.
	Hidden(pixels []float32, gridTHW [][3]int) ([]float32, error)
	Close() error
}

// gridTowerRegistry is one encoder type's named device-tower factories.
type gridTowerRegistry[E any] struct {
	mu sync.Mutex
	m  map[string]func(*E) (GridTowerAccelerator, error)
}

func (r *gridTowerRegistry[E]) register(name string, f func(*E) (GridTowerAccelerator, error)) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.m == nil {
		r.m = map[string]func(*E) (GridTowerAccelerator, error){}
	}
	r.m[name] = f
}

func (r *gridTowerRegistry[E]) unregister(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.m, name)
}

func (r *gridTowerRegistry[E]) names() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	var n []string
	for k := range r.m {
		n = append(n, k)
	}
	sort.Strings(n)
	return n
}

func (r *gridTowerRegistry[E]) build(family, name string, enc *E) (GridTowerAccelerator, error) {
	r.mu.Lock()
	f, ok := r.m[name]
	r.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("multimodal: no %q %s tower in this binary (registered: %v)", name, family, r.names())
	}
	return f(enc)
}

var (
	qwen3Towers  gridTowerRegistry[vision.Qwen3VisionEncoder]
	glmOcrTowers gridTowerRegistry[vision.GlmOcrVisionEncoder]
)

// RegisterQwen3Tower makes a Qwen3.5+ tower accelerator available by name, from a backend module's init.
func RegisterQwen3Tower(name string, f func(*vision.Qwen3VisionEncoder) (GridTowerAccelerator, error)) {
	qwen3Towers.register(name, f)
}

// UnregisterQwen3Tower removes a registration (tests that register a fake clean up with it).
func UnregisterQwen3Tower(name string) { qwen3Towers.unregister(name) }

// Qwen3Towers lists the registered Qwen3.5+ tower accelerators, sorted.
func Qwen3Towers() []string { return qwen3Towers.names() }

// NewQwen3Tower builds the named accelerator over a loaded tower (float32: LoadQwen3VisionEncoder quant=false).
func NewQwen3Tower(name string, enc *vision.Qwen3VisionEncoder) (GridTowerAccelerator, error) {
	return qwen3Towers.build("Qwen3.5", name, enc)
}

// Qwen3TowerFeatures runs the tower on acc (nil: aikit's CPU Forward) and aikit's tail: the merged image embeddings.
func Qwen3TowerFeatures(enc *vision.Qwen3VisionEncoder, acc GridTowerAccelerator, pixels []float32, gridTHW [][3]int) ([]float32, error) {
	if acc == nil {
		return enc.Forward(pixels, gridTHW)
	}
	h, err := acc.Hidden(pixels, gridTHW)
	if err != nil {
		return nil, err
	}
	return enc.FinishHidden(h, gridTHW)
}

// RegisterGlmOcrTower makes a GLM-OCR tower accelerator available by name, from a backend module's init.
func RegisterGlmOcrTower(name string, f func(*vision.GlmOcrVisionEncoder) (GridTowerAccelerator, error)) {
	glmOcrTowers.register(name, f)
}

// UnregisterGlmOcrTower removes a registration (tests that register a fake clean up with it).
func UnregisterGlmOcrTower(name string) { glmOcrTowers.unregister(name) }

// GlmOcrTowers lists the registered GLM-OCR tower accelerators, sorted.
func GlmOcrTowers() []string { return glmOcrTowers.names() }

// NewGlmOcrTower builds the named accelerator over a loaded tower (float32: LoadGlmOcrVisionEncoder quant=false).
func NewGlmOcrTower(name string, enc *vision.GlmOcrVisionEncoder) (GridTowerAccelerator, error) {
	return glmOcrTowers.build("GLM-OCR", name, enc)
}

// GlmOcrTowerFeatures runs the tower on acc (nil: aikit's CPU Forward) and aikit's tail: the merged image embeddings.
func GlmOcrTowerFeatures(enc *vision.GlmOcrVisionEncoder, acc GridTowerAccelerator, pixels []float32, gridTHW [][3]int) ([]float32, error) {
	if acc == nil {
		return enc.Forward(pixels, gridTHW)
	}
	h, err := acc.Hidden(pixels, gridTHW)
	if err != nil {
		return nil, err
	}
	return enc.FinishHidden(h, gridTHW)
}
