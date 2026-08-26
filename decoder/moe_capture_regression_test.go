package decoder

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"testing"
)

// TestForwardN_capturesMoELayers is the C-07 regression gate: forwardN's batched MoE branch must write
// the hidden-state capture (cache.captured[ci]) for MoE layers. It used to `continue` before the
// capture block, so against a sparse-MoE EAGLE target (Mixtral/Mellum) captured stayed all-nil and
// fuseAt sliced a nil slice → "slice bounds out of range" panic inside the generation goroutine. Every
// mixtral-tiny layer is MoE, so this exercises exactly the fixed path.
func TestForwardN_capturesMoELayers(t *testing.T) {
	const dir = "../testdata/mixtral-tiny"
	if _, err := os.Stat(dir + "/model.safetensors"); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no mixtral-tiny fixture at %s", dir)
	}
	m, err := Load(dir, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if m.w.arch.MoE == nil {
		t.Fatalf("expected a MoE arch, got %q", m.w.arch.Name)
	}
	ids := []int{1, 2, 3}
	layers := make([]int, m.w.arch.NumLayers)
	for i := range layers {
		layers[i] = i
	}
	cache := m.NewCache(len(ids) + 1)
	cache.captureLayers = layers
	cache.captured = make([][]float32, len(layers))
	if _, err := m.forwardN(context.Background(), ids, cache); err != nil {
		t.Fatalf("forwardN: %v", err)
	}
	hidden := m.w.arch.HiddenDim
	for ci, l := range layers {
		if cache.captured[ci] == nil {
			t.Fatalf("layer %d (MoE): captured hidden is nil — C-07 reintroduced (EAGLE fuseAt would panic)", l)
		}
		if got := len(cache.captured[ci]); got != len(ids)*hidden {
			t.Errorf("layer %d: captured len %d, want %d (rows×hidden)", l, got, len(ids)*hidden)
		}
	}
}
