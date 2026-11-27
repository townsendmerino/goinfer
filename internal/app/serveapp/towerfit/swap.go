package towerfit

import (
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// TowerSwap runs a Gemma 4 device tower that occupies device memory only during
// encoding, shrinking and regrowing the model's MoE expert cache around each call.
type TowerSwap struct {
	mu      sync.Mutex
	enc     *vision.Gemma4Encoder
	backend string
	model   *decoder.Model
	reserve int64
}

// NewTowerSwap constructs a TowerSwap for enc on backend beside model.
func NewTowerSwap(enc *vision.Gemma4Encoder, backend string, model *decoder.Model, reserve int64) *TowerSwap {
	return &TowerSwap{
		enc:     enc,
		backend: backend,
		model:   model,
		reserve: reserve,
	}
}

// Features encodes patches at pos into soft tokens, shrinking the expert cache
// to leave reserve bytes before uploading the tower and restoring the cache afterward.
func (t *TowerSwap) Features(patches []float32, pos [][2]int) ([]float32, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t0 := time.Now()
	small, err := t.model.ResizeExpertCache(t.reserve)
	if err != nil {
		fmt.Fprintf(os.Stderr, "vision: the expert cache could not make room for the Gemma 4 tower (%v); this image encodes on the CPU\n", err)
		return multimodal.Gemma4TowerFeatures(t.enc, nil, patches, pos)
	}
	defer func() {
		big, err := t.model.ResizeExpertCache(0)
		if err != nil {
			fmt.Fprintf(os.Stderr, "vision: the expert cache stays at %d slots/layer after the encode: %v\n", small, err)
			return
		}
		fmt.Fprintf(os.Stderr, "vision: the Gemma 4 tower ran beside a %d-slot expert cache and was released (%s with its upload); the cache is back to %d slots/layer\n",
			small, time.Since(t0).Round(time.Millisecond), big)
	}()
	tw, err := multimodal.NewGemma4Tower(t.backend, t.enc)
	if err != nil {
		fmt.Fprintf(os.Stderr, "vision: the Gemma 4 tower could not start on %s (%v); this image encodes on the CPU\n", t.backend, err)
		return multimodal.Gemma4TowerFeatures(t.enc, nil, patches, pos)
	}
	defer tw.Close()
	return multimodal.Gemma4TowerFeatures(t.enc, tw, patches, pos)
}
