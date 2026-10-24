package serveapp

import (
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/multimodal"
)

// towerSwap runs a device Gemma 4 tower that holds device memory only while it encodes, for a model whose MoE expert cache
// (decoder.ResidentExpertCache) uses that memory between images (docs/tasks/task-multimodal-support-2026-10.md, S20).
// Before an encode the cache shrinks to leave reserve bytes free and the tower uploads from the encoder's host weights, so
// those weights stay loaded; after it the tower is closed and the cache grows back. An image whose tower cannot come up
// encodes on the CPU. It is called inside a generation, and a vision model runs one generation at a time (setConcurrency);
// mu also keeps encodes one at a time.
type towerSwap struct {
	mu      sync.Mutex
	enc     *vision.Gemma4Encoder
	backend string
	model   *decoder.Model
	reserve int64
}

// features encodes one image's patches at their positions into soft tokens.
func (t *towerSwap) features(patches []float32, pos [][2]int) ([]float32, error) {
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
