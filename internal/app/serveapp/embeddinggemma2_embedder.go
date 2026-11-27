package serveapp

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/townsendmerino/goinfer/embeddinggemma2"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/embedders"
)

var (
	isEmbeddingGemma2 = embedders.IsEmbeddingGemma2
	chooseEG2Device   = embedders.ChooseEG2Device
)

// loadEmbeddingGemma2 loads an EmbeddingGemma 2 checkpoint as the /v1/embeddings model.
func (s *server) loadEmbeddingGemma2(cfg config) error {
	switch strings.ToLower(cfg.embedQuant) {
	case "", "f32":
	default:
		return fmt.Errorf("invalid -embed-quant %q for EmbeddingGemma 2: it runs in f32 only (omit -embed-quant, or pass f32)", cfg.embedQuant)
	}
	var resize embeddinggemma2.ImageResize
	if cfg.embedResize != "" {
		r, err := embeddinggemma2.ParseImageResize(cfg.embedResize)
		if err != nil {
			return fmt.Errorf("-embed-image-resize: %w", err)
		}
		resize = r
	}
	t0 := time.Now()
	e, err := embeddinggemma2.LoadEncoder(cfg.embedPath)
	if err != nil {
		return fmt.Errorf("load embedding model: %w", err)
	}
	if resize != "" {
		if err := e.SetImageResize(resize); err != nil {
			return fmt.Errorf("-embed-image-resize: %w", err)
		}
	}
	where, err := chooseEG2Device(e, cfg.load.Backend, cfg.requireBE)
	if err != nil {
		return err
	}
	name := cfg.embedName
	if name == "" {
		name = filepath.Base(strings.TrimRight(cfg.embedPath, "/"))
	}
	s.embed, s.embedTok, s.embedID, s.embedDim = e, nil, name, e.HiddenDim()
	s.embedMRLMin, s.embedWidths = 0, nil
	for _, w := range embeddinggemma2.MatryoshkaWidths {
		if w <= s.embedDim {
			s.embedWidths = append(s.embedWidths, w)
		}
	}
	tower, atower := "CPU", "CPU"
	if a := e.Accelerator(); a != nil && slices.Contains(embeddinggemma2.VisionAccelerators(), a.Name()) {
		tower = a.Name()
	}
	if a := e.Accelerator(); a != nil && slices.Contains(embeddinggemma2.AudioAccelerators(), a.Name()) {
		atower = a.Name()
	}
	e.OnTowerLoad(func(kind, device string, took time.Duration) {
		fmt.Fprintf(os.Stderr, "EmbeddingGemma 2 %s tower loaded on %s in %s\n", kind, device, took.Round(time.Millisecond))
	})
	fmt.Fprintf(os.Stderr, "loaded embedding model %q (EmbeddingGemma 2, dim %d, f32, %s, dimensions %v, %d task prompts; no prompt unless the request names one with task or input_type; images resized %s, image tower on %s and audio tower on %s, each loaded on first use) in %s\n",
		name, s.embedDim, where, s.embedWidths, len(e.PromptNames()), e.ImageResizeMode(), tower, atower, time.Since(t0).Round(time.Millisecond))
	return nil
}
