package serveapp

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/townsendmerino/goinfer/embeddinggemma2"
)

// isEmbeddingGemma2 reports whether dir is an EmbeddingGemma 2 checkpoint (config.json's model_type).
func isEmbeddingGemma2(dir string) bool {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return false
	}
	var c struct {
		ModelType string `json:"model_type"`
	}
	return json.Unmarshal(raw, &c) == nil && c.ModelType == "embedding_gemma2"
}

// loadEmbeddingGemma2 loads an EmbeddingGemma 2 checkpoint as the /v1/embeddings model
// (docs/tasks/task-embeddinggemma2.md). It runs in float32, on the CPU or, where chooseEG2Device picks it, on Metal,
// so -embed-quant other than f32 is refused rather than dropped (a quant flag that silently does nothing).
// `dimensions` takes exactly the model's Matryoshka widths.
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

// eg2Accelerable is the part of the EmbeddingGemma 2 encoder chooseEG2Device uses (a seam for its test).
type eg2Accelerable interface {
	UseAccelerator(name string) (embeddinggemma2.Accelerator, error)
}

// chooseEG2Device puts the encoder on the GPU when serve's resolved backend is Metal and says where it runs. Metal
// failing to start falls back to the CPU with the reason, unless -require-backend asks for a refusal instead. Other
// backends run it on the CPU: its only GPU path is Metal.
func chooseEG2Device(e eg2Accelerable, backend string, require bool) (string, error) {
	switch backend {
	case "metal":
		if _, err := e.UseAccelerator("metal"); err != nil {
			if require {
				return "", fmt.Errorf("-require-backend: EmbeddingGemma 2 could not start on Metal: %w", err)
			}
			return "CPU (Metal declined: " + err.Error() + ")", nil
		}
		return "Metal", nil
	case "", "cpu":
		return "CPU", nil
	default:
		if require {
			return "", fmt.Errorf("-require-backend: EmbeddingGemma 2 has no %s path; it runs on Metal or the CPU", backend)
		}
		return "CPU (its GPU path is Metal only; -backend is " + backend + ")", nil
	}
}
