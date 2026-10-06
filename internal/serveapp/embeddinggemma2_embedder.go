package serveapp

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
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

// loadEmbeddingGemma2 loads an EmbeddingGemma 2 checkpoint as the /v1/embeddings model (docs/tasks/task-embeddinggemma2.md,
// Gate 3). It runs in float32 on the CPU only, so -embed-quant other than f32 is refused rather than dropped (the M-17
// class: a quant flag that silently does nothing). `dimensions` takes exactly the model's Matryoshka widths.
func (s *server) loadEmbeddingGemma2(cfg config) error {
	switch strings.ToLower(cfg.embedQuant) {
	case "", "f32":
	default:
		return fmt.Errorf("invalid -embed-quant %q for EmbeddingGemma 2: it runs in f32 only (omit -embed-quant, or pass f32)", cfg.embedQuant)
	}
	t0 := time.Now()
	e, err := embeddinggemma2.LoadEncoder(cfg.embedPath)
	if err != nil {
		return fmt.Errorf("load embedding model: %w", err)
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
	fmt.Fprintf(os.Stderr, "loaded embedding model %q (EmbeddingGemma 2, dim %d, f32, CPU, dimensions %v, %d task prompts; no prompt unless the request names one with task or input_type) in %s\n",
		name, s.embedDim, s.embedWidths, len(e.PromptNames()), time.Since(t0).Round(time.Millisecond))
	return nil
}
