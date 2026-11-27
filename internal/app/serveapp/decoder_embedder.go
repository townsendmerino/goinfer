package serveapp

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/embedders"
	"github.com/townsendmerino/goinfer/internal/runtime/modelload"
)

const (
	qwen3EmbedQueryPrompt = embedders.Qwen3EmbedQueryPrompt
	qwen3EmbedDocPrompt   = embedders.Qwen3EmbedDocPrompt
	qwen3EmbedEOD         = embedders.Qwen3EmbedEOD
)

type decoderEmbedder = embedders.DecoderEmbedder

var (
	newDecoderEmbedder = embedders.NewDecoderEmbedder
	truncateForContext = embedders.TruncateForContext
)

// loadDecoderEmbedder wires a causal decoder (.gguf) in as the /v1/embeddings model.
func (s *server) loadDecoderEmbedder(cfg config) error {
	t0 := time.Now()
	tk, err := modelload.Tokenizer(cfg.embedPath)
	if err != nil {
		return fmt.Errorf("load embedding tokenizer (%s): %w", cfg.embedPath, err)
	}
	quant := ""
	if strings.EqualFold(cfg.embedQuant, "q8") {
		quant = "int8"
	}
	m, err := decoder.Load(cfg.embedPath, decoder.Options{
		Backend:      cfg.load.Backend,
		BackendAuto:  cfg.load.Auto != nil,
		Quant:        quant,
		ExactPrefill: cfg.load.ExactPrefill,
		Knobs:        cfg.load.Knobs(),
		DirectLoad:   true,
	})
	if err != nil {
		return fmt.Errorf("load embedding model (%s): %w", cfg.embedPath, err)
	}
	name := cfg.embedName
	if name == "" {
		name = strings.TrimSuffix(filepath.Base(cfg.embedPath), ".gguf")
	}
	e := newDecoderEmbedder(m, tk, qwen3EmbedQueryPrompt, qwen3EmbedDocPrompt)
	if e.AppendID() < 0 {
		fmt.Fprintf(os.Stderr, "warning: embedding model %q has no %s token — last-token pooling will pool the final content token, which will NOT match the reference\n",
			cfg.embedPath, qwen3EmbedEOD)
	}
	s.embed, s.embedTok, s.embedID, s.embedDim = e, nil, name, e.HiddenDim()
	s.embedMRLMin = 0
	fmt.Fprintf(os.Stderr, "loaded decoder-backed embedding model %q (dim %d, last-token pooling) in %s\n",
		name, s.embedDim, time.Since(t0).Round(time.Millisecond))
	return nil
}
