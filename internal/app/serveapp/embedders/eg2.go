package embedders

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/townsendmerino/goinfer/embeddinggemma2"
)

// IsEmbeddingGemma2 reports whether dir's config.json declares model_type == "embedding_gemma2".
func IsEmbeddingGemma2(dir string) bool {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return false
	}
	var c struct {
		ModelType string `json:"model_type"`
	}
	return json.Unmarshal(raw, &c) == nil && c.ModelType == "embedding_gemma2"
}

// EG2Accelerable is the subset of embeddinggemma2.Encoder used by ChooseEG2Device.
type EG2Accelerable interface {
	UseAccelerator(name string) (embeddinggemma2.Accelerator, error)
}

// ChooseEG2Device attaches the Metal accelerator when backend == "metal", falling
// back to CPU with an explanation unless require is true.
func ChooseEG2Device(e EG2Accelerable, backend string, require bool) (string, error) {
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
