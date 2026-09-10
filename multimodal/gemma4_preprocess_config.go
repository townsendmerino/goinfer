package multimodal

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Gemma4PreprocessConfig holds the one per-checkpoint knob
// vision.Gemma4Preprocess needs: the soft-token budget (config.json's top-level
// vision_soft_tokens_per_image, one of aikit's five legal values
// {70,140,280,560,1120}). Unlike Qwen, Gemma 4 checkpoints carry this in the
// SAME config.json the text decoder and vision tower already read — there is
// no separate preprocessor_config.json (confirmed absent on real checkpoint
// directories).
type Gemma4PreprocessConfig struct {
	MaxSoftTokens int
}

// LoadGemma4PreprocessConfig reads dir/config.json's vision_soft_tokens_per_image,
// defaulting to 280 (HF's own default, and vision.Gemma4Preprocess's own
// zero-value default) when absent or zero.
func LoadGemma4PreprocessConfig(dir string) (Gemma4PreprocessConfig, error) {
	cfg := Gemma4PreprocessConfig{MaxSoftTokens: 280}
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return cfg, fmt.Errorf("multimodal(gemma4): read config.json: %w", err)
	}
	var c struct {
		VisionSoftTokensPerImage int `json:"vision_soft_tokens_per_image"`
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return cfg, fmt.Errorf("multimodal(gemma4): parse config.json: %w", err)
	}
	if c.VisionSoftTokensPerImage > 0 {
		cfg.MaxSoftTokens = c.VisionSoftTokensPerImage
	}
	return cfg, nil
}
