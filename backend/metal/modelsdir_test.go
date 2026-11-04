//go:build darwin

package metal

import (
	"os"
	"path/filepath"
)

// modelPath resolves a fixture model under GOINFER_MODELS_DIR (default $HOME/models), so the
// metal tests are not pinned to one developer's home directory and can run wherever the model
// zoo lives (audit G-06). Mirrors the tokenizer/chat/decoder helpers of the same name.
func modelPath(name string) string {
	root := os.Getenv("GOINFER_MODELS_DIR")
	if root == "" {
		if h, err := os.UserHomeDir(); err == nil {
			root = filepath.Join(h, "models")
		} else {
			root = filepath.Join("/Users/francistownsend-merino", "models")
		}
	}
	return filepath.Join(root, name)
}
