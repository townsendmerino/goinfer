package tokenizer

import (
	"os"
	"path/filepath"
)

// modelPath resolves a fixture model under GOINFER_MODELS_DIR (default $HOME/models), so tests are not
// pinned to one developer's home directory and can run wherever the model zoo lives (audit G-06).
func modelPath(name string) string {
	root := os.Getenv("GOINFER_MODELS_DIR")
	if root == "" {
		if h, err := os.UserHomeDir(); err == nil {
			root = filepath.Join(h, "models")
		} else {
			root = "/home/francis/models"
		}
	}
	return filepath.Join(root, name)
}
