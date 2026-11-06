package multimodal

import (
	"os"
	"path/filepath"
	"testing"
)

// A preprocess config that is absent means the defaults. One that is there and cannot be read is an error: defaults in its place would
// size images differently with no word to anyone. A directory in the file's place is the read failure every platform gives.
func TestPreprocessConfigs_unreadableIsNotAbsent(t *testing.T) {
	qwen := func(dir string) error { _, err := LoadQwenPreprocessConfig(dir); return err }
	lfm2 := func(dir string) error { _, err := LoadLfm2VLPreprocessConfig(dir); return err }
	pixtral := func(dir string) error { _, err := LoadPixtralPreprocessConfig(dir); return err }
	for _, c := range []struct {
		name, file string
		load       func(dir string) error
	}{
		{"qwen", "preprocessor_config.json", qwen},
		{"lfm2-vl", "processor_config.json", lfm2},
		{"pixtral config", "config.json", pixtral},
		{"pixtral processor", "processor_config.json", pixtral},
		{"pixtral preprocessor", "preprocessor_config.json", pixtral},
	} {
		t.Run(c.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := c.load(dir); err != nil {
				t.Errorf("no %s: %v, want the defaults", c.file, err)
			}
			if err := os.Mkdir(filepath.Join(dir, c.file), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := c.load(dir); err == nil {
				t.Errorf("an unreadable %s loaded as the defaults", c.file)
			}
		})
	}
}
