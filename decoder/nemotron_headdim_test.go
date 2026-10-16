package decoder

import (
	"os"
	"testing"
	"testing/fstest"
)

// nemotronHeadDim builds the architecture from a config.json the way the loader does and returns its attention head dimension.
func nemotronHeadDim(t *testing.T, path string) int {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := loadConfig(fstest.MapFS{"config.json": {Data: raw}}, "config.json")
	if err != nil {
		t.Fatalf("%s: %v", path, err)
	}
	arch, _, err := nemotronhArchitecture(cfg)
	if err != nil {
		t.Fatalf("%s: %v", path, err)
	}
	return arch.HeadDim
}

// The nemotron_h adapter reads attention_head_dim when the config has it, and head_dim otherwise: the tiny fixtures and the 9B (head_dim only)
// give what they gave before, the 4B (attention_head_dim only; hidden/heads would be 96) gives its own 128.
func TestNemotronH_attentionHeadDim(t *testing.T) {
	for _, c := range []struct {
		path string
		want int
	}{
		{"../testdata/nemotron-tiny/config.json", 16},
		{"../testdata/nemotron3nano-tiny/config.json", 16},
		{"../testdata/nemotron-h-configs/nano-9b-v2.config.json", 128}, // hidden/heads is 112: head_dim is load-bearing here
		{"../testdata/nemotron-h-configs/h-4b-instruct-128k.config.json", 128},
	} {
		if got := nemotronHeadDim(t, c.path); got != c.want {
			t.Errorf("%s: head dimension %d, want %d", c.path, got, c.want)
		}
	}
}
