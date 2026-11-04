package serveapp

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestLoadVisionTower_discoversInResolvedSource is task-checkpoint-fetch P7: vision auto-discovery looks in the sole
// --model AS RESOLVED. A `hf:<repo>:safetensors` reference is fetched into a checkpoint directory by loadDecoder, and
// that directory is where a VL repo's tower lives; stat-ing the typed string instead finds nothing and serves the model
// text-only, with no error to say so. The resolved directory here declares a Qwen2.5-VL config with no tower weights,
// so a discovery that looks there fails loudly naming it; one that looks at the typed reference returns nil.
func TestLoadVisionTower_discoversInResolvedSource(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"qwen2_5_vl"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	s := &server{models: map[string]*loadedModel{"vl": {name: "vl", source: dir}}}
	cfg := config{models: modelFlag{{path: "hf:owner/vl:safetensors"}}}
	err := s.loadVisionTower(cfg)
	if err == nil || !strings.Contains(err.Error(), dir) {
		t.Fatalf("loadVisionTower = %v; want discovery to try the resolved checkpoint dir %s", err, dir)
	}
}
