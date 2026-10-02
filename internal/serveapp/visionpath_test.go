package serveapp

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// R22: `-vision <mmproj>.gguf` used to fail inside the encoder loader with ".../config.json: not a directory". Through the caller
// (loadVisionTower, with the path set and nothing else), the refusal must say what -vision takes and that GGUF mmproj is not supported.
func TestLoadVisionTower_ggufMmprojIsRefusedPlainly(t *testing.T) {
	dir := t.TempDir()
	mm := filepath.Join(dir, "mmproj-google_gemma-3-4b-it-f16.gguf")
	if err := os.WriteFile(mm, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	err := (&server{}).loadVisionTower(config{visionPath: mm})
	if err == nil {
		t.Fatal("a GGUF mmproj passed to -vision was accepted")
	}
	for _, want := range []string{"mmproj", "not supported yet", "directory with a vision tower", "config.json and safetensors"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal lacks %q: %v", want, err)
		}
	}
	if strings.Contains(err.Error(), "not a directory") {
		t.Errorf("the file-system error leaked through: %v", err)
	}

	// Some other plain file: still a file, still told what -vision takes; and a directory is left to the loaders.
	other := filepath.Join(dir, "weights.bin")
	if err := os.WriteFile(other, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if e := visionPathError(other); e == nil || strings.Contains(e.Error(), "mmproj") || !strings.Contains(e.Error(), "directory with a vision tower") {
		t.Errorf("plain file: %v", e)
	}
	if e := visionPathError(dir); e != nil {
		t.Errorf("a directory was refused by the path check: %v", e)
	}
	if e := visionPathError(filepath.Join(dir, "missing")); e != nil {
		t.Errorf("a missing path is the loaders' to judge: %v", e)
	}
}
