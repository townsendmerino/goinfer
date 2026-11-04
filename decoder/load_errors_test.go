package decoder

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A caller can tell Load's refusals apart without reading their text, and the text they had is unchanged: harnesses and people read it.
func TestLoad_errorKinds(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"no-such-family"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	gguf := filepath.Join(t.TempDir(), "x.gguf")
	if err := os.WriteFile(gguf, buildGGUF([]ggufKV{kvStr("general.architecture", "no-such-arch")}, nil), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name string
		path string
		opts Options
		kind error
		text string
	}{
		{"model_type in config.json", dir, Options{}, ErrUnsupportedModel, `decoder: unsupported model_type "no-such-family" (have: `},
		{"architecture in a GGUF", gguf, Options{}, ErrUnsupportedModel, `decoder(gguf): architecture "no-such-arch" unsupported (have: `},
		{"quant", dir, Options{Quant: "no-such-quant"}, ErrUnknownQuant, `decoder: unknown quant "no-such-quant" (have: `},
		{"backend", dir, Options{Backend: "no-such-backend"}, ErrUnknownBackend, `decoder: unknown backend "no-such-backend" (have: `},
	} {
		t.Run(c.name, func(t *testing.T) {
			m, err := Load(c.path, c.opts)
			if err == nil {
				m.Close()
				t.Fatal("loaded")
			}
			if !errors.Is(err, c.kind) {
				t.Errorf("error %q is not %v", err, c.kind)
			}
			if !strings.Contains(err.Error(), c.text) {
				t.Errorf("error %q, want its text to hold %q", err, c.text)
			}
		})
	}
	if err := (Options{Backend: "no-such-backend"}).Validate(); !errors.Is(err, ErrUnknownBackend) {
		t.Errorf("Validate: %v, want ErrUnknownBackend", err)
	}
}
