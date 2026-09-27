package modelload

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestActivationSafeQuant: a Phi-3 source (queue-engineering.md H2) loaded at the default int4 gets
// int8int8 with per-32 activation scales, the configuration that passed the quality gate; an explicit
// int8int8 gets per-32 added; an explicit int4 is honoured with a warning; weight-only int8 / f32 and a
// family with no known hazard are left alone. The Phi-3 source is the committed tiny fixture
// (config.json model_type "phi3"), so this runs without a real checkpoint.
func TestActivationSafeQuant(t *testing.T) {
	const phi3 = "../../testdata/phi3-tiny"
	for _, c := range []struct {
		src, quant, explicit string
		wantQ                string
		wantG                int
		msg                  string
	}{
		{phi3, "int4", "", "int8int8", 32, "note: loading at --quant int8int8 with per-32"},
		{phi3, "int8int8", "int8int8", "int8int8", 32, "note: using per-32"},
		{phi3, "int4", "int4", "int4", 0, "warning: --quant int4"},
		{phi3, "int8", "", "int8", 0, ""},
		{phi3, "f32", "", "f32", 0, ""},
		{"../../testdata/nonexistent-model-dir", "int4", "", "int4", 0, ""},
	} {
		q, g, msg := activationSafeQuant(c.src, c.quant, 0, c.explicit, "cpu")
		if q != c.wantQ || g != c.wantG || !strings.HasPrefix(msg, c.msg) || (c.msg == "") != (msg == "") {
			t.Errorf("activationSafeQuant(%s, %q, explicit %q) = %q, %d, %q; want %q, %d, prefix %q", c.src, c.quant, c.explicit, q, g, msg, c.wantQ, c.wantG, c.msg)
		}
	}
}

// TestActivationSafeQuant_ggufBackend: a Phi-3 .gguf defaults to q4k on the CPU and CUDA backends
// (Phases 1a and 1b of docs/tasks/task-int4-weight-quality-2026-09.md) and to int8int8 + per-32 on
// Metal and WebGPU, which have no Q4_K kernel yet and run int8int8 resident. An explicit quant is never replaced by q4k. Needs
// the real Phi-3 GGUF for PeekModelType; skips without it (a skip is not a pass).
func TestActivationSafeQuant_ggufBackend(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skipf("no home directory: %v", err)
	}
	gguf := filepath.Join(home, "models", "phi3-mini-4k-gguf", "Phi-3-mini-4k-instruct-q4.gguf")
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no Phi-3 GGUF at %s", gguf)
	}
	for _, c := range []struct {
		backend, explicit, wantQ string
		wantG                    int
	}{
		{"cpu", "", "q4k", 0},
		{"", "", "q4k", 0},
		{"cuda", "", "q4k", 0},
		{"webgpu", "", "int8int8", 32},
		{"metal", "", "int8int8", 32},
		{"cpu", "int8int8", "int8int8", 32},
		{"cpu", "int4", "int4", 0},
	} {
		quant := "int4"
		if c.explicit != "" {
			quant = c.explicit
		}
		q, g, _ := activationSafeQuant(gguf, quant, 0, c.explicit, c.backend)
		if q != c.wantQ || g != c.wantG {
			t.Errorf("backend %q explicit %q: got %q, %d; want %q, %d", c.backend, c.explicit, q, g, c.wantQ, c.wantG)
		}
	}
}
