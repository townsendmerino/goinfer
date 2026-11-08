package modelload

import (
	"os"
	"strings"
	"testing"
)

// TestActivationSafeQuant: a Phi-3 source (queue-engineering.md H2) loaded at the default int4 gets
// int8int8 with per-32 activation scales, the configuration that passed the quality gate; an explicit
// int8int8 gets per-32 added; an explicit int4 is honoured with a warning; weight-only int8 / f32 and a
// family with no known hazard are left alone. The Phi-3 source is the committed tiny fixture
// (config.json model_type "phi3"), so this runs without a real checkpoint.
func TestActivationSafeQuant(t *testing.T) {
	const phi3 = "../../../testdata/phi3-tiny"
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

// TestActivationSafeQuant_ggufBackend: a Phi-3 .gguf (Phi-4's too: llama.cpp files both as phi3) defaults to q4k on the CPU, CUDA
// and Metal backends (docs/tasks/task-int4-weight-quality-2026-09.md's phases 1a and 1b; Metal's q4k lane, OP6's default switch) and
// to int8int8 + per-32 on WebGPU, which has no Q4_K kernel. An explicit quant is never replaced by q4k, on Metal as elsewhere. The
// file is testdata/phi3-q4k-tiny.gguf; what -backend auto hands this function is TestPhi3Default_autoRunsResidentOnTheQ4KLane's
// (backend/metal), through loadflags.Resolve.
func TestActivationSafeQuant_ggufBackend(t *testing.T) {
	const gguf = "../../../testdata/phi3-q4k-tiny.gguf" // read for its header only (PeekModelType): nothing is loaded beside it
	if _, err := os.Stat(gguf); err != nil {
		t.Fatalf("the committed Phi-3 fixture: %v", err)
	}
	for _, c := range []struct {
		backend, explicit, wantQ string
		wantG                    int
	}{
		{"cpu", "", "q4k", 0},
		{"", "", "q4k", 0},
		{"cuda", "", "q4k", 0},
		{"webgpu", "", "int8int8", 32},
		{"metal", "", "q4k", 0},
		{"cpu", "int8int8", "int8int8", 32},
		{"cpu", "int4", "int4", 0},
		{"metal", "int8int8", "int8int8", 32},
		{"metal", "int4", "int4", 0},
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
