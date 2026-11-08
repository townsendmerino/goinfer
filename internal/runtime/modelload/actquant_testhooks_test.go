//go:build goinfer_testhooks

package modelload

import (
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestActivationSafeQuant_ggufBackend: a Phi-3 .gguf (Phi-4's too: llama.cpp files both as phi3) defaults to q4k on the CPU, CUDA
// and Metal backends (docs/tasks/task-int4-weight-quality-2026-09.md's phases 1a and 1b; Metal's q4k lane, OP6's default switch) and
// to int8int8 + per-32 on WebGPU, which has no Q4_K kernel. An explicit quant is never replaced by q4k, on Metal as elsewhere. The
// file is decoder.WritePhi3Q4KTinyGGUFForTest's. What -backend auto hands this function is
// TestPhi3Default_autoRunsResidentOnTheQ4KLane's (backend/metal), through loadflags.Resolve.
func TestActivationSafeQuant_ggufBackend(t *testing.T) {
	gguf := filepath.Join(t.TempDir(), "phi3.gguf")
	if err := decoder.WritePhi3Q4KTinyGGUFForTest(gguf); err != nil {
		t.Fatal(err)
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
