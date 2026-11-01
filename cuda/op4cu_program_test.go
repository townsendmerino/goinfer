//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestResident_holdsTheModelsProgram is OP4cu-a's A-1 (docs/tasks/task-forward-architecture-2026-10.md §4.9a): a CUDA resident
// holds the model's own program, the pointer decoder.Model.Program returns, with one operation list per layer. Both fixtures
// are tracked, so a missing one fails.
func TestResident_holdsTheModelsProgram(t *testing.T) {
	for _, dir := range []string{"../testdata/llama-tiny", "../testdata/qwen35-tiny"} {
		t.Run(filepath.Base(dir), func(t *testing.T) {
			if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
				t.Fatalf("no fixture (it is tracked): %v", err)
			}
			requireDeviceAndFixture(t, dir)
			m, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			r, ok := m.ResidentForwardForTest().(*cudaResident)
			if !ok {
				t.Fatalf("not a CUDA resident: %s", m.ResidentDecline())
			}
			want := m.Program()
			if r.prog != want {
				t.Fatalf("the resident holds %p, not the model's program %p", r.prog, want)
			}
			ops := 0
			for _, l := range r.prog.Layers {
				ops += len(l)
			}
			if len(r.prog.Layers) != len(m.Weights().Layers) || ops == 0 {
				t.Fatalf("the resident's program has %d layers and %d operations, for a %d-layer model", len(r.prog.Layers), ops, len(m.Weights().Layers))
			}
			t.Logf("%d layers, %d operations", len(r.prog.Layers), ops)
		})
	}
}
