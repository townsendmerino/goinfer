//go:build goinfer_testhooks

package fitcmd

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/runtime/modelload"
)

// TestSourceFor_opensWhatTheLoadOpens: fit plans each backend on what a load on that backend opens. The quant is the load's own
// (modelload.Plan: with no -quant, the per-family default, which for a Phi-3 .gguf is not int4), and a Metal line reads through the
// metal sidecar a Metal load builds, when it is fresh, not the .gguf.
func TestSourceFor_opensWhatTheLoadOpens(t *testing.T) {
	src := filepath.Join(t.TempDir(), "phi3.gguf")
	if err := decoder.WritePhi3Q4KTinyGGUFForTest(src); err != nil {
		t.Fatal(err)
	}
	old := time.Now().Add(-time.Hour)
	if err := os.Chtimes(src, old, old); err != nil {
		t.Fatal(err)
	}
	for _, be := range []string{"cpu", "cuda", "metal", "webgpu"} {
		s, err := sourceFor(src, "int4", "", be)
		if err != nil {
			t.Fatal(err)
		}
		pl, err := modelload.Plan(modelload.Request{Spec: src, Opts: decoder.Options{Quant: "int4", Backend: be}})
		if err != nil {
			t.Fatal(err)
		}
		if s.quant != pl.Opts.Quant || s.group != pl.Opts.ActQuantGroup {
			t.Errorf("%s, no -quant: fit plans %s/%d, the load takes %s/%d", be, s.quant, s.group, pl.Opts.Quant, pl.Opts.ActQuantGroup)
		}
	}

	sidecar := decoder.SidecarPath(src, "q4k", "metal", false)
	if err := decoder.TranscodeGIW(context.Background(), src, sidecar, "q4k", false, decoder.GIWTargetForBackend("metal")); err != nil {
		t.Fatal(err)
	}
	s, err := sourceFor(src, "q4k", "q4k", "metal")
	if err != nil {
		t.Fatal(err)
	}
	if s.path != sidecar {
		t.Fatalf("metal at q4k: fit opens %s, want the fresh metal sidecar %s", s.path, sidecar)
	}
	m, err := decoder.Load(s.path, fitLoadOptions(s, 0))
	if err != nil {
		t.Fatalf("the metal sidecar does not open as fit opens it: %v", err)
	}
	m.Close()
}
