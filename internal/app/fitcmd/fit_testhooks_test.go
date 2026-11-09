//go:build goinfer_testhooks

package fitcmd

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/decoder/fit"
	"github.com/townsendmerino/goinfer/internal/runtime/modelload"
)

// TestSourceFor_opensWhatTheLoadOpens: fit plans each backend on what a load on that backend opens. The quant is the load's own
// (modelload.Plan: with no -quant, the per-family default, which for a Phi-3 .gguf is not int4), and a Metal line reads through the
// fresh metal sidecar where a load reads one (decoder.DefaultToSidecar), the .gguf where it reads that.
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
	want := src // where a load reads a .gguf directly (decoder.DefaultToSidecar: Windows), it opens the .gguf
	if decoder.DefaultToSidecar(false) {
		want = sidecar
	}
	if s.path != want {
		t.Fatalf("metal at q4k: fit opens %s, want %s, what a load on %s opens", s.path, want, runtime.GOOS)
	}
	m, err := decoder.Load(s.path, fitLoadOptions(s, 0))
	if err != nil {
		t.Fatalf("the metal sidecar does not open as fit opens it: %v", err)
	}
	m.Close()
}

// TestPlanFor_metalPricesASourceAsItsSidecarRoute: a Metal load of a .gguf or a safetensors directory with no fresh sidecar builds
// the metal sidecar first and runs on it (modelload's sidecar routes), so fit's Metal line prices such a source as that .giw holds it:
// Metal's layout (canonical int4, no row4 repack) and a host copy the bundle maps. Each source is planned as Run plans it (sourceFor,
// fitLoadOptions, planFor) before and after its sidecar is built, and the two plans agree to the byte. The sources are qwen35-tiny's
// directory and a synthetic llama q4_k .gguf (no Qwen3.5 GGUF is committed, and glm-tiny.gguf has no tokenizer to transcode with).
func TestPlanFor_metalPricesASourceAsItsSidecarRoute(t *testing.T) {
	if !decoder.DefaultToSidecar(false) {
		t.Skipf("a load on %s reads its source directly and builds no sidecar (decoder.DefaultToSidecar), so there is no sidecar route to price", runtime.GOOS)
	}
	plan := func(t *testing.T, path string) (fit.Plan, string) {
		t.Helper()
		s, err := sourceFor(path, "int4", "int4", "metal")
		if err != nil {
			t.Fatal(err)
		}
		m, err := decoder.Load(s.path, fitLoadOptions(s, 64))
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		return planFor(m, "metal", 1<<40, fit.PlanRequest{Ctx: 64, CtxPinned: true}), s.path
	}
	copyTo := func(t *testing.T, from, to string) {
		t.Helper()
		b, err := os.ReadFile(from)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(to, b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for _, c := range []struct{ name, fixture string }{
		{"qwen35-tiny's directory", "../../../testdata/qwen35-tiny"},
		{"a llama q4_k .gguf", ""}, // decoder.WriteLlamaQ4KTinyGGUFForTest's: a .gguf the sidecar transcode takes needs a tokenizer
	} {
		t.Run(c.name, func(t *testing.T) {
			src := filepath.Join(t.TempDir(), "model.gguf") // its sidecar is written beside it
			if c.fixture == "" {
				if err := decoder.WriteLlamaQ4KTinyGGUFForTest(src); err != nil {
					t.Fatal(err)
				}
			} else if st, err := os.Stat(c.fixture); err != nil {
				t.Fatal(err)
			} else if src = filepath.Join(filepath.Dir(src), filepath.Base(c.fixture)); st.IsDir() {
				if err := os.Mkdir(src, 0o755); err != nil {
					t.Fatal(err)
				}
				ents, err := os.ReadDir(c.fixture)
				if err != nil {
					t.Fatal(err)
				}
				for _, e := range ents {
					if e.Type().IsRegular() {
						copyTo(t, filepath.Join(c.fixture, e.Name()), filepath.Join(src, e.Name()))
					}
				}
			} else {
				copyTo(t, c.fixture, src)
			}
			old := time.Now().Add(-time.Hour)
			if err := os.Chtimes(src, old, old); err != nil {
				t.Fatal(err)
			}
			before, from := plan(t, src)
			if from != src || before.Placement != fit.PlacementResident {
				t.Fatalf("before the sidecar: fit planned %s (%v, %s), want the source, resident", from, before.Placement, before.Reason)
			}
			// The control: opened as the source itself, not as its sidecar's stand-in, it holds a heap host copy, and on a core that builds
			// the CPU's row4 repack beside canonical int4 (arm64 with its kernels), that repack too.
			plain, err := decoder.Load(src, decoder.Options{Quant: "int4", DirectLoad: true, ResidentContext: 64})
			if err != nil {
				t.Fatal(err)
			}
			p := planFor(plain, "metal", 1<<40, fit.PlanRequest{Ctx: 64, CtxPinned: true})
			plain.Close()
			row4 := linalg.Int4Row4Usable(256, 256, 32)
			if p.HostCopyBytes <= before.HostCopyBytes || p.DenseBytes < before.DenseBytes || row4 && p.DenseBytes == before.DenseBytes {
				t.Fatalf("the source opened as itself prices dense %d, host copy %d B, against its stand-in's %d, %d B (row4 on this core: %v): the test cannot see the layout",
					p.DenseBytes, p.HostCopyBytes, before.DenseBytes, before.HostCopyBytes, row4)
			}
			sidecar := decoder.SidecarPath(src, "int4", "metal", false)
			if err := decoder.TranscodeGIW(context.Background(), src, sidecar, "int4", false, decoder.GIWTargetForBackend("metal")); err != nil {
				t.Fatal(err)
			}
			after, from := plan(t, src)
			if from != sidecar {
				t.Fatalf("after the sidecar is built fit plans %s, want it", from)
			}
			if before.DenseBytes != after.DenseBytes || before.ExpertBytesFull != after.ExpertBytesFull ||
				before.HostCopyBytes != after.HostCopyBytes || before.NeedBytes() != after.NeedBytes() {
				t.Errorf("the source prices dense %d, experts %d, host copy %d, need %d B; its sidecar dense %d, experts %d, host copy %d, need %d B",
					before.DenseBytes, before.ExpertBytesFull, before.HostCopyBytes, before.NeedBytes(),
					after.DenseBytes, after.ExpertBytesFull, after.HostCopyBytes, after.NeedBytes())
			}
		})
	}
}
