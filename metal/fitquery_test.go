//go:build darwin && goinfer_testhooks

package metal

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/giw"
)

// tinyMetalGIW writes the wide-window tiny model as a Metal-target .giw at quant.
func tinyMetalGIW(t *testing.T, quant string) string {
	t.Helper()
	src, err := decoder.Load(wideWindowTiny(t), decoder.Options{Quant: quant})
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	blob, err := decoder.SerializeWeightsForTarget(src.Weights(), "fit-test", decoder.GIWTargetForBackend("metal"))
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "tiny.giw")
	if err := os.WriteFile(path, giw.Write(blob, []byte("tok")), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func syncMapLen(m *sync.Map) int {
	n := 0
	m.Range(func(any, any) bool { n++; return true })
	return n
}

// TestFitQuery_metalGIWPlansWhatTheLoadBuilds: for a .giw on Metal, FitQuery's decode path is the one the load then
// reports, from the checks BuildResident makes before it allocates (PriceResident). With room, it names the resident and
// the context the resident holds; with a memory figure the guard refuses even at the floor context, it names the CPU path
// with the guard's own decline, the same text the load records. The plan leaves no per-model state behind.
func TestFitQuery_metalGIWPlansWhatTheLoadBuilds(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	path := tinyMetalGIW(t, "int4")
	live := metalLiveAvailable
	t.Cleanup(func() { metalLiveAvailable = live })
	opts := decoder.Options{Backend: "metal"}
	for _, c := range []struct {
		name     string
		budget   int64
		resident bool
	}{
		{"room", 8 << 30, true},
		{"the guard refuses", 4 << 10, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			metalLiveAvailable = func() int64 { return c.budget }
			a, err := decoder.FitQuery(path, opts)
			if err != nil {
				t.Fatal(err)
			}
			if !a.Fits || a.DecodePath == "" {
				t.Fatalf("fits %v, decode path %q (unknown: %s)", a.Fits, a.DecodePath, a.DecodeUnknown)
			}
			if n, k := syncMapLen(&metalFitChoice), syncMapLen(&metalCtxCeiling); n+k != 0 {
				t.Errorf("the plan left %d fit choices and %d context ceilings behind", n, k)
			}
			m, err := decoder.Load(path, opts)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if got := m.DecodePath(); got != a.DecodePath {
				t.Errorf("the plan says %q; the load reports %q", a.DecodePath, got)
			}
			res, ok := m.ResidentForwardForTest().(*metalResident)
			if ok != c.resident {
				t.Fatalf("resident %v, want %v (%s)", ok, c.resident, m.ResidentDecline())
			}
			if c.resident {
				if a.DecodePath != "metal-resident (int4)" || a.ResidentCtx != res.ContextCap() {
					t.Errorf("plan %q at ctx %d; the resident holds %d", a.DecodePath, a.ResidentCtx, res.ContextCap())
				}
			} else if !strings.Contains(a.DecodePath, residentMemoryDeclinePrefix) || a.ResidentCtx != 0 {
				t.Errorf("plan %q at ctx %d: want the memory guard's decline and no resident context", a.DecodePath, a.ResidentCtx)
			}
		})
	}
}

// TestFitQuery_metalInt8LeavesTheLabelToTheBuild: an int8int8 .giw's resident quant is decided inside the build (native
// int8 or re-quantized to int4), so the plan reports the path as unknown and says so, rather than guess the label.
func TestFitQuery_metalInt8LeavesTheLabelToTheBuild(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	live := metalLiveAvailable
	metalLiveAvailable = func() int64 { return 8 << 30 }
	t.Cleanup(func() { metalLiveAvailable = live })
	path := tinyMetalGIW(t, "int8int8")
	a, err := decoder.FitQuery(path, decoder.Options{Backend: "metal"})
	if err != nil {
		t.Fatal(err)
	}
	if a.DecodePath != "" || !strings.Contains(a.DecodeUnknown, "decided by the build") || a.ResidentCtx == 0 {
		t.Errorf("decode path %q, unknown %q, ctx %d", a.DecodePath, a.DecodeUnknown, a.ResidentCtx)
	}

	// With per-32 activation scales, which Metal's resident projections do not implement, decoder's own admission sends the
	// load to the CPU before the backend is asked: the plan names that path and reason, as the load does.
	opts := decoder.Options{Backend: "metal", ActQuantGroup: 32}
	a, err = decoder.FitQuery(path, opts)
	if err != nil {
		t.Fatal(err)
	}
	m, err := decoder.Load(path, opts)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if !strings.Contains(a.DecodePath, "ActQuantGroup 32") || a.DecodePath != m.DecodePath() {
		t.Errorf("the plan says %q; the load reports %q", a.DecodePath, m.DecodePath())
	}
}
