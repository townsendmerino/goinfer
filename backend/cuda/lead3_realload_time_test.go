//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestLead3_realLoadTime is Lead 3's pre-registered decision measurement
// (docs/measurements/lead3-pin-order-2026-09-22.md): one real 26B load, timed, in ITS OWN
// process — not repeated in-process trials, which the microbenchmark step found had highly
// variable cache-pressure effects that a single production load (once per server lifetime)
// would not average out. Run this as a SEPARATE `go test` invocation per sample, alternating
// GOINFER_MOE_PIN_REGISTER between runs (ABBA at the shell level), so each sample sees
// whatever cache state a freshly-started process actually finds — the real scenario, not a
// repeated-allocation artifact.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_MOE_PIN_REGISTER=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestLead3_realLoadTime -v -timeout 10m
//	GOINFER_HEAVY_TESTS=1                             go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestLead3_realLoadTime -v -timeout 10m
func TestLead3_realLoadTime(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("GOINFER_HEAVY_TESTS unset — loads the real 26B")
	}
	path := os.Getenv("GOINFER_GEMMA4_26B_GIW")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", "gemma4-26b-int4.giw")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no 26B .giw at %s: %v", path, err)
	}
	t.Setenv("GOINFER_MOE_CACHE_EXPERTS", "1")

	t0 := time.Now()
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda"})
	dur := time.Since(t0)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	rf := m.ResidentForwardForTest()
	if rf == nil {
		t.Fatal("cuda resident DECLINED the 26B with C′ on")
	}
	r := rf.(*cudaResident)
	if !r.cacheExperts {
		t.Fatal("C′ expert staging is OFF")
	}
	t.Logf("LOAD_TIME arm=%s dur_ms=%d cacheSlots=%d", os.Getenv("GOINFER_MOE_PIN_REGISTER"), dur.Milliseconds(), r.cacheSlots)
}
