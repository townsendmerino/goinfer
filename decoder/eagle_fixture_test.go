package decoder

import (
	"os"
	"runtime/debug"
	"strconv"
	"sync"
	"testing"
	"time"
)

// Shared, process-lifetime EAGLE fixtures. The EAGLE tests each reloaded the
// ~1.8 GB base (+ the 785 MB head) — 7 loads for one opt-in run
// (GOINFER_HEAVY_TESTS=1). These accessors load each distinct config AT MOST
// ONCE across the whole test binary and hand back a SHARED, READ-ONLY handle.
//
// Read-only is the contract, and it holds because per-stream state lives on the
// KVCache (base.NewCache), not on *Model — the tests only call NewCache /
// ForwardCapture / embedToken / be, none of which mutate the model, and the head
// methods are pure over its weights. The one thing a test must NOT do is Close a
// shared handle (the old `defer base.Close()`) — teardown is TestMain's job
// (closeEagleFixtures), so the first test can't pull the model out from under the
// rest.
//
// This is cleanup (reload waste), NOT the memory fix: the accept test's 81 GB was
// ≫ 7× a load, i.e. retention in the per-position capture loop, which sharing does
// not touch. Nor does it mask a per-load leak — a single test still triggers
// exactly one load (this one), so the HeapInuse-vs-RSS discriminator on one test
// is unchanged; sharing only collapses the multi-test load COUNT.
//
// Keyed by (loadPath, quant) so GOINFER_EAGLE_BASE overrides still get their own
// base rather than colliding with the default int8int8 one.
var (
	eagleBaseMu    sync.Mutex
	eagleBaseCache = map[string]*Model{}
	eagleHeadMu    sync.Mutex
	eagleHeadCache = map[string]*EagleHead{}
)

// sharedEagleBase returns the shared read-only base for (loadPath, quant), loading
// it once. Do NOT Close the result; see closeEagleFixtures.
func sharedEagleBase(t *testing.T, loadPath, quant string) *Model {
	t.Helper()
	key := loadPath + "\x00" + quant
	eagleBaseMu.Lock()
	defer eagleBaseMu.Unlock()
	if m := eagleBaseCache[key]; m != nil {
		return m
	}
	m, err := Load(loadPath, Options{Quant: quant})
	if err != nil {
		t.Fatalf("Load base %s (quant %q): %v", loadPath, quant, err)
	}
	eagleBaseCache[key] = m
	return m
}

// sharedEagleHead returns the shared read-only EAGLE head for dir, loading it once.
// Do NOT Close the result; see closeEagleFixtures.
func sharedEagleHead(t *testing.T, dir string) *EagleHead {
	t.Helper()
	eagleHeadMu.Lock()
	defer eagleHeadMu.Unlock()
	if h := eagleHeadCache[dir]; h != nil {
		return h
	}
	h, err := LoadEagleHead(dir)
	if err != nil {
		t.Fatalf("LoadEagleHead %s: %v", dir, err)
	}
	eagleHeadCache[dir] = h
	return h
}

// closeEagleFixtures releases every shared base/head. Called from TestMain after
// all tests run, so no test observes a closed handle.
func closeEagleFixtures() {
	for _, m := range eagleBaseCache {
		m.Close()
	}
	for _, h := range eagleHeadCache {
		h.Close()
	}
}

func TestMain(m *testing.M) {
	boundSuiteMemory()
	stopReclaim := startMemReclaimer() // return freed pages promptly so RSS tracks the working set, not the high-water
	stop := startMemTicker()           // GOINFER_MEM_PROBE: per-2s RSS/heap ticks (interleave with -v to name the accumulator)
	code := m.Run()
	stopReclaim()
	stop()
	memProbeSuite() // heap-vs-RSS discriminator over the whole suite (fixtures still mapped here)
	closeEagleFixtures()
	os.Exit(code)
}

// boundSuiteMemory caps the suite's peak RSS with a soft GOMEMLIMIT. The heap-vs-RSS probe
// (mem_probe_test.go) delivered the verdict: `go test ./decoder/` balloons to ~31 GB RSS pre-GC but
// collapses to ~4 MB HeapInuse / ~8 GB RSS after a forced GC — the "in-use" 31 GB is transient
// allocation churn across the many tiny logic tests, freed but neither collected (pacing-based GC
// only triggers on heap doubling) nor returned (lazy MADV_FREE). NOT a leak (HeapInuse→4 MB) and NOT
// mmap (nonGo RSS-Sys negative). A soft memory limit paces GC to keep that garbage from piling up, so
// RSS tracks the working set (a few MB live) instead of the high-water — no thrash on a memory-tight
// box, no swap. It cannot starve a real test: nothing holds more than a few MB live. Respects a
// user-set GOMEMLIMIT (the runtime already applied it); tune or disable with GOINFER_TEST_MEMLIMIT
// (GiB, or "off").
func boundSuiteMemory() {
	if os.Getenv("GOMEMLIMIT") != "" { // user/runtime already set one — don't override
		return
	}
	gib := int64(8)
	switch v := os.Getenv("GOINFER_TEST_MEMLIMIT"); v {
	case "off":
		return
	case "":
		// default 8 GiB
	default:
		if n, err := strconv.ParseInt(v, 10, 64); err == nil && n > 0 {
			gib = n
		}
	}
	debug.SetMemoryLimit(gib << 30)
}

// startMemReclaimer returns freed heap pages to the OS on a short interval, so the suite's RSS
// tracks the CURRENT working set instead of accumulating a high-water. The probe verdict: the peak
// (~24-31 GB) is transient garbage from a cluster of legitimately heavy tests (serialize
// round-trips, GGUF loaders — each briefly a few GB live) whose freed pages the runtime holds under
// lazy MADV_FREE and never reclaims across the following tests. GODEBUG=madvdontneed=1 would return
// them eagerly but can only be set at process start; debug.FreeOSMemory() does the same on demand,
// and is CHEAP here because it does a GC over a live heap that is only ~4 MB — the cost is the
// madvise of freed spans, not a scan. Every ~3s keeps the peak near a single test's working set.
// stop() ends it (before the post-suite probe, so its GCs don't muddy the FreeOSMemory delta).
// Disable with GOINFER_TEST_MEMLIMIT=off (same knob as the soft limit).
func startMemReclaimer() (stop func()) {
	if os.Getenv("GOINFER_TEST_MEMLIMIT") == "off" {
		return func() {}
	}
	done := make(chan struct{})
	go func() {
		tk := time.NewTicker(3 * time.Second)
		defer tk.Stop()
		for {
			select {
			case <-done:
				return
			case <-tk.C:
				debug.FreeOSMemory()
			}
		}
	}()
	return func() { close(done) }
}
