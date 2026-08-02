package decoder

import (
	"os"
	"sync"
	"testing"
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
// Keyed by (loadPath, quant) so GINFER_EAGLE_BASE overrides still get their own
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
	code := m.Run()
	closeEagleFixtures()
	os.Exit(code)
}
