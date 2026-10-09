package serveapp

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestSessionLRU_restoreBindsTheLRUsAdapter drives serve's -session-dir round trip with a
// compute-time adapter: a session saved from an adapter LRU and loaded back through
// sessionLRU.load continues as that adapter when the LRU binds it, and as the base when the LRU
// is a base pool. Since snapshot format v3 the restored session arrives bound to the adapter it
// was built under, so bindAdapter has to clear it for a base pool; before, it only ever set an
// adapter, and a base pool would have answered with the adapter's projections.
//
// Production keeps each adapter's snapshots in their own fingerprint namespace, so a base pool
// does not meet an adapter's blob there; the test shares one fingerprint on purpose, to check the
// binding rule rather than the namespacing.
func TestSessionLRU_restoreBindsTheLRUsAdapter(t *testing.T) {
	base := buildSyntheticBase(t)
	m, err := decoder.Load(base, decoder.Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("decoder.Load base: %v", err)
	}
	defer m.Close()
	if err := m.LoadAdapter("a1", buildSyntheticAdapter(t, +1)); err != nil {
		t.Fatalf("LoadAdapter: %v", err)
	}
	const fp = "shared-fp"
	lru := func(adapter string) *loadedModel {
		l := newSessionLRU(m, 4, 0, fp)
		l.adapter = adapter
		return &loadedModel{model: m, name: "x", fp: fp, adapter: adapter, sessions: l}
	}

	// The first turn under a1, saved.
	src := lru("a1")
	prompt := []int{1, 4, 2, 7, 3}
	genGreedy(t, src, prompt, 6)
	if len(src.sessions.order) != 1 {
		t.Fatalf("source LRU holds %d sessions, want 1", len(src.sessions.order))
	}
	turn2 := append(append([]int(nil), src.sessions.order[0].Tokens()...), 5, 6)
	dir := t.TempDir()
	if err := src.sessions.save(dir); err != nil {
		t.Fatalf("save: %v", err)
	}

	// References from cold, no snapshot involved.
	wantA1 := genGreedy(t, lru("a1"), turn2, 8)
	wantBase := genGreedy(t, lru(""), turn2, 8)
	if eqTokens(wantA1, wantBase) {
		t.Fatalf("a1 and the base continue alike (%v); the test cannot tell the bindings apart", wantA1)
	}

	for _, tc := range []struct {
		name, adapter string
		want          []int
	}{
		{"loaded into the adapter's LRU continues as the adapter", "a1", wantA1},
		{"loaded into a base LRU continues as the base", "", wantBase},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dst := lru(tc.adapter)
			if n := dst.sessions.load(dir); n != 1 {
				t.Fatalf("load restored %d sessions, want 1", n)
			}
			if got := genGreedy(t, dst, turn2, 8); !eqTokens(got, tc.want) {
				t.Errorf("continuation %v, want %v", got, tc.want)
			}
		})
	}
}
