package decoder

import (
	"reflect"
	"sort"
	"testing"
)

// Tests for the cache-state grid (cachestate.go). The first three keep the grid complete; the rest
// check each declaration against what the code does, with properties that do not consult the grid
// — production reads the grid, so a test that only asked the grid would agree with itself.

// markState puts kind s into c the way the cache would hold it, minimally. Every kind needs an entry:
// TestCacheStateGrid_holdsStateKnowsEveryKind fails a kind without one.
var markState = map[cacheState]func(c *KVCache){
	statePositional: func(c *KVCache) {
		if len(c.keys) == 0 {
			c.keys, c.vals, c.stride = make([][]float32, 1), make([][]float32, 1), make([]int, 1)
		}
	},
	stateRing:        func(c *KVCache) { c.localAny = true },
	stateInt8KV:      func(c *KVCache) { c.quant = kvI8 },
	stateMLALatent:   func(c *KVCache) { c.mlaLatent = [][]float32{{}} },
	stateDeltaNet:    func(c *KVCache) { c.delta = []*deltaState{} },
	stateShortConv:   func(c *KVCache) { c.conv = []*shortConvState{} },
	stateMamba2:      func(c *KVCache) { c.mamba = []*mamba2State{} },
	stateKDA:         func(c *KVCache) { c.kda = []*kdaState{} },
	stateImageBlocks: func(c *KVCache) { c.imgBlocks = [][2]int{{1, 3}} },
	stateMRoPE:       func(c *KVCache) { c.mropePos = [][3]int{{0, 0, 0}, {1, 1, 1}, {1, 2, 2}}; c.mropeDelta = -1 },
	stateDeepstack:   func(c *KVCache) { c.deepstack = &deepstackRows{} },
	stateCapture:     func(c *KVCache) { c.captureLayers = []int{0} },
	stateTree:        func(c *KVCache) { c.treeMask = [][]bool{{true}} },
	stateAdapter:     func(c *KVCache) { c.lora = &loraRuntime{} },
}

// TestKVCache_everyFieldHasAState is the registration test: a KVCache field lands only with a
// declaration of which kind of state it is, or why it is not state. KDA (audit 2026-09-10 C-03)
// landed as a field nothing classified and was missed at every lifecycle site; this fails that commit.
func TestKVCache_everyFieldHasAState(t *testing.T) {
	ty := reflect.TypeOf(KVCache{})
	seen := map[string]bool{}
	for i := 0; i < ty.NumField(); i++ {
		name := ty.Field(i).Name
		seen[name] = true
		_, isState := kvCacheFieldState[name]
		why, notState := kvCacheNotState[name]
		switch {
		case isState && notState:
			t.Errorf("KVCache.%s is in both kvCacheFieldState and kvCacheNotState; pick one", name)
		case !isState && !notState:
			t.Errorf("KVCache.%s has no declaration: add it to kvCacheFieldState with its kind of state "+
				"(and give that kind a row in cacheStateGrid if it is new), or to kvCacheNotState with a "+
				"reason it is not per-sequence state (cachestate.go)", name)
		case notState && why == "":
			t.Errorf("KVCache.%s is declared not-state with no reason", name)
		}
	}
	for name := range kvCacheFieldState {
		if !seen[name] {
			t.Errorf("kvCacheFieldState names KVCache.%s, which no longer exists", name)
		}
	}
	for name := range kvCacheNotState {
		if !seen[name] {
			t.Errorf("kvCacheNotState names KVCache.%s, which no longer exists", name)
		}
	}
	for name, s := range kvCacheFieldState {
		if _, ok := cacheStateGrid[s]; !ok {
			t.Errorf("KVCache.%s is declared as kind %q, which has no row in cacheStateGrid", name, s)
		}
	}
}

// TestCacheStateGrid_everyCellDeclared: every kind has a cell on every lifecycle path, with a reason
// wherever the handling is not self-evident, and cacheStates (the order paths iterate) is the grid's
// key set exactly.
func TestCacheStateGrid_everyCellDeclared(t *testing.T) {
	listed := map[cacheState]bool{}
	for _, s := range cacheStates {
		if listed[s] {
			t.Errorf("cacheStates lists %q twice", s)
		}
		listed[s] = true
		if _, ok := cacheStateGrid[s]; !ok {
			t.Errorf("cacheStates lists %q, which has no row in cacheStateGrid", s)
		}
	}
	for s, row := range cacheStateGrid {
		if !listed[s] {
			t.Errorf("cacheStateGrid has a row for %q that cacheStates does not list — the lifecycle "+
				"paths iterate cacheStates, so this row is never read", s)
		}
		for _, p := range lifecyclePaths {
			cell, ok := row[p]
			if !ok || cell.h == "" {
				t.Errorf("%q has no declaration for %q", s, p)
				continue
			}
			switch cell.h {
			case hExact, hCleared, hPersisted:
			default:
				if cell.why == "" {
					t.Errorf("%q × %q is %q with no reason; say why that is correct", s, p, cell.h)
				}
			}
		}
		if len(row) != len(lifecyclePaths) {
			t.Errorf("%q declares %d cells for %d lifecycle paths", s, len(row), len(lifecyclePaths))
		}
		for p, cell := range row {
			if !allowedHandling[p][cell.h] {
				t.Errorf("%q × %q is %q, which that path cannot do (allowed: %v)", s, p, cell.h, handlingNames(allowedHandling[p]))
			}
		}
	}
}

// allowedHandling is what each path can do with state at all. A snapshot either writes a kind,
// refuses while it is held, leaves it to the caller, or never sees it live; it cannot "keep" one,
// so a snapshot cell changed from refused to kept fails here instead of passing as a no-op.
var allowedHandling = map[lifecyclePath]map[stateHandling]bool{
	lcRewind:       {hExact: true, hExactUnlessWrapped: true, hInexact: true, hKept: true, hTransient: true},
	lcReset:        {hCleared: true, hKept: true, hTransient: true},
	lcSnapshot:     {hPersisted: true, hRefused: true, hCallerBound: true, hTransient: true},
	lcReuse:        {hExact: true, hExactUnlessWrapped: true, hInexact: true, hColdOnMismatch: true, hTransient: true},
	lcSpecRollback: {hExact: true, hRefused: true, hKept: true, hTransient: true},
}

func handlingNames(m map[stateHandling]bool) []string {
	var out []string
	for h := range m {
		out = append(out, string(h))
	}
	sort.Strings(out)
	return out
}

// TestCacheStateGrid_holdsStateKnowsEveryKind: a zero cache holds nothing, and marking each kind makes
// exactly that kind held. holdsState's default returns true (fail closed), so a kind missing from its
// switch shows up here as held by the zero cache.
func TestCacheStateGrid_holdsStateKnowsEveryKind(t *testing.T) {
	for _, s := range cacheStates {
		if (&KVCache{}).holdsState(s) {
			t.Errorf("a zero KVCache holds %q — holdsState has no case for it", s)
		}
		mark, ok := markState[s]
		if !ok {
			t.Errorf("markState has no entry for %q; the behaviour tests below cannot place it", s)
			continue
		}
		c := &KVCache{}
		mark(c)
		for _, other := range cacheStates {
			if got, want := c.holdsState(other), other == s; got != want {
				t.Errorf("after marking %q, holdsState(%q) = %v, want %v", s, other, got, want)
			}
		}
	}
}

// TestCacheStateGrid_recurrentRowMatchesHasRecurrentState: the kinds the grid gives the recurrent row
// are exactly the kinds hasRecurrentState() sees. hasRecurrentState is read on its own at the reset in
// TruncateTo and in Session.reconcile, so a fifth recurrent kind declared in the grid but missing there
// (the KDA failure, one predicate along) is caught here.
func TestCacheStateGrid_recurrentRowMatchesHasRecurrentState(t *testing.T) {
	var recurrent []string
	for _, s := range cacheStates {
		isRow := reflect.ValueOf(cacheStateGrid[s]).Pointer() == reflect.ValueOf(recurrentCells).Pointer()
		c := &KVCache{}
		markState[s](c)
		if got := c.hasRecurrentState(); got != isRow {
			t.Errorf("%q: hasRecurrentState() = %v, but the grid %s it the recurrent row", s, got,
				map[bool]string{true: "gives", false: "does not give"}[isRow])
		}
		if isRow {
			recurrent = append(recurrent, string(s))
		}
	}
	sort.Strings(recurrent)
	if len(recurrent) != 4 {
		t.Logf("recurrent kinds now: %v (was four on 2026-10-08: DeltaNet, short-conv, Mamba-2, KDA)", recurrent)
	}
}

// TestCacheStateGrid_rewindNeverLeavesMultimodalStatePastPos checks the multimodal rewind cells by
// the property they protect, not by the grid: after a rewind that reports EXACT, no image block may
// end beyond the new position and no m-RoPE entry may sit at or beyond it. Declaring either kind
// exact would make TruncateTo report exact here with the state still in place.
func TestCacheStateGrid_rewindNeverLeavesMultimodalStatePastPos(t *testing.T) {
	for _, s := range []cacheState{stateImageBlocks, stateMRoPE} {
		c := NewKVCache(1, 1, 1, 0, 4, nil)
		markState[s](c)
		for range 3 {
			c.Advance()
		}
		exact := c.TruncateTo(2)
		if !exact {
			continue // the caller goes cold; nothing stale is reused
		}
		for _, b := range c.imgBlocks {
			if b[1] > 2 {
				t.Errorf("%q: an exact rewind to 2 left image block %v in place; new text at position 2 "+
					"would attend bidirectionally", s, b)
			}
		}
		if len(c.mropePos) > 2 {
			t.Errorf("%q: an exact rewind to 2 left %d m-RoPE positions; the ones at 2 and beyond belong "+
				"to the dropped tokens", s, len(c.mropePos))
		}
	}
}

// TestCacheStateGrid_rewindCellsOnRealCaches runs every own-forward family's real cache (the
// allocation production uses) through a partial rewind. A family holding a kind declared inexact
// must get an inexact report; one holding none must get an exact one (nothing has run, so no ring
// has wrapped).
func TestCacheStateGrid_rewindCellsOnRealCaches(t *testing.T) {
	for _, f := range ownForwards {
		t.Run(f.Name, func(t *testing.T) {
			_, c := realCacheFor(t, f)
			var inexact []cacheState
			for _, s := range cacheStates {
				if c.holdsState(s) && cacheStateGrid[s][lcRewind].h == hInexact {
					inexact = append(inexact, s)
				}
			}
			for range 3 {
				c.Advance()
			}
			if got, want := c.TruncateTo(2), len(inexact) == 0; got != want {
				t.Errorf("TruncateTo(2) = %v with %v held; want %v", got, inexact, want)
			}
		})
	}
}

// TestCacheStateGrid_snapshotCells checks each kind's snapshot cell against Snapshot and LoadSession:
// a refused kind makes Snapshot return nil; any other kind lets it through, and the blob LoadSession
// restores does not carry a kind the grid says is not persisted (so a "persisted" declaration for a
// kind the format drops fails). The persisted kinds' own round trips are TestSession_snapshotRoundTrip
// (positional), TestKVI8_snapshotRoundtrip (int8) and the ring cases in session_test.go.
func TestCacheStateGrid_snapshotCells(t *testing.T) {
	m := plainModel(t)
	for _, s := range cacheStates {
		if s == statePositional || s == stateRing || s == stateInt8KV {
			continue // need real storage behind the flag; covered by the round-trip tests named above
		}
		t.Run(string(s), func(t *testing.T) {
			sess := m.NewSession(4)
			sess.cache.Append(0, make([]float32, sess.cache.kvDim), make([]float32, sess.cache.kvDim))
			for l := 1; l < sess.cache.numLayers; l++ {
				sess.cache.Append(l, make([]float32, sess.cache.kvDim), make([]float32, sess.cache.kvDim))
			}
			sess.tokens = []int{7}
			markState[s](sess.cache)
			blob := sess.Snapshot("id")
			cell := cacheStateGrid[s][lcSnapshot]
			if cell.h == hRefused {
				if blob != nil {
					t.Errorf("declared refused, but Snapshot wrote %d bytes", len(blob))
				}
				return
			}
			if blob == nil {
				t.Fatalf("declared %q, but Snapshot refused", cell.h)
			}
			back, err := m.LoadSession(blob, "id")
			if err != nil {
				t.Fatalf("LoadSession: %v", err)
			}
			if cell.h != hPersisted && back.cache.holdsState(s) {
				t.Errorf("declared %q (not persisted), yet the restored session holds it", cell.h)
			}
			if cell.h == hPersisted && !back.cache.holdsState(s) {
				t.Errorf("declared persisted, but the restored session does not hold it — the format drops it")
			}
		})
	}
}

// TestCacheStateGrid_specRollbackCells: a family whose real cache holds a kind declared refused for
// speculative rollback must have specRollbackSafe() false.
func TestCacheStateGrid_specRollbackCells(t *testing.T) {
	for _, f := range ownForwards {
		t.Run(f.Name, func(t *testing.T) {
			m, c := realCacheFor(t, f)
			for _, s := range cacheStates {
				if c.holdsState(s) && cacheStateGrid[s][lcSpecRollback].h == hRefused && m.specRollbackSafe() {
					t.Errorf("the cache holds %q, declared refused for speculative rollback, but "+
						"specRollbackSafe() says yes", s)
				}
			}
		})
	}
}

// TestSession_adapterSwitchGoesCold is the adapter × prefix-reuse cell driven through the session's
// own reuse decision (rewindForReuse is the first thing Session.Generate and both speculative session
// entry points do with a non-empty prompt; reconcile is the last). Before 2026-10-08 a library caller
// that switched adapters on a warm session reused K/V the previous adapter built.
func TestSession_adapterSwitchGoesCold(t *testing.T) {
	m := plainModel(t)
	a, b := &loraRuntime{}, &loraRuntime{}
	warm := func(under *loraRuntime) *Session {
		s := m.NewSession(8)
		s.cache.lora = under
		s.rewindForReuse([]int{1, 2, 3})
		for range 3 {
			s.cache.Advance()
		}
		s.reconcile([]int{1, 2, 3})
		return s
	}
	cases := []struct {
		name       string
		built, now *loraRuntime
		wantReuse  bool
	}{
		{"same adapter reuses", a, a, true},
		{"base stays base reuses", nil, nil, true},
		{"adapter A to adapter B goes cold", a, b, false},
		{"base to adapter goes cold", nil, a, false},
		{"adapter to base goes cold", a, nil, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := warm(tc.built)
			s.cache.lora = tc.now
			got := s.rewindForReuse([]int{1, 2, 3, 4})
			if tc.wantReuse && got != 3 {
				t.Errorf("reused %d positions, want 3", got)
			}
			if !tc.wantReuse && (got != 0 || s.cache.Pos() != 0) {
				t.Errorf("reused %d positions (cache at %d) of a prefix built under another adapter; want 0 and a cold cache",
					got, s.cache.Pos())
			}
		})
	}
	t.Run("a restored session adopts the adapter bound before its first turn", func(t *testing.T) {
		s := warm(nil)
		s.adapterKnown = false // as LoadSession leaves it: the blob does not record the adapter
		s.cache.lora = a       // serve's bindAdapter
		if got := s.rewindForReuse([]int{1, 2, 3, 4}); got != 3 {
			t.Errorf("reused %d, want 3: the snapshot carries no adapter, so the first binding is adopted", got)
		}
	})
}

// plainModel is a dense, full-attention, f32 model with no recurrent, ring, MLA or multimodal state.
func plainModel(t *testing.T) *Model {
	t.Helper()
	cfg := representativeConfig("llama")
	if cfg == nil {
		t.Fatal(`representativeConfig("llama") is gone`)
	}
	arch, _, err := resolveArchitecture(cfg)
	if err != nil {
		t.Fatalf("resolveArchitecture(llama): %v", err)
	}
	m := &Model{w: &Weights{arch: arch}}
	for _, s := range cacheStates {
		if s != statePositional && m.NewCache(4).holdsState(s) {
			t.Fatalf("the plain llama cache holds %q; the tests using it assume only positional KV", s)
		}
	}
	return m
}
