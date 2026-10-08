package decoder

import (
	"context"
	"math"
	"testing"
)

// TestSession_snapshotAdapterContinuation drives the adapter × snapshot cell through real
// generation: a session built under a real compute-time adapter is snapshotted, restored, and
// continued, and the K/V it ends with must equal a session that was never snapshotted. The
// bookkeeping tests in cachestate_test.go use stand-in runtimes and a weightless model; this one
// runs the adapter's deltas through the forward, so a restore that dropped the adapter, or reused a
// prefix built under another one, shows up in the cache.
//
// The cache, not the tokens: the synthetic model's weights are periodic, and greedy decoding picks
// the same token under every adapter, so tokens cannot tell the projections apart. The adapter
// targets v_proj, so the V rows differ by adapter at every position.
func TestSession_snapshotAdapterContinuation(t *testing.T) {
	base, writeAdapter := loraFixture(t)
	be, _ := NewBackend("")
	w, err := loadWeights(base, quantNone, false, true, false, nil, nil)
	if err != nil {
		t.Fatalf("loadWeights: %v", err)
	}
	m := &Model{w: w, be: be, eosIDs: w.Cfg.EOSIDs()}
	defer m.Close()
	if err := m.LoadAdapter("a", writeAdapter(0)); err != nil {
		t.Fatalf("LoadAdapter a: %v", err)
	}
	if err := m.LoadAdapter("b", writeAdapter(3)); err != nil {
		t.Fatalf("LoadAdapter b: %v", err)
	}

	ctx := context.Background()
	greedy := SamplingParams{}
	const n = 10
	prompt := []int{1, 5, 3, 9, 2, 7}

	// kv flattens every layer's K and V up to the cache position.
	kv := func(s *Session) []float32 {
		c := s.cache
		var out []float32
		for l := 0; l < c.numLayers; l++ {
			out = append(out, c.keys[l][:c.pos*c.kvDim]...)
			out = append(out, c.vals[l][:c.pos*c.kvDim]...)
		}
		return out
	}
	dist := func(x, y []float32) float64 {
		if len(x) != len(y) {
			return math.Inf(1)
		}
		d := 0.0
		for i := range x {
			d = math.Max(d, math.Abs(float64(x[i]-y[i])))
		}
		return d
	}
	// fresh runs prompt from a new session under adapter (or the base for ""), with no snapshot
	// anywhere: the reference every restored continuation is compared with.
	fresh := func(adapter string, prompt []int) []float32 {
		s := m.NewSession(0)
		if adapter != "" {
			if err := s.UseAdapter(adapter); err != nil {
				t.Fatal(err)
			}
		}
		gen(t, func() (<-chan int, *Generation) { return s.Generate(ctx, prompt, n, greedy) })
		return kv(s)
	}
	// same is within float noise: a fresh run prefills turn2 in one pass, while the snapshotted
	// session built its first turn by prefill plus decode steps, so the sums are ordered differently.
	const same, apart = 1e-4, 1e-2

	// The first turn, under a, then a snapshot of it.
	s := m.NewSession(0)
	if err := s.UseAdapter("a"); err != nil {
		t.Fatal(err)
	}
	gen(t, func() (<-chan int, *Generation) { return s.Generate(ctx, prompt, n, greedy) })
	seq := append([]int(nil), s.Tokens()...)
	turn2 := append(append([]int(nil), seq...), 4, 11)
	blob := s.Snapshot("id")
	if blob == nil {
		t.Fatal("Snapshot refused a session under a compute-time adapter")
	}

	wantA, wantB, wantBase := fresh("a", turn2), fresh("b", turn2), fresh("", turn2)
	// Not vacuous: the three projections must leave caches far apart, or a wrong binding could pass.
	for _, p := range []struct {
		name string
		x, y []float32
	}{{"a/base", wantA, wantBase}, {"a/b", wantA, wantB}, {"b/base", wantB, wantBase}} {
		if d := dist(p.x, p.y); d < apart {
			t.Fatalf("%s caches differ by only %g; the test cannot tell the adapters apart", p.name, d)
		}
	}

	t.Run("restored without rebinding continues under the recorded adapter", func(t *testing.T) {
		r, err := m.LoadSession(blob, "id")
		if err != nil {
			t.Fatalf("LoadSession: %v", err)
		}
		gen(t, func() (<-chan int, *Generation) { return r.Generate(ctx, turn2, n, greedy) })
		if d := dist(kv(r), wantA); d > same {
			t.Errorf("K/V differs by %g from the reference (adapter a, never snapshotted)", d)
		}
	})
	t.Run("restored then bound to another adapter matches that adapter from cold", func(t *testing.T) {
		r, err := m.LoadSession(blob, "id")
		if err != nil {
			t.Fatalf("LoadSession: %v", err)
		}
		if err := r.UseAdapter("b"); err != nil {
			t.Fatal(err)
		}
		gen(t, func() (<-chan int, *Generation) { return r.Generate(ctx, turn2, n, greedy) })
		if d := dist(kv(r), wantB); d > same {
			t.Errorf("K/V differs by %g from the reference (adapter b from cold): the prefix built under a was reused", d)
		}
	})
	t.Run("restored then cleared to the base matches the base from cold", func(t *testing.T) {
		r, err := m.LoadSession(blob, "id")
		if err != nil {
			t.Fatalf("LoadSession: %v", err)
		}
		r.ClearAdapter()
		gen(t, func() (<-chan int, *Generation) { return r.Generate(ctx, turn2, n, greedy) })
		if d := dist(kv(r), wantBase); d > same {
			t.Errorf("K/V differs by %g from the reference (base from cold): the prefix built under a was reused", d)
		}
	})
}
