package decoder

import "testing"

// TestNewKVCache_skipLayerZeroesCapacity gates P-02's second half (docs/audit-2026-09-10.md):
// a layer NewKVCache is told to skip must get zero keys/vals CAPACITY, not just an empty length —
// the whole point is eliminating the allocation itself, not merely leaving it unused.
func TestNewKVCache_skipLayerZeroesCapacity(t *testing.T) {
	const capHint = 1000
	skip := func(l int) bool { return l == 1 }
	c := NewKVCache(3, 8, 128, 0, capHint, skip)
	for l := 0; l < 3; l++ {
		wantZero := l == 1
		gotZero := cap(c.keys[l]) == 0 && cap(c.vals[l]) == 0
		if wantZero != gotZero {
			t.Errorf("layer %d: cap(keys)=%d cap(vals)=%d, want zero=%v", l, cap(c.keys[l]), cap(c.vals[l]), wantZero)
		}
	}
	// The non-skipped layers must be UNAFFECTED — same capacity as before this parameter existed.
	want := capHint * 8 * 128
	if cap(c.keys[0]) != want || cap(c.vals[0]) != want {
		t.Errorf("layer 0 (not skipped): cap(keys)=%d cap(vals)=%d, want %d", cap(c.keys[0]), cap(c.vals[0]), want)
	}
}

// TestNewKVCache_nilSkipLayerReservesEveryLayer confirms the nil-safe default (every existing
// caller before this fix, and decoder/mtp.go's single-layer cache) is unaffected: no layer is
// ever skipped, matching the exact pre-fix behavior.
func TestNewKVCache_nilSkipLayerReservesEveryLayer(t *testing.T) {
	const capHint = 16
	c := NewKVCache(4, 2, 4, 0, capHint, nil)
	want := capHint * 2 * 4
	for l := 0; l < 4; l++ {
		if cap(c.keys[l]) != want || cap(c.vals[l]) != want {
			t.Errorf("layer %d: cap(keys)=%d cap(vals)=%d, want %d", l, cap(c.keys[l]), cap(c.vals[l]), want)
		}
	}
}

// TestNewCache_hybridRealFixtureSkipsRecurrentLayerAllocation is the real-fixture integration
// test: (*Model).NewCache on a real Nemotron hybrid checkpoint (mamba/attention/mlp/mamba/
// attention, testdata/nemotron-tiny) must reserve zero keys/vals capacity for its mamba and mlp
// layers, matching hasNoAttentionKVAt exactly — proving the wiring from Model down to
// NewKVCache's new parameter, not just the parameter's own isolated logic above.
func TestNewCache_hybridRealFixtureSkipsRecurrentLayerAllocation(t *testing.T) {
	m, err := Load("../testdata/nemotron-tiny", Options{Quant: "f32"})
	if err != nil {
		t.Skipf("no nemotron-tiny fixture: %v", err)
	}
	defer m.Close()
	arch := m.w.arch
	if arch.nemotron == nil {
		t.Fatal("test bug: nemotron-tiny did not resolve as a Nemotron architecture")
	}
	const capHint = 4096
	c := m.NewCache(capHint)
	for l := 0; l < arch.NumLayers; l++ {
		wantZero := arch.hasNoAttentionKVAt(l)
		gotZero := cap(c.keys[l]) == 0 && cap(c.vals[l]) == 0
		if wantZero != gotZero {
			t.Errorf("layer %d (blockKind=%d): cap(keys)=%d cap(vals)=%d, want zero=%v",
				l, arch.nemotron.blockKind[l], cap(c.keys[l]), cap(c.vals[l]), wantZero)
		}
	}
}
