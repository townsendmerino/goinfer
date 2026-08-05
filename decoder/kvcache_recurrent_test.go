package decoder

import "testing"

// TestTruncateTo_resetsRecurrent is the break-it-first gate for audit C-01: a reused
// cache must not leak the prior sequence's Mamba-2 / DeltaNet rolling state. A full
// clear (Session.Reset → TruncateTo(0)) re-zeroes it; any rewind reports inexact so
// rewindForReuse cold-prefills instead of decoding from stale recurrent state.
func TestTruncateTo_resetsRecurrent(t *testing.T) {
	newCache := func() *KVCache {
		c := NewKVCache(1, 1, 1, 0, 4)
		c.pos = 5
		c.mamba = []*mamba2State{{ssm: []float32{1, 2, 3}, convWin: [][]float32{{9}}}}
		c.delta = []*deltaState{{s: []float32{4, 5}, convWin: [][]float32{{7}}}}
		return c
	}

	// TruncateTo(0) zeroes the rolling state (the leak fix) and is exact.
	c := newCache()
	if exact := c.TruncateTo(0); !exact {
		t.Errorf("TruncateTo(0) reported inexact")
	}
	for _, v := range c.mamba[0].ssm {
		if v != 0 {
			t.Fatalf("mamba ssm not zeroed after TruncateTo(0): %v", c.mamba[0].ssm)
		}
	}
	if c.mamba[0].convWin != nil {
		t.Errorf("mamba convWin not cleared: %v", c.mamba[0].convWin)
	}
	for _, v := range c.delta[0].s {
		if v != 0 {
			t.Fatalf("delta state not zeroed after TruncateTo(0): %v", c.delta[0].s)
		}
	}

	// A partial rewind can't be exactly represented by a single rolling state → inexact,
	// which drives the cold-prefill path.
	c2 := newCache()
	if exact := c2.TruncateTo(2); exact {
		t.Errorf("TruncateTo(2) on a recurrent cache reported exact; want inexact (cold-prefill)")
	}
}
