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

// TestTruncateTo_resetsMultimodal is the break-it-first gate for audit M-25: a reused
// cache must not leak the prior sequence's multimodal state — image attention blocks and
// Qwen2.5-VL m-RoPE positions/delta. A full clear (Session.Reset → TruncateTo(0)) must
// clear them, alongside the recurrent reset (C-01), so a future image-in-chat with warm-KV
// prefix reuse can't inherit stale image blocks / m-RoPE offsets. This runs on a plain
// (non-recurrent) cache, proving the reset happens outside the recurrent guard.
func TestTruncateTo_resetsMultimodal(t *testing.T) {
	c := NewKVCache(1, 1, 1, 0, 4)
	c.pos = 5
	c.SetImageBlocks([][2]int{{1, 4}})
	c.mropePos = [][3]int{{0, 0, 0}, {1, 1, 1}}
	c.mropeDelta = -3

	if exact := c.TruncateTo(0); !exact {
		t.Errorf("TruncateTo(0) reported inexact on a text/VL cache")
	}
	if c.imgBlocks != nil {
		t.Errorf("imgBlocks not cleared after TruncateTo(0): %v", c.imgBlocks)
	}
	if c.mropePos != nil {
		t.Errorf("mropePos not cleared after TruncateTo(0): %v", c.mropePos)
	}
	if c.mropeDelta != 0 {
		t.Errorf("mropeDelta not zeroed after TruncateTo(0): %d", c.mropeDelta)
	}
}
