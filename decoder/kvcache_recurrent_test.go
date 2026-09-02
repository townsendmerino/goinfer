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

// A THIRD KIND OF RECURRENT STATE, ALONE. TestTruncateTo_resetsRecurrent above sets mamba AND
// delta on the same cache, so a guard checking either one stays green — and a third kind is
// invisible to it. That is exactly what happened: LFM2's short-conv window is mutated in place per
// token like the other two, resetRecurrent() already cleared it, and the guard that CALLS
// resetRecurrent named only mamba and delta. On an LFM2-only cache the reset was unreachable, so
// TruncateTo(0) left the window intact and conversation B's first K-1 tokens convolved over
// conversation A's last Bx vectors at every conv layer (audit-2026-09-02 C-02).
//
// This cache carries ONLY conv, which is the whole point of it being a separate test.
func TestTruncateTo_resetsConvWindowAlone(t *testing.T) {
	newCache := func() *KVCache {
		c := NewKVCache(1, 1, 1, 0, 4)
		c.pos = 5
		c.conv = []*shortConvState{{convWin: [][]float32{{9}, {8}}}}
		return c
	}
	if c := newCache(); c.mamba != nil || c.delta != nil {
		t.Fatalf("the premise broke: this cache must carry conv ONLY (mamba=%v delta=%v)", c.mamba, c.delta)
	}

	c := newCache()
	if exact := c.TruncateTo(0); !exact {
		t.Errorf("TruncateTo(0) reported inexact")
	}
	if c.conv[0].convWin != nil {
		t.Errorf("conv window not cleared by TruncateTo(0): %v — the prior sequence's last Bx "+
			"vectors are still what the next sequence's first tokens convolve over", c.conv[0].convWin)
	}

	c2 := newCache()
	if exact := c2.TruncateTo(2); exact {
		t.Error("TruncateTo(2) on a conv cache reported EXACT; want inexact. An exact report is " +
			"what lets rewindForReuse warm-reuse a prefix whose windows still hold the dropped " +
			"positions, and the attention layers then append K/V computed from that residual")
	}
}

// Snapshot must refuse a conv cache for the same reason it refuses mamba/delta/MLA: LoadSession
// rebuilds through NewCache(pos) with empty windows, so the session comes back "warm" and wrong.
// Reachable through -session-dir and -kv-idle-demote (audit-2026-09-02 C-02, the C-05 shape).
func TestSnapshot_refusesConvCache(t *testing.T) {
	c := NewKVCache(1, 1, 1, 0, 4)
	c.pos = 2
	c.conv = []*shortConvState{{convWin: [][]float32{{9}}}}
	s := &Session{cache: c}
	if b := s.Snapshot("id"); b != nil {
		t.Errorf("Snapshot returned %d bytes for a cache with a short-conv window; it must refuse "+
			"so the caller cold-prefills", len(b))
	}
}
