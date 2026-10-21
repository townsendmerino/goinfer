//go:build cuda && goinfer_testhooks

package cuda

import (
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestKV2_optionKVShareOff_resident drives Options.KVShareOff through a CUDA resident's prefill (the option grid's
// GPU-resident-prefill cell): conversation B's first turn shares A's 1,000-token preamble, so with sharing on it copies
// A's rows, and with KVShareOff it copies none. B's reply equals the cold reference either way.
func TestKV2_optionKVShareOff_resident(t *testing.T) {
	const dir = "../testdata/llama-attnfa-tiny"
	requireDeviceAndFixture(t, dir)
	for _, off := range []bool{false, true} {
		s := &kv2Sched{t: t, path: dir, opts: decoder.Options{ResidentContext: 2048, KVPrecision: "f16", KVShareOff: off},
			maxTok: 16, pre: kv2Preamble(1000), own: 70}
		g := s.load(2)
		a := s.gen(g, s.first(0))
		b := s.gen(g, s.first(1))
		_ = g.m.Close()
		s.colds = map[string]kv2Turn{}
		ref := s.cold(s.first(1))
		if !slices.Equal(b.ids, ref.ids) {
			t.Errorf("KVShareOff=%v: B's reply %v differs from the cold reference %v", off, b.ids, ref.ids)
		}
		switch {
		case !off && b.shared == 0:
			t.Errorf("sharing on: B copied no rows from A (A reused %d)", a.reused)
		case off && b.shared != 0:
			t.Errorf("KVShareOff: B copied %d rows from another conversation, want none", b.shared)
		}
		t.Logf("KVShareOff=%v: B shared %d, reused %d, declined %q", off, b.shared, b.reused, b.declined)
	}
}
