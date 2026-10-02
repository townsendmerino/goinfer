//go:build darwin

package metal

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestPrefillFloor (A-G01, docs/audit-metal-2026-09-30.md): batched prefill declines a prompt shorter than the floor,
// and the positions already in the cache count toward the prompt, so a short suffix after a long cached prefix is
// admitted. GOINFER_METAL_FAST_PREFILL_FLOOR sets the floor, 0 turns it off, and a value that does not parse leaves
// the default; PrefillPath names the floor in force. Many tests set the knob to 0, so nothing checked the decision.
func TestPrefillFloor(t *testing.T) {
	for _, c := range []struct {
		v    string
		want int
	}{{"", 64}, {"0", 0}, {"128", 128}, {"-1", 64}, {"x", 64}, {" 32", 64}} {
		if got := metalFastPrefillFloorFor(c.v); got != c.want {
			t.Errorf("metalFastPrefillFloorFor(%q) = %d, want %d", c.v, got, c.want)
		}
	}
	for _, c := range []struct {
		knob        string
		startPos, M int
		decline     bool
	}{
		{"", 0, 63, true},     // one short of the floor
		{"", 0, 64, false},    // at it
		{"", 56, 7, true},     // 63 positions in all: the cached ones count
		{"", 56, 8, false},    // 64 in all
		{"", 1000, 8, false},  // a short suffix after a long cached prefix
		{"0", 0, 8, false},    // the floor off
		{"128", 0, 100, true}, // the knob raises it
		{"128", 0, 128, false},
		{"x", 0, 63, true}, // a value that does not parse leaves 64
	} {
		name := fmt.Sprintf("floor knob %q, startPos %d, M %d", c.knob, c.startPos, c.M)
		a := tinyPrefillResident(t, decoder.Options{Quant: "int8int8",
			Knobs: &decoder.Knobs{"GOINFER_METAL_FAST_PREFILL_FLOOR": c.knob}}, 2048)
		// Nothing has written the cached prefix; zero it so an admitted call reads defined keys. The KV buffers are
		// byte-sized, so Int8s is the view whose length matches them.
		for l := range a.r.kc {
			clear(a.r.kc[l].Int8s())
			clear(a.r.vc[l].Int8s())
		}
		want := "GOINFER_METAL_FAST_PREFILL_FLOOR=0"
		if floor := metalFastPrefillFloorFor(c.knob); floor > 0 {
			want = fmt.Sprintf("above %d prompt tokens", floor)
		}
		if ok, why := a.PrefillPath(); !ok || !strings.Contains(why, want) {
			t.Errorf("%s: PrefillPath = %v, %q; want batched, naming %q", name, ok, why, want)
		}
		_, err := a.PrefillLast(context.Background(), tinyEmbs(c.M), c.startPos)
		short := err != nil && strings.Contains(err.Error(), "prompt too short")
		switch {
		case c.decline && !short:
			t.Errorf("%s: err %v, want the floor's decline", name, err)
		case !c.decline && err != nil:
			t.Errorf("%s: err %v, want the batched pass", name, err)
		}
		a.r.Close()
	}
}
