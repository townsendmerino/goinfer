//go:build darwin && goinfer_testhooks

package metal

import (
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestMetalKVSlots_default is E-P09's backend half: a slot request marked as the caller's default
// (Options.ResidentKVSlotsDefault) builds metalDefaultKVSlots slots, and the same request given by the operator builds
// what it asked (the memory guard permitting, which the fixture's few MB always does); a default of 1 stays 1.
func TestMetalKVSlots_default(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	path := writeMC3Fixture(t, 4096)
	for _, tc := range []struct {
		ask       int
		byDefault bool
		want      int
	}{{4, true, metalDefaultKVSlots}, {4, false, 4}, {1, true, 1}, {2, true, 2}} {
		m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 1024,
			ResidentKVSlots: tc.ask, ResidentKVSlotsDefault: tc.byDefault})
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		got := m.ResidentKVSlots()
		m.Close()
		if got != tc.want {
			t.Errorf("ask %d (default %v): %d resident KV slots, want %d", tc.ask, tc.byDefault, got, tc.want)
		}
	}
}
