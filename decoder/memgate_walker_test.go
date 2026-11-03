//go:build goinfer_testhooks

package decoder

import (
	"reflect"
	"testing"

	"github.com/townsendmerino/goinfer/internal/mem"
)

// TestHoldsWalker_chargesItsLine: every byte the walker can see lands on the line of the field it came from, in each shape that once went uncounted
// or to scratch, and only the words that name other memory (slice and pointer headers) are scratch. Fields are walked as a struct would hold them.
func TestHoldsWalker_chargesItsLine(t *testing.T) {
	type elem struct {
		N int64
		A [4]float32
	}
	type target struct {
		A [8]float32
		N int64
	}
	type withSlice struct{ S []float32 }
	const header, word = 24, 8
	spare := make([]withSlice, 1, 2)
	spare[:2][1].S = make([]float32, 4)
	for _, c := range []struct {
		name           string
		v              any
		state, scratch int64
	}{
		{"[64]float32 field", &struct{ A [64]float32 }{}, 256, 0},
		{"scalar fields", &struct {
			N int64
			F float32
		}{}, 12, 0},
		{"[][64]float32, with spare capacity", &struct{ S [][64]float32 }{make([][64]float32, 1, 2)}, 512, header},
		{"[]struct, with spare capacity", &struct{ S []elem }{make([]elem, 1, 2)}, 48, header},
		{"*T", &struct{ P *target }{&target{}}, 40, word},
		{"[]*T", &struct{ S []*target }{[]*target{{}, {}}}, 80, header + 2*word},
		{"[][]float32", &struct{ S [][]float32 }{[][]float32{make([]float32, 4)}}, 16, 2 * header},
		{"a slice inside spare capacity", &struct{ S []withSlice }{spare}, 16, 3 * header},
		{"map[int][64]float32", &struct{ M map[int][64]float32 }{map[int][64]float32{0: {}}}, 264, word},
		{"[2][4]float32", &struct{ A [2][4]float32 }{}, 32, 0},
	} {
		t.Run(c.name, func(t *testing.T) {
			var l mem.Ledger
			h := &holdsWalker{l: &l}
			h.root(func(string) mem.LineID { return mem.LineState }, "f", reflect.ValueOf(c.v).Elem())
			release := h.finish()
			defer release()
			u := l.Usage()
			if got := u.ByLine[mem.LineState]; got != c.state {
				t.Errorf("state: %d bytes, want %d", got, c.state)
			}
			if got := u.ByLine[mem.LineScratch]; got != c.scratch {
				t.Errorf("scratch: %d bytes, want %d", got, c.scratch)
			}
		})
	}
}
