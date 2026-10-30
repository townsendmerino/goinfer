package decoder

import (
	"context"
	"errors"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// op3tGemma4Fixtures are the tiny Gemma 4 checkpoints OP3t's TW1 runs on: the five gate identity lists for the family, and
// emodel-tiny (PLE, cross-layer KV sharing) and vl-bidir-tiny, which no other fixture covers.
var op3tGemma4Fixtures = []string{
	"../testdata/gemma4-moe-tiny",
	"../testdata/gemma4-moe-unified-tiny",
	"../testdata/gemma4-dense-twogeom-tiny",
	"../testdata/gemma4-dense-scaled",
	"../testdata/gemma4-moe-kv-tiny",
	"../testdata/gemma4-emodel-tiny",
	"../testdata/gemma4-vl-bidir-tiny",
}

// op3tG4Cell is one Gemma 4 TW1 cell: op3tCell's dimensions and a span layout.
type op3tG4Cell struct {
	op3tCell
	spans string // "none", "one", or "two" (on a primed cell, one span ends on the first pass's last row, the other starts the second pass)
}

func (c op3tG4Cell) String() string { return c.op3tCell.String() + "/spans=" + c.spans }

// op3tG4Prime is the priming pass's length.
const op3tG4Prime = 9

// layout returns the cell's spans in absolute positions; they never overlap and lie inside the prompt.
func (c op3tG4Cell) layout() []ImageSpan {
	start := 0
	if c.primed {
		start = op3tG4Prime
	}
	k := c.k
	switch c.spans {
	case "one":
		return []ImageSpan{{Pos: start + k/4, Len: max(2, k/2)}}
	case "two":
		if c.primed {
			// The caller cuts passes outside blocks (cutOutsideBlocks), so no span crosses a pass boundary: these two meet it.
			return []ImageSpan{{Pos: op3tG4Prime - 3, Len: 3}, {Pos: start, Len: max(1, k/2)}}
		}
		return []ImageSpan{{Pos: 0, Len: max(1, k/4)}, {Pos: k / 2, Len: max(1, k/4)}}
	}
	return nil
}

// op3tG4Pass embeds ids (the prompt's rows from position from), splices deterministic stand-ins for projected image features
// into the rows inside spans, and runs the pass through runLayersGemma4FromEmbedN.
func op3tG4Pass(t *testing.T, m *Model, ids []int, from int, spans []ImageSpan, cache *KVCache) []float32 {
	t.Helper()
	hidden := m.w.arch.HiddenDim
	h := m.embedN(ids)
	for i := range ids {
		if _, in := imageSpanRow(spans, from+i); in {
			for j := range hidden {
				h[i*hidden+j] = float32(math.Sin(float64((from+i)*131+j))) * 0.5
			}
		}
	}
	out, err := m.runLayersGemma4FromEmbedN(context.Background(), h, ids, spans, cache)
	if err != nil {
		t.Fatal(err)
	}
	return append([]float32(nil), out...)
}

// op3tGemma4Run runs one cell's passes through the walker (twin false) or the twin and returns what the last pass leaves.
func op3tGemma4Run(t *testing.T, m *Model, c op3tG4Cell, twin bool) op3tState {
	t.Helper()
	gemma4TwinForTest = twin
	defer func() { gemma4TwinForTest = false }()
	vocab := m.w.arch.VocabSize
	ids := func(n, salt int) []int {
		out := make([]int, n)
		for i := range out {
			out[i] = (i*37 + salt) % vocab
		}
		return out
	}
	spans := c.layout()
	cache := m.NewCache(op3tG4Prime + c.k)
	from := 0
	if c.primed {
		op3tG4Pass(t, m, ids(op3tG4Prime, 5), 0, spans, cache)
		from = op3tG4Prime
	}
	st := op3tState{rows: op3tG4Pass(t, m, ids(c.k, 11), from, spans, cache), pos: cache.Pos()}
	for l := range m.w.arch.NumLayers {
		st.kv = append(st.kv, append(append([]float32(nil), cache.Keys(l)...), cache.Vals(l)...))
	}
	return st
}

// op3tGemma4Cells runs every Gemma 4 TW1 cell, walker against twin, and calls check with each cell and its differences.
func op3tGemma4Cells(t *testing.T, check func(c op3tG4Cell, diffs []string)) int {
	ran := 0
	for _, fx := range op3tGemma4Fixtures {
		if _, err := os.Stat(filepath.Join(fx, "model.safetensors")); errors.Is(err, fs.ErrNotExist) {
			t.Logf("NOT RUN: %s (no checkpoint on disk)", fx)
			continue
		}
		for _, quant := range []string{"", "int8int8", "int4"} {
			m, err := Load(fx, Options{Quant: quant})
			if err != nil {
				t.Fatalf("%s/%s: %v", fx, quant, err)
			}
			if m.w.arch.gemma4 == nil {
				t.Fatalf("%s: not a Gemma 4 checkpoint", fx)
			}
			for _, k := range []int{2, 17, 64} {
				for _, primed := range []bool{false, true} {
					for _, sp := range []string{"none", "one", "two"} {
						c := op3tG4Cell{op3tCell: op3tCell{fixture: fx, quant: quant, k: k, primed: primed}, spans: sp}
						check(c, op3tDiff(op3tGemma4Run(t, m, c, true), op3tGemma4Run(t, m, c, false)))
						ran++
					}
				}
			}
			m.Close()
		}
	}
	return ran
}

// TestOP3t_gemma4WalkerMatchesTwin is OP3t's TW1 for Gemma 4 (docs/tasks/task-forward-architecture-2026-10.md §4.8):
// runLayersGemma4FromEmbedN on the batched walker against the twin it replaced, bit for bit, on every fixture above at f32,
// int8int8 and int4, for passes of 2, 17 and 64 rows from position 0 and after a primed first pass, with no span, one span and
// two (on a primed cell, one each side of the pass boundary). It compares the last row after the head's norm, every layer's K/V and
// the position.
func TestOP3t_gemma4WalkerMatchesTwin(t *testing.T) {
	ran := op3tGemma4Cells(t, func(c op3tG4Cell, diffs []string) {
		if len(diffs) > 0 {
			t.Errorf("%v: DIFFERENT: %v", c, diffs)
		}
	})
	if ran == 0 {
		t.Fatal("no cell ran: no Gemma 4 tiny fixture on disk")
	}
	t.Logf("TW1 Gemma 4: %d cells compared", ran)
}

// TestOP3t_gemma4Plants is OP3t's TW3 for Gemma 4: each plant turns TW1 red on at least one cell. Dropping the bidirectional
// block can only show where a span holds two or more rows, so it must be red on span cells and on no span-free cell; the image
// row's PLE can only show on a PLE fixture's span cells, so it must be red on no span-free cell either.
func TestOP3t_gemma4Plants(t *testing.T) {
	for _, p := range []struct {
		name  string
		plant op3tPlant
	}{
		{"bidirectional block dropped", op3tPlantNoBidirectional},
		{"image row PLE from its token id", op3tPlantImagePLEToken},
	} {
		t.Run(p.name, func(t *testing.T) {
			op3tPlantForTest = p.plant
			defer func() { op3tPlantForTest = op3tPlantNone }()
			red, redNoSpan := 0, 0
			ran := op3tGemma4Cells(t, func(c op3tG4Cell, diffs []string) {
				if len(diffs) > 0 {
					red++
					if c.spans == "none" {
						redNoSpan++
					}
				}
			})
			t.Logf("plant %q: red on %d of %d cells (%d without a span)", p.name, red, ran, redNoSpan)
			if red == 0 {
				t.Errorf("plant %q: TW1 stayed green on all %d cells", p.name, ran)
			}
			if redNoSpan != 0 {
				t.Errorf("plant %q: red on %d cells without a span, which it cannot touch", p.name, redNoSpan)
			}
		})
	}
}
