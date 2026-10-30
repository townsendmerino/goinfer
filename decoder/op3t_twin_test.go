package decoder

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// op3tQwen35Fixtures are the tracked tiny qwen3.5 checkpoints OP3t's TW1 runs on (docs/tasks/task-forward-architecture-2026-10.md
// §4.8, "OP3t").
var op3tQwen35Fixtures = []string{
	"testdata/qwen3_5-tiny",
	"testdata/qwen3_5-tiny-normw",
	"testdata/qwen3_5_moe-tiny",
	"../testdata/qwen35-tiny",
}

// op3tCell is one TW1 cell: a fixture at a quant, a pass of K rows, from position 0 or after a primed first pass.
type op3tCell struct {
	fixture, quant string
	k              int
	primed         bool
}

func (c op3tCell) String() string {
	return fmt.Sprintf("%s/%s/K=%d/primed=%v", filepath.Base(c.fixture), c.quant, c.k, c.primed)
}

// op3tState is what a pass leaves behind: the rows it returned and the cache after it.
type op3tState struct {
	rows []float32
	kv   [][]float32 // per layer: its K rows then its V rows
	conv [][]float32 // per DeltaNet layer: its conv window, oldest first, flattened
	s    [][]float32 // per DeltaNet layer: its recurrent state
	pos  int
}

// op3tQwen35Run runs one cell's passes through the walker (twin false) or the twin, and returns the second pass's state.
func op3tQwen35Run(t *testing.T, m *Model, c op3tCell, twin bool) op3tState {
	t.Helper()
	qwen35TwinForTest = twin
	defer func() { qwen35TwinForTest = false }()
	vocab := m.w.arch.VocabSize
	ids := func(n, salt int) []int {
		out := make([]int, n)
		for i := range out {
			out[i] = (i*37 + salt) % vocab
		}
		return out
	}
	const prime = 9
	cache := m.NewCache(prime + c.k)
	ctx := context.Background()
	if c.primed {
		if !m.qwen35BatchN(prime, cache) {
			t.Fatalf("%v: the batched path does not apply to the priming pass", c)
		}
		if _, err := m.runLayersQwen35N(ctx, m.embedN(ids(prime, 5)), cache); err != nil {
			t.Fatal(err)
		}
	}
	if !m.qwen35BatchN(c.k, cache) {
		t.Fatalf("%v: the batched path does not apply", c)
	}
	h, err := m.runLayersQwen35N(ctx, m.embedN(ids(c.k, 11)), cache)
	if err != nil {
		t.Fatal(err)
	}
	st := op3tState{rows: append([]float32(nil), h...), pos: cache.Pos()}
	for l := range m.w.arch.NumLayers {
		st.kv = append(st.kv, append(append([]float32(nil), cache.Keys(l)...), cache.Vals(l)...))
		if d := cache.delta[l]; d != nil {
			var w []float32
			for _, r := range d.convWin {
				w = append(w, r...)
			}
			st.conv = append(st.conv, w)
			st.s = append(st.s, append([]float32(nil), d.s...))
		}
	}
	return st
}

// op3tDiff names every part of got that differs from want in any bit.
func op3tDiff(want, got op3tState) []string {
	var d []string
	same := func(a, b []float32) bool {
		if len(a) != len(b) {
			return false
		}
		for i := range a {
			if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
				return false
			}
		}
		return true
	}
	if !same(want.rows, got.rows) {
		d = append(d, "rows")
	}
	if want.pos != got.pos {
		d = append(d, fmt.Sprintf("pos %d vs %d", want.pos, got.pos))
	}
	for l := range want.kv {
		if !same(want.kv[l], got.kv[l]) {
			d = append(d, fmt.Sprintf("layer %d K/V", l))
		}
	}
	for i := range want.conv {
		if !same(want.conv[i], got.conv[i]) {
			d = append(d, fmt.Sprintf("DeltaNet %d conv window", i))
		}
		if !same(want.s[i], got.s[i]) {
			d = append(d, fmt.Sprintf("DeltaNet %d state", i))
		}
	}
	return d
}

// op3tQwen35Cells runs every TW1 cell, walker against twin, and calls check with each cell and its differences. It returns how
// many cells ran; a fixture not on disk is logged NOT RUN.
func op3tQwen35Cells(t *testing.T, check func(c op3tCell, diffs []string)) int {
	ran := 0
	for _, fx := range op3tQwen35Fixtures {
		if _, err := os.Stat(filepath.Join(fx, "model.safetensors")); errors.Is(err, fs.ErrNotExist) {
			t.Logf("NOT RUN: %s (no checkpoint on disk)", fx)
			continue
		}
		for _, quant := range []string{"", "int8int8", "int4"} {
			m, err := Load(fx, Options{Quant: quant})
			if err != nil {
				t.Fatalf("%s/%s: %v", fx, quant, err)
			}
			for _, k := range []int{2, 17, 64} {
				for _, primed := range []bool{false, true} {
					c := op3tCell{fixture: fx, quant: quant, k: k, primed: primed}
					check(c, op3tDiff(op3tQwen35Run(t, m, c, true), op3tQwen35Run(t, m, c, false)))
					ran++
				}
			}
			m.Close()
		}
	}
	return ran
}

// TestOP3t_qwen35WalkerMatchesTwin is OP3t's TW1 for qwen3.5: runLayersQwen35N on the batched walker against the twin it
// replaced, bit for bit, on every tracked tiny fixture at f32, int8int8 and int4, for passes of 2, 17 and 64 rows from position 0
// and after a primed first pass. It compares the rows returned and the cache after: every layer's K/V, each DeltaNet layer's
// conv window and recurrent state, and the position.
func TestOP3t_qwen35WalkerMatchesTwin(t *testing.T) {
	ran := op3tQwen35Cells(t, func(c op3tCell, diffs []string) {
		if len(diffs) > 0 {
			t.Errorf("%v: DIFFERENT: %v", c, diffs)
		}
	})
	if ran == 0 {
		t.Fatal("no cell ran: no qwen3.5 tiny fixture on disk")
	}
	t.Logf("TW1 qwen3.5: %d cells compared", ran)
}

// TestOP3t_qwen35Plants is OP3t's TW3 for qwen3.5: each plant turns TW1 red on at least one cell. The conv-window plant can only
// show on a primed cell (an unprimed pass starts from an empty window), so it must be red there and nowhere else.
func TestOP3t_qwen35Plants(t *testing.T) {
	for _, p := range []struct {
		name  string
		plant op3tPlant
	}{
		{"attention gate skipped", op3tPlantNoAttnGate},
		{"conv window dropped", op3tPlantDropConvWindow},
	} {
		t.Run(p.name, func(t *testing.T) {
			op3tPlantForTest = p.plant
			defer func() { op3tPlantForTest = op3tPlantNone }()
			red, redUnprimed := 0, 0
			ran := op3tQwen35Cells(t, func(c op3tCell, diffs []string) {
				if len(diffs) > 0 {
					red++
					if !c.primed {
						redUnprimed++
					}
				}
			})
			t.Logf("plant %q: red on %d of %d cells (%d unprimed)", p.name, red, ran, redUnprimed)
			if red == 0 {
				t.Errorf("plant %q: TW1 stayed green on all %d cells", p.name, ran)
			}
			if p.plant == op3tPlantDropConvWindow && redUnprimed != 0 {
				t.Errorf("plant %q: red on %d unprimed cells, where the window is empty anyway", p.name, redUnprimed)
			}
		})
	}
}
