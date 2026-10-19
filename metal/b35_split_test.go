//go:build darwin

package metal

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestB35SplitInvariance is B35's gate on Metal (docs/queue-engineering.md, B35): a Gated-DeltaNet prompt prefilled in several
// PrefillLast calls, cut at a save point, must leave bit for bit what one call leaves: the last row's logits, every DeltaNet layer's
// conv window and recurrent state, and every attention layer's K/V rows. Metal runs a call's rows in one command buffer, so a call
// boundary is the only place the state exists between rows, and the decoder may put one at a save point only if this holds. The save
// points: mid-prompt (an arbitrary row), on a 64-row block boundary with a second cut after it, and a 4-row tail (Qwen3.5's empty
// think block). A plant (the state reset between the calls) must differ, so the comparison can see a difference.
//
// It runs on testdata/qwen35-tiny; GOINFER_B35_METAL_MODEL points it at a real checkpoint (~/models/qwen3.5-0.8b).
func TestB35SplitInvariance(t *testing.T) {
	path := os.Getenv("GOINFER_B35_METAL_MODEL")
	if path == "" {
		path = filepath.Join("..", "testdata", "qwen35-tiny")
	}
	if _, err := os.Stat(filepath.Join(path, "config.json")); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "int4", ResidentContext: 1024})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	prevDnet := dnetPrefillOn
	dnetPrefillOn = true
	defer func() { dnetPrefillOn = prevDnet }()
	r, err := buildResident(m)
	if err != nil {
		t.Fatalf("resident: %v", err)
	}
	defer r.Close()
	if !r.recurrentPrefillOK() {
		t.Skip("this model's DeltaNet layers do not take Metal's batched prefill")
	}
	const N = 160
	_, _, _, _, _, _, vocab := m.Dims()
	ids := make([]int, N)
	for i := range ids {
		ids[i] = (i*37 + 5) % vocab
	}
	embs := getEmbs(r, ids)
	words := func() (out [][]uint32) {
		for l := range r.layers {
			L := &r.layers[l]
			if L.delta != nil {
				out = append(out, append([]uint32(nil), L.delta.win.U32s()...), append([]uint32(nil), L.delta.state.U32s()...))
				continue
			}
			if L.geom == nil || l >= len(r.kc) {
				continue
			}
			n := N * L.geom.kvDim / 2 // f16 halves, read as words
			out = append(out, append([]uint32(nil), r.kc[l].U32s()[:n]...), append([]uint32(nil), r.vc[l].U32s()[:n]...))
		}
		return
	}
	run := func(plant bool, cuts ...int) ([]float32, [][]uint32) {
		r.resetDeltaNet()
		var last []float32
		from := 0
		for _, to := range append(cuts, N) {
			if plant && from > 0 {
				r.resetDeltaNet()
			}
			last = append([]float32(nil), r.PrefillLast(embs[from:to], from)...)
			from = to
		}
		return last, words()
	}
	wantLg, wantW := run(false)
	nDelta := 0
	for l := range r.layers {
		if r.layers[l].delta != nil {
			nDelta++
		}
	}
	if nDelta == 0 || len(wantW) == 2*nDelta {
		t.Fatalf("compared %d DeltaNet layers and %d attention layers: the model must have both", nDelta, (len(wantW)-2*nDelta)/2)
	}
	for _, c := range []struct {
		name  string
		cuts  []int
		plant bool
	}{
		{"mid-prompt save at row 97", []int{97}, false},
		{"on a 64-row block boundary, then row 128", []int{64, 128}, false},
		{"a 4-row tail (B35's shape)", []int{N - 4}, false},
		{"plant: state reset between the calls at row 97", []int{97}, true},
	} {
		gotLg, gotW := run(c.plant, c.cuts...)
		lg, st := 0, 0
		for i := range wantLg {
			if wantLg[i] != gotLg[i] {
				lg++
			}
		}
		for i := range wantW {
			for j := range wantW[i] {
				if wantW[i][j] != gotW[i][j] {
					st++
				}
			}
		}
		t.Logf("%s: %d of %d logits and %d state/window/K/V words differ from one pass", c.name, lg, len(wantLg), st)
		switch {
		case c.plant && lg == 0 && st == 0:
			t.Errorf("%s: resetting the state between the calls changed nothing, so the comparison cannot see a difference", c.name)
		case !c.plant && (lg != 0 || st != 0):
			t.Errorf("%s: a call boundary at a save point is not bit-neutral on Metal", c.name)
		}
	}
}
