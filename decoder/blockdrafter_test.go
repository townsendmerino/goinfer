package decoder

import (
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestBlockDrafterWeights_matchesTrunk checks the interface reports what the drafter actually
// holds, on a real checkpoint.
//
// The compile-time assertions in blockdrafter.go prove both families SATISFY the interface; they
// prove nothing about the values. A DrafterGeometry that reported the wrong head count or handed
// back layer 0's weights for every index would satisfy the compiler and produce a resident
// drafter that runs, allocates plausibly, and drafts noise — exactly the failure this program has
// hit repeatedly (a wrong tensor that still has the right shape).
//
// So this cross-checks every field against the trunk's own state, and specifically requires that
// DIFFERENT LAYERS RETURN DIFFERENT WEIGHTS, which is the assertion an index bug fails.
func TestBlockDrafterWeights_matchesTrunk(t *testing.T) {
	dir := assetPath(t, "GOINFER_DFLASH_F32")
	d, err := LoadDFlashDrafter(dir)
	if err != nil {
		t.Fatalf("LoadDFlashDrafter: %v", err)
	}
	defer d.Close()

	var w BlockDrafterWeights = d
	g := w.DrafterGeometry()
	for _, c := range []struct {
		name     string
		got, exp int
	}{
		{"Layers", g.Layers, len(d.layers)},
		{"Hidden", g.Hidden, d.hidden},
		{"NumHeads", g.NumHeads, d.nHeads},
		{"NumKVHeads", g.NumKVHeads, d.nKV},
		{"HeadDim", g.HeadDim, d.headDim},
		{"Intermediate", g.Intermediate, d.inter},
		{"BlockSize", w.BlockSize(), d.blockSize},
	} {
		if c.got != c.exp {
			t.Errorf("%s = %d, want %d", c.name, c.got, c.exp)
		}
	}
	if g.NormEps != d.normEps {
		t.Errorf("NormEps = %v, want %v", g.NormEps, d.normEps)
	}
	if len(g.InvFreq) != len(d.invFreq) {
		t.Errorf("InvFreq len %d, want %d", len(g.InvFreq), len(d.invFreq))
	}

	// fc is the projection with no counterpart in a normal layer — the one most likely to be
	// mis-wired, and the one a backend cannot infer from geometry alone.
	if fc := w.DrafterFC(); fc.Rows() != d.hidden || fc.Cols() != d.fc.Cols() {
		t.Errorf("FC is %dx%d, want %dx%d", fc.Rows(), fc.Cols(), d.hidden, d.fc.Cols())
	}
	if got := len(w.DrafterHiddenNorm()); got != len(d.hiddenNorm) {
		t.Errorf("HiddenNorm len %d, want %d", got, len(d.hiddenNorm))
	}
	if got := len(w.DrafterFinalNorm()); got != len(d.finalNorm) {
		t.Errorf("FinalNorm len %d, want %d", got, len(d.finalNorm))
	}

	// Per-layer: shapes, and that the index is honoured. An accessor returning layer 0 for
	// every i has correct shapes everywhere and is catastrophically wrong.
	seen := map[*float32]int{}
	for i := 0; i < g.Layers; i++ {
		lw := w.DrafterLayer(i)
		if lw.Q.Rows() != d.nHeads*d.headDim {
			t.Errorf("layer %d Q rows %d, want %d", i, lw.Q.Rows(), d.nHeads*d.headDim)
		}
		if lw.K.Rows() != d.nKV*d.headDim {
			t.Errorf("layer %d K rows %d, want %d", i, lw.K.Rows(), d.nKV*d.headDim)
		}
		if lw.Gate.Rows() != d.inter {
			t.Errorf("layer %d Gate rows %d, want %d", i, lw.Gate.Rows(), d.inter)
		}
		if lw.Down.Cols() != d.inter {
			t.Errorf("layer %d Down cols %d, want %d", i, lw.Down.Cols(), d.inter)
		}
		if len(lw.InputNorm) != d.hidden {
			t.Errorf("layer %d InputNorm len %d, want %d", i, len(lw.InputNorm), d.hidden)
		}
		// identity check: each layer's norm slice must be a DISTINCT backing array
		if len(lw.InputNorm) > 0 {
			p := &lw.InputNorm[0]
			if prev, dup := seen[p]; dup {
				t.Fatalf("layer %d returns the SAME InputNorm storage as layer %d — "+
					"DrafterLayer ignores its index", i, prev)
			}
			seen[p] = i
		}
	}
	t.Logf("%d layers, hidden %d, heads %d/%d, headDim %d, inter %d, block %d, fc %dx%d — "+
		"interface agrees with the trunk on every field",
		g.Layers, g.Hidden, g.NumHeads, g.NumKVHeads, g.HeadDim, g.Intermediate,
		w.BlockSize(), w.DrafterFC().Rows(), w.DrafterFC().Cols())
}

// syntheticDrafter builds a small, entirely fabricated DFlashDrafter with nLayers layers — no
// checkpoint needed, so this exercises DrafterResidentBytesEstimate's arithmetic unconditionally
// in CI rather than only wherever GOINFER_DFLASH_F32 happens to be present. Values are zeroed;
// only shapes matter for a byte count.
func syntheticDrafter(t *testing.T, nLayers int) *DFlashDrafter {
	t.Helper()
	const hidden, nH, nKV, hd, inter, nTaps = 8, 2, 1, 4, 16, 2
	mat := func(n, k int) linalg.WeightMat { return linalg.WrapF32(make([]float32, n*k), n, k) }
	d := &DFlashDrafter{blockTrunk: blockTrunk{
		hidden: hidden, nHeads: nH, nKV: nKV, headDim: hd, inter: inter,
		fc:         mat(hidden, nTaps*hidden),
		hiddenNorm: make([]float32, hidden),
		finalNorm:  make([]float32, hidden),
	}}
	for i := 0; i < nLayers; i++ {
		d.layers = append(d.layers, dflashLayer{
			q: mat(nH*hd, hidden), k: mat(nKV*hd, hidden), v: mat(nKV*hd, hidden), o: mat(hidden, nH*hd),
			gate: mat(inter, hidden), up: mat(inter, hidden), down: mat(hidden, inter),
			inputNorm: make([]float32, hidden), postAttnNorm: make([]float32, hidden),
		})
	}
	return d
}

// wmInt8Bytes is DrafterResidentBytesEstimate's own per-matrix formula, reimplemented
// independently here so the test is not just calling the function and checking it agrees with
// itself — same discipline decoder/weightbytes_test.go's cross-checks use.
func wmInt8Bytes(n, k int) int64 { return int64(n)*int64(k) + int64(n)*4 }

// TestDrafterResidentBytesEstimate_matchesHandComputedTotal pins the arithmetic against an
// independently-summed total for a small synthetic geometry, so a future change to which
// matrices are counted (or a copy-paste layer omission) shows up as a wrong number, not a
// plausible-looking one.
func TestDrafterResidentBytesEstimate_matchesHandComputedTotal(t *testing.T) {
	const hidden, nH, nKV, hd, inter, nTaps, nLayers = 8, 2, 1, 4, 16, 2, 2
	d := syntheticDrafter(t, nLayers)

	perLayer := wmInt8Bytes(nH*hd, hidden) + wmInt8Bytes(nKV*hd, hidden)*2 + wmInt8Bytes(hidden, nH*hd) +
		wmInt8Bytes(inter, hidden)*2 + wmInt8Bytes(hidden, inter)
	want := wmInt8Bytes(hidden, nTaps*hidden) + int64(nLayers)*perLayer

	if got := DrafterResidentBytesEstimate(d); got != want {
		t.Errorf("DrafterResidentBytesEstimate = %d, want %d (hand-computed)", got, want)
	}
}

// TestDrafterResidentBytesEstimate_growsWithLayers guards the shape of the estimate, not just one
// pinned number: more layers must cost strictly more, and a drafter with zero layers (fc/norms
// only) must still return a positive figure rather than 0 — a caller pricing "nothing else is
// attaching" as 0 (Options.ExtraResidentBytes' own doc comment) must never be confused with a
// genuinely-loaded drafter that this function under-counted to zero.
func TestDrafterResidentBytesEstimate_growsWithLayers(t *testing.T) {
	zero := DrafterResidentBytesEstimate(syntheticDrafter(t, 0))
	if zero <= 0 {
		t.Fatalf("a 0-layer drafter (fc + norms only) estimated %d bytes, want > 0", zero)
	}
	one := DrafterResidentBytesEstimate(syntheticDrafter(t, 1))
	two := DrafterResidentBytesEstimate(syntheticDrafter(t, 2))
	if !(zero < one && one < two) {
		t.Errorf("estimate did not grow monotonically with layer count: 0L=%d 1L=%d 2L=%d", zero, one, two)
	}
}

// TestExtraResidentBytes_reachesTheModel closes the loop the estimate feeds: internal/serveapp's
// loadDecoder sets Options.ExtraResidentBytes before calling Load, and a backend
// (cuda/resident.go's checkWeightsFit/checkKVFits/allocSlots) reads it back off *Model, not off
// Options — so a struct-literal typo at either of Load's two &Model{...} sites would silently
// zero it and no backend-level test could ever see why a --drafter attach still ran out of room.
// Load()-based (a real tracked fixture) rather than constructing *Model directly, so this proves
// the actual production path, matching TestFitDisabled/TestResolveCtxCapFit_shortcuts' own
// "load a real model, don't pass nil" precedent (cuda/resident_cap_test.go).
func TestExtraResidentBytes_reachesTheModel(t *testing.T) {
	m, err := Load("../testdata/llama-tiny", Options{Quant: "f32", ExtraResidentBytes: 123456789})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	if got := m.ExtraResidentBytes(); got != 123456789 {
		t.Errorf("ExtraResidentBytes() = %d, want 123456789", got)
	}
}
