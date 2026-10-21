package decoder

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// B27's tier 0 (docs/queue-engineering.md, B27, registered before any code): a quantized Nemotron-H load holds its Mamba
// projections at int8, its .giw carries them so, the codes are the ones the GPU backends made before, and v17 refuses an
// encoding it does not know by name.

const b27Tiny = "../testdata/nemotron-tiny"

// b27Load loads nemotron-tiny at quant, with the f32 projections kept when keepF32 (the f32 arm, and the plant).
func b27Load(t *testing.T, quant string, keepF32 bool) *Model {
	t.Helper()
	prev := mambaKeepF32ForTest
	mambaKeepF32ForTest = keepF32
	defer func() { mambaKeepF32ForTest = prev }()
	m, err := Load(b27Tiny, Options{Quant: quant})
	if err != nil {
		t.Fatalf("load %s at %s: %v", b27Tiny, quant, err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

// b27MambaLayers lists the layers holding a Mamba mixer.
func b27MambaLayers(t *testing.T, m *Model) []int {
	t.Helper()
	var ls []int
	for i := range m.w.Layers {
		if m.w.Layers[i].mamba != nil {
			ls = append(ls, i)
		}
	}
	if len(ls) == 0 {
		t.Fatal("nemotron-tiny has no Mamba layer")
	}
	return ls
}

// b27Dims is the two projections' [rows, cols].
func b27Dims(t *testing.T, m *Model) (d struct{ in, out [2]int }) {
	t.Helper()
	nh, hd, ds, ng, _, _, _, ok := m.NemotronResidentParams()
	if !ok {
		t.Fatal("not a Nemotron-H model")
	}
	hidden, dInner := m.w.arch.HiddenDim, nh*hd
	d.in = [2]int{2*dInner + 2*ng*ds + nh, hidden}
	d.out = [2]int{hidden, dInner}
	return d
}

// b27ProjBytes is what a bundle spends on the Mamba projections: its size less the size of the same weights with the
// projections emptied.
func b27ProjBytes(t *testing.T, w *Weights) int {
	t.Helper()
	full, err := SerializeWeights(w, "b27")
	if err != nil {
		t.Fatal(err)
	}
	cp := *w
	cp.Layers = slices.Clone(w.Layers)
	for i := range cp.Layers {
		if mw := cp.Layers[i].mamba; mw != nil {
			e := *mw
			e.inProj, e.outProj, e.inQ, e.outQ = nil, nil, linalg.WeightMat{}, linalg.WeightMat{}
			cp.Layers[i].mamba = &e
		}
	}
	empty, err := SerializeWeights(&cp, "b27")
	if err != nil {
		t.Fatal(err)
	}
	return len(full) - len(empty)
}

// b27SizeCheck is the bundle gate: the projections take at most their int8 size (codes, row scales, and 64 bytes a matrix
// for its record and padding) and at least their codes. It returns the measured bytes and the bounds.
func b27SizeCheck(t *testing.T, m *Model) (got, lo, hi int, ok bool) {
	t.Helper()
	for _, l := range b27MambaLayers(t, m) {
		mw := m.w.Layers[l].mamba
		for _, d := range [][2]int{{mw.inQ.Rows(), mw.inQ.Cols()}, {mw.outQ.Rows(), mw.outQ.Cols()}} {
			if d[0] == 0 { // an f32 projection: its int8 size is still the bound
				continue
			}
			lo += d[0] * d[1]
			hi += d[0]*d[1] + 4*d[0] + 64
		}
		if mw.inQ.Rows() == 0 {
			np := b27Dims(t, m)
			lo += np.in[0]*np.in[1] + np.out[0]*np.out[1]
			hi += np.in[0]*np.in[1] + 4*np.in[0] + 64 + np.out[0]*np.out[1] + 4*np.out[0] + 64
		}
	}
	got = b27ProjBytes(t, m.w)
	return got, lo, hi, got >= lo && got <= hi
}

// TestB27_bundleHoldsInt8MambaProjections: an int4 nemotron-tiny holds both projections of every Mamba layer as int8 (and
// no f32 copy), its .giw round-trips them so, and the bundle spends their int8 size on them. With the f32 copies kept (the
// planted defect) the size check is red.
func TestB27_bundleHoldsInt8MambaProjections(t *testing.T) {
	m := b27Load(t, "int4", false)
	for _, l := range b27MambaLayers(t, m) {
		mw := m.w.Layers[l].mamba
		if _, _, _, ok := mw.inQ.Int8(); !ok || mw.inProj != nil || mw.outProj != nil {
			t.Fatalf("layer %d: int8 in_proj %v, f32 copies %d and %d values", l, ok, len(mw.inProj), len(mw.outProj))
		}
	}
	blob, err := SerializeWeights(m.w, "b27")
	if err != nil {
		t.Fatal(err)
	}
	w2, err := LoadSerializedWeights(blob)
	if err != nil {
		t.Fatalf("load the bundle: %v", err)
	}
	for _, l := range b27MambaLayers(t, m) {
		a, b := m.w.Layers[l].mamba, w2.Layers[l].mamba
		for _, p := range [][2]*linalg.WeightMat{{&a.inQ, &b.inQ}, {&a.outQ, &b.outQ}} {
			qa, sa, _, _ := p[0].Int8()
			qb, sb, _, ok := p[1].Int8()
			if !ok || !slices.Equal(qa, qb) || !slices.Equal(sa, sb) || b.inProj != nil {
				t.Errorf("layer %d: the bundle's projection is not the load's int8 (int8 %v)", l, ok)
			}
		}
	}
	got, lo, hi, ok := b27SizeCheck(t, m)
	t.Logf("the bundle spends %d bytes on the Mamba projections; their int8 size is %d..%d", got, lo, hi)
	if !ok {
		t.Errorf("the Mamba projections take %d bytes of the bundle, outside their int8 size %d..%d", got, lo, hi)
	}
	planted := b27Load(t, "int4", true)
	pgot, plo, phi, pok := b27SizeCheck(t, planted)
	t.Logf("planted (f32 kept): %d bytes against %d..%d", pgot, plo, phi)
	if pok {
		t.Error("with the f32 projections kept the size check still passes: it cannot see the defect")
	}
}

// TestB27_int8MatchesTheUploadBefore: the int8 codes and row scales an int4 load holds, and its bundle round-trip, equal
// linalg.QuantizeRowsInt8 of the f32 load's weights byte for byte: what metal/mamba.go and gpu/residency.go uploaded before.
func TestB27_int8MatchesTheUploadBefore(t *testing.T) {
	f := b27Load(t, "", false)
	q := b27Load(t, "int4", false)
	blob, err := SerializeWeights(q.w, "b27")
	if err != nil {
		t.Fatal(err)
	}
	w2, err := LoadSerializedWeights(blob)
	if err != nil {
		t.Fatal(err)
	}
	np := b27Dims(t, f)
	for _, l := range b27MambaLayers(t, f) {
		fw := f.w.Layers[l].mamba
		if fw.inQ.Rows() != 0 || fw.inProj == nil {
			t.Fatalf("layer %d: an f32 load holds int8 projections", l)
		}
		for _, c := range []struct {
			name string
			f32  []float32
			rows int
			cols int
			got  [2]*linalg.WeightMat
		}{
			{"in_proj", fw.inProj, np.in[0], np.in[1], [2]*linalg.WeightMat{&q.w.Layers[l].mamba.inQ, &w2.Layers[l].mamba.inQ}},
			{"out_proj", fw.outProj, np.out[0], np.out[1], [2]*linalg.WeightMat{&q.w.Layers[l].mamba.outQ, &w2.Layers[l].mamba.outQ}},
		} {
			wantQ, wantS := linalg.QuantizeRowsInt8(c.f32, c.rows, c.cols)
			for i, wm := range c.got {
				gq, gs, _, ok := wm.Int8()
				if !ok || !slices.Equal(gq, wantQ) || !slices.EqualFunc(gs, wantS, func(a, b float32) bool { return math.Float32bits(a) == math.Float32bits(b) }) {
					t.Errorf("layer %d %s (%s): not the codes and scales the backends uploaded before", l, c.name, []string{"load", "bundle"}[i])
				}
			}
		}
	}
}

// TestB27_unknownEncodingRefusedByName: a standalone array or a weight matrix whose encoding this build does not read is a
// *SerializeError naming the encoding, the error a sidecar's freshness check rebuilds on.
func TestB27_unknownEncodingRefusedByName(t *testing.T) {
	for _, c := range []struct {
		name string
		read func(r *giwReader)
	}{
		{"array", func(r *giwReader) { r.f32() }},
		{"matrix", func(r *giwReader) { r.weightMat() }},
	} {
		r := &giwReader{data: []byte{9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}
		c.read(r)
		var se *SerializeError
		if !errors.As(r.err, &se) || !strings.Contains(se.Error(), "encoding 9") {
			t.Errorf("%s with encoding 9: err %v, want a *SerializeError naming encoding 9", c.name, r.err)
		}
	}
	r := &giwReader{data: []byte{giwEncF32, 1, 0, 0, 0, 0, 0, 128, 63}}
	if got := r.f32(); r.err != nil || len(got) != 1 || got[0] != 1 {
		t.Errorf("a known encoding: %v, err %v", got, r.err)
	}
}

// b27Lockstep runs the int8-Mamba model against the f32-Mamba one (same quant otherwise) in greedy lockstep off the f32
// arm for steps tokens, and returns the worst logit cosine from position 2 and the argmax agreements.
func b27Lockstep(t *testing.T, ref, got *Model, seed []int, steps int) (minCos float64, agree int) {
	t.Helper()
	cr, cg := ref.NewCache(0), got.NewCache(0)
	minCos, tok := 1, seed[0]
	for i := range steps {
		lr, err := ref.forward(tok, cr)
		if err != nil {
			t.Fatal(err)
		}
		lg, err := got.forward(tok, cg)
		if err != nil {
			t.Fatal(err)
		}
		if i >= 2 {
			minCos = min(minCos, float64(logitCosine(lg, lr)))
		}
		if argmax(lr) == argmax(lg) {
			agree++
		}
		if i+1 < len(seed) {
			tok = seed[i+1]
		} else {
			tok = argmax(lr)
		}
	}
	return minCos, agree
}

// TestB27_cpuFidelityTiny is B27's CPU fidelity bar on nemotron-tiny at int4: the int8-Mamba CPU against the f32-Mamba CPU,
// 32 steps of greedy lockstep, min logit cosine >= 0.999 from position 2 and argmax agreement >= 30/32.
func TestB27_cpuFidelityTiny(t *testing.T) {
	ref := b27Load(t, "int4", true)
	got := b27Load(t, "int4", false)
	_, _, _, _, _, _, vocab := ref.Dims()
	minCos, agree := b27Lockstep(t, ref, got, []int{1, 17 % vocab, 113 % vocab, 29 % vocab, 71 % vocab, 5 % vocab}, 32)
	t.Logf("nemotron-tiny at int4, int8 Mamba against f32: min cosine %.6f, argmax %d/32", minCos, agree)
	if minCos < 0.999 || agree < 30 {
		t.Errorf("below B27's bar (min cosine >= 0.999, argmax >= 30/32)")
	}
}

// TestB27_cpuFidelitySlice is B27's CPU fidelity bar on the three-layer slice of NVIDIA-Nemotron-Nano-9B-v2 (layers 0, 1
// and 14 of the Q8_0 GGUF; docs/tasks/task-forward-architecture-2026-10.md §4.7) at int4: the int8-Mamba CPU against the
// f32-Mamba CPU, 32 steps of greedy lockstep, min logit cosine >= 0.99 from position 2 and argmax agreement >= 28/32.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -run '^TestB27_cpuFidelitySlice$' -v ./decoder/
func TestB27_cpuFidelitySlice(t *testing.T) {
	requireHeavyModel(t)
	path := os.Getenv("GOINFER_NEMOTRON_SLICE_GGUF")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", "nemotron", "nemotron-nano-9b-v2-Q8_0.slice-0-1-14.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no slice at %s", path)
	}
	load := func(keepF32 bool) *Model {
		prev := mambaKeepF32ForTest
		mambaKeepF32ForTest = keepF32
		defer func() { mambaKeepF32ForTest = prev }()
		m, err := Load(path, Options{Quant: "int4", ResidentContext: 64})
		if err != nil {
			t.Fatalf("load the slice (f32 Mamba %v): %v", keepF32, err)
		}
		t.Cleanup(func() { m.Close() })
		return m
	}
	ref, got := load(true), load(false)
	if in, _ := got.MambaProjInt8(0); in == nil {
		t.Fatal("the int8 arm's layer 0 holds no int8 projections")
	}
	minCos, agree := b27Lockstep(t, ref, got, []int{1, 17, 113, 29, 71, 5}, 32)
	t.Logf("the slice at int4, int8 Mamba against f32: min cosine %.6f, argmax %d/32", minCos, agree)
	if minCos < 0.99 || agree < 28 {
		t.Errorf("below B27's bar (min cosine >= 0.99, argmax >= 28/32)")
	}
}
