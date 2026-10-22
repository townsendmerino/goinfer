package decoder

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestMEM11_lfm2ResidentBytes is B16-G0 (docs/tasks/task-memory-campaign-2026-10.md, MEM11-bf16 gates): LFM2.5-2.6B from
// ~/models at int4 on the CPU, its ResidentWeightBytes and the short-conv projections' share. Not timed; a real checkpoint.
//
//	GOINFER_MEM11_LFM2=1 go test -count=1 -run '^TestMEM11_lfm2ResidentBytes$' -v ./decoder/
func TestMEM11_lfm2ResidentBytes(t *testing.T) {
	if os.Getenv("GOINFER_MEM11_LFM2") == "" {
		t.Skip("set GOINFER_MEM11_LFM2=1 (loads LFM2.5-2.6B from ~/models)")
	}
	home, _ := os.UserHomeDir()
	m, err := Load(filepath.Join(home, "models", "lfm2.5-2.6b"), Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	total := m.ResidentWeightBytes()
	var conv int64
	nConv := 0
	for i := range m.w.Layers {
		if sc := m.w.Layers[i].shortConv; sc != nil {
			conv += shortConvProjBytes(sc)
			nConv++
		}
	}
	fmt.Fprintf(os.Stderr, "[B16-G0] LFM2.5-2.6B int4: ResidentWeightBytes %.3f GB; %d short-conv layers' in/out projections %.3f GB (%.1f%%)\n",
		float64(total)/1e9, nConv, float64(conv)/1e9, 100*float64(conv)/float64(total))
}

// shortConvProjBytes is the bytes a short-conv layer's in and out projections hold.
func shortConvProjBytes(sc *shortConvWeights) int64 {
	return wmBytes(&sc.inProj) + wmBytes(&sc.outProj)
}

// mem11Logits decodes 16 fixed tokens from a fresh cache and returns every position's logits.
func mem11Logits(t *testing.T, m *Model) [][]float32 {
	t.Helper()
	_, _, _, _, _, _, vocab := m.Dims()
	cache := m.NewCache(32)
	var out [][]float32
	for i := range 16 {
		lg, err := m.forward((i*37+11)%vocab, cache)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg...))
	}
	return out
}

func mem11SameBits(a, b [][]float32) (diff int) {
	for i := range a {
		for j := range a[i] {
			if math.Float32bits(a[i][j]) != math.Float32bits(b[i][j]) {
				diff++
			}
		}
	}
	return diff
}

// TestMEM11_lfm2BF16Islands is B16-G1 and B16-G2 for LFM2 (docs/tasks/task-memory-campaign-2026-10.md, MEM11-bf16 gates): the
// tiny fixture's short-conv projections rounded to bf16 and held as f32 ("off") against the same values through islandMat
// ("on", the bf16 kind): logits over 16 tokens bit-identical, each projection's bytes halved in ResidentWeightBytes, and a .giw
// serialization round trip of the "on" weights that reads them as bf16 again and decodes the same logits.
func TestMEM11_lfm2BF16Islands(t *testing.T) {
	m, err := Load("../testdata/lfm2-tiny", Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	var convs []*shortConvWeights
	for l := range m.w.Layers {
		if sc := m.w.Layers[l].shortConv; sc != nil {
			convs = append(convs, sc)
			if k := sc.inProj.Kind(); k != "f32" {
				t.Fatalf("the committed fixture's in_proj is %s: its f32 islands are bf16-exact, so B16-G3 would not cover the f32 rule", k)
			}
		}
	}
	if len(convs) == 0 {
		t.Fatal("the fixture has no conv layer")
	}
	round := func(w *linalg.WeightMat) []float32 {
		f := slices.Clone(islandF32(w))
		for i, x := range f {
			f[i] = linalg.BF16ToF32(linalg.F32ToBF16(x))
		}
		return f
	}
	type pair struct{ in, out []float32 }
	rounded := make([]pair, len(convs))
	for i, sc := range convs {
		rounded[i] = pair{round(&sc.inProj), round(&sc.outProj)}
	}
	set := func(on bool) {
		for i, sc := range convs {
			r, c := sc.inProj.Rows(), sc.inProj.Cols()
			ro, co := sc.outProj.Rows(), sc.outProj.Cols()
			if on {
				sc.inProj, sc.outProj = islandMat(slices.Clone(rounded[i].in), r, c), islandMat(slices.Clone(rounded[i].out), ro, co)
			} else {
				sc.inProj, sc.outProj = linalg.WrapF32(slices.Clone(rounded[i].in), r, c), linalg.WrapF32(slices.Clone(rounded[i].out), ro, co)
			}
		}
	}
	set(false)
	offBytes, off := m.ResidentWeightBytes(), mem11Logits(t, m)
	set(true)
	for _, sc := range convs {
		if sc.inProj.Kind() != "bf16" || sc.outProj.Kind() != "bf16" {
			t.Fatalf("islandMat kept a bf16-exact projection as %s / %s", sc.inProj.Kind(), sc.outProj.Kind())
		}
	}
	onBytes, on := m.ResidentWeightBytes(), mem11Logits(t, m)
	var island int64
	for i := range rounded {
		island += 4 * int64(len(rounded[i].in)+len(rounded[i].out))
	}
	d := mem11SameBits(off, on)
	fmt.Fprintf(os.Stderr, "[B16-G1] lfm2-tiny: %d conv layers; logits over 16 tokens: %d differ between f32 and bf16 islands; bytes %d -> %d (islands %d -> %d)\n",
		len(convs), d, offBytes, onBytes, island, island/2)
	if d != 0 {
		t.Errorf("%d logits differ between the f32 and bf16 islands", d)
	}
	if offBytes-onBytes != island/2 {
		t.Errorf("ResidentWeightBytes fell by %d, want %d (half the islands)", offBytes-onBytes, island/2)
	}
	blob, err := SerializeWeights(m.w, "mem11-lfm2")
	if err != nil {
		t.Fatalf("SerializeWeights: %v", err)
	}
	w2, err := LoadSerializedWeights(blob)
	if err != nil {
		t.Fatalf("LoadSerializedWeights: %v", err)
	}
	for l := range w2.Layers {
		sc := w2.Layers[l].shortConv
		if sc == nil {
			continue
		}
		if sc.inProj.Kind() != "bf16" || sc.outProj.Kind() != "bf16" {
			t.Errorf("layer %d: the round-tripped projections are %s / %s, want bf16", l, sc.inProj.Kind(), sc.outProj.Kind())
		}
		m.w.Layers[l].shortConv = sc // decode through the round-tripped mixers
	}
	if d := mem11SameBits(on, mem11Logits(t, m)); d != 0 {
		t.Errorf("the round-tripped bundle's logits: %d differ from the direct load's", d)
	}
}
