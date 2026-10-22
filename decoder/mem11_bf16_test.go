package decoder

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
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
	runtime.GC()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	fmt.Fprintf(os.Stderr, "[B16-G0] LFM2.5-2.6B int4: ResidentWeightBytes %.3f GB; %d short-conv layers' in/out projections %.3f GB (%.1f%%); Go heap in use after GC %.3f GB\n",
		float64(total)/1e9, nConv, float64(conv)/1e9, 100*float64(conv)/float64(total), float64(ms.HeapInuse)/1e9)
	runtime.KeepAlive(m)
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

// TestMEM11_drafterBF16Islands is B16-G1 and B16-G2 for the block drafters (DFlash; DSpark loads through the same stIsland):
// a two-layer trunk with bf16-exact random weights, built once with f32 WeightMats and once with the bf16 kind, drafts the same
// block bit for bit, its fc projection matches too, and the bf16 trunk's projections hold half the bytes. stIsland's three
// F32 and BF16 branches are pinned on a safetensors file: F32 stays the f32 view, BF16 holds the bf16 bits.
func TestMEM11_drafterBF16Islands(t *testing.T) {
	const h, nH, nKV, hd, inter, ctxLen, block = 64, 4, 2, 16, 96, 7, 5
	rng := rand.New(rand.NewPCG(11, 12))
	rnd := func(n int, scale float32) []float32 {
		f := make([]float32, n)
		for i := range f {
			f[i] = linalg.BF16ToF32(linalg.F32ToBF16(scale * float32(rng.NormFloat64())))
		}
		return f
	}
	ones := func(n int) []float32 { return slices.Repeat([]float32{1}, n) }
	build := func(bf16 bool) *blockTrunk {
		rng = rand.New(rand.NewPCG(11, 12)) // the same weights for both builds
		mk := func(rows, cols int) linalg.WeightMat {
			f := rnd(rows*cols, 0.08)
			if bf16 {
				return islandMat(f, rows, cols)
			}
			return linalg.WrapF32(f, rows, cols)
		}
		d := &blockTrunk{hidden: h, nHeads: nH, nKV: nKV, headDim: hd, inter: inter, normEps: 1e-6,
			invFreq: computeInvFreq(1e6, hd, nil), fc: mk(h, 2*h), hiddenNorm: ones(h), finalNorm: ones(h)}
		for range 2 {
			d.layers = append(d.layers, dflashLayer{q: mk(nH*hd, h), k: mk(nKV*hd, h), v: mk(nKV*hd, h), o: mk(h, nH*hd),
				gate: mk(inter, h), up: mk(inter, h), down: mk(h, inter), qNorm: ones(hd), kNorm: ones(hd), inputNorm: ones(h), postAttnNorm: ones(h)})
		}
		return d
	}
	f32, bf := build(false), build(true)
	if k := bf.layers[0].q.Kind(); k != "bf16" {
		t.Fatalf("islandMat kept bf16-exact weights as %s", k)
	}
	rows := func(n, w int) [][]float32 {
		out := make([][]float32, n)
		for i := range out {
			out[i] = rnd(w, 1)
		}
		return out
	}
	fused, blockIn := rows(ctxLen, h), rows(block, h)
	be := &cpuBackend{}
	a, err := f32.DraftBlock(be, fused, blockIn)
	if err != nil {
		t.Fatal(err)
	}
	b, err := bf.DraftBlock(be, fused, blockIn)
	if err != nil {
		t.Fatal(err)
	}
	taps := rnd(2*h, 1)
	fa, fb := make([]float32, h), make([]float32, h)
	matmul(be, &f32.fc, taps, fa, 1)
	matmul(be, &bf.fc, taps, fb, 1)
	d := mem11SameBits(a, b) + mem11SameBits([][]float32{fa}, [][]float32{fb})
	var bytesF, bytesB int64
	for i := range f32.layers {
		for _, p := range [][2]*linalg.WeightMat{{&f32.layers[i].q, &bf.layers[i].q}, {&f32.layers[i].down, &bf.layers[i].down}} {
			bytesF, bytesB = bytesF+wmBytes(p[0]), bytesB+wmBytes(p[1])
		}
	}
	fmt.Fprintf(os.Stderr, "[B16-G1] block drafter trunk: %d of %d values differ between f32 and bf16 weights; q+down bytes %d -> %d\n", d, block*h+h, bytesF, bytesB)
	if d != 0 {
		t.Errorf("%d drafted values differ between the f32 and bf16 trunks", d)
	}
	if bytesB*2 != bytesF {
		t.Errorf("bf16 projections hold %d bytes, want half of %d", bytesB, bytesF)
	}

	// stIsland on a safetensors file: an F32 tensor stays its f32 view, a BF16 one holds its bf16 bits.
	vals := rnd(8*4, 1)
	raw16 := make([]byte, 2*len(vals))
	for i, v := range vals {
		u := linalg.F32ToBF16(v)
		raw16[2*i], raw16[2*i+1] = byte(u), byte(u>>8)
	}
	st := loadST(t, buildSafetensors(map[string]stTyped{"f32": {"F32", []int{8, 4}, f32Bytes(vals)}, "bf16": {"BF16", []int{8, 4}, raw16}}))
	defer st.Close()
	for name, want := range map[string]string{"f32": "f32", "bf16": "bf16"} {
		m, err := stIsland(st, name, 8, 4)
		if err != nil {
			t.Fatalf("stIsland(%s): %v", name, err)
		}
		if m.Kind() != want {
			t.Errorf("stIsland(%s) = %s, want %s", name, m.Kind(), want)
		}
		row := make([]float32, 4)
		for r := range 8 {
			m.Row(r, row)
			if !slices.Equal(row, vals[r*4:(r+1)*4]) {
				t.Fatalf("stIsland(%s) row %d = %v, want %v", name, r, row, vals[r*4:(r+1)*4])
			}
		}
	}
}

// TestMEM11_islandMatRule pins islandMat's rule: bf16 when every value is exactly a bf16 value, f32 when one is not, and f32
// for a NaN whose payload bf16 would not keep.
func TestMEM11_islandMatRule(t *testing.T) {
	exact := []float32{1, -2.5, 0, 3.140625, float32(math.Inf(1)), -0}
	if k := islandMat(slices.Clone(exact), 2, 3); k.Kind() != "bf16" {
		t.Errorf("bf16-exact values held as %s", k.Kind())
	}
	notExact := slices.Clone(exact)
	notExact[3] = 3.14159
	if k := islandMat(notExact, 2, 3); k.Kind() != "f32" {
		t.Errorf("a value bf16 cannot hold, held as %s", k.Kind())
	}
	nan := slices.Clone(exact)
	nan[0] = math.Float32frombits(0x7fc00001) // a payload in the low 16 bits
	if k := islandMat(nan, 2, 3); k.Kind() != "f32" {
		t.Errorf("a NaN with a low payload, held as %s", k.Kind())
	}
}
