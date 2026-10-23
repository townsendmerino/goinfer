package decoder

import (
	"fmt"
	"hash/fnv"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// mlaMainF32Hash is main's deepseek-tiny f32 CPU logits over 32 positions, hashed, before MEM11-int8 (the f32 load must not move).
// The CPU reference is bit-identical within an architecture, not across (arm64 fuses multiply-adds), so the hash is arm64's; another
// architecture's f32 path is held to the tiny forward golden's tolerance instead.
const mlaMainF32Hash uint64 = 0xbd704d523f8878ff

// mlaLoad loads path at quant, with MLA's f32 projections kept when keepF32 (the f32 arm, and the plant's control).
func mlaLoad(t *testing.T, path, quant string, keepF32 bool) *Model {
	t.Helper()
	prev := mlaKeepF32ForTest
	mlaKeepF32ForTest = keepF32
	defer func() { mlaKeepF32ForTest = prev }()
	m, err := Load(path, Options{Quant: quant, ResidentContext: 64})
	if err != nil {
		t.Fatalf("load %s at %q (f32 MLA %v): %v", path, quant, keepF32, err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

// mlaPairs is each int8-able projection's f32 (from the f32 arm) and int8 (from the int8 arm), with its shape.
func mlaPairs(ref, got *mlaWeights) []struct {
	name string
	f    []float32
	q    *linalg.WeightMat
} {
	return []struct {
		name string
		f    []float32
		q    *linalg.WeightMat
	}{{"q_a", ref.qAProj, &got.qAQ}, {"q_b", ref.qBProj, &got.qBQ}, {"q", ref.qProj, &got.qQ}, {"kv_a", ref.kvAProj, &got.kvAQ}, {"o", ref.oProj, &got.oQ}}
}

// TestMEM11_mlaInt8Holding is I8M-G1 and I8M-G2's host half (docs/tasks/task-memory-campaign-2026-10.md, "MEM11-int8 gates"): on
// deepseek-tiny, a quantized load holds q (or the q-LoRA pair), kv-A and o as int8 and kv-B f32, an f32 load holds all six f32 with
// logits unchanged from main's, ResidentWeightBytes falls by exactly the difference, and the int8 codes and scales, and a bundle
// round trip's, equal linalg.QuantizeInt8 of the f32 arm's projections.
func TestMEM11_mlaInt8Holding(t *testing.T) {
	const dir = "../testdata/deepseek-tiny"
	f32 := mlaLoad(t, dir, "", false)
	_, _, _, _, _, _, vocab := f32.Dims()
	cache := f32.NewCache(40)
	h := fnv.New64a()
	for i := range 32 {
		lg, err := f32.forward((i*37+11)%vocab, cache)
		if err != nil {
			t.Fatal(err)
		}
		for _, x := range lg {
			b := math.Float32bits(x)
			h.Write([]byte{byte(b), byte(b >> 8), byte(b >> 16), byte(b >> 24)})
		}
	}
	if got := h.Sum64(); runtime.GOARCH == "arm64" && got != mlaMainF32Hash {
		t.Errorf("the f32 load's logits hash %#x, main's %#x", got, mlaMainF32Hash)
	}
	for _, quant := range []string{"int4", "int8int8"} {
		ref, got := mlaLoad(t, dir, quant, true), mlaLoad(t, dir, quant, false)
		blob, err := SerializeWeights(got.w, "mem11-mla")
		if err != nil {
			t.Fatal(err)
		}
		back, err := LoadSerializedWeights(blob)
		if err != nil {
			t.Fatal(err)
		}
		var saved int64
		layers := 0
		for l := range got.w.Layers {
			rm, gm, bm := ref.w.Layers[l].mla, got.w.Layers[l].mla, back.Layers[l].mla
			if gm == nil {
				continue
			}
			layers++
			if gm.kvBProj == nil || !slices.Equal(gm.kvBProj, rm.kvBProj) {
				t.Errorf("%s layer %d: kv-B is not the f32 kv-B", quant, l)
			}
			for _, p := range mlaPairs(rm, gm) {
				if p.f == nil {
					continue // the other q form
				}
				want := linalg.QuantizeInt8(p.f, p.q.Rows(), p.q.Cols(), false)
				wq, ws, _, _ := want.Int8()
				for i, q := range []*linalg.WeightMat{p.q, mlaPairs(rm, bm)[slices.IndexFunc(mlaPairs(rm, bm), func(x struct {
					name string
					f    []float32
					q    *linalg.WeightMat
				}) bool {
					return x.name == p.name
				})].q} {
					gq, gs, _, ok := q.Int8()
					if !ok || !slices.Equal(gq, wq) || !slices.Equal(gs, ws) {
						t.Errorf("%s layer %d %s (%s): not linalg.QuantizeInt8 of the f32", quant, l, p.name, []string{"load", "bundle"}[i])
					}
				}
				saved += f32SliceBytes(p.f) - wmBytes(p.q)
			}
		}
		rb, gb := ref.ResidentWeightBytes(), got.ResidentWeightBytes()
		fmt.Fprintf(os.Stderr, "[I8M-G1] deepseek-tiny at %s: %d MLA layers int8 (kv-B f32); ResidentWeightBytes %d -> %d (the projections save %d)\n", quant, layers, rb, gb, saved)
		if layers == 0 || rb-gb != saved {
			t.Errorf("%s: ResidentWeightBytes fell by %d, the projections save %d", quant, rb-gb, saved)
		}
	}
}

// TestMEM11_mlaInt8FidelityTiny is I8M-G3: deepseek-tiny at int4, the int8-MLA CPU against the f32-MLA CPU, 32 steps of greedy
// lockstep (B27's instrument), min logit cosine >= 0.999 from position 2 and argmax agreement >= 30/32.
func TestMEM11_mlaInt8FidelityTiny(t *testing.T) {
	ref := mlaLoad(t, "../testdata/deepseek-tiny", "int4", true)
	got := mlaLoad(t, "../testdata/deepseek-tiny", "int4", false)
	_, _, _, _, _, _, vocab := ref.Dims()
	minCos, agree := b27Lockstep(t, ref, got, []int{1, 17 % vocab, 113 % vocab, 29 % vocab, 71 % vocab, 5 % vocab}, 32)
	fmt.Fprintf(os.Stderr, "[I8M-G3] deepseek-tiny at int4, int8 MLA against f32: min cosine %.6f, argmax %d/32\n", minCos, agree)
	if minCos < 0.999 || agree < 30 {
		t.Errorf("below the bar (min cosine >= 0.999, argmax >= 30/32)")
	}
}

// TestMEM11_mlaInt8FidelitySlice is I8M-G4: the first two layers of DeepSeek-V2-Lite-Chat's Q4_K_M GGUF at int4, the same
// lockstep, min logit cosine >= 0.99 and argmax >= 28/32.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -run '^TestMEM11_mlaInt8FidelitySlice$' -v ./decoder/
func TestMEM11_mlaInt8FidelitySlice(t *testing.T) {
	requireHeavyModel(t)
	home, _ := os.UserHomeDir()
	path := filepath.Join(home, "models", "deepseek-v2-lite-gguf", "DeepSeek-V2-Lite-Chat-Q4_K_M.slice-0-1.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no slice at %s", path)
	}
	ref, got := mlaLoad(t, path, "int4", true), mlaLoad(t, path, "int4", false)
	if got.w.Layers[0].mla.qQ.Rows() == 0 {
		t.Fatal("the int8 arm's layer 0 holds no int8 q projection")
	}
	minCos, agree := b27Lockstep(t, ref, got, []int{1, 17, 113, 29, 71, 5}, 32)
	fmt.Fprintf(os.Stderr, "[I8M-G4] V2-Lite slice at int4, int8 MLA against f32: min cosine %.6f, argmax %d/32; MLA bytes %d -> %d\n",
		minCos, agree, ref.ResidentWeightBytes(), got.ResidentWeightBytes())
	if minCos < 0.99 || agree < 28 {
		t.Errorf("below the bar (min cosine >= 0.99, argmax >= 28/32)")
	}
}
