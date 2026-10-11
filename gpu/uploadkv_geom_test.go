//go:build gpu && goinfer_testhooks

package gpu

import (
	"context"
	"errors"
	"io/fs"
	"math/rand"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestUploadKV_perLayerGeometry is G-S12.0a (docs/tasks/task-multimodal-support-2026-10.md, S12): the procedure of
// TestUploadKV_matchesSequentialForward on a model whose layers do not share one attention geometry. The tiny dense
// Gemma 4 has a sliding layer (head_dim 16, window 4) and a global one (head_dim 512), so an upload that sizes or
// groups a layer's rows by the model's geometry is wrong on one of them. Both prompt lengths are run: one inside the
// window and one past it, where the sliding layer's rows start at a nonzero base.
//
// The bar is the float32-KV test's, 0.999999, only at float32 KV. At f16 and int8 KV the two arms are not the same
// computation: the resident's own prefill attends over quantized rows and the CPU's over exact ones, and a model with
// one geometry reads 0.9998-0.9999 the same way (the task doc's G-S12.0a reading). There the floor is 0.99, which
// the wrong geometry misses (0.88-0.93, or a refused upload).
func TestUploadKV_perLayerGeometry(t *testing.T) {
	uploadKVGeomRun(t, false)
}

// TestUploadKV_perLayerGeometry_planted is the gate's planted defect: with every layer's rows sized and grouped by
// the model's geometry, each precision must refuse the upload or diverge at the longer prompt.
func TestUploadKV_perLayerGeometry_planted(t *testing.T) {
	uploadKVModelGeomForTest = true
	defer func() { uploadKVModelGeomForTest = false }()
	uploadKVGeomRun(t, true)
}

// uploadKVGeomRun runs the comparison at every KV precision and both prompt lengths. With wantRed, a case passes when
// the upload is refused or a decode step falls under the bar, and the longer prompt must do one of the two.
func uploadKVGeomRun(t *testing.T, wantRed bool) {
	if _, err := os.Stat(twoGeomDir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture (%s)", twoGeomDir)
	}
	if _, err := New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	for _, kv := range []string{"", "f16", "i8"} {
		for _, n := range []int{3, 7} {
			name := "kv-f32"
			if kv != "" {
				name = "kv-" + kv
			}
			t.Run(name+"/"+map[int]string{3: "inside-the-window", 7: "past-the-window"}[n], func(t *testing.T) {
				mc, err := decoder.Load(twoGeomDir, decoder.Options{Backend: "webgpu", Quant: "int4", KVPrecision: kv})
				if err != nil {
					t.Fatalf("load: %v", err)
				}
				defer mc.Close()
				rf := mc.ResidentForwardForTest()
				if rf == nil {
					t.Fatal("webgpu declined the dense Gemma 4 fixture")
				}
				_, nLayers, _, _, _, _, vocab := mc.Dims()
				rng := rand.New(rand.NewSource(7))
				ids := make([]int, n)
				for i := range ids {
					ids[i] = rng.Intn(vocab - 1)
				}
				const nDecode = 4
				// Sequential on the resident: the reference.
				var last []float32
				for i, id := range ids {
					l, err := rf.Forward(mc.EmbedResidentForTest(id), i)
					if err != nil {
						t.Fatalf("sequential pos %d: %v", i, err)
					}
					last = l
				}
				want := make([][]float32, nDecode)
				tok := argmax(last)
				seq := []int{tok}
				for i := range nDecode {
					l, err := rf.Forward(mc.EmbedResidentForTest(tok), n+i)
					if err != nil {
						t.Fatalf("sequential decode step %d: %v", i, err)
					}
					want[i] = append([]float32(nil), l...)
					tok = argmax(l)
					seq = append(seq, tok)
				}
				rf.Reset()
				// CPU prefill, uploaded layer by layer, then the SAME tokens through the resident.
				cache := mc.NewCache(n + nDecode)
				if _, err := mc.PrefillLogitsForTest(context.Background(), ids, cache); err != nil {
					t.Fatalf("CPU prefill: %v", err)
				}
				for l := range nLayers {
					k, v, base := cache.LayerKVForTest(l)
					if len(k) == 0 {
						t.Fatalf("layer %d: no K rows from the CPU cache", l)
					}
					if err := rf.UploadKV(l, base, k, v); err != nil {
						if wantRed {
							t.Logf("planted: UploadKV layer %d refused: %v", l, err)
							return
						}
						t.Fatalf("UploadKV layer %d (base %d, %d floats): %v", l, base, len(k), err)
					}
				}
				red, bar := false, 0.999999
				if kv != "" {
					bar = 0.99
				}
				for i := range nDecode {
					l, err := rf.Forward(mc.EmbedResidentForTest(seq[i]), n+i)
					if err != nil {
						t.Fatalf("uploaded decode step %d: %v", i, err)
					}
					cos, maxAbs := cosSim(want[i], l)
					t.Logf("decode step %d: cosine(sequential, uploaded) = %.8f, max |diff| %.4g", i, cos, maxAbs)
					if cos < bar {
						red = true
						if !wantRed {
							t.Errorf("decode step %d: cosine %.8f, want at least %v", i, cos, bar)
						}
					}
				}
				if wantRed && !red && n > 4 {
					t.Errorf("planted: the model's geometry on every layer still matched the sequential forward at %d tokens", n)
				}
			})
		}
	}
}
