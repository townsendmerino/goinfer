//go:build gpu && goinfer_testhooks

package webgpu

import (
	"context"
	"errors"
	"github.com/townsendmerino/goinfer/backend/webgpu/internal/kernels"
	"io/fs"
	"math/rand"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestUploadKV_perLayerGeometry is G-S12.0a: the procedure of
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
	uploadKVGeomRun(t, twoGeomDir, false, "", "f16", "i8")
}

// TestUploadKV_gemma4MoE is the same procedure on a Gemma 4 model whose layers carry the parallel dense and MoE
// feed-forward (S12.2): a CPU prefill uploaded layer by layer, then decoded on the resident, against the resident's own
// sequential walk. It is what an image turn on the 26B does after its CPU prefill. Float32 and f16 KV only: at int8 KV
// this fixture's sequential walk is itself 0.93-0.99 from its float32-KV result (the router turns KV rounding into a
// different expert), further than the uploaded arm is, so the two arms' agreement says nothing about the upload there.
func TestUploadKV_gemma4MoE(t *testing.T) {
	uploadKVGeomRun(t, g4MoEFixtures[0], false, "", "f16")
}

// TestUploadKV_perLayerGeometry_planted is the gate's planted defect: with every layer's rows sized and grouped by
// the model's geometry, each precision must refuse the upload or diverge at the longer prompt.
func TestUploadKV_perLayerGeometry_planted(t *testing.T) {
	kernels.UploadKVModelGeomForTest = true
	defer func() { kernels.UploadKVModelGeomForTest = false }()
	uploadKVGeomRun(t, twoGeomDir, true, "", "f16", "i8")
}

// uploadKVGeomRun runs the comparison at the given KV precisions ("" is float32) and both prompt lengths. With wantRed,
// a case passes when the upload is refused or a decode step falls under the bar, and the longer prompt must do one of
// the two. Without it, a step under the bar SKIPs, naming the steps, when the uploaded arm agrees with the CPU decode
// over the uploaded cache and the sequential reference does not (referenceRoutedApart): on an adapter whose own prefill
// rounds differently from the CPU's, a near-tied MoE router choice can go the other way in the reference alone.
func uploadKVGeomRun(t *testing.T, dir string, wantRed bool, kvs ...string) {
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no fixture (%s)", dir)
	}
	if _, err := New(); err != nil {
		t.Skipf("no WebGPU adapter: %v", err)
	}
	for _, kv := range kvs {
		for _, n := range []int{3, 7} {
			name := "kv-f32"
			if kv != "" {
				name = "kv-" + kv
			}
			t.Run(name+"/"+map[int]string{3: "inside-the-window", 7: "past-the-window"}[n], func(t *testing.T) {
				mc, err := decoder.Load(dir, decoder.Options{Backend: "webgpu", Quant: "int4", KVPrecision: kv})
				if err != nil {
					t.Fatalf("load: %v", err)
				}
				defer mc.Close()
				rf := mc.ResidentForwardForTest()
				if rf == nil {
					t.Fatalf("webgpu declined the Gemma 4 fixture %s", dir)
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
				got := make([][]float32, nDecode)
				var under []int
				for i := range nDecode {
					l, err := rf.Forward(mc.EmbedResidentForTest(seq[i]), n+i)
					if err != nil {
						t.Fatalf("uploaded decode step %d: %v", i, err)
					}
					got[i] = append([]float32(nil), l...)
					cos, maxAbs := cosSim(want[i], l)
					t.Logf("decode step %d: cosine(sequential, uploaded) = %.8f, max |diff| %.4g", i, cos, maxAbs)
					if cos < bar {
						red = true
						under = append(under, i)
					}
				}
				if !wantRed && len(under) > 0 {
					// The CPU decode over the cache that was uploaded: the source the uploaded arm must agree with.
					cpu := make([][]float32, nDecode)
					for i := range nDecode {
						l, err := mc.ForwardForTest(seq[i], cache)
						if err != nil {
							t.Fatalf("CPU decode step %d: %v", i, err)
						}
						cpu[i] = append([]float32(nil), l...)
					}
					if referenceRoutedApart(under, cpu, got, want, bar) {
						t.Skipf("SKIP: decode steps %v: the uploaded arm agrees with the CPU decode it was uploaded from (cosine >= %v) and the "+
							"sequential reference does not: this adapter's own prefill rows round differently from the CPU's and a near-tied "+
							"MoE router choice goes the other way, so the reference cannot judge the upload at those steps", under, bar)
					}
					for _, i := range under {
						cos, _ := cosSim(want[i], got[i])
						t.Errorf("decode step %d: cosine %.8f, want at least %v", i, cos, bar)
					}
				}
				if wantRed && !red && n > 4 {
					t.Errorf("planted: the model's geometry on every layer still matched the sequential forward at %d tokens", n)
				}
			})
		}
	}
}

// referenceRoutedApart says whether every step under the bar is the sequential reference's disagreement and not the
// upload's: the uploaded arm's logits (up) agree with the CPU decode over the uploaded cache (cpu) at bar, and the
// reference's (ref) do not. A wrong upload disagrees with the CPU it came from, so it never qualifies.
func referenceRoutedApart(under []int, cpu, up, ref [][]float32, bar float64) bool {
	for _, i := range under {
		if c, _ := cosSim(cpu[i], up[i]); c < bar {
			return false
		}
		if c, _ := cosSim(cpu[i], ref[i]); c >= bar {
			return false
		}
	}
	return len(under) > 0
}

// TestReferenceRoutedApart: the skip in uploadKVGeomRun is taken only when the uploaded arm matches the CPU and the reference
// does not; an upload that disagrees with the CPU at any step under the bar stays red.
func TestReferenceRoutedApart(t *testing.T) {
	a, b := []float32{1, 0, 0}, []float32{0, 1, 0}
	for _, tc := range []struct {
		name         string
		cpu, up, ref []float32
		want         bool
	}{
		{"the reference routed apart: skip", a, a, b, true},
		{"the upload disagrees with the CPU: red", a, b, a, false},
		{"both disagree with the CPU: red", a, b, b, false},
		{"both agree with the CPU: red (nothing explains the miss)", a, a, a, false},
	} {
		got := referenceRoutedApart([]int{0}, [][]float32{tc.cpu}, [][]float32{tc.up}, [][]float32{tc.ref}, 0.999999)
		if got != tc.want {
			t.Errorf("%s: referenceRoutedApart = %v, want %v", tc.name, got, tc.want)
		}
	}
	if referenceRoutedApart(nil, nil, nil, nil, 0.999999) {
		t.Error("no step under the bar: referenceRoutedApart = true, want false")
	}
}
