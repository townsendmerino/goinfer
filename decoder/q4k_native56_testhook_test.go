//go:build goinfer_testhooks

package decoder

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// TestQ4K_native56: a Backend "cpu" q4k load of a file with Phi-3's type mix (qkv Q5_K, down Q6_K) keeps those projections as
// the file's own blocks where linalg has a kernel for them (q4kNative56), and its logits are closer to an f32 load of the same
// file than the int8 widening's are. A .giw of such a load is refused rather than written without them.
func TestQ4K_native56(t *testing.T) {
	src := filepath.Join(t.TempDir(), "phi3-kmix.gguf")
	if err := WritePhi3KMixTinyGGUFForTest(src); err != nil {
		t.Fatal(err)
	}
	load := func(quant string) *Model {
		m, err := Load(src, Options{Backend: "cpu", Quant: quant, DirectLoad: true})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { m.Close() })
		return m
	}
	native := load("q4k")
	wantQKV, wantDown := "q5k", "q6k"
	if !linalg.KQuantNative() {
		wantQKV, wantDown = "int8", "int8"
		t.Log("no K-quant kernel on this CPU: the load widens Q5_K and Q6_K to int8")
	}
	for i := range native.w.Layers {
		l := &native.w.Layers[i]
		for _, w := range []*linalg.WeightMat{&l.QProj, &l.KProj, &l.VProj} {
			if w.Kind() != wantQKV {
				t.Errorf("layer %d: a q/k/v split of the Q5_K qkv loads as %q, want %q", i, w.Kind(), wantQKV)
			}
		}
		if l.DownProj.Kind() != wantDown {
			t.Errorf("layer %d: the Q6_K down projection loads as %q, want %q", i, l.DownProj.Kind(), wantDown)
		}
		for _, w := range []*linalg.WeightMat{&l.OProj, &l.GateProj, &l.UpProj} {
			if w.Kind() != "q4k" {
				t.Errorf("layer %d: a Q4_K projection loads as %q", i, w.Kind())
			}
		}
	}
	if !linalg.KQuantNative() {
		return
	}
	ref := d8bLogits(t, load("f32"))
	got := d8bLogits(t, native)
	q4kWiden56ForTest = true
	widened := d8bLogits(t, load("q4k"))
	q4kWiden56ForTest = false
	eN, eW := relL2(got, ref), relL2(widened, ref)
	t.Logf("logits rel L2 vs f32: native %.3g, int8-widened %.3g", eN, eW)
	if !(eN < eW) {
		t.Errorf("native Q5_K/Q6_K logits are not closer to f32 than the int8 widening's: %.3g vs %.3g", eN, eW)
	}
	if _, err := SerializeWeights(native.w, "kmix"); err == nil || !strings.Contains(err.Error(), "native") {
		t.Errorf("serializing a load holding native Q5_K/Q6_K: err %v, want a refusal naming the native kind", err)
	}
}
