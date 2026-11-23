//go:build goinfer_testhooks

package decoder

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/internal/testgate"
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
	testgate.SetForTest(t, &q4kWiden56ForTest, true) // the last load in this test: restored when it ends
	widened := d8bLogits(t, load("q4k"))
	eN, eW := relL2(got, ref), relL2(widened, ref)
	t.Logf("logits rel L2 vs f32: native %.3g, int8-widened %.3g", eN, eW)
	if !(eN < eW) {
		t.Errorf("native Q5_K/Q6_K logits are not closer to f32 than the int8 widening's: %.3g vs %.3g", eN, eW)
	}
	if _, err := SerializeWeights(native.w, "kmix"); err == nil || !strings.Contains(err.Error(), "native") {
		t.Errorf("serializing a load holding native Q5_K/Q6_K: err %v, want a refusal naming the native kind", err)
	}
}

// TestQ4K_native56Sidecar is the default load (sidecar on, no Backend named: serve's CPU path): the q4k sidecar is written for
// cpu-amd64 under the ".k56" name, carries the Q5_K and Q6_K projections as kinds 11 and 12, loads them as native kinds from
// the mapping, and forwards bit-identically to a direct native load. Where linalg has no kernel for them the sidecar widens,
// under the plain name.
func TestQ4K_native56Sidecar(t *testing.T) {
	mem1NeedsSidecarPlatform(t)
	src := filepath.Join(t.TempDir(), "phi3-kmix.gguf")
	if err := WritePhi3KMixTinyGGUFForTest(src); err != nil {
		t.Fatal(err)
	}
	old := time.Now().Add(-time.Hour)
	if err := os.Chtimes(src, old, old); err != nil {
		t.Fatal(err)
	}
	direct, err := Load(src, Options{Backend: "cpu", Quant: "q4k", DirectLoad: true})
	if err != nil {
		t.Fatal(err)
	}
	defer direct.Close()
	want := d8bLogits(t, direct)

	sidecarDefaultOn(t)
	wantQKV, wantDown, tag := "q5k", "q6k", ".k56."
	if !linalg.KQuantNative() {
		wantQKV, wantDown, tag = "int8", "int8", ""
	}
	for round := range 2 { // the first load writes the sidecar, the second maps the one already there
		m, err := Load(src, Options{Quant: "q4k"})
		if err != nil {
			t.Fatal(err)
		}
		for i := range m.w.Layers {
			l := &m.w.Layers[i]
			if l.QProj.Kind() != wantQKV || l.DownProj.Kind() != wantDown {
				t.Errorf("round %d layer %d: sidecar load holds q %q, down %q; want %q, %q", round, i, l.QProj.Kind(), l.DownProj.Kind(), wantQKV, wantDown)
			}
		}
		if got := d8bLogits(t, m); !slices.Equal(got, want) {
			t.Errorf("round %d: the sidecar load's logits differ from the direct load's", round)
		}
		m.Close()
	}
	var sidecars []string
	for _, n := range dirNames(t, filepath.Dir(src)) {
		if strings.HasSuffix(n, ".giw") {
			sidecars = append(sidecars, n)
		}
	}
	if len(sidecars) != 1 || (tag != "" && !strings.Contains(sidecars[0], tag)) || (tag == "" && strings.Contains(sidecars[0], ".k56.")) {
		t.Errorf("sidecars %v; want one, named with %q", sidecars, tag)
	}
}

// TestQ4K_native56CUDATarget: a q4k load holding the file's Q5_K and Q6_K blocks serializes for the cuda target as kinds 11 and 12
// and loads back holding them, as for cpu-amd64, while every other target refuses it naming the native kind; kquantTag names a q4k
// sidecar ".k56" for exactly those two targets. Tier 0: the tiny Phi-3 type-mix fixture, no device.
func TestQ4K_native56CUDATarget(t *testing.T) {
	for _, c := range []struct {
		quant  string
		target GIWTarget
		want   bool
	}{{"q4k", GIWTargetCPUAmd64, true}, {"q4k", GIWTargetCUDA, true}, {"q4k", GIWTargetMetal, false}, {"q4k", GIWTargetWebGPU, false},
		{"q4k", GIWTargetCPUArm64, false}, {"q4k", GIWTargetNone, false}, {"int8", GIWTargetCUDA, false}, {"int4", GIWTargetCPUAmd64, false}} {
		want := ""
		if c.want && linalg.KQuantNative() {
			want = ".k56"
		}
		if got := kquantTag(c.quant, c.target); got != want {
			t.Errorf("kquantTag(%q, %q) = %q, want %q", c.quant, c.target, got, want)
		}
		if c.quant == "q4k" && giwCarriesKQuant56(c.target) != c.want {
			t.Errorf("giwCarriesKQuant56(%q) = %v, want %v", c.target, !c.want, c.want)
		}
	}
	if !linalg.KQuantNative() {
		t.Skip("no K-quant kernel on this CPU: a q4k load widens Q5_K and Q6_K, so there is nothing native to serialize")
	}
	src := filepath.Join(t.TempDir(), "phi3-kmix.gguf")
	if err := WritePhi3KMixTinyGGUFForTest(src); err != nil {
		t.Fatal(err)
	}
	m, err := Load(src, Options{Backend: "cpu", Quant: "q4k", DirectLoad: true})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	blob, err := SerializeWeightsForTarget(m.w, "kmix", GIWTargetCUDA)
	if err != nil {
		t.Fatalf("serializing native Q5_K/Q6_K for the cuda target: %v", err)
	}
	back, err := LoadSerializedWeights(blob)
	if err != nil {
		t.Fatal(err)
	}
	for i := range back.Layers {
		l := &back.Layers[i]
		if l.QProj.Kind() != "q5k" || l.DownProj.Kind() != "q6k" {
			t.Errorf("layer %d from the cuda .giw: q %q, down %q; want q5k, q6k", i, l.QProj.Kind(), l.DownProj.Kind())
		}
	}
	for _, tgt := range []GIWTarget{GIWTargetMetal, GIWTargetWebGPU, GIWTargetCPUArm64} {
		if _, err := SerializeWeightsForTarget(m.w, "kmix", tgt); err == nil || !strings.Contains(err.Error(), "native") {
			t.Errorf("serializing native Q5_K/Q6_K for %q: err %v, want a refusal naming the native kind", tgt, err)
		}
	}
}
