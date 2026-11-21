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
