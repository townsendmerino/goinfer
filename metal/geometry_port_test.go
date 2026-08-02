//go:build darwin

package metal

import (
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGeometryPortIsLive is the negative gate for the per-layer attention-geometry seam (9c
// Step 1). On a UNIFORM model every layer's geometry equals the model's, so byte-identical
// parity (TestDenseResidentParity) CANNOT distinguish a correct port — one where every dispatch
// reads residLayer.geom — from a decorative one that still bound a model-level source: the output
// is identical either way. Removing the model-level geometry fields from *Resident made the wrong
// source a COMPILE error; this is the runtime companion: it poisons ONE layer's geometry and
// asserts the logits move, proving L.geom is actually consumed (not dead-stored) before K=V,
// softcap, and Gemma-4 admission land on top of the seam. Mirrors cuda's TestGeometryPortIsLive.
//
// It gives layer 1 its OWN geom (uniform layers share one via geomFor's value dedup, so this also
// exercises the per-layer indirection r.layers[l].geom) whose rope work-count uQtotal/uKtotal is
// zeroed. Those are the rope kernel's `total` early-out bound (`if(gid>=total) return`), so every
// rope thread returns without rotating — disabling that layer's RoPE. This is the safe poison: it
// needs no 0-thread dispatch (the grid stays r.nH*g.half) and does not touch rhalf, which the
// kernel divides by (`head=gid/rhalf`) — a zeroed width would fault, not disable. A disabled RoPE
// must change the output at any position > 0 (RoPE at pos 0 is the identity), so we prime past it.
func TestGeometryPortIsLive(t *testing.T) {
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	m, err := decoder.Load(path, decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	r, err := BuildResident(m)
	if err != nil {
		t.Fatalf("BuildResident: %v", err)
	}
	defer r.Close()
	if len(r.layers) < 2 {
		t.Skip("need >= 2 layers to poison layer 1")
	}
	if r.layers[1].geom.half == 0 {
		t.Skip("layer 1 has no rotary to disable")
	}

	// Prime positions 0..3 with the correct geometry; the last forward (pos 3, RoPE non-identity,
	// attention over 4 keys) is the baseline.
	ids := []int{7, 11, 42, 100}
	var base []float32
	for i, id := range ids {
		base = append([]float32(nil), r.ForwardEmb(m.EmbedResidentForTest(id), i)...)
	}

	// Poison layer 1's rope work-count → 0 (disables its RoPE), on its OWN geom so no other layer
	// is affected. Shallow-copy shares every other read-only uniform buffer; only uQtotal/uKtotal
	// are replaced. Re-run the last position and compare.
	orig := r.layers[1].geom
	poisoned := *orig
	poisoned.uQtotal = r.d.NewBufferU32(0)
	poisoned.uKtotal = r.d.NewBufferU32(0)
	r.layers[1].geom = &poisoned
	got := append([]float32(nil), r.ForwardEmb(m.EmbedResidentForTest(ids[len(ids)-1]), len(ids)-1)...)
	r.layers[1].geom = orig // restore (defensive; the runner is torn down after this test)

	identical := len(got) == len(base)
	for i := range got {
		if i >= len(base) || got[i] != base[i] {
			identical = false
			break
		}
	}
	if identical {
		t.Fatal("poisoning layers[1].geom (rope work-count → 0) left the logits byte-identical — the " +
			"per-layer geometry is not being read; the seam is decorative")
	}
	t.Logf("seam live: disabling layer 1's RoPE via its own geom moved the logits, as a consumed per-layer field must")
}
