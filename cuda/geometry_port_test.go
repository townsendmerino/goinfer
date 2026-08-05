//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGeometryPortIsLive is the negative gate for the per-layer attention-geometry port
// (9a-P2). On a UNIFORM model every layer's geometry equals the model's, so byte-identical
// parity against the CPU path CANNOT distinguish a correct port (launchToken reads
// Ly.hd/nKV/qDim/kvDim/rhalf) from a decorative one that still read a model-level source —
// the output is identical either way. Removing the model-level fields from cudaResident
// makes the wrong source a COMPILE error; this test is the runtime companion: it poisons one
// layer's per-layer geometry and asserts the logits move, proving the field is actually
// consumed (not dead-stored) before Split A layers K=V, softcap, and admission on top.
//
// It poisons rhalf (rotated pairs per head) → 0, which disables that layer's RoPE: a safe,
// in-bounds corruption that must change the output at any position > 0 (RoPE at pos 0 is the
// identity, so we prime past it). rhalf rides the same per-layer plumbing as hd/nKV/qDim/
// kvDim — they are read at the same launch sites — so its liveness demonstrates the port's.
func TestGeometryPortIsLive(t *testing.T) {
	requireHeavyModel(t)
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cuda): %v", err)
	}
	defer mc.Close()
	rf, ok := mc.ResidentForwardForTest().(*cudaResident)
	if !ok || rf == nil {
		t.Skip("cuda resident not admitted for this checkpoint")
	}
	if len(rf.layers) < 2 {
		t.Skip("need >= 2 layers to poison layer 1")
	}
	if rf.layers[1].rhalf == 0 {
		t.Skip("layer 1 has no rotary to disable")
	}

	// Prime positions 0..2 with the correct geometry, then baseline-forward at pos 3 (RoPE
	// non-identity, attention over 4 keys — hd/nKV/rhalf all matter here).
	ids := []int{7, 11, 42, 100}
	var base []float32
	for i, id := range ids {
		lg, err := rf.Forward(mc.EmbedResidentForTest(id), i)
		if err != nil {
			t.Fatalf("baseline forward pos %d: %v", i, err)
		}
		base = append([]float32(nil), lg...)
	}

	// Poison layer 1's rotary width and re-run the last position.
	saved := rf.layers[1].rhalf
	rf.layers[1].rhalf = 0
	got, err := rf.Forward(mc.EmbedResidentForTest(ids[len(ids)-1]), len(ids)-1)
	rf.layers[1].rhalf = saved // restore (defensive; the runner is torn down after this test)
	if err != nil {
		t.Fatalf("poisoned forward: %v", err)
	}

	identical := len(got) == len(base)
	for i := range got {
		if i >= len(base) || got[i] != base[i] {
			identical = false
			break
		}
	}
	if identical {
		t.Fatal("poisoning layers[1].rhalf left the logits byte-identical — the per-layer geometry " +
			"is not being read; the port is decorative")
	}
	t.Logf("port live: disabling layer 1's RoPE (rhalf 0) moved the logits, as a consumed per-layer field must")
}
