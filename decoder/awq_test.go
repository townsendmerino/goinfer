package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

// TestAWQ_parity loads a real AWQ checkpoint (TheBloke/TinyLlama-1.1B-Chat-v1.0
// -AWQ, 4-bit group-128 GEMM) and checks its forward against the committed f32
// oracle for the SAME model — argmax preserved and cosine clearing the 4-bit
// floor. Exercises the AWQ unpack: output-dim packing, the [0,4,1,5,2,6,3,7]
// nibble de-interleave, and the (no-+1) zero-point. Loads ~0.8 GB (gitignored),
// so skips when absent or under -short.
//
//	hf download TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ config.json model.safetensors \
//	  --local-dir testdata/tinyllama-awq
func TestAWQ_parity(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: loads + reconstructs an AWQ checkpoint")
	}
	raw, err := os.ReadFile(llamaForwardGolden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no llama golden at %s", llamaForwardGolden)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g forwardGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	dir := "../testdata/tinyllama-awq"
	if _, err := os.Stat(dir + "/model.safetensors"); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no AWQ checkpoint at %s", dir)
	}

	m, err := Load(dir, Options{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	cache := m.NewCache(len(g.IDs))
	for _, id := range g.IDs[:len(g.IDs)-1] {
		if _, err := m.runLayers(id, cache); err != nil {
			t.Fatalf("runLayers: %v", err)
		}
	}
	logits, err := m.forward(g.IDs[len(g.IDs)-1], cache)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if got := argmax(logits); got != g.Argmax {
		t.Errorf("AWQ argmax = %d, want %d (f32)", got, g.Argmax)
	}
	cos := cosineToFull(t, logits, llamaForwardFullPath)
	if !math.IsNaN(cos) && cos < 0.98 {
		t.Errorf("AWQ cosine vs f32 oracle = %v, want ≥ 0.98", cos)
	}
	t.Logf("AWQ TinyLlama (4-bit g128 GEMM): argmax=%d (want %d) | cosine vs f32 = %v", argmax(logits), g.Argmax, cos)
}
