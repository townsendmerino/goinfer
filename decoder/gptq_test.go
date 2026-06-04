package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"testing"
)

func TestParseQuantConfig(t *testing.T) {
	if p, err := parseQuantConfig(nil); p != nil || err != nil {
		t.Errorf("nil config → (%v,%v), want (nil,nil)", p, err)
	}
	good := `{"quant_method":"gptq","bits":4,"group_size":128,"desc_act":true,"sym":true}`
	p, err := parseQuantConfig(json.RawMessage(good))
	if err != nil || p == nil {
		t.Fatalf("good config: %v", err)
	}
	if p.method != "gptq" || p.bits != 4 || p.groupSize != 128 || !p.descAct || !p.sym {
		t.Errorf("parsed %+v", p)
	}
	aw, err := parseQuantConfig(json.RawMessage(`{"quant_method":"awq","bits":4,"group_size":128,"version":"gemm"}`))
	if err != nil || aw == nil || aw.method != "awq" || aw.groupSize != 128 {
		t.Errorf("awq config: %v %+v", err, aw)
	}
	for _, bad := range []string{
		`{"quant_method":"bitsandbytes","bits":4,"group_size":128}`,
		`{"quant_method":"gptq","bits":8,"group_size":128}`,
		`{"quant_method":"awq","bits":4,"group_size":-1}`,
	} {
		if _, err := parseQuantConfig(json.RawMessage(bad)); err == nil {
			t.Errorf("expected error for %s", bad)
		}
	}
}

// TestGPTQ_parity loads a real GPTQ checkpoint (TheBloke/TinyLlama-1.1B-Chat-v1.0
// -GPTQ, 4-bit group-128 act-order) and checks its forward against the committed
// f32 oracle for the SAME model — the argmax must hold and the cosine clear the
// 4-bit floor. Validates the qweight/qzeros/scales/g_idx reconstruction
// end-to-end. Loads ~0.8 GB (gitignored), so skips when absent or under -short.
//
//	hf download TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ config.json model.safetensors \
//	  --local-dir testdata/tinyllama-gptq
func TestGPTQ_parity(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: loads + reconstructs a GPTQ checkpoint")
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
	dir := "../testdata/tinyllama-gptq"
	if _, err := os.Stat(dir + "/model.safetensors"); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no GPTQ checkpoint at %s", dir)
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
		t.Errorf("GPTQ argmax = %d, want %d (f32)", got, g.Argmax)
	}
	cos := cosineToFull(t, logits, llamaForwardFullPath)
	if !math.IsNaN(cos) && cos < 0.98 {
		t.Errorf("GPTQ cosine vs f32 oracle = %v, want ≥ 0.98", cos)
	}
	t.Logf("GPTQ TinyLlama (4-bit g128 act-order): argmax=%d (want %d) | cosine vs f32 = %v", argmax(logits), g.Argmax, cos)
}
