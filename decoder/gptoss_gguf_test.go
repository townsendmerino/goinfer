package decoder

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"testing"
)

// TestGptOssGGUF_parity gates the gpt-oss forward against an HF reference on a TINY
// model (scripts/gptoss_tiny_golden.py). The GGUF fixture is F32 — no quant — so any
// divergence is the FORWARD or the LOADER, not quant noise: the per-head attention
// SINK in the softmax, the clamped interleaved-SwiGLU experts (+ per-expert biases),
// the router-logit bias, the alternating sliding/full attention (seq_len 8 > window
// 4, so the sliding layers actually clip), and YaRN (truncate=false). The MXFP4
// dequant is NOT exercised here — the committed bit-exact unpacker test covers that
// (a real gpt-oss GGUF's experts are MXFP4 and load once aikit dequants ggml type 39).
func TestGptOssGGUF_parity(t *testing.T) {
	const golden = "testdata/gptoss_tiny_golden.json"
	const gguf = "testdata/gptoss_tiny.gguf"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden — run scripts/gptoss_tiny_golden.py")
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	if _, err := os.Stat(gguf); errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no GGUF fixture at %s — run scripts/gptoss_tiny_golden.py", gguf)
	}
	var g struct {
		InputIDs []int     `json:"input_ids"`
		Argmax   int       `json:"argmax"`
		Logits   []float32 `json:"logits"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	m, err := Load(gguf, Options{})
	if err != nil {
		t.Fatalf("Load(%s): %v", gguf, err)
	}
	defer m.Close()

	if m.w.arch.Name != "gpt-oss" {
		t.Fatalf("arch = %q, want gpt-oss", m.w.arch.Name)
	}
	if m.w.arch.gptoss == nil {
		t.Fatalf("gptoss params not set")
	}
	// The loader must populate the gpt-oss-specific weights on every layer.
	for i := range m.w.Layers {
		lw := &m.w.Layers[i]
		if len(lw.AttnSinks) != m.w.arch.NumHeads {
			t.Fatalf("layer %d AttnSinks len = %d, want %d", i, len(lw.AttnSinks), m.w.arch.NumHeads)
		}
		if len(lw.RouterBias) != m.w.arch.MoE.NumExperts {
			t.Fatalf("layer %d RouterBias len = %d, want %d", i, len(lw.RouterBias), m.w.arch.MoE.NumExperts)
		}
		if lw.Experts == nil {
			t.Fatalf("layer %d has no experts", i)
		}
		if lw.Experts[0].GateBias == nil || lw.Experts[0].DownBias == nil {
			t.Fatalf("layer %d expert 0 missing per-expert biases", i)
		}
		if lw.QBias == nil || lw.OBias == nil {
			t.Fatalf("layer %d missing q/o biases", i)
		}
	}

	cache := m.NewCache(len(g.InputIDs))
	var logits []float32
	for _, id := range g.InputIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.Logits)
	t.Logf("gpt-oss GGUF parity: argmax got=%d want=%d | logit cosine=%.6f", gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("last argmax = %d, want %d", gotArg, g.Argmax)
	}
	if cos < 0.9999 {
		t.Errorf("last-logit cosine %.6f < 0.9999 (F32 GGUF — should be ~1.0)", cos)
	}
}
