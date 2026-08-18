//go:build realckpt

// Real-model gate for gpt-oss:20b (gpt_oss, 21B-total / 3.6B-active, 32 experts, MXFP4)
// — the llama.cpp gpt-oss GGUF loader (MXFP4 experts dequantized via aikit's ggml
// type-39 support) + the sink/clamped-SwiGLU forward on actual released weights. A full
// f32 load is ~80 GB (OOM on this box), so goinfer loads at int8 (~20 GB resident): the
// MXFP4 experts dequant to f32 then requantize per row, near-lossless since MXFP4 is
// already 4-bit. Coherence is the gate (a wrong loader — expert de-interleave, sinks,
// router bias, sliding/full pattern — yields garbage); the logit-parity vs HF is the
// separate golden (scripts/gptoss_hf_oracle.py).
//
//	GOINFER_GPTOSS_GGUF=~/models/gpt-oss-20b-MXFP4.gguf \
//	  go test -tags realckpt ./decoder/ -run TestGptOssReal -v -timeout 60m
package decoder

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

func TestGptOssReal_gate(t *testing.T) {
	requireHeavyModel(t)
	// assetPath, not a hand-rolled env+fallback: the asset registry is what makes the
	// gate and the sweep preflight apply the SAME predicate to the same candidate paths
	// (testdata/assets.json). This file predated GOINFER_GPTOSS_GGUF being registered.
	gguf := assetPath(t, "GOINFER_GPTOSS_GGUF")
	m, err := Load(gguf, Options{Quant: "int8"})
	if err != nil {
		t.Fatalf("Load(%s): %v", gguf, err)
	}
	defer m.Close()

	a := m.w.arch
	if a.Name != "gpt-oss" || a.gptoss == nil {
		t.Fatalf("arch = %q (gptoss=%v), want gpt-oss", a.Name, a.gptoss != nil)
	}
	// MXFP4 experts + sinks + router bias must have loaded on every layer.
	for l := 0; l < a.NumLayers; l++ {
		lw := &m.w.Layers[l]
		if lw.Experts == nil || len(lw.Experts) != a.MoE.NumExperts {
			t.Fatalf("layer %d: experts not loaded (MXFP4 dequant failed?)", l)
		}
		if len(lw.AttnSinks) != a.NumHeads || len(lw.RouterBias) != a.MoE.NumExperts {
			t.Fatalf("layer %d: sinks/router-bias missing", l)
		}
	}
	t.Logf("gpt-oss:20b loaded int8: %d layers, H=%d kv=%d headDim=%d experts=%d top%d window=%d",
		a.NumLayers, a.NumHeads, a.NumKVHeads, a.HeadDim, a.MoE.NumExperts, a.MoE.TopK, a.SlidingWindow)

	tk, err := tokenizer.LoadGGUF(gguf)
	if err != nil {
		t.Fatalf("LoadGGUF tokenizer: %v", err)
	}
	prompt := "The capital of France is"
	ids, err := tk.Encode(prompt, false) // base completion, no BOS (gpt-oss is a harmony model; a raw BOS+prompt derails the base)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	out, _ := m.Generate(context.Background(), ids, 16, SamplingParams{})
	gen := make([]int, 0, 16)
	for id := range out {
		gen = append(gen, id)
	}
	distinct := map[int]bool{}
	for _, id := range gen {
		distinct[id] = true
	}
	text, _ := tk.Decode(gen)
	t.Logf("gpt-oss:20b gen: %q", text)
	if len(gen) == 0 || len(distinct) < 3 {
		t.Errorf("degenerate output: %d distinct in %d tokens", len(distinct), len(gen))
	}
	if !strings.Contains(text, "Paris") {
		t.Logf("note: continuation did not contain \"Paris\" — coherence is the gate")
	}
}

// TestGptOssReal_logitParity is the §6.2 numeric gate: goinfer's next-token logits on
// the real gpt-oss:20b must match the HF reference (scripts/gptoss_hf_oracle.py) on the
// SAME token ids — argmax-exact + high logit cosine. Feeding the golden's prompt_ids
// directly removes the tokenizer as a variable. goinfer is int8-resident vs HF bf16, so
// the bar is argmax-exact + cosine ~0.99 (the deepseek/llama4 real-model bar), not
// bit-exact.
func TestGptOssReal_logitParity(t *testing.T) {
	requireHeavyModel(t)
	const golden = "testdata/gptoss_20b_golden.json"
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skipf("no golden at %s — run scripts/gptoss_hf_oracle.py", golden)
	}
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g struct {
		Prompt     string    `json:"prompt"`
		PromptIDs  []int     `json:"prompt_ids"`
		Argmax     int       `json:"argmax"`
		LastLogits []float32 `json:"last_logits"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	// assetPath, not a hand-rolled env+fallback: the asset registry is what makes the
	// gate and the sweep preflight apply the SAME predicate to the same candidate paths
	// (testdata/assets.json). This file predated GOINFER_GPTOSS_GGUF being registered.
	gguf := assetPath(t, "GOINFER_GPTOSS_GGUF")
	m, err := Load(gguf, Options{Quant: "int8"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	cache := m.NewCache(len(g.PromptIDs))
	var logits []float32
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	gotArg := argmax(logits)
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("gpt-oss:20b logit parity (%q): argmax got=%d want=%d | cosine=%.5f", g.Prompt, gotArg, g.Argmax, cos)
	if gotArg != g.Argmax {
		t.Errorf("argmax = %d, want %d (HF)", gotArg, g.Argmax)
	}
	if cos < 0.99 {
		t.Errorf("logit cosine %.5f < 0.99 (int8 resident vs HF bf16)", cos)
	}
}
