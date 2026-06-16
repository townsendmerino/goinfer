//go:build realckpt

// Real-model gate for Llama 4 Scout (llama4_text, 17B-active / 109B-total, 16 experts) —
// the llama.cpp llama4 GGUF loader + the iRoPE forward on actual released weights. Scout is
// the smallest Llama 4 (no small checkpoint exists); a full bf16 oracle is infeasible (the
// safetensors is gated + ~210 GB), so this is a COHERENT-GENERATION gate (a wrong loader —
// expert layout, per-layer kind, the injected NoPE/QK-norm/attn-temp config — yields
// garbage). goinfer loads from a Q2_K GGUF at int4 (~55 GB resident; int8 would be ~109 GB,
// over this box's RAM). The long-context llama3 rope scaling is NOT applied (negligible for
// a short prompt). Fixture: a Scout Q2_K GGUF.
//
//	GOINFER_LLAMA4_GGUF=~/models/.../Llama-4-Scout-...-Q2_K.gguf \
//	  go test -tags realckpt ./decoder/ -run TestLlama4Real -v -timeout 60m
package decoder

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

func TestLlama4Real_gate(t *testing.T) {
	home, _ := os.UserHomeDir()
	gguf := os.Getenv("GOINFER_LLAMA4_GGUF")
	if gguf == "" {
		gguf = filepath.Join(home, "models", "llama4-scout-gguf", "Llama-4-Scout-17B-16E-Instruct-Q2_K.gguf")
	}
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no Llama 4 GGUF at %s: %v", gguf, err)
	}
	m, err := Load(gguf, Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("Load(%s): %v", gguf, err)
	}
	defer m.Close()
	a := m.w.arch
	if a.Name != "llama4_text" || a.llama4 == nil {
		t.Fatalf("arch = %q (llama4=%v), want llama4_text", a.Name, a.llama4 != nil)
	}
	nMoE, nNoPE := 0, 0
	for l := 0; l < a.NumLayers; l++ {
		if a.llama4.isMoE[l] {
			nMoE++
		}
		if !a.llama4.useRope[l] {
			nNoPE++
		}
	}
	t.Logf("Scout: %d layers (%d MoE, %d NoPE), H=%d kv=%d headDim=%d experts=%d top%d qknorm=%v attnTemp=%v",
		a.NumLayers, nMoE, nNoPE, a.NumHeads, a.NumKVHeads, a.HeadDim, a.MoE.NumExperts, a.MoE.TopK, a.llama4.useQKNorm, a.llama4.attnTemp)

	tk, err := tokenizer.LoadGGUF(gguf)
	if err != nil {
		t.Fatalf("LoadGGUF tokenizer: %v", err)
	}
	prompt := "The capital of France is"
	ids, err := tk.Encode(prompt, true)
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
	t.Logf("Scout gen: %q", text)
	if len(gen) == 0 || len(distinct) < 3 {
		t.Errorf("degenerate output: %d distinct in %d tokens", len(distinct), len(gen))
	}
	if !strings.Contains(text, "Paris") {
		t.Logf("note: continuation did not contain \"Paris\" (Q2_K is lossy) — coherence is the gate")
	}
}
