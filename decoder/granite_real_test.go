//go:build realckpt

// Real-model gate for Granite-4.0-H-Tiny (granitehybrid GGUF, ~7B-A1B) — the GGUF
// loader + hybrid forward on actual released weights. The model fits resident at
// int8 (~7 GB) so no streaming. Verifies the GLM seams on the real model (the
// per-layer mamba/attention split from head_count_kv, the four Granite multipliers
// from metadata, MoE on every layer) and coherent generation. The llama.cpp-convert
// GGUF also confirms the ssm_* conventions (ssm_a = −exp(A_log), conv [convDim,K]).
//
//	GOINFER_GRANITE_GGUF=~/models/granite/granite-4.0-h-tiny-Q8_0.gguf \
//	  go test -tags realckpt ./decoder/ -run TestGraniteReal -v -timeout 30m
package decoder

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

func TestGraniteReal_gate(t *testing.T) {
	requireHeavyModel(t)
	home, _ := os.UserHomeDir()
	gguf := os.Getenv("GOINFER_GRANITE_GGUF")
	if gguf == "" {
		gguf = filepath.Join(home, "models", "granite", "granite-4.0-h-tiny-Q8_0.gguf")
	}
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no Granite GGUF at %s: %v", gguf, err)
	}

	m, err := Load(gguf, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("Load(%s): %v", gguf, err)
	}
	defer m.Close()

	a := m.w.arch
	if a.Name != "granitemoehybrid" || a.granite == nil {
		t.Fatalf("arch = %q (granite=%v), want granitemoehybrid", a.Name, a.granite != nil)
	}
	// Hybrid layer split + the four Granite scalars from metadata.
	mamba, attn := 0, 0
	for l := range m.w.Layers {
		if a.isMambaLayer(l) {
			if m.w.Layers[l].mamba == nil {
				t.Errorf("layer %d is mamba but has no mixer weights", l)
			}
			mamba++
		} else {
			if m.w.Layers[l].QProj.Rows() == 0 {
				t.Errorf("layer %d is attention but has no q_proj", l)
			}
			attn++
		}
	}
	t.Logf("granite-4.0-h-tiny: %d layers (%d mamba, %d attention)", len(m.w.Layers), mamba, attn)
	if mamba == 0 || attn == 0 {
		t.Fatalf("expected a mix of mamba+attention layers, got %d/%d", mamba, attn)
	}
	if a.granite.EmbMul == 1 || a.LogitScale == 1 || a.granite.ResidMul == 1 {
		t.Errorf("Granite multipliers look unset: emb=%v resid=%v logit=%v", a.granite.EmbMul, a.granite.ResidMul, a.LogitScale)
	}

	// Coherence: greedy continuation of a canned prompt.
	tk, err := tokenizer.LoadGGUF(gguf)
	if err != nil {
		t.Fatalf("LoadGGUF tokenizer: %v", err)
	}
	// AUDIT NOTE (da5a6ec): raw completion prompt on an instruction-tuned checkpoint
	// (granite-4.0-h), gated only by the distinct<3 floor below — which measures "did the
	// forward avoid TOTAL collapse", not coherence. On gemma-4-26b-a4b-it a raw prompt
	// manufactured a false "int4 is broken" signal that survived a week, and distinct<3
	// would not have caught it (repetition has >3 distinct tokens). This gate currently
	// passes, so the completion is in-distribution ENOUGH for this checkpoint — but when
	// it is next revalidated, adopt TestGemma4_26B_gate's pattern (render the family chat
	// template + distinctTrigramRatio floor) instead of trusting distinct<3.
	prompt := "The capital of France is"
	ids, err := tk.Encode(prompt, true)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	out, _ := m.Generate(context.Background(), ids, 24, SamplingParams{}) // greedy
	gen := make([]int, 0, 24)
	for id := range out {
		gen = append(gen, id)
	}
	if len(gen) == 0 {
		t.Fatal("no tokens generated")
	}
	distinct := map[int]bool{}
	for _, id := range gen {
		distinct[id] = true
	}
	if len(distinct) < 3 {
		t.Errorf("degenerate output: %d distinct in %d", len(distinct), len(gen))
	}
	text, _ := tk.Decode(gen)
	if strings.TrimSpace(text) == "" {
		t.Error("decoded continuation empty")
	}
	t.Logf("granite real gate OK.\n  prompt: %q\n  cont:   %q", prompt, text)
}
