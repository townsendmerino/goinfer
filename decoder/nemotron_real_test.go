//go:build realckpt

// Real-model gate for Nemotron-Nano-9B-v2 (nemotron_h GGUF, ~9B) — the GGUF loader +
// single-op-block forward on actual released weights. Fits resident at int8 (~9.5 GB)
// so no streaming. Verifies the seams on the real model (per-layer kind from the
// head_count_kv / feed_forward_length arrays, the grouped gated norm with n_groups=8,
// NoPE attention, relu² MLP) and coherent generation. The llama.cpp-convert GGUF also
// confirms the ssm_* conventions reused from granite.
//
//	GOINFER_NEMOTRON_GGUF=~/models/nemotron/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf \
//	  go test -tags realckpt ./decoder/ -run TestNemotronReal -v -timeout 30m
package decoder

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

func TestNemotronReal_gate(t *testing.T) {
	requireHeavyModel(t)
	home, _ := os.UserHomeDir()
	gguf := os.Getenv("GOINFER_NEMOTRON_GGUF")
	if gguf == "" {
		gguf = filepath.Join(home, "models", "nemotron", "nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf")
	}
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no Nemotron GGUF at %s: %v", gguf, err)
	}

	m, err := Load(gguf, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("Load(%s): %v", gguf, err)
	}
	defer m.Close()

	a := m.w.arch
	if a.Name != "nemotron_h" || a.nemotron == nil {
		t.Fatalf("arch = %q (nemotron=%v), want nemotron_h", a.Name, a.nemotron != nil)
	}
	mamba, attn, mlp := 0, 0, 0
	for _, k := range a.nemotron.blockKind {
		switch k {
		case nemoMamba:
			mamba++
		case nemoAttn:
			attn++
		case nemoMLP:
			mlp++
		}
	}
	t.Logf("nemotron-nano-9b: %d layers (%d mamba, %d attn, %d mlp), n_groups=%d", len(a.nemotron.blockKind), mamba, attn, mlp, a.nemotron.NGroups)
	if mamba == 0 || attn == 0 || mlp == 0 {
		t.Fatalf("expected all three block kinds, got mamba=%d attn=%d mlp=%d", mamba, attn, mlp)
	}
	if a.nemotron.NGroups < 2 {
		t.Errorf("expected grouped mamba (n_groups≥2), got %d", a.nemotron.NGroups)
	}

	tk, err := tokenizer.LoadGGUF(gguf)
	if err != nil {
		t.Fatalf("LoadGGUF tokenizer: %v", err)
	}
	// AUDIT NOTE (da5a6ec): raw completion prompt on an instruction-tuned checkpoint,
	// gated only by the distinct<3 floor below — which measures "did the forward avoid
	// TOTAL collapse", not coherence. On gemma-4-26b-a4b-it a raw prompt manufactured a
	// false "int4 is broken" signal that survived a week, and distinct<3 would not have
	// caught it (repetition has >3 distinct tokens). This gate currently passes, so the
	// completion is in-distribution ENOUGH for this checkpoint — but when it is next
	// revalidated, adopt TestGemma4_26B_gate's pattern (render the family chat template +
	// distinctTrigramRatio floor) instead of trusting distinct<3.
	prompt := "The capital of France is"
	ids, err := tk.Encode(prompt, true)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	out, _ := m.Generate(context.Background(), ids, 24, SamplingParams{})
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
	t.Logf("nemotron real gate OK.\n  prompt: %q\n  cont:   %q", prompt, text)
}

// TestNemotronReal_oracle is the T3 row: the released bf16 safetensors loaded at int8 and
// matched against an HF bf16 forward of the SAME weights (argmax + cosine + greedy
// continuation). The GGUF gate above is a coherence gate on a different artifact (a
// llama.cpp Q8_0 convert) and cannot produce a cosine, which is why nemotron_h sat at
// `pending` in the manifest while having a passing "real gate" — coherent-generation is not
// a T3 method. Structural assertions stay above; this covers the numerics.
//
// Fixture: scripts/pin_nemotron_real.py.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestNemotronReal_oracle -v -timeout 60m
func TestNemotronReal_oracle(t *testing.T) {
	requireHeavyModel(t)
	ckpt := assetPath(t, "GOINFER_NEMOTRON_HF")
	realLogitOracle(t, ckpt, "../testdata/nemotron_real_golden.json", "nemotron_h", "nemotron_h",
		"HF bf16 (NVIDIA-Nemotron-Nano-9B-v2; int8 resident)")
}
