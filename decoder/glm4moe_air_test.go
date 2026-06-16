//go:build realckpt

// Real-model gate for GLM-4.5-Air (glm4moe, 106B-A12B) — the GGUF loader on actual
// released weights, not the tiny synthetic. There is no full bf16 logit oracle (a
// 106B forward reference won't fit the box), so this is a LOADER + COHERENCE gate:
// the real GGUF must load through the stream-weights (.giw) path under a RAM budget,
// expose the GLM seams correctly (46 layers after dropping the NextN block, q/k/v
// bias, the dense layer-0 prefix, the e_score_correction_bias on MoE layers), and
// generate coherent text on a canned prompt. The unsloth GGUF is produced by
// llama.cpp convert, so it also confirms the real tensor conventions the tiny
// fixture had to assume (exp_probs_b.bias, post_attention_norm, stacked experts).
//
// Build the int4 .giw first (the test can't import internal/prequant — it would
// cycle through decoder):
//
//	go run ./cmd/prequant -o ~/models/glm-air/GLM-4.5-Air-Q2_K.int4.giw \
//	  -quant int4 ~/models/glm-air/GLM-4.5-Air-Q2_K.gguf
//	GOINFER_GLM_GGUF=~/models/glm-air/GLM-4.5-Air-Q2_K.gguf \
//	  go test -tags realckpt ./decoder/ -run TestGlm4MoeAir -v -timeout 30m
package decoder

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

func TestGlm4MoeAir_gate(t *testing.T) {
	home, _ := os.UserHomeDir()
	gguf := os.Getenv("GOINFER_GLM_GGUF")
	if gguf == "" {
		gguf = filepath.Join(home, "models", "glm-air", "GLM-4.5-Air-Q2_K.gguf")
	}
	if _, err := os.Stat(gguf); err != nil {
		t.Skipf("no GLM-4.5-Air GGUF at %s: %v", gguf, err)
	}
	giw := os.Getenv("GOINFER_GLM_GIW")
	if giw == "" {
		giw = strings.TrimSuffix(gguf, ".gguf") + ".int4.giw"
	}
	if _, err := os.Stat(giw); err != nil {
		t.Skipf("no int4 .giw at %s — build with: go run ./cmd/prequant -o %s -quant int4 %s", giw, giw, gguf)
	}

	// Stream the experts out of an mmap'd int4 .giw under a RAM budget — a 106B
	// model won't sit resident in 62 GB. Bound the load fan-out (each in-flight
	// layer briefly holds its experts), as the qwen35 gate does.
	prev := runtime.GOMAXPROCS(2)
	m, err := Load(giw, Options{StreamWeights: true, Quant: "int4", WeightCacheBytes: 24 << 30})
	runtime.GOMAXPROCS(prev)
	if err != nil {
		t.Fatalf("Load(%s, stream int4): %v", giw, err)
	}
	defer m.Close()

	// --- GLM seams on the real model ---
	if m.w.arch.Name != "glm4_moe" {
		t.Fatalf("arch = %q, want glm4_moe", m.w.arch.Name)
	}
	if got := len(m.w.Layers); got != 46 {
		t.Fatalf("loaded %d layers, want 46 (block_count 47 minus the NextN block)", got)
	}
	if !m.w.arch.QKVBias {
		t.Errorf("QKVBias false; GLM-4.5-Air carries q/k/v bias")
	}
	if m.w.Layers[0].Experts != nil {
		t.Errorf("layer 0 (first_k_dense_replace) has experts; want dense")
	}
	if m.w.Layers[1].Experts == nil || len(m.w.Layers[1].RouterBias) == 0 {
		t.Errorf("layer 1 (MoE) missing experts/RouterBias: experts=%v bias=%d",
			m.w.Layers[1].Experts != nil, len(m.w.Layers[1].RouterBias))
	}
	if len(m.w.Layers[1].QBias) == 0 {
		t.Errorf("layer 1 q_proj bias not loaded")
	}

	// --- coherence: greedy continuation of a canned prompt ---
	tk, err := tokenizer.LoadGGUF(gguf)
	if err != nil {
		t.Fatalf("LoadGGUF tokenizer: %v", err)
	}
	prompt := "The capital of France is"
	ids, err := tk.Encode(prompt, true)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	out, _ := m.Generate(context.Background(), ids, 40, SamplingParams{}) // greedy
	gen := make([]int, 0, 40)
	for id := range out {
		gen = append(gen, id)
	}
	if len(gen) == 0 {
		t.Fatal("no tokens generated")
	}
	// Non-degenerate: not a single repeated token.
	distinct := map[int]bool{}
	for _, id := range gen {
		distinct[id] = true
	}
	if len(distinct) < 3 {
		t.Errorf("degenerate output: only %d distinct tokens in %d", len(distinct), len(gen))
	}
	text, derr := tk.Decode(gen)
	if derr != nil {
		t.Fatalf("decode: %v", derr)
	}
	if strings.TrimSpace(text) == "" {
		t.Error("decoded continuation is empty")
	}
	t.Logf("GLM-4.5-Air gate OK — 46 layers, qkv-bias, dense L0, MoE bias.\n  prompt: %q\n  cont:   %q", prompt, text)
}
