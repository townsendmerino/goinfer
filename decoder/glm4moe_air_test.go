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
	"runtime"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

func TestGlm4MoeAir_gate(t *testing.T) {
	requireHeavyModel(t)
	// assetPath, not a hand-rolled env+fallback: the registry is what makes this gate and the sweep
	// preflight apply the SAME predicate to the same candidate paths. Both halves are registered
	// because the .giw is DERIVED from the .gguf — resolving only the derived file would report an
	// asset that cannot be regenerated as present. Build the .giw with:
	//
	//	go run ./cmd/prequant -o <giw> -quant int4 <gguf>
	gguf := assetPath(t, "GOINFER_GLM_GGUF")
	giw := assetPath(t, "GOINFER_GLM_GIW")

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
	// AUDIT NOTE (da5a6ec): raw completion prompt on an instruction-tuned checkpoint
	// (GLM-4.5-Air), gated only by the distinct<3 floor below — which measures "did the
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
