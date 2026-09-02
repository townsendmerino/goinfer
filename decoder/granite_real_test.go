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
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

func TestGraniteReal_gate(t *testing.T) {
	requireHeavyModel(t)
	// assetPath, not a hand-rolled env+fallback: the registry is what makes this gate and the
	// sweep preflight apply the SAME predicate to the same candidate paths. Required by the sweep
	// since 2026-09-02, and a required gate whose presence check disagrees with the preflight's is
	// a SKIP nobody can attribute.
	gguf := assetPath(t, "GOINFER_GRANITE_GGUF")

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

	// NoPE on the GGUF path. The Q8_0 convert carries the same base weights as the
	// safetensors release, so the bf16 golden is a valid reference for it too — and it is
	// the ONLY check that the rope.scaling.finetuned → NoPE mapping in ggufGraniteConfig is
	// right, since llama.cpp writes rope.dimension_count/freq_base on this model regardless.
	// Roped, this reads ~0.9936 with a diverging continuation; NoPE, ~0.9958 and exact. Not
	// a parity row: the T3 row is the safetensors oracle below, on weights HF actually ran.
	if !a.isNoPELayer(0) {
		t.Errorf("granitehybrid GGUF resolved to roped attention; the released granite-4.0-h models are NoPE")
	}
	graniteGGUFvsOracle(t, m)
}

// graniteGGUFvsOracle replays the bf16 golden's prompt through an already-loaded Granite
// model and gates the last-token argmax + cosine + greedy continuation. Split out so the
// GGUF gate above reuses its own load rather than quantizing the model a second time.
func graniteGGUFvsOracle(t *testing.T, m *Model) {
	t.Helper()
	raw, err := os.ReadFile("../testdata/granite_real_golden.json")
	if err != nil {
		t.Skipf("no golden (%v) — run scripts/pin_granite_real.py", err)
	}
	var g struct {
		PromptIDs       []int     `json:"prompt_ids"`
		Argmax          int       `json:"argmax"`
		LastLogits      []float32 `json:"last_logits"`
		NNew            int       `json:"n_new"`
		ContinuationIDs []int     `json:"continuation_ids"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	cache := m.NewCache(len(g.PromptIDs) + g.NNew)
	var logits []float32
	for _, id := range g.PromptIDs {
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("forward: %v", err)
		}
	}
	cos := logitCosine(logits, g.LastLogits)
	t.Logf("granite GGUF vs bf16 oracle: argmax got=%d want=%d | cosine=%.6f", argmax(logits), g.Argmax, cos)
	if got := argmax(logits); got != g.Argmax {
		t.Errorf("GGUF last argmax = %d, want %d", got, g.Argmax)
	}
	if cos < 0.99 { // Q8_0 convert + int8 activations vs bf16
		t.Errorf("GGUF last-logit cosine %.6f < 0.99", cos)
	}
	for i := range g.ContinuationIDs {
		id := argmax(logits)
		if id != g.ContinuationIDs[i] {
			t.Errorf("GGUF continuation[%d] = %d, want %d", i, id, g.ContinuationIDs[i])
			break
		}
		if logits, err = m.forward(id, cache); err != nil {
			t.Fatalf("continuation forward: %v", err)
		}
	}
}

// TestGraniteReal_oracle is the T3 row: the released bf16 safetensors loaded at int8 and
// matched against an HF bf16 forward of the SAME weights. The gate above runs on a
// DIFFERENT artifact (a llama.cpp Q8_0 convert), and until this golden existed it had no
// reference to compare against at all — which is why granitemoehybrid sat at `pending`
// while having a passing "real gate": coherent-generation is not a T3 method. The row is
// recorded here, on the weights HF actually ran.
//
// Fixture: scripts/pin_granite_real.py.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestGraniteReal_oracle -v -timeout 60m
func TestGraniteReal_oracle(t *testing.T) {
	requireHeavyModel(t)
	ckpt := assetPath(t, "GOINFER_GRANITE_HF")
	realLogitOracle(t, ckpt, "../testdata/granite_real_golden.json", "granitemoehybrid", "granitemoehybrid",
		"HF bf16 (Granite-4.0-H-Tiny 7B-A1B; int8 resident)")
}
