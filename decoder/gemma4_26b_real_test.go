//go:build realckpt

// Real-model gate for Gemma 4 26B-A4B (gemma4, shipped as a unified
// Gemma4ForConditionalGeneration). There is no feasible bf16 logit oracle — a 26B CPU
// forward reference is ~52 GB and glacial in torch — so this is a LOADER + COHERENCE
// gate (the same class as GLM-4.5-Air / Kimi K2): the real safetensors checkpoint must
// load through the unified layout (model.language_model.* prefix auto-detect, the MoE
// arch nested in text_config, the vision tower ignored), expose the gemma4 MoE seams
// (128 experts top-8, the parallel dense+MoE FFN on every layer, K=V global attention),
// and generate coherent, non-degenerate text through the model's REAL chat template.
//
//	GOINFER_GEMMA4_26B=~/models/gemma-4-26b-a4b-it \
//	  go test -tags realckpt ./decoder/ -run TestGemma4_26B -v -timeout 30m
package decoder

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// distinctTrigramRatio returns |distinct 3-rune windows| / |total| over s — a
// language-agnostic degeneracy metric. Coherent prose cycles through many trigrams (high
// ratio); a looping forward reuses a few ("water-water-water", "얓숌면-얓숌면-얓숌면",
// "true or true or") so the ratio collapses. This is what the old printable-ASCII-majority
// check was blind to: it passed int8's English-ish repetition and flagged int4's CJK
// repetition, though both are the SAME degeneracy — see the RESOLUTION in
// docs/task-gemma4-moe.md (da5a6ec).
func distinctTrigramRatio(s string) float64 {
	r := []rune(s)
	if len(r) < 3 {
		return 1
	}
	seen := make(map[string]struct{})
	total := 0
	for i := 0; i+3 <= len(r); i++ {
		seen[string(r[i:i+3])] = struct{}{}
		total++
	}
	return float64(len(seen)) / float64(total)
}

// wantGemma4Prompt is the exact rendered generation prompt for the gate's user turn.
// Asserted verbatim so a template/marker change (e.g. the <|turn>/<turn|> rename that
// 681db0c made optional in tokenizer/sentencepiece.go) surfaces HERE as a clear diff,
// not downstream as mystery garbage.
const wantGemma4Prompt = "<bos><|turn>user\nWhat is the capital of France, and what is it famous for?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"

const gemma4GateUserTurn = "What is the capital of France, and what is it famous for?"

// degeneracyFloor is the distinct-trigram floor for a 64-token greedy continuation,
// calibrated on the REAL recorded samples for this gate:
//
//	coherent int4, templated prompt = 0.841  ("…the Eiffel Tower, the Louvre Museum…")
//	garbage  int4, RAW prompt        = 0.413  (" than than … 얓숌면-얓숌면-얓숌면-…")
//	garbage  int8, RAW prompt        = 0.584  ("…water-water-water is 100°C…")
//
// 0.70 sits above BOTH garbage flavors and below coherent. The rawPromptControl subtest
// re-derives the low end on the live checkpoint so the floor stays empirically anchored.
const degeneracyFloor = 0.70

func TestGemma4_26B_gate(t *testing.T) {
	dir := os.Getenv("GOINFER_GEMMA4_26B")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, "models", "gemma-4-26b-a4b-it")
	}
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		t.Skipf("no 26B checkpoint at %s: %v", dir, err)
	}

	// Run the gate at BOTH precisions. int4 (~13 GB) is the config Phase 5 benchmarks and
	// is fully coherent under the chat template — the earlier "int4 is incoherent" claim
	// was a raw-prompt artifact (retracted, da5a6ec). int8 (~26 GB) is the control.
	for _, quant := range []string{"int4", "int8int8"} {
		t.Run(quant, func(t *testing.T) {
			m, err := Load(dir, Options{Quant: quant})
			if err != nil {
				t.Fatalf("Load(%s, %s): %v", dir, quant, err)
			}
			defer m.Close()
			tk, err := tokenizer.Load(dir)
			if err != nil {
				t.Fatalf("Load tokenizer: %v", err)
			}
			gemma4CheckSeams(t, m)
			gemma4CoherenceGate(t, m, tk, quant)
			if quant == "int4" {
				gemma4RawPromptControl(t, m, tk) // the deliberate mutation must trip the gate
			}
		})
	}
}

// gemma4CheckSeams asserts the gemma4 MoE structure on the real, unified checkpoint.
func gemma4CheckSeams(t *testing.T, m *Model) {
	t.Helper()
	a := m.w.arch
	if a.Name != "gemma4" {
		t.Fatalf("arch = %q, want gemma4", a.Name)
	}
	if a.MoE == nil || a.MoE.NumExperts != 128 || a.MoE.TopK != 8 {
		t.Fatalf("MoE = %+v, want 128 experts top-8", a.MoE)
	}
	if got := len(m.w.Layers); got != 30 {
		t.Fatalf("loaded %d layers, want 30", got)
	}
	nMoE, nVFromK := 0, 0
	for i := range m.w.Layers {
		if m.w.Layers[i].gemma4moe != nil {
			nMoE++
		}
		if m.w.Layers[i].VFromK {
			nVFromK++
		}
	}
	if nMoE != 30 {
		t.Fatalf("gemma4moe present on %d/30 layers (parallel dense+MoE FFN expected on all)", nMoE)
	}
	if nVFromK == 0 {
		t.Errorf("no K=V global layers (attention_k_eq_v) — expected some")
	}
	t.Logf("loaded 26B-A4B: 30 layers, %d MoE (128 experts top-8), %d K=V-global", nMoE, nVFromK)
}

// gemma4CoherenceGate is the real gate: a greedy continuation of a PROPERLY TEMPLATED
// chat turn must contain the known answer and clear the degeneracy floor.
//
// The prompt MUST go through Gemma 4's chat template. A raw completion prompt ("The
// capital of France is") is off-distribution for this instruction-tuned checkpoint: BOTH
// int8 and int4 degenerate there (int8 → "water-water-water is 100°C", int4 → CJK
// repetition), and the old printable-ASCII gate mistook int8's English-ish garbage for
// coherence while flagging int4's CJK — a false precision signal that cost a week
// (resolved da5a6ec). DO NOT reintroduce a raw prompt here as a quality check;
// gemma4RawPromptControl keeps the raw case as a NEGATIVE control only. Greedy + fixed
// prompt is deterministic (sampling does not rescue a raw prompt anyway — verified), so
// the known-answer substring is a legitimate, non-flaky signal.
func gemma4CoherenceGate(t *testing.T, m *Model, tk *tokenizer.Tokenizer, quant string) {
	t.Helper()
	turns := []chat.Turn{{Role: "user", Content: gemma4GateUserTurn}}
	if got := chat.Gemma4().Render("", turns); got != wantGemma4Prompt {
		t.Fatalf("rendered prompt drifted (template change?):\n got: %q\nwant: %q", got, wantGemma4Prompt)
	}
	ids, err := tk.EncodeSegments(chat.Gemma4().RenderSegments("", turns), false)
	if err != nil {
		t.Fatalf("encode segments: %v", err)
	}
	text := gemma4Generate(t, m, tk, ids, 64)

	// Signal 1 — known answer. Greedy + fixed prompt is deterministic, so "Paris" is a
	// non-flaky substring gate for this question (one signal, not the whole check).
	if !strings.Contains(text, "Paris") {
		t.Errorf("[%s] continuation lacks the known answer %q: %q", quant, "Paris", text)
	}
	// Signal 2 — degeneracy floor (the repetition failure mode ASCII-majority missed).
	if r := distinctTrigramRatio(text); r < degeneracyFloor {
		t.Errorf("[%s] continuation is degenerate (distinct-trigram %.3f < %.2f): %q", quant, r, degeneracyFloor, text)
	}
	t.Logf("gemma4 26B gate OK [%s]: distinct-trigram %.3f\n  cont: %q", quant, distinctTrigramRatio(text), text)
}

// gemma4RawPromptControl is the deliberate mutation: feed the RAW completion prompt the
// old gate used and confirm it now TRIPS the degeneracy floor. This proves, on the live
// checkpoint (not just recorded samples), that the gate catches the off-distribution
// failure the printable-ASCII check let through — and documents inline why raw prompts
// are banned as a quality check.
func gemma4RawPromptControl(t *testing.T, m *Model, tk *tokenizer.Tokenizer) {
	t.Helper()
	ids, err := tk.Encode("The capital of France is", true)
	if err != nil {
		t.Fatalf("encode raw: %v", err)
	}
	text := gemma4Generate(t, m, tk, ids, 64)
	r := distinctTrigramRatio(text)
	if r >= degeneracyFloor {
		t.Errorf("raw-prompt control did NOT degenerate (distinct-trigram %.3f ≥ %.2f) — the gate's "+
			"discriminating power is unproven on this checkpoint: %q", r, degeneracyFloor, text)
	}
	t.Logf("raw-prompt control degenerates as expected: distinct-trigram %.3f < %.2f\n  cont: %q", r, degeneracyFloor, text)
}

func gemma4Generate(t *testing.T, m *Model, tk *tokenizer.Tokenizer, ids []int, maxTok int) string {
	t.Helper()
	out, _ := m.Generate(context.Background(), ids, maxTok, SamplingParams{}) // greedy, fixed
	gen := make([]int, 0, maxTok)
	for id := range out {
		gen = append(gen, id)
	}
	if len(gen) == 0 {
		t.Fatal("no tokens generated")
	}
	text, derr := tk.Decode(gen)
	if derr != nil {
		t.Fatalf("decode: %v", derr)
	}
	if strings.TrimSpace(text) == "" {
		t.Fatal("decoded continuation is empty")
	}
	return text
}
