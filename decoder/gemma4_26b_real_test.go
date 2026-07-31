//go:build realckpt

// Real-model gate for Gemma 4 26B-A4B (gemma4, shipped as a unified
// Gemma4ForConditionalGeneration). There is no feasible bf16 logit oracle — a 26B CPU
// forward reference is ~52 GB and glacial in torch — so this is a LOADER + COHERENCE
// gate (the same class as GLM-4.5-Air / Kimi K2): the real safetensors checkpoint must
// load through the unified layout (model.language_model.* prefix auto-detect, the MoE
// arch nested in text_config, the vision tower ignored), expose the gemma4 MoE seams
// (128 experts top-8, the parallel dense+MoE FFN on every layer, K=V global attention),
// and generate coherent, non-degenerate text.
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

	"github.com/townsendmerino/goinfer/tokenizer"
)

func TestGemma4_26B_gate(t *testing.T) {
	dir := os.Getenv("GOINFER_GEMMA4_26B")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, "models", "gemma-4-26b-a4b-it")
	}
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		t.Skipf("no 26B checkpoint at %s: %v", dir, err)
	}

	// int8 (~26 GB resident) fits a 64 GB box; the bf16 safetensors is mmap'd and
	// quantized per-tensor at load (experts streamed one at a time via SubF32), so peak
	// stays near the int8 resident, not the 52 GB source. NOTE: int4 is too aggressive
	// for this model — the 128-expert top-8 MoE compounds int4's per-expert error into
	// incoherent output (measured); int8 generates coherent English. So the real-model
	// gate runs int8. (The int4 path is exercised on the tiny checkpoint instead.)
	m, err := Load(dir, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("Load(%s, int8): %v", dir, err)
	}
	defer m.Close()

	// --- gemma4 MoE seams on the real, unified checkpoint ---
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

	// --- coherence: greedy continuation of a canned prompt ---
	tk, err := tokenizer.Load(dir)
	if err != nil {
		t.Fatalf("Load tokenizer: %v", err)
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
	text, derr := tk.Decode(gen)
	if derr != nil {
		t.Fatalf("decode: %v", derr)
	}
	if strings.TrimSpace(text) == "" {
		t.Error("decoded continuation is empty")
	}
	// Coherence: the continuation must be mostly printable ASCII (English prose), which a
	// broken forward is NOT — the pre-fix run emitted CJK/control-char garbage
	// ("…얓숌면…", all-dashes). A distinct-token count alone passed on that garbage; this
	// gates the actual failure mode. A working forward here reads e.g. " France.\n**Q:…".
	printable := 0
	for _, r := range text {
		if r == '\n' || r == '\t' || (r >= 0x20 && r < 0x7f) {
			printable++
		}
	}
	frac := float64(printable) / float64(len([]rune(text))+1)
	if frac < 0.85 {
		t.Errorf("continuation is not coherent English (%.0f%% printable-ASCII): %q", 100*frac, text)
	}
	t.Logf("Gemma 4 26B-A4B gate OK\n  prompt: %q\n  cont:   %q", prompt, text)
}
