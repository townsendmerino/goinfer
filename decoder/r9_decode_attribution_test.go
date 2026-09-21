//go:build goinfer_testhooks

package decoder

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestR9_decodeAttribution is docs/tasks/red-october.md R9, step 1, Mac half: the per-component
// ms/token table step 1 asks for, using the already-shipped GOINFER_DECODE_TIMING diagnostic
// (decoder/model.go's decodeTiming var — forward/sample/logitProc/embed, already split; this test
// adds nothing new there) at the brief's own decision-cell depth (128), greedy, on 0.5B/1.5B/7B.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_DECODE_TIMING=1 go test -tags goinfer_testhooks ./decoder/ -run TestR9_decodeAttribution -v -timeout 20m
func TestR9_decodeAttribution(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads real checkpoints on CPU)")
	}
	if os.Getenv("GOINFER_DECODE_TIMING") == "" {
		t.Skip("set GOINFER_DECODE_TIMING=1 so decoder.decodeTiming prints the per-component split")
	}

	models := []struct {
		name string
		env  string
		def  string
	}{
		{"0.5B", "GOINFER_CPU_MODEL_05B", "$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"},
		{"1.5B", "GOINFER_CPU_MODEL", "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"},
		{"7B", "GOINFER_CPU_MODEL_D7", "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf"},
	}
	const promptDepth = 128 // R9's own decision-cell depth
	const decodeN = 24      // enough tokens for decodeTiming's per-token average to settle

	for _, mc := range models {
		t.Run(mc.name, func(t *testing.T) {
			path := os.Getenv(mc.env)
			if path == "" {
				path = os.ExpandEnv(mc.def)
			}
			if _, err := os.Stat(path); err != nil {
				t.Skipf("no fixture at %s (set %s)", path, mc.env)
			}
			m, err := Load(path, Options{Backend: "cpu", Quant: "int4"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			tk, err := tokenizer.LoadGGUF(path)
			if err != nil {
				t.Fatalf("load tokenizer: %v", err)
			}
			// A calibrated-shape filler prompt (scripts/bench_prompts_calibrate.py's own
			// convention: " the" repeats at ~1 token/word) tokenized for real by this model's own
			// tokenizer -- content doesn't matter for a timing run, real tokenization does.
			text := "Continue this text. " + strings.Repeat(" the", promptDepth)
			ids, err := tk.Encode(text, true)
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			if len(ids) > promptDepth {
				ids = ids[:promptDepth]
			}
			t.Logf("%s: prompt %d tokens, decoding %d greedy", mc.name, len(ids), decodeN)
			out, g := m.Generate(context.Background(), ids, decodeN, SamplingParams{Temperature: 0})
			n := 0
			for range out {
				n++
			}
			if g.err != nil {
				t.Fatalf("generate: %v", g.err)
			}
			t.Logf("%s: %d tokens generated (DECODE TIMING line printed above by decoder.decodeTiming)", mc.name, n)
		})
	}
}
