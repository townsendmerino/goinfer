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
//
// TestR9_w4a8BatchAt7B is a follow-up to R-06 (docs/tasks/task-recompute-audit.md), which measured
// GOINFER_W4A8_BATCH's fused q/k/v and gate/up matmuls on the 1.5B only (1.071x arm64/Metal,
// 1.066x amd64/CPU -- both ambiguous, parked). R9 step 1's own finding (MLP's share of the token
// grows with model size, largest at 7B) raises the natural follow-up: does R-06 behave differently
// at 7B? w4a8BatchEnabled (decoder/weightmat.go) is a package-level var read once from
// GOINFER_W4A8_BATCH at process start, so this test cannot toggle it mid-process -- it reports
// this run's own mean/stdev over repeated decode windows (fresh KV cache each, same loaded model,
// no reload between repeats) and is meant to be run TWICE, once per env setting, the results
// compared by hand (or by a small script) rather than paired within one process. That is a real
// limitation next to R-06's own interleaved design -- disclosed, not hidden -- see the record.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_DECODE_TIMING=1 GOINFER_NO_FIT_GUARD=1 GOINFER_W4A8_BATCH=0 go test -tags goinfer_testhooks ./decoder/ -run TestR9_w4a8BatchAt7B -v -timeout 20m
//	GOINFER_HEAVY_TESTS=1 GOINFER_DECODE_TIMING=1 GOINFER_NO_FIT_GUARD=1 GOINFER_W4A8_BATCH=1 go test -tags goinfer_testhooks ./decoder/ -run TestR9_w4a8BatchAt7B -v -timeout 20m
func TestR9_w4a8BatchAt7B(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a real 7B checkpoint on CPU)")
	}
	if os.Getenv("GOINFER_DECODE_TIMING") == "" {
		t.Skip("set GOINFER_DECODE_TIMING=1 so decoder.decodeTiming prints the per-window forward ms/token")
	}
	path := os.Getenv("GOINFER_CPU_MODEL_D7")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s (set GOINFER_CPU_MODEL_D7)", path)
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
	text := "Continue this text. " + strings.Repeat(" the", 128)
	ids, err := tk.Encode(text, true)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if len(ids) > 128 {
		ids = ids[:128]
	}
	const repeats = 6
	const decodeN = 16
	t.Logf("w4a8Batch env=%q, %d repeats x %d tokens, fresh cache each (see individual DECODE TIMING lines above)", os.Getenv("GOINFER_W4A8_BATCH"), repeats, decodeN)
	for i := range repeats {
		out, g := m.Generate(context.Background(), ids, decodeN, SamplingParams{Temperature: 0})
		n := 0
		for range out {
			n++
		}
		if g.err != nil {
			t.Fatalf("repeat %d: generate: %v", i, g.err)
		}
	}
}

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
