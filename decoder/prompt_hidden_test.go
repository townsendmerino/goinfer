package decoder

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestPromptHidden_matchesHF is D2's gate (docs/tasks/task-constrained-confidence.md): the final-norm hidden state at the last
// prompt position must reach cosine >= 0.9999 against HF's output_hidden_states[-1] on five prompts, in f32, for the family the
// JEV decision models are (Qwen3.5), dense and MoE. The reference is scripts/pin_prompt_hidden.py, run on the existing tiny
// checkpoints; its prompts are 2..64 tokens, never 1, where attention is the identity. Cosine alone cannot see the final norm when
// its weights are uniform (it only rescales), and the tiny checkpoints' are all 0, a scale of 1 under the add-one RMSNorm. So the
// golden's first fixture, qwen3_5-tiny-normw, carries a random final-norm weight, and every prompt also has to meet a relative L2
// error bound, which a wrong scale fails.
func TestPromptHidden_matchesHF(t *testing.T) {
	golden := filepath.Join("testdata", "prompt_hidden_golden.json")
	raw, err := os.ReadFile(golden)
	if errors.Is(err, fs.ErrNotExist) {
		t.Skip("no golden: run scripts/pin_prompt_hidden.py")
	}
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Fixtures []struct {
			Checkpoint string
			ModelType  string `json:"model_type"`
			Prompts    []struct {
				IDs    []int
				Hidden []float64
			}
		}
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if len(g.Fixtures) == 0 {
		t.Fatal("golden has no fixtures")
	}
	for _, fx := range g.Fixtures {
		t.Run(fx.Checkpoint, func(t *testing.T) {
			ckpt := filepath.Join("testdata", fx.Checkpoint)
			if _, err := os.Stat(filepath.Join(ckpt, "model.safetensors")); errors.Is(err, fs.ErrNotExist) {
				t.Skipf("no checkpoint at %s (scripts/pin_qwen3_5_forward.py)", ckpt)
			}
			m, err := Load(ckpt, Options{})
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if _, own := m.w.arch.ownForward(); !own {
				t.Fatalf("%s is expected to have its own layer loop (the path HiddenLast refuses); the test would not cover it", fx.Checkpoint)
			}
			for i, p := range fx.Prompts {
				h, err := m.PromptHidden(context.Background(), p.IDs)
				if err != nil {
					t.Fatalf("prompt %d: %v", i, err)
				}
				if len(h) != len(p.Hidden) {
					t.Fatalf("prompt %d: hidden size %d, want %d", i, len(h), len(p.Hidden))
				}
				var dot, na, nb, ne, maxd float64
				for j := range h {
					x, y := float64(h[j]), p.Hidden[j]
					dot, na, nb, ne = dot+x*y, na+x*x, nb+y*y, ne+(x-y)*(x-y)
					maxd = math.Max(maxd, math.Abs(x-y))
				}
				cos, rel := dot/math.Sqrt(na*nb), math.Sqrt(ne/nb)
				if cos < 0.9999 {
					t.Errorf("prompt %d (%d tokens): cosine %.8f < 0.9999 (max |diff| %.3g)", i, len(p.IDs), cos, maxd)
				}
				if rel > 1e-5 {
					t.Errorf("prompt %d (%d tokens): relative L2 error %.3g > 1e-5 (cosine %.8f): the scale is wrong", i, len(p.IDs), rel, cos)
				}
				t.Logf("prompt %d (%2d tokens): cosine %.8f, relative L2 %.2g, max |diff| %.3g", i, len(p.IDs), cos, rel, maxd)
			}
		})
	}
}

// PromptHidden refuses what it cannot score and stops when asked, rather than running a long CPU prefill to the end.
func TestPromptHidden_refusesAndCancels(t *testing.T) {
	ckpt := filepath.Join("testdata", "qwen3_5-tiny")
	if _, err := os.Stat(filepath.Join(ckpt, "model.safetensors")); errors.Is(err, fs.ErrNotExist) {
		t.Skip("no tiny qwen3_5 checkpoint")
	}
	m, err := Load(ckpt, Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if _, err := m.PromptHidden(context.Background(), nil); err == nil {
		t.Error("an empty prompt must be refused")
	}
	if _, err := m.PromptHidden(context.Background(), []int{1, m.w.arch.VocabSize}); err == nil {
		t.Error("an out-of-vocab id must be refused")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := m.PromptHidden(ctx, []int{1, 2, 3}); !errors.Is(err, context.Canceled) {
		t.Errorf("a cancelled context must stop the prefill, got %v", err)
	}
}
