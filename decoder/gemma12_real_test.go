//go:build realckpt

// Real-checkpoint gate for Gemma 1 (gemma-2b-it), CodeGemma (codegemma-2b) and Gemma 2 (gemma-2-2b-it): the tiny-fixture
// parity (gemma12_test.go) on released weights. Against scripts/pin_gemma12_real.py's HF f32 eager golden:
//
//   - safetensors at f32: argmax and the 8-token greedy continuation exact, last-logit cosine >= 0.999 on both the
//     per-token and the batched prefill path (the final-logit softcap and GELU-tanh keep a Gemma's f32 cosine a little
//     under 1, as gemma3_real_test.go measured);
//
//   - the llama.cpp Q8_0 GGUF, when present: argmax and the first 4 continuation tokens exact, cosine >= 0.99 against the
//     same f32 golden (a loader check through a real llama.cpp conversion, where Q8 rounding is the only expected
//     difference).
//
//     GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestGemma12Real -v -timeout 30m
package decoder

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestGemma12Real_gate(t *testing.T) {
	requireHeavyModel(t)
	raw, err := readGolden("../testdata/gemma12_real_golden.json.gz")
	if err != nil {
		t.Skipf("no golden (%v): run scripts/pin_gemma12_real.py", err)
	}
	var g struct {
		Models []struct {
			Name            string
			ModelType       string    `json:"model_type"`
			PromptIDs       []int     `json:"prompt_ids"`
			Argmax          int       `json:"argmax"`
			LastLogits      []float32 `json:"last_logits"`
			NNew            int       `json:"n_new"`
			ContinuationIDs []int     `json:"continuation_ids"`
		}
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	home, _ := os.UserHomeDir()
	envs := map[string]string{"gemma-2b-it": "GEMMA_2B", "codegemma-2b": "CODEGEMMA_2B", "gemma-2-2b-it": "GEMMA2_2B"}
	ggufs := map[string]string{"gemma-2b-it": "gemma-2b-it-q8_0.gguf", "codegemma-2b": "codegemma-2b-Q8_0.gguf", "gemma-2-2b-it": "gemma-2-2b-it-Q8_0.gguf"}
	for _, gm := range g.Models {
		t.Run(gm.Name, func(t *testing.T) {
			run := func(path string, opts Options, cosBar float64, contN int, label string) float64 {
				m, err := Load(path, opts)
				if err != nil {
					t.Fatalf("%s: Load(%s): %v", label, path, err)
				}
				defer m.Close()
				if m.w.arch.Name != gm.ModelType {
					t.Fatalf("%s: arch %q, want %q", label, m.w.arch.Name, gm.ModelType)
				}
				cache := m.NewCache(len(gm.PromptIDs) + gm.NNew)
				var logits []float32
				for _, id := range gm.PromptIDs {
					if logits, err = m.forward(id, cache); err != nil {
						t.Fatal(err)
					}
				}
				// A GGUF can carry more tokens than the HF checkpoint (lmstudio's gemma-2b-it: 256,128 against 256,000, the
				// extra entries at the end), so the cosine is over the shared vocabulary; the argmax is over everything,
				// so an extra token winning would still fail.
				ref := gm.LastLogits
				shared := func(l []float32) []float32 {
					if len(l) > len(ref) {
						return l[:len(ref)]
					}
					return l
				}
				if len(logits) != len(ref) {
					t.Logf("%s: %d logits against the reference's %d; the cosine is over the first %d", label, len(logits), len(ref), len(ref))
				}
				cos := logitCosine(shared(logits), ref)
				bl, err := m.prefillLogits(context.Background(), gm.PromptIDs, m.NewCache(len(gm.PromptIDs)))
				if err != nil {
					t.Fatal(err)
				}
				bcos := logitCosine(shared(bl), ref)
				got := make([]int, 0, gm.NNew)
				for range gm.NNew {
					id := argmax(logits)
					got = append(got, id)
					if logits, err = m.forward(id, cache); err != nil {
						t.Fatal(err)
					}
				}
				t.Logf("%s %s: argmax %d (want %d), cosine per-token %.6f batched %.6f, continuation %v (want %v)",
					gm.Name, label, got[0], gm.Argmax, cos, bcos, got, gm.ContinuationIDs)
				if argmax(bl) != gm.Argmax {
					t.Errorf("%s: batched argmax %d, want %d", label, argmax(bl), gm.Argmax)
				}
				if cos < cosBar || bcos < cosBar {
					t.Errorf("%s: cosine %.6f / %.6f < %.3f", label, cos, bcos, cosBar)
				}
				for i := range contN {
					if got[i] != gm.ContinuationIDs[i] {
						t.Errorf("%s: continuation[%d] = %d, want %d", label, i, got[i], gm.ContinuationIDs[i])
						break
					}
				}
				return min(cos, bcos)
			}
			ckpt := os.Getenv(envs[gm.Name])
			if ckpt == "" {
				ckpt = filepath.Join(home, "models", gm.Name)
			}
			if _, err := os.Stat(ckpt); err != nil {
				t.Skipf("no checkpoint at %s", ckpt)
			}
			cos := run(ckpt, Options{}, 0.999, gm.NNew, "safetensors f32")
			if p := filepath.Join(home, "models", ggufs[gm.Name]); fileExists(p) {
				run(p, Options{}, 0.99, 4, "GGUF Q8_0")
			} else {
				t.Logf("no %s: the GGUF half is skipped", p)
			}
			switch gm.Name {
			case "gemma-2b-it":
				emitParityRow(t, "gemma", "full-forward-oracle", "HF f32 eager (gemma-2b-it; codegemma-2b and the Q8_0 GGUFs also checked)", 100.0, cos, cos)
			case "gemma-2-2b-it":
				emitParityRow(t, "gemma2", "full-forward-oracle", "HF f32 eager (gemma-2-2b-it; the Q8_0 GGUF also checked)", 100.0, cos, cos)
			}
		})
	}
}

func fileExists(p string) bool { _, err := os.Stat(p); return err == nil }
