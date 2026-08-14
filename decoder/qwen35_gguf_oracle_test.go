//go:build realckpt

// Definitive GGUF-loader gate against the task's LITERAL oracle: the
// SAFETENSORS-loaded int8 path on the same released weights. Both models are
// loaded int8 and fed IDENTICAL teacher-forced inputs (the golden token
// sequence), so their per-step logits are directly comparable — the only
// remaining difference is the container (GGUF Q8_0→int8 double-quant vs
// safetensors bf16→int8) plus whatever the GGUF loader's transform-reversal does.
// A correct loader ⇒ per-step cosine ≈ 1 (only the tiny Q8_0 double-quant delta);
// a V-reorder / norm / A_log bug ⇒ cosine craters. This isolates LOADER from
// quant in a way the bf16-golden comparison (qwen35_gguf_gate_test.go) cannot.
//
//	GOINFER_QWEN35_DIR=~/models/qwen3.6-35b-a3b \
//	GOINFER_QWEN35_GGUF=~/models/qwen3.6-35b-a3b-Q8_0.gguf \
//	  go test -tags realckpt ./decoder/ -run TestQwen35GGUF_vsSafetensors -v -timeout 60m
package decoder

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"testing"
)

func TestQwen35GGUF_vsSafetensors(t *testing.T) {
	requireHeavyModel(t)
	gguf := assetPath(t, "GOINFER_QWEN35_GGUF")
	dir := realQwen35Dir(t) // safetensors dir (skips if absent)
	goldenDir := assetPath(t, "GOINFER_QWEN35_GOLDEN")
	raw, err := os.ReadFile(filepath.Join(goldenDir, "manifest.json"))
	if err != nil {
		t.Skipf("no golden manifest: %v", err)
	}
	var man gate2Manifest
	if err := json.Unmarshal(raw, &man); err != nil {
		t.Fatalf("manifest: %v", err)
	}

	// teacherForced runs each prompt with the golden token sequence as input
	// (prefill prompt[:-1], then feed prompt[-1] and each golden GenID), returning
	// the last-position logits per (prompt, step). Identical inputs for both models.
	teacherForced := func(label, path string) [][]float32 {
		prev := runtime.GOMAXPROCS(2)
		m, err := Load(path, Options{Quant: "int8int8"})
		runtime.GOMAXPROCS(prev)
		if err != nil {
			t.Fatalf("%s Load: %v", label, err)
		}
		defer func() {
			m.Close()
			debug.FreeOSMemory() // return the ~39 GB before the next model loads
		}()
		if m.w.arch.qwen35 == nil {
			t.Fatalf("%s arch = %q, want qwen3_5_moe", label, m.w.arch.Name)
		}
		var out [][]float32
		for pi := range man.Prompts {
			p := man.Prompts[pi]
			cache := m.NewCache(len(p.PromptIDs) + len(p.GenIDs))
			for _, id := range p.PromptIDs[:len(p.PromptIDs)-1] {
				if _, err := m.runLayers(id, cache); err != nil {
					t.Fatalf("%s prompt %d prefill: %v", label, pi, err)
				}
			}
			cur := p.PromptIDs[len(p.PromptIDs)-1]
			for s := range p.GenIDs {
				logits, err := m.forward(cur, cache)
				if err != nil {
					t.Fatalf("%s prompt %d step %d: %v", label, pi, s, err)
				}
				out = append(out, append([]float32(nil), logits...))
				cur = p.GenIDs[s] // teacher force with the golden token
			}
		}
		return out
	}

	// Safetensors oracle first; freed (FreeOSMemory) before the GGUF loads so the
	// two ~39 GB models never coexist.
	ref := teacherForced("safetensors", dir)
	got := teacherForced("gguf", gguf)
	if len(ref) != len(got) {
		t.Fatalf("step count mismatch: safetensors %d vs gguf %d", len(ref), len(got))
	}

	minCos, sumCos := 1.0, 0.0
	worst := -1
	for i := range ref {
		c := cosineFull(got[i], ref[i])
		sumCos += c
		if c < minCos {
			minCos, worst = c, i
		}
	}
	mean := sumCos / float64(len(ref))
	t.Logf("=== GGUF int8 vs SAFETENSORS int8 (teacher-forced, %d steps): cosine min=%.6f (step %d) mean=%.6f ===",
		len(ref), minCos, worst, mean)

	// A correct loader makes the GGUF int8 logits track the safetensors int8 oracle
	// to within the Q8_0-vs-bf16 double-quant delta (~1e-3). Any loader bug in the
	// V-head un-tile, the −exp(A_log)/norm(+1) un-bake, or the q‖gate handling
	// shifts logits far more and craters this cosine.
	if minCos < 0.998 {
		t.Errorf("GGUF int8 vs safetensors int8 min cosine %.6f < 0.998 — loader bug (not Q8_0 quant)", minCos)
	}
}
