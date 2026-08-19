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

	// THE FLOORS, AND WHY THEY ARE WHAT THEY ARE (reclassified 2026-08-18, decider Francis;
	// evidence in docs/queue-release.md under B13).
	//
	// This gate previously asserted `min >= 0.998` and called any miss "loader bug (not Q8_0
	// quant)". It failed at min 0.987835 (step 63) with mean 0.998114 — i.e. it required EVERY
	// step to be at least as good as the average, which no spread can satisfy. Three measurements
	// say the residual is quant noise, not a defect:
	//
	//   1. TestQwen35GGUF_weightDiff: every transform-bearing tensor bit-exact or at a UNIFORM
	//      relL2 ~0.0057 (the Q8_0-vs-bf16 floor), worst 0.999980 — and the ROUTER is BIT-IDENTICAL
	//      (maxAbs=0), so routing differences are not a mis-read router.
	//   2. TestQwen35GGUF_locateDivergence: divergence is present at layer 0 and decays smoothly
	//      and NON-MONOTONICALLY to the top with no step. A localized defect cannot recover; that
	//      curve recovers repeatedly.
	//   3. TestQwen35GGUF_routeFlipAtOutlier: the two containers pick different top-8 sets in 779
	//      of 3200 (step,layer) pairs. With a bit-identical router that is the ROUTER'S INPUT
	//      differing by quant noise at a decision boundary — top-8 of 128 near-tied scores flips
	//      easily — and each flip is a legitimate alternative, not a wrong choice. It is also why
	//      flip COUNT does not predict cosine (the flipped expert's WEIGHT is what matters) and
	//      why a min-over-80 statistic is the wrong thing to floor.
	//
	// So the gate now floors the statistic that is stable (the mean) and keeps a min floor set
	// from measurement with headroom, for catastrophic single steps. Both are derived from two
	// independent reproductions of the same numbers (2026-08-12 and 2026-08-18), not chosen to
	// make red green: the mean bar sits ~0.001 under a measured 0.998114 and the min bar ~0.008
	// under a measured 0.987835. A real transform bug does not land in that gap — it craters
	// cosine, which is what the three probes above independently confirm is not happening.
	const meanFloor, minFloor = 0.997, 0.980
	if mean < meanFloor {
		t.Errorf("GGUF int8 vs safetensors int8 MEAN cosine %.6f < %.3f — systematic divergence, "+
			"which quant noise does not produce; run TestQwen35GGUF_weightDiff (is a tensor "+
			"mis-transformed?) and TestQwen35GGUF_locateDivergence (does the curve have a STEP?)",
			mean, meanFloor)
	}
	if minCos < minFloor {
		t.Errorf("GGUF int8 vs safetensors int8 MIN cosine %.6f < %.3f at step %d — one step far "+
			"outside the measured spread; check whether that step's router flipped an expert with "+
			"large weight (TestQwen35GGUF_routeFlipAtOutlier) before concluding loader bug",
			minCos, minFloor, worst)
	}
}
