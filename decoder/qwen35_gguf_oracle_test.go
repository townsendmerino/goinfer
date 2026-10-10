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
	"errors"
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
		if errors.Is(err, ErrWontFitResident) {
			t.Skipf("%s: fit-guard declined on this box: %v", label, err)
		}
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

	// THE FLOORS (reclassified by the owner; evidence in docs/queue-release.md under B13). The gate once required `min >= 0.998`
	// and called any miss a loader bug, which asks every step to be at least as good as the average, and no spread can satisfy
	// that. Three probes say the residual is quant noise, not a defect:
	//   1. TestQwen35GGUF_weightDiff: every transform-bearing tensor is bit-exact or at a UNIFORM relL2 (the Q8_0-vs-bf16
	//      floor), and the ROUTER is BIT-IDENTICAL, so routing differences are not a mis-read router.
	//   2. TestQwen35GGUF_locateDivergence: divergence is present at layer 0 and decays smoothly and NON-MONOTONICALLY to the
	//      top with no step; a localized defect cannot recover.
	//   3. TestQwen35GGUF_routeFlipAtOutlier: the containers pick different top-8 sets in many (step,layer) pairs. With a
	//      bit-identical router that is the router's INPUT differing by quant noise at a decision boundary, each flip a
	//      legitimate alternative. It is also why flip COUNT does not predict cosine and why a min-over-80 statistic is the wrong
	//      thing to floor.
	//
	// So the gate floors the stable statistic (the mean) and keeps a min floor, set from measurement with headroom, for
	// catastrophic single steps. Both bars come from reproduced measurements, not from making red green, and a real transform
	// bug does not land in the gap between them and the measured values: it craters cosine, which the three probes confirm is
	// not happening. Move a bar only with a mechanism.
	//
	// The mean was re-baselined once, with the mechanism confirmed by commit bisection: both loaders now quantize the DeltaNet
	// in/out-proj and gated-softmax q/k/v/o projections to int8 via quantizeWM, so each side re-quantizes its own slightly
	// different f32 view and a delta small in f32 can land in different int8 buckets. The min floor kept its value. This gate
	// lived outside the gate-ledger manifest at the re-baseline, so its red was not listed as a blocker and went unnoticed
	// until a parity sweep caught it. Measured values and commits: docs/code-notes/decoder.md#TestQwen35GGUF_vsSafetensors.floors.
	const meanFloor, minFloor = 0.9946, 0.980
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
