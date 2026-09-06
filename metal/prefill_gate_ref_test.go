//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestPrefillGateVsReference is Phase B of docs/task-prefill-gap.md §3.1's L1 re-run — the
// corrected §3 gate. TestPrefillGate (prefill_gate_test.go) scored Metal's fast (f16-activation)
// path against Metal's own exact (int8-per-row-activation) path and treated the exact path as
// truth; §3.1 found that comparison cannot distinguish a defect in the fast path from the exact
// path's own quantisation loss, since both are lossy relative to f32 activations. That test's
// numbers stand as a measurement of fast-vs-exact distance (still printed by
// runPrefillGateModel) but no longer gate anything.
//
// This test instead scores BOTH Metal arms — exact (sequential Forward) and fast (batched
// PrefillLast) — against a THIRD, external reference: the CPU backend's own sequential forward
// with f32 activations (decoder/prefill_ref_gen_test.go, TestPrefillGateReference, run separately
// and in its own process — see that file for why). Reference files live under
// ~/goinfer-logs/prefill-ref/ (not in the repo) and MUST be generated first; a missing file skips
// that cell with a clear message rather than silently falling back to the withdrawn exact-as-
// oracle scoring — that fallback is exactly the mistake §3.1 corrected.
//
// Both arms are teacher-forced on the SAME reference-supplied continuation tokens (not on either
// arm's own greedy output, and not on each other's), so a difference between the two arms'
// per-position agreement is attributable to the arm alone, not to which one's tokens happened to
// be used as the "ground truth" stream.
//
// Gate, pre-registered in §3 (as amended) — per (model, K) DECISION cell (K ∈ {256, 1024}; S's
// K=3900 is a confirmation cell, scored and reported the same way but not part of the ship
// decision): the fast arm ships as Metal's new default for that cell if ALL of:
//
//	(a) fast's hard-flip count vs the reference ≤ exact's hard-flip count (over the same 640
//	    continuation positions, decoder.NearTieArgmaxForTest against refLogits)
//	(b) fast's mean teacher-forced agreement (decoder.TeacherForcedTop1AgreementForTest against
//	    refTokens) ≥ exact's mean − 1.0 percentage point, AND fast ≥ exact on ≥ half the prompts
//	    (the paired per-prompt comparison, not just the cell mean)
//	(c) fast's mean continuation KL(reference ‖ arm) ≤ 1.1 × exact's mean
//
// A model ships only if every decision-set cell ships; the confirmation cell is reported but does
// not veto or approve on its own.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./metal/ -run TestPrefillGateVsReference -v -timeout 4h
func TestPrefillGateVsReference(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads real checkpoints; needs Phase A's reference files)")
	}
	if testing.Short() {
		t.Skip("long-running gate: skipped in -short")
	}
	t.Setenv("GOINFER_METAL_BATCHED_PREFILL", "1")

	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("home dir: %v", err)
	}
	refDir := filepath.Join(home, "goinfer-logs", "prefill-ref")

	models := []struct {
		name        string
		pathEnv     string
		defaultPath string
	}{
		{"S", "GOINFER_METAL_MODEL", "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"},
		{"D7", "GOINFER_METAL_MODEL_D7", "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf"},
	}
	decisionKs := []int{256, 1024}
	confirmKsByModel := map[string][]int{"S": {3900}}

	for _, mc := range models {
		t.Run(mc.name, func(t *testing.T) {
			path := os.Getenv(mc.pathEnv)
			if path == "" {
				path = os.ExpandEnv(mc.defaultPath)
			}
			if _, err := os.Stat(path); err != nil {
				t.Skipf("no fixture at %s (set %s)", path, mc.pathEnv)
			}
			m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			rf, ok := m.ResidentForwardForTest().(*metalResident)
			if !ok {
				t.Skipf("metal resident not built for this model")
			}
			tk, err := tokenizer.LoadGGUF(path)
			if err != nil {
				t.Fatalf("load tokenizer: %v", err)
			}

			allKs := append(append([]int{}, decisionKs...), confirmKsByModel[mc.name]...)
			maxK := 0
			for _, k := range allKs {
				if k > maxK {
					maxK = k
				}
			}
			prompts := make([][]int, 0, len(decoder.PrefillGateProseFiles))
			for _, f := range decoder.PrefillGateProseFiles {
				prompts = append(prompts, decoder.PrefillGateProseIDsForTest(t, tk, f, maxK))
			}

			shipVerdict := true
			anyDecisionCellRun := false
			for _, K := range allKs {
				ran := runPrefillRefGateCellK(t, rf, m, mc.name, K, prompts, refDir)
				if ran == nil {
					continue // missing reference files, already logged
				}
				if slices.Contains(decisionKs, K) {
					anyDecisionCellRun = true
					if !*ran {
						shipVerdict = false
					}
				}
			}
			if !anyDecisionCellRun {
				t.Skip("no decision-set reference cells found under " + refDir +
					" — run TestPrefillGateReference (decoder package) first")
			}
			verdict := map[bool]string{true: "SHIPS", false: "DOES NOT SHIP"}[shipVerdict]
			fmt.Printf("=== %s: §3.1 OVERALL DECISION-SET VERDICT = %s ===\n", mc.name, verdict)
			t.Logf("%s: §3.1 decision-set verdict = %s", mc.name, verdict)
			if !shipVerdict {
				t.Fatalf("§3.1 gate: model %s does not ship on the decision set — see log above for which cell/criterion", mc.name)
			}
		})
	}
}

// runPrefillRefGateCellK runs one (model, K) cell over every prompt, reporting per-prompt and
// per-cell numbers for both arms. Returns nil (and logs why) if the reference files for this cell
// are missing; otherwise returns a pointer to whether this cell ships under the §3.1 gate.
func runPrefillRefGateCellK(t *testing.T, rf *metalResident, m *decoder.Model, modelName string, K int, prompts [][]int, refDir string) *bool {
	t.Helper()
	for pi := range prompts {
		p := filepath.Join(refDir, fmt.Sprintf("%s-K%d-p%d.bin", modelName, K, pi))
		if _, err := os.Stat(p); err != nil {
			t.Logf("%s K=%d: reference file missing (%s) — SKIPPING this cell; no fallback to the "+
				"withdrawn exact-as-oracle scoring. Run TestPrefillGateReference (decoder package) first.",
				modelName, K, p)
			return nil
		}
	}

	var (
		sumExactAgree, sumFastAgree, sumExactKL, sumFastKL float64
		exactHFtotal, fastHFtotal, contN                   int
		worstExactGap, worstFastGap                        float64
		agreeAtLeastHalf, n                                int
	)
	t0 := time.Now()
	for pi, ids := range prompts {
		refPath := filepath.Join(refDir, fmt.Sprintf("%s-K%d-p%d.bin", modelName, K, pi))
		seedRef, refTokens, refLogitsRef, err := decoder.ReadPrefillReferenceForTest(refPath)
		if err != nil {
			t.Fatalf("read reference %s: %v", refPath, err)
		}
		contN = len(refTokens)
		res := runPrefillRefCell(t, rf, m, ids[:K], K, seedRef, refTokens, refLogitsRef)
		n++
		sumExactAgree += res.exactAgreement
		sumFastAgree += res.fastAgreement
		sumExactKL += res.exactMeanKL
		sumFastKL += res.fastMeanKL
		exactHFtotal += res.exactHardFails
		fastHFtotal += res.fastHardFails
		if res.exactWorstGap > worstExactGap {
			worstExactGap = res.exactWorstGap
		}
		if res.fastWorstGap > worstFastGap {
			worstFastGap = res.fastWorstGap
		}
		if res.fastAgreement >= res.exactAgreement {
			agreeAtLeastHalf++
		}
		fmt.Printf("[ref-gate] %s K=%d prompt %2d/%2d exact(seed=%s agree=%.1f%% HF=%d/%d KL=%.4f) "+
			"fast(seed=%s agree=%.1f%% HF=%d/%d KL=%.4f) diff(agree=%+.1fpt KL=%+.4f) elapsed=%s\n",
			modelName, K, pi+1, len(prompts),
			map[bool]string{true: "AGREE", false: "FLIP"}[res.exactSeedAgree], res.exactAgreement*100, res.exactHardFails, contN, res.exactMeanKL,
			map[bool]string{true: "AGREE", false: "FLIP"}[res.fastSeedAgree], res.fastAgreement*100, res.fastHardFails, contN, res.fastMeanKL,
			(res.fastAgreement-res.exactAgreement)*100, res.diffMeanKL, time.Since(t0).Round(time.Second))
	}

	meanExactAgree := sumExactAgree / float64(n) * 100
	meanFastAgree := sumFastAgree / float64(n) * 100
	meanExactKL := sumExactKL / float64(n)
	meanFastKL := sumFastKL / float64(n)

	critA := fastHFtotal <= exactHFtotal
	critB := meanFastAgree >= meanExactAgree-1.0 && agreeAtLeastHalf*2 >= n
	critC := meanFastKL <= 1.1*meanExactKL
	cellShips := critA && critB && critC

	fmt.Printf("=== %s K=%d SUMMARY (n=%d prompts, %d continuation positions each): "+
		"exact(meanAgree=%.1f%% HF=%d/%d worstGap=%.3f%% meanKL=%.4f) "+
		"fast(meanAgree=%.1f%% HF=%d/%d worstGap=%.3f%% meanKL=%.4f) "+
		"critA(hardFlips fast<=exact)=%v critB(agree fast>=exact-1pt & >=half prompts)=%v "+
		"critC(KL fast<=1.1x exact)=%v — CELL %s\n",
		modelName, K, n, contN,
		meanExactAgree, exactHFtotal, n*contN, worstExactGap*100, meanExactKL,
		meanFastAgree, fastHFtotal, n*contN, worstFastGap*100, meanFastKL,
		critA, critB, critC, map[bool]string{true: "SHIPS", false: "DOES NOT SHIP"}[cellShips])
	t.Logf("%s K=%d: exact(agree=%.1f%% HF=%d KL=%.4f) fast(agree=%.1f%% HF=%d KL=%.4f) critA=%v critB=%v critC=%v cell=%s",
		modelName, K, meanExactAgree, exactHFtotal, meanExactKL, meanFastAgree, fastHFtotal, meanFastKL,
		critA, critB, critC, map[bool]string{true: "SHIPS", false: "DOES NOT SHIP"}[cellShips])
	return &cellShips
}

type prefillRefCellResult struct {
	exactSeedAgree, fastSeedAgree       bool
	exactSeedGapPct, fastSeedGapPct     float64
	exactSeedHardFail, fastSeedHardFail bool
	exactSeedKL, fastSeedKL             float64
	exactAgreement, fastAgreement       float64
	exactHardFails, fastHardFails       int
	exactWorstGap, fastWorstGap         float64
	exactMeanKL, fastMeanKL             float64
	diffAgreement, diffMeanKL           float64
}

// runPrefillRefCell runs the exact and fast Metal arms over one K-token prompt, teacher-forces
// BOTH on refTokens (the external CPU-f32-activation reference's own greedy continuation — neither
// arm's own output), and scores both against the reference's seedLogits/refLogits. Shares one
// resident KV store with itself run exact-then-fast, overwritten in place per call — the same
// arrangement metal/prefill_gate_test.go's runPrefillGateCell already established as safe (nothing
// from the exact pass survives into the fast pass's reads because every read happens before the
// next backend call that would overwrite it, and every kept value is cloned at capture time).
func runPrefillRefCell(t *testing.T, rf *metalResident, m *decoder.Model, ids []int, K int, seedRef []float32, refTokens []int, refLogitsRef [][]float32) prefillRefCellResult {
	t.Helper()
	ctx := context.Background()
	embs := make([][]float32, K)
	for i, id := range ids {
		embs[i] = m.EmbedResidentForTest(id)
	}

	lastLog := time.Now()
	var exactSeed []float32
	for i := range K {
		lg, err := rf.Forward(embs[i], i)
		if err != nil {
			t.Fatalf("exact Forward pos=%d: %v", i, err)
		}
		exactSeed = cloneF32(lg)
		if time.Since(lastLog) > 20*time.Second {
			fmt.Printf("[ref-gate]   ... exact prefill K=%d pos %d/%d\n", K, i+1, K)
			lastLog = time.Now()
		}
	}
	exactCont := teacherForceOnRef(t, rf, m, exactSeed, refTokens, K)

	fastSeed, err := rf.PrefillLast(ctx, embs, 0)
	if err != nil {
		t.Fatalf("fast PrefillLast: %v", err)
	}
	fastSeed = cloneF32(fastSeed)
	fastCont := teacherForceOnRef(t, rf, m, fastSeed, refTokens, K)

	exactSeedAgree, exactSeedGap, exactSeedHF := decoder.NearTieArgmaxForTest(seedRef, exactSeed)
	fastSeedAgree, fastSeedGap, fastSeedHF := decoder.NearTieArgmaxForTest(seedRef, fastSeed)
	exactSeedKL := decoder.KLDivergenceForTest(seedRef, exactSeed)
	fastSeedKL := decoder.KLDivergenceForTest(seedRef, fastSeed)

	exactAgreement, _ := decoder.TeacherForcedTop1AgreementForTest(exactCont, refTokens)
	fastAgreement, _ := decoder.TeacherForcedTop1AgreementForTest(fastCont, refTokens)

	exactHF, exactWorstGap, exactMeanKL := scoreContinuationVsRef(refLogitsRef, exactCont)
	fastHF, fastWorstGap, fastMeanKL := scoreContinuationVsRef(refLogitsRef, fastCont)

	return prefillRefCellResult{
		exactSeedAgree: exactSeedAgree, fastSeedAgree: fastSeedAgree,
		exactSeedGapPct: exactSeedGap, fastSeedGapPct: fastSeedGap,
		exactSeedHardFail: exactSeedHF, fastSeedHardFail: fastSeedHF,
		exactSeedKL: exactSeedKL, fastSeedKL: fastSeedKL,
		exactAgreement: exactAgreement, fastAgreement: fastAgreement,
		exactHardFails: exactHF, fastHardFails: fastHF,
		exactWorstGap: exactWorstGap, fastWorstGap: fastWorstGap,
		exactMeanKL: exactMeanKL, fastMeanKL: fastMeanKL,
		diffAgreement: fastAgreement - exactAgreement,
		diffMeanKL:    fastMeanKL - exactMeanKL,
	}
}

// teacherForceOnRef continues an arm from its already-computed seedLogits (position K-1) through
// len(refTokens)-1 more positions, feeding refTokens[i-1] (the REFERENCE's token, not this arm's
// own prediction) as the input at each step. seedLogits is returned as continuation position 0
// verbatim (already cloned by the caller); every subsequent position is cloned here.
func teacherForceOnRef(t *testing.T, rf *metalResident, m *decoder.Model, seedLogits []float32, refTokens []int, K int) [][]float32 {
	t.Helper()
	n := len(refTokens)
	out := make([][]float32, n)
	out[0] = seedLogits
	pos := K - 1
	for i := 1; i < n; i++ {
		pos++
		lg, err := rf.Forward(m.EmbedResidentForTest(refTokens[i-1]), pos)
		if err != nil {
			t.Fatalf("teacher-forced Forward pos=%d: %v", pos, err)
		}
		out[i] = cloneF32(lg)
	}
	return out
}

// scoreContinuationVsRef compares each of an arm's continuation logits against the reference's own
// logits at that position (decoder.NearTieArgmaxForTest, the same 3%-near-tie rule used
// throughout this gate), returning the hard-flip count, the worst gap seen, and the mean KL
// divergence over all positions.
func scoreContinuationVsRef(refLogits, armLogits [][]float32) (hardFails int, worstGap, meanKL float64) {
	var klSum float64
	for i := range armLogits {
		_, gap, hf := decoder.NearTieArgmaxForTest(refLogits[i], armLogits[i])
		if hf {
			hardFails++
		}
		if gap > worstGap {
			worstGap = gap
		}
		klSum += decoder.KLDivergenceForTest(refLogits[i], armLogits[i])
	}
	return hardFails, worstGap, klSum / float64(len(armLogits))
}
