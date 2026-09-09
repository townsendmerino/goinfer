//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestPrefillGateVsReference is Phase B of docs/task-prefill-gap.md §4 L1's fresh-prompt decision
// run (2026-09-09), superseding the §3.1 per-cell form this file used before. TestPrefillGate
// (prefill_gate_test.go) scored Metal's fast (f16-activation) path against Metal's own exact
// (int8-per-row-activation) path and treated the exact path as truth; §3.1 found that comparison
// cannot distinguish a defect in the fast path from the exact path's own quantisation loss, since
// both are lossy relative to f32 activations. That test's numbers stand as a measurement of
// fast-vs-exact distance (still printed by runPrefillGateModel) but no longer gate anything.
//
// This test scores BOTH Metal arms — exact (sequential Forward) and fast (batched PrefillLast) —
// against a THIRD, external reference: the CPU backend's own sequential forward with f32
// activations (decoder/prefill_ref_gen_test.go, TestPrefillGateReference, run separately and in
// its own process — see that file for why). Reference files live under
// ~/goinfer-logs/prefill-ref[-<set>]/ (not in the repo) and MUST be generated first; a missing
// file skips that cell with a clear message rather than silently falling back to the withdrawn
// exact-as-oracle scoring — that fallback is exactly the mistake §3.1 corrected.
//
// Both arms are teacher-forced on the SAME reference-supplied continuation tokens (not on either
// arm's own greedy output, and not on each other's), so a difference between the two arms'
// per-position agreement is attributable to the arm alone, not to which one's tokens happened to
// be used as the "ground truth" stream.
//
// §3.2's POOLED form (docs/task-prefill-gap.md §3, as amended by §3.1 and §3.2) — the per-cell
// binary form this file used on 2026-09-05 has NO resolving power at these sample sizes (§3.2's
// own arithmetic: a per-cell veto over N criteria fails an arm of EQUAL quality most of the time).
// The decision is now pooled over every decision-set cell for one model (K ∈ {256, 512, 1024} —
// 512 added 2026-09-09 as the measured floor candidate, matching how CUDA set its own floor from
// a measured K=512 cell rather than interpolating one; S's K=3900 is a confirmation cell, scored
// and reported but never part of the pooled decision):
//
//	(a) hard flips (decoder.NearTieArgmaxForTest, the 3%-near-tie rule CUDA decode is already held
//	    to against CPU), pooled over every decision-cell position: fast's total <= exact's total +
//	    2*sqrt(exact's total) — the Poisson-noise-aware form of "no worse", not a strict inequality
//	(b) teacher-forced top-1 agreement (exact argmax match against the reference's own recorded
//	    token), pooled: fast's rate >= exact's rate - 2*sqrt(d)/N, where d is the number of pooled
//	    positions on which EXACTLY ONE arm matches the reference (the McNemar-shaped paired noise)
//	    and N is the total pooled positions — d is RECORDED and used directly, not approximated by
//	    the conservative independent-errors bound
//	(c) mean KL(reference ‖ arm) — the gating continuous measure: pooled mean fast <= pooled mean
//	    exact, AND fast lower on >= half the pooled (K, prompt) pairs (paired sign test), AND no
//	    single cell's fast mean KL exceeds 1.1x that cell's exact mean KL
//
// Per-cell values are reported for all three (a table, printed and logged) but NEVER veto the
// decision individually — that is the exact defect §3.2 found in the 2026-09-05 form.
//
// A model ships iff all three pooled criteria hold over its full decision set.
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

	primaryLabel, primaryFiles := decoder.PrefillGatePromptSet()

	models := []struct {
		name        string
		pathEnv     string
		defaultPath string
	}{
		{"S", "GOINFER_METAL_MODEL", "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"},
		{"D7", "GOINFER_METAL_MODEL_D7", "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf"},
	}
	decisionKs := []int{256, 512, 1024}
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

			// The DECIDING run: whatever GOINFER_PREFILL_GATE_PROMPTS selects (default set A).
			shipped := runPrefillGateSet(t, rf, m, tk, mc.name, primaryLabel, primaryFiles,
				refDirFor(home, primaryLabel), decisionKs, confirmKsByModel[mc.name], true)

			// Set A's stored reference files, RE-SCORED under this same pooled form and printed
			// beside the primary run's — informational only, per the brief ("not deciding"). Only
			// when the primary run is NOT already set A (no point re-scoring a set against itself).
			// Set A has no K=512 reference (it predates this session's floor work), so its decision
			// set here is {256, 1024} only — narrower than the real run's, and that narrowing is
			// itself part of why this is reported, not decided.
			if primaryLabel != "a" {
				aFiles := decoder.PrefillGatePromptSetFor("a")
				runPrefillGateSet(t, rf, m, tk, mc.name, "a", aFiles,
					refDirFor(home, "a"), []int{256, 1024}, confirmKsByModel[mc.name], false)
			}

			if !shipped {
				t.Fatalf("§3 gate: model %s does not ship on the %q decision set — see log above for which criterion", mc.name, primaryLabel)
			}
		})
	}
}

// refDirFor mirrors decoder/prefill_ref_gen_test.go's own directory choice EXACTLY (set A keeps
// the historical ~/goinfer-logs/prefill-ref/ name; any other set gets prefill-ref-<label>/) — the
// two must agree or Phase B looks for reference files Phase A never wrote there.
func refDirFor(home, label string) string {
	name := "prefill-ref"
	if label != "a" {
		name = "prefill-ref-" + label
	}
	return filepath.Join(home, "goinfer-logs", name)
}

// runPrefillGateSet runs every decision-set + confirmation cell for one (model, prompt-set),
// pools the decision-set cells under §3.2's form, prints the per-cell table and the pooled
// verdict, and returns whether the pooled decision set ships. deciding=false labels the output
// "re-scored, not deciding" and never fails the test on its own — see the caller.
func runPrefillGateSet(t *testing.T, rf *metalResident, m *decoder.Model, tk *tokenizer.Tokenizer,
	modelName, setLabel string, promptFiles []string, refDir string,
	decisionKs, confirmKs []int, deciding bool) bool {
	t.Helper()

	allKs := append(append([]int{}, decisionKs...), confirmKs...)
	maxK := 0
	for _, k := range allKs {
		if k > maxK {
			maxK = k
		}
	}
	prompts := make([][]int, 0, len(promptFiles))
	for _, f := range promptFiles {
		prompts = append(prompts, decoder.PrefillGateProseIDsForTest(t, tk, f, maxK))
	}

	role := "DECIDING"
	if !deciding {
		role = "re-scored, NOT deciding"
	}
	fmt.Printf("--- %s prompt set %q (%s) ---\n", modelName, setLabel, role)

	var decisionCells []*cellSummary
	var confirmCells []*cellSummary
	for _, K := range decisionKs {
		cs := runPrefillRefGateCellK(t, rf, m, modelName, setLabel, K, prompts, refDir)
		if cs != nil {
			decisionCells = append(decisionCells, cs)
		}
	}
	for _, K := range confirmKs {
		cs := runPrefillRefGateCellK(t, rf, m, modelName, setLabel, K, prompts, refDir)
		if cs != nil {
			confirmCells = append(confirmCells, cs)
		}
	}

	if len(decisionCells) == 0 {
		t.Logf("%s (%s, set %q): no decision-set reference cells found under %s — run TestPrefillGateReference (decoder package) first", modelName, role, setLabel, refDir)
		return false
	}

	pooled := poolCells(decisionCells)
	ships := pooled.critA && pooled.critB && pooled.critC
	verdict := map[bool]string{true: "SHIPS", false: "DOES NOT SHIP"}[ships]

	fmt.Printf("=== %s (set %q, %s) POOLED decision-set verdict (K=%v, %d cells, %d positions): "+
		"critA(hardFlips fast<=exact+2*sqrt(exact))=%v (exact=%d fast=%d) "+
		"critB(agree fast>=exact-2*sqrt(d)/N)=%v (exactAgree=%.2f%% fastAgree=%.2f%% d=%d N=%d) "+
		"critC(KL pooled fast<=exact & fast lower on >=half prompts & no cell >1.1x)=%v "+
		"(exactMeanKL=%.4f fastMeanKL=%.4f fastLowerPrompts=%d/%d ceilingOK=%v) — %s ===\n",
		modelName, setLabel, role, decisionKs, len(decisionCells), pooled.n,
		pooled.critA, pooled.exactHF, pooled.fastHF,
		pooled.critB, pooled.exactAgreeRate*100, pooled.fastAgreeRate*100, pooled.d, pooled.n,
		pooled.critC, pooled.exactMeanKL, pooled.fastMeanKL, pooled.promptFastLowerKL, pooled.promptsCounted, pooled.klCeilingOK,
		verdict)
	t.Logf("%s (set %q, %s): pooled verdict = %s (critA=%v critB=%v critC=%v)",
		modelName, setLabel, role, verdict, pooled.critA, pooled.critB, pooled.critC)

	for _, cs := range confirmCells {
		fmt.Printf("[confirm, not gating] %s set %q K=%d: exact(agree=%.1f%% HF=%d/%d meanKL=%.4f) fast(agree=%.1f%% HF=%d/%d meanKL=%.4f)\n",
			modelName, setLabel, cs.K, cs.exactAgreeRate*100, cs.exactHF, cs.n*cs.contN, cs.exactMeanKL,
			cs.fastAgreeRate*100, cs.fastHF, cs.n*cs.contN, cs.fastMeanKL)
	}

	return ships
}

// cellSummary is one (model, set, K) cell's pooled-ready statistics — everything §3.2's criteria
// need, summed/meaned over the cell's prompts so the caller can pool across cells without re-reading
// per-position data.
type cellSummary struct {
	K                 int
	n, contN          int     // prompts scored, positions per prompt
	exactHF, fastHF   int     // hard-flip counts (NearTieArgmaxForTest), summed over n*contN positions
	exactMatch        int     // exact TOP-1 MATCH count (argmax == reference's own token), summed over n*contN
	fastMatch         int     // same, fast arm
	d                 int     // positions where exactly one of {exactMatch, fastMatch} is true, summed over n*contN
	exactKLsum        float64 // sum over n*contN positions
	fastKLsum         float64
	exactMeanKL       float64 // = exactKLsum / (n*contN), this cell only — for the 1.1x ceiling and reporting
	fastMeanKL        float64
	exactAgreeRate    float64 // = exactMatch / (n*contN) — reporting
	fastAgreeRate     float64
	promptFastLowerKL int // count of prompts in THIS cell where fast's per-prompt mean KL <= exact's
	promptsCounted    int // = n, named separately so pooling reads "prompts", not "cells"
	worstExactGap     float64
	worstFastGap      float64
}

// runPrefillRefGateCellK runs one (model, set, K) cell over every prompt, printing per-prompt
// numbers, and returns the cell's pooled-ready summary — or nil (having logged why) if the
// reference files for this cell are missing.
func runPrefillRefGateCellK(t *testing.T, rf *metalResident, m *decoder.Model, modelName, setLabel string, K int, prompts [][]int, refDir string) *cellSummary {
	t.Helper()
	for pi := range prompts {
		p := filepath.Join(refDir, fmt.Sprintf("%s-K%d-p%d.bin", modelName, K, pi))
		if _, err := os.Stat(p); err != nil {
			t.Logf("%s set %q K=%d: reference file missing (%s) — SKIPPING this cell; no fallback to the "+
				"withdrawn exact-as-oracle scoring. Run TestPrefillGateReference (decoder package, GOINFER_PREFILL_GATE_PROMPTS=%s) first.",
				modelName, setLabel, K, p, setLabel)
			return nil
		}
	}

	cs := &cellSummary{K: K}
	t0 := time.Now()
	for pi, ids := range prompts {
		refPath := filepath.Join(refDir, fmt.Sprintf("%s-K%d-p%d.bin", modelName, K, pi))
		seedRef, refTokens, refLogitsRef, err := decoder.ReadPrefillReferenceForTest(refPath)
		if err != nil {
			t.Fatalf("read reference %s: %v", refPath, err)
		}
		_ = seedRef // folded into refLogitsRef[0] already (see decoder/prefill_ref_gen_test.go)
		_ = refTokens
		res := runPrefillRefCell(t, rf, m, ids[:K], K, refLogitsRef)
		cs.n++
		cs.contN = res.contN
		cs.exactHF += res.exactHF
		cs.fastHF += res.fastHF
		cs.exactMatch += res.exactMatch
		cs.fastMatch += res.fastMatch
		cs.d += res.d
		cs.exactKLsum += res.exactKLsum
		cs.fastKLsum += res.fastKLsum
		cs.promptsCounted++
		if res.fastMeanKL <= res.exactMeanKL {
			cs.promptFastLowerKL++
		}
		if res.exactWorstGap > cs.worstExactGap {
			cs.worstExactGap = res.exactWorstGap
		}
		if res.fastWorstGap > cs.worstFastGap {
			cs.worstFastGap = res.fastWorstGap
		}
		fmt.Printf("[ref-gate] %s set %q K=%d prompt %2d/%2d exact(agree=%.1f%% HF=%d/%d KL=%.4f) "+
			"fast(agree=%.1f%% HF=%d/%d KL=%.4f) diff(agree=%+.1fpt KL=%+.4f) elapsed=%s\n",
			modelName, setLabel, K, pi+1, len(prompts),
			res.exactMatchRate()*100, res.exactHF, res.contN, res.exactMeanKL,
			res.fastMatchRate()*100, res.fastHF, res.contN, res.fastMeanKL,
			(res.fastMatchRate()-res.exactMatchRate())*100, res.fastMeanKL-res.exactMeanKL, time.Since(t0).Round(time.Second))
	}
	total := float64(cs.n * cs.contN)
	cs.exactMeanKL = cs.exactKLsum / total
	cs.fastMeanKL = cs.fastKLsum / total
	cs.exactAgreeRate = float64(cs.exactMatch) / total
	cs.fastAgreeRate = float64(cs.fastMatch) / total

	fmt.Printf("=== %s set %q K=%d cell SUMMARY (n=%d prompts, %d positions each, %d total): "+
		"exact(agree=%.1f%% HF=%d meanKL=%.4f worstGap=%.3f%%) "+
		"fast(agree=%.1f%% HF=%d meanKL=%.4f worstGap=%.3f%%) — reported, no per-cell veto\n",
		modelName, setLabel, K, cs.n, cs.contN, int(total),
		cs.exactAgreeRate*100, cs.exactHF, cs.exactMeanKL, cs.worstExactGap*100,
		cs.fastAgreeRate*100, cs.fastHF, cs.fastMeanKL, cs.worstFastGap*100)

	return cs
}

// pooledStats is the §3.2 pooled decision over a model's whole decision set.
type pooledStats struct {
	n                                 int // total pooled positions (sum of n*contN over decision cells)
	exactHF, fastHF                   int
	exactAgreeRate, fastAgreeRate     float64
	d                                 int
	exactMeanKL, fastMeanKL           float64
	promptFastLowerKL, promptsCounted int
	klCeilingOK                       bool // no single cell's fast mean KL > 1.1x that cell's exact mean KL
	critA, critB, critC               bool
}

// poolCells implements §3.2's pooled criteria over a model's decision-set cells. No per-cell veto:
// every quantity is summed/meaned across cells FIRST, and the three criteria are evaluated once,
// on the pooled totals — the exact repair §3.2 made after the 2026-09-05 per-cell form failed an
// arm of equal quality most of the time by construction (a veto per cell per criterion multiplies
// the false-fail rate by the cell count).
func poolCells(cells []*cellSummary) pooledStats {
	var p pooledStats
	var exactMatch, fastMatch, exactKLsum, fastKLsum float64
	klCeilingOK := true
	for _, c := range cells {
		n := c.n * c.contN
		p.n += n
		p.exactHF += c.exactHF
		p.fastHF += c.fastHF
		exactMatch += float64(c.exactMatch)
		fastMatch += float64(c.fastMatch)
		p.d += c.d
		exactKLsum += c.exactKLsum
		fastKLsum += c.fastKLsum
		p.promptFastLowerKL += c.promptFastLowerKL
		p.promptsCounted += c.promptsCounted
		// 1.1x ceiling: per-cell, not pooled — a single cell far off the reference must not be
		// diluted into invisibility by cells that are fine (§3's own "hard ceiling... in any single
		// cell" wording).
		if c.fastMeanKL > 1.1*c.exactMeanKL {
			klCeilingOK = false
		}
	}
	total := float64(p.n)
	p.exactAgreeRate = exactMatch / total
	p.fastAgreeRate = fastMatch / total
	p.exactMeanKL = exactKLsum / total
	p.fastMeanKL = fastKLsum / total
	p.klCeilingOK = klCeilingOK

	p.critA = float64(p.fastHF) <= float64(p.exactHF)+2*math.Sqrt(float64(p.exactHF))
	p.critB = p.fastAgreeRate >= p.exactAgreeRate-2*math.Sqrt(float64(p.d))/total
	fastLowerAtLeastHalf := p.promptFastLowerKL*2 >= p.promptsCounted
	p.critC = p.fastMeanKL <= p.exactMeanKL && fastLowerAtLeastHalf && p.klCeilingOK
	return p
}

type prefillRefCellResult struct {
	contN                       int
	exactHF, fastHF             int
	exactMatch, fastMatch       int // TOP-1 match count (argmax == reference token), not near-tie hard-flip
	d                           int // positions where exactly one of {exactMatch,fastMatch} matched
	exactKLsum, fastKLsum       float64
	exactMeanKL, fastMeanKL     float64
	exactWorstGap, fastWorstGap float64
}

func (r prefillRefCellResult) exactMatchRate() float64 {
	return float64(r.exactMatch) / float64(r.contN)
}
func (r prefillRefCellResult) fastMatchRate() float64 { return float64(r.fastMatch) / float64(r.contN) }

// runPrefillRefCell runs the exact and fast Metal arms over one K-token prompt, teacher-forces
// BOTH on refTokens (the external CPU-f32-activation reference's own greedy continuation — neither
// arm's own output), and scores both against the reference's per-position logits
// (refLogitsRef[0] is the seed; refLogitsRef[1:] the continuation — see
// decoder/prefill_ref_gen_test.go's prefillReferenceCell, which stores the seed as refLogits[0]
// too). Shares one resident KV store with itself run exact-then-fast, overwritten in place per
// call — the same arrangement metal/prefill_gate_test.go's runPrefillGateCell already established
// as safe (nothing from the exact pass survives into the fast pass's reads because every read
// happens before the next backend call that would overwrite it, and every kept value is cloned at
// capture time).
func runPrefillRefCell(t *testing.T, rf *metalResident, m *decoder.Model, ids []int, K int, refLogitsRef [][]float32) prefillRefCellResult {
	t.Helper()
	ctx := context.Background()
	embs := make([][]float32, K)
	for i, id := range ids {
		embs[i] = m.EmbedResidentForTest(id)
	}
	// refTokens (the teacher-forcing stream) is the reference's own argmax at every position,
	// which is exactly what decoder.NearTieArgmaxForTest's `agree` return already tells us against
	// refLogitsRef — no separate refTokens value is needed here (see readNote for why).
	refTokens := make([]int, len(refLogitsRef))
	for i, lg := range refLogitsRef {
		refTokens[i] = argmaxF(lg)
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

	exactHF, exactWorstGap, exactMeanKL, exactMatches := scoreContinuationVsRef(refLogitsRef, exactCont)
	fastHF, fastWorstGap, fastMeanKL, fastMatches := scoreContinuationVsRef(refLogitsRef, fastCont)

	d := 0
	for i := range exactMatches {
		if exactMatches[i] != fastMatches[i] {
			d++
		}
	}
	exactMatchCount, fastMatchCount := 0, 0
	for _, ok := range exactMatches {
		if ok {
			exactMatchCount++
		}
	}
	for _, ok := range fastMatches {
		if ok {
			fastMatchCount++
		}
	}

	return prefillRefCellResult{
		contN:         len(exactCont),
		exactHF:       exactHF,
		fastHF:        fastHF,
		exactMatch:    exactMatchCount,
		fastMatch:     fastMatchCount,
		d:             d,
		exactKLsum:    exactMeanKL * float64(len(exactCont)),
		fastKLsum:     fastMeanKL * float64(len(fastCont)),
		exactMeanKL:   exactMeanKL,
		fastMeanKL:    fastMeanKL,
		exactWorstGap: exactWorstGap,
		fastWorstGap:  fastWorstGap,
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
// throughout this gate), returning the hard-flip count, the worst gap seen, the mean KL divergence
// over all positions, AND the per-position EXACT top-1 match (argmax(ref) == argmax(arm) — the
// `agree` NearTieArgmaxForTest already computes, reused rather than re-derived) needed for §3.2's
// pooled agreement bound (d = positions where exactly one arm matches).
func scoreContinuationVsRef(refLogits, armLogits [][]float32) (hardFails int, worstGap, meanKL float64, matches []bool) {
	matches = make([]bool, len(armLogits))
	var klSum float64
	for i := range armLogits {
		agree, gap, hf := decoder.NearTieArgmaxForTest(refLogits[i], armLogits[i])
		matches[i] = agree
		if hf {
			hardFails++
		}
		if gap > worstGap {
			worstGap = gap
		}
		klSum += decoder.KLDivergenceForTest(refLogits[i], armLogits[i])
	}
	return hardFails, worstGap, klSum / float64(len(armLogits)), matches
}

// argmaxF (metal/testshared_test.go) recovers refTokens (the reference's own greedy pick)
// directly from refLogitsRef without a second stored copy — decoder.WritePrefillReferenceForTest
// already writes refTokens separately, but this test only has refLogitsRef in scope at the point
// it needs the teacher-forcing stream, and the two are guaranteed identical by construction
// (decoder/prefill_ref_gen_test.go's prefillReferenceCell sets refTokens[i] = argmax(refLogits[i])
// for the exact same slice).
