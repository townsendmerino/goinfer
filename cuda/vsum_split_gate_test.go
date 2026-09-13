//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestVsumSplitGateVsReference is the §3.2 fidelity gate for the flash-decode V-sum spike
// (GOINFER_SPLITKV_VSUM_SPLIT), pre-registered in
// docs/measurements/vsum-split-fidelity-PREREGISTERED.md and owed by
// docs/measurements/vsum-split-spike-2026-09-13.md's "What this does NOT establish".
//
// THE ORACLE IS NOT THE BIT-IDENTICAL PATH. Both arms are scored against a THIRD thing: the CPU
// backend's own forward with f32 activations and the exact f64-accumulating attention
// (decoder/prefill_ref_gen_test.go, Phase A, its own process). Scoring the spike against the
// shipped path and calling the shipped path truth is what cuda/vsum_split_fidelity_test.go does,
// and it is the exact mistake docs/task-prefill-gap.md §3.1 corrected: the shipped path is itself
// one particular summation order, guaranteed to disagree with any other for reasons that have
// nothing to do with a defect, and the distance gets booked against whichever arm is newer.
//
// WHY THIS REUSES THE PREFILL GATE'S REFERENCE. That reference stores the prompt-final logits plus
// 64 teacher-forced continuation rows, and the continuation is generated one token at a time
// through the resident DECODE path — which is the only path this change touches. So the reference
// is already the right shape; only the arms differ.
//
// THE ARMS. Both run split-KV (skMinKeys forced to 0) and are teacher-forced on the same
// reference-supplied tokens. They differ in one field:
//
//	exact  r.skVsumSplit = 0  — today's default, bit-identical to attn_batched(M=1)
//	spike  r.skVsumSplit = S  — the key-axis split; NOT bit-identical, opt-in, unreachable in a stock binary
//
// GATE, per (model, K) cell, all three:
//
//	(a) spike's hard-flip count vs the reference <= exact + 2·√exact   (AMENDED — see below)
//	(b) spike's mean teacher-forced agreement >= exact's mean - 1.0 pt AND spike >= exact on >= half
//	    the prompts (PAIRED, not pooled — CLAUDE.md rule 7)
//	(c) spike's mean continuation KL(reference || arm) <= 1.1 x exact's mean
//
// CRITERION (a) WAS AMENDED BY OWNER DECISION, 2026-09-13, after the S confirmation cell was scored
// and before any D7 reference existed. As pre-registered it read `spike HF <= exact HF`, §3's strict
// form — which the pre-registration mislabelled "§3.2". S failed it 8 v 7 over 640 positions, a
// difference well inside Poisson noise (σ ≈ √7 ≈ 2.6), so the strict count cannot resolve the
// question it is asking. It now uses the ceiling task-prefill-gap.md §3.2 specifies and the Metal
// pooled gate implements (metal/prefill_gate_ref_test.go, `exact + 2*math.Sqrt(exact)`). The STRICT
// result is still computed and printed for every cell, so the amendment is auditable rather than
// silent. (b) and (c) are NOT amended: §3.2's noise-aware (b) would be looser than the registered
// 1.0 pt, and that bar stays.
//
// AMBIGUOUS -> PARKED: (b) inside its last 0.2 pt, or (c) in 1.05-1.10x, is inconclusive rather
// than a pass. The band is pre-registered because the zone just under a threshold is where
// motivated reasoning lives.
//
// A PASS DOES NOT FLIP THE DEFAULT. cuda/prefill.go's exact-path note records the invariant: the
// exact path "remains bit-identical to the M=1 decode kernels, remains what spec-decode verify and
// the parity gates run". Promotion breaks TestSplitKV_bitIdentical by construction, breaks
// spec-decode losslessness, and moves the decode goldens.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_SPLITKV_VSUM_SPLIT=4 \
//	  go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestVsumSplitGateVsReference -v -timeout 2h
func TestVsumSplitGateVsReference(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (real checkpoints; needs Phase A's reference files)")
	}
	if testing.Short() {
		t.Skip("long-running gate: skipped in -short")
	}
	if strings.TrimSpace(os.Getenv("GOINFER_SPLITKV_VSUM_SPLIT")) == "" {
		t.Skip("set GOINFER_SPLITKV_VSUM_SPLIT=4 — the spike pipelines are loaded at backend setup, " +
			"so without it there is no spike arm to score and the gate would compare the exact path to itself")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("home dir: %v", err)
	}
	refDir := filepath.Join(home, "goinfer-logs", "prefill-ref")

	ks := []int{8000} // the depth the +40% was measured at; overridable for a re-run at another depth
	if v := strings.TrimSpace(os.Getenv("GOINFER_VSUM_GATE_KS")); v != "" {
		ks = ks[:0]
		for _, f := range strings.Split(v, ",") {
			if k, err := strconv.Atoi(strings.TrimSpace(f)); err == nil && k > 0 {
				ks = append(ks, k)
			}
		}
	}

	models := []struct {
		name        string
		pathEnv     string
		defaultPath string
		decision    bool
	}{
		{"D7", "GOINFER_CUDA_GATE_MODEL_D7", "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf", true},
		{"S", "GOINFER_CUDA_GATE_MODEL", "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", false},
	}

	for _, mc := range models {
		t.Run(mc.name, func(t *testing.T) {
			path := os.Getenv(mc.pathEnv)
			if path == "" {
				path = os.ExpandEnv(mc.defaultPath)
			}
			if _, err := os.Stat(path); err != nil {
				t.Skipf("no fixture at %s", path)
			}
			m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
			if err != nil {
				t.Fatalf("load: %v", err)
			}
			defer m.Close()
			rf, ok := m.ResidentForwardForTest().(*cudaResident)
			if !ok {
				t.Fatal("cuda resident did not engage — an earlier run silently profiled the CPU " +
					"fallback for ten minutes after a 0-byte allocation took the resident path down")
			}

			// PRECONDITION 1: the spike must be loaded and reachable. A missing pipeline here would
			// make every "spike" arm silently run the exact kernel and score a perfect pass.
			if rf.skVsumSplit <= 1 || rf.skVsumPartial == (Pipeline{}) || rf.skVsumCombine == (Pipeline{}) {
				t.Fatalf("spike not loaded (skVsumSplit=%d partial=%v combine=%v) — the gate would "+
					"score the exact path twice and call it a pass",
					rf.skVsumSplit, rf.skVsumPartial != (Pipeline{}), rf.skVsumCombine != (Pipeline{}))
			}
			nSplit := rf.skVsumSplit

			// PRECONDITION 2: split-KV itself must be taken, in BOTH arms, at every depth this run
			// touches. Forced rather than assumed: the per-geometry threshold is exactly the thing
			// this campaign has been changing.
			if !rf.splitkvAttn || rf.skScores == (Pipeline{}) {
				t.Fatalf("split-KV decode attention is not active (splitkvAttn=%v scores=%v) — "+
					"neither arm would reach the V-sum under test", rf.splitkvAttn, rf.skScores != (Pipeline{}))
			}
			rf.skMinKeys = 0
			if got := rf.splitkvMin(rf.layers[0].nKV, rf.layers[0].hd); got != 0 {
				t.Fatalf("skMinKeys=0 did not force split-KV: splitkvMin still %d", got)
			}

			tk, err := tokenizer.LoadGGUF(path)
			if err != nil {
				t.Fatalf("load tokenizer: %v", err)
			}
			// The SNAPSHOTS Phase A built the reference from, never the live docs — see the same
			// note in prefill_gate_ref_test.go for what reading the live ones costs.
			setLabel, promptFiles := decoder.PrefillGatePromptSet()
			maxK := 0
			for _, k := range ks {
				maxK = max(maxK, k)
			}
			prompts := make([][]int, 0, len(promptFiles))
			for _, f := range promptFiles {
				prompts = append(prompts, decoder.PrefillGateProseIDsForTest(t, tk, f, maxK))
			}
			t.Logf("%s: prompt set %q, %d prompts, S=%d, cells %v", mc.name, setLabel, len(prompts), nSplit, ks)

			for _, K := range ks {
				runVsumGateCell(t, rf, m, mc.name, K, nSplit, prompts, refDir, mc.decision)
			}
		})
	}
}

// vsumArmScore is one arm's distance from the reference over a prompt's continuation. Agreement is
// not a field here: it comes from decoder.TeacherForcedTop1AgreementForTest, which scores against
// the reference's chosen TOKENS rather than against its logits, and keeping a second copy of it
// alongside these three would invite the two to drift apart.
type vsumArmScore struct {
	hardFail int
	worstGap float64
	kl       float64
}

// runVsumGateCell scores both arms over every prompt at one depth and applies the pre-registered
// three-part rule. A missing reference SKIPS the cell — there is no fallback to scoring the spike
// against the bit-identical path, because that fallback is the thing §3.1 withdrew.
func runVsumGateCell(t *testing.T, rf *cudaResident, m *decoder.Model, model string, K, nSplit int,
	prompts [][]int, refDir string, decision bool) {
	t.Helper()
	for pi := range prompts {
		p := filepath.Join(refDir, fmt.Sprintf("%s-K%d-p%d.bin", model, K, pi))
		if _, err := os.Stat(p); err != nil {
			t.Logf("%s K=%d: reference missing (%s) — SKIPPING the cell; run Phase A "+
				"(GOINFER_CPU_REF_KS=%d TestPrefillGateReference) first", model, K, p, K)
			return
		}
	}
	var (
		sumEA, sumFA, sumEKL, sumFKL float64
		eHF, fHF, n, spikeWins       int
		worstEGap, worstFGap         float64
		diffPositions, contN         int
	)
	t0 := time.Now()
	for pi, ids := range prompts {
		seedRef, refTokens, refLogits, err := decoder.ReadPrefillReferenceForTest(
			filepath.Join(refDir, fmt.Sprintf("%s-K%d-p%d.bin", model, K, pi)))
		if err != nil {
			t.Fatalf("read reference: %v", err)
		}
		_ = seedRef
		contN = len(refTokens)

		exactCont := vsumArm(t, rf, m, ids[:K], K, refTokens, 0)
		spikeCont := vsumArm(t, rf, m, ids[:K], K, refTokens, nSplit)

		// PRECONDITION 3a, per prompt: the prefill seed row must be IDENTICAL across arms. The spike
		// touches the M=1 decode V-sum and nothing else; a seed that moves means something other
		// than the kernel under test changed between the two runs.
		if !sameLogits(exactCont[0], spikeCont[0]) {
			t.Fatalf("%s K=%d p%d: the prefill seed row differs between arms — the spike is not the "+
				"only difference between these two runs", model, K, pi)
		}
		// PRECONDITION 3b, per prompt: the DECODE rows must differ somewhere. If they never do, the
		// spike did not run, and every metric below would be the exact path scored against itself.
		nd := 0
		for i := 1; i < len(spikeCont); i++ {
			if !sameLogits(exactCont[i], spikeCont[i]) {
				nd++
			}
		}
		diffPositions += nd

		// PRECONDITION 3c, first prompt only: A/A. The spike gives up bit-identity to HISTORY, never
		// determinism — its combine is a fixed ascending order and never atomics. Two runs of the
		// same arm on the same input must agree bit for bit, or the kernel is wrong and no fidelity
		// number from it means anything.
		if pi == 0 {
			again := vsumArm(t, rf, m, ids[:K], K, refTokens, nSplit)
			for i := range again {
				if !sameLogits(again[i], spikeCont[i]) {
					t.Fatalf("%s K=%d: A/A FAILED at continuation row %d — the spike is not "+
						"deterministic; determinism is the property it must keep", model, K, i)
				}
			}
			t.Logf("%s K=%d: A/A bit-identical over %d rows", model, K, len(again))
		}

		ea, _ := decoder.TeacherForcedTop1AgreementForTest(exactCont, refTokens)
		fa, _ := decoder.TeacherForcedTop1AgreementForTest(spikeCont, refTokens)
		e := scoreVsumArm(refLogits, exactCont)
		s := scoreVsumArm(refLogits, spikeCont)
		n++
		sumEA += ea
		sumFA += fa
		sumEKL += e.kl
		sumFKL += s.kl
		eHF += e.hardFail
		fHF += s.hardFail
		worstEGap = max(worstEGap, e.worstGap)
		worstFGap = max(worstFGap, s.worstGap)
		if fa >= ea {
			spikeWins++
		}
		fmt.Printf("[vsum-gate] %s K=%d prompt %2d/%2d exact(agree=%.1f%% HF=%d/%d KL=%.5f) "+
			"spike(agree=%.1f%% HF=%d/%d KL=%.5f) diff(agree=%+.2fpt KL=%+.5f) rowsDiffering=%d/%d elapsed=%s\n",
			model, K, pi+1, len(prompts), ea*100, e.hardFail, contN, e.kl,
			fa*100, s.hardFail, contN, s.kl, (fa-ea)*100, s.kl-e.kl, nd, contN-1,
			time.Since(t0).Round(time.Second))
	}

	if diffPositions == 0 {
		t.Fatalf("%s K=%d: VACUOUS — the two arms produced bit-identical decode logits at every "+
			"position of every prompt. The spike did not run; every metric in this cell is the "+
			"exact path scored against itself", model, K)
	}

	mEA, mFA := sumEA/float64(n)*100, sumFA/float64(n)*100
	mEKL, mFKL := sumEKL/float64(n), sumFKL/float64(n)
	critAStrict := fHF <= eHF // as pre-registered; reported, no longer decisive
	critA := float64(fHF) <= float64(eHF)+2*math.Sqrt(float64(eHF))
	critB := mFA >= mEA-1.0 && spikeWins*2 >= n
	critC := mFKL <= 1.1*mEKL
	// The pre-registered ambiguous band: a pass whose margin is inside it is INCONCLUSIVE, not a pass.
	parked := (critB && mFA < mEA-0.8) || (critC && mFKL > 1.05*mEKL)
	verdict := "DOES NOT PASS"
	switch {
	case critA && critB && critC && parked:
		verdict = "AMBIGUOUS — PARKED"
	case critA && critB && critC:
		verdict = "PASSES"
	}
	label := "confirmation"
	if decision {
		label = "DECISION"
	}
	fmt.Printf("=== VSUM-SPLIT GATE %s K=%d (%s cell, n=%d prompts x %d positions, S=%d): "+
		"exact(meanAgree=%.2f%% HF=%d/%d worstGap=%.3f%% meanKL=%.6f) "+
		"spike(meanAgree=%.2f%% HF=%d/%d worstGap=%.3f%% meanKL=%.6f) "+
		"pairedWins=%d/%d rowsDiffering=%d "+
		"critA(HF spike<=exact+2*sqrt(exact))=%v [strict spike<=exact, as pre-registered: %v] "+
		"critB(agree>=exact-1pt & >=half)=%v critC(KL<=1.1x)=%v — %s ===\n",
		model, K, label, n, contN, nSplit,
		mEA, eHF, n*contN, worstEGap*100, mEKL,
		mFA, fHF, n*contN, worstFGap*100, mFKL,
		spikeWins, n, diffPositions, critA, critAStrict, critB, critC, verdict)
	t.Logf("%s K=%d (%s): %s — exact(agree=%.2f%% HF=%d KL=%.6f) spike(agree=%.2f%% HF=%d KL=%.6f)",
		model, K, label, verdict, mEA, eHF, mEKL, mFA, fHF, mFKL)
	if decision && verdict != "PASSES" {
		t.Errorf("%s K=%d is the DECISION cell and it did not pass (%s) — the spike stays opt-in, "+
			"which is a RESULT, not a failure of the run", model, K, verdict)
	}
}

// vsumArm runs one arm over one prompt and returns its continuation rows, cloned.
//
// EVERY ROW IS CLONED AT CAPTURE: the resident returns a view onto device-backed scratch that the
// next launch overwrites, so keeping the slice would score the LAST call's logits at every
// position — a bug that produces entirely plausible numbers.
func vsumArm(t *testing.T, rf *cudaResident, m *decoder.Model, ids []int, K int, refTokens []int, split int) [][]float32 {
	t.Helper()
	ctx := context.Background()
	embs := make([][]float32, K)
	for i, id := range ids {
		embs[i] = m.EmbedResidentForTest(id)
	}
	rf.skVsumSplit = split
	seed, err := rf.PrefillLast(ctx, embs, 0)
	if err != nil {
		t.Fatalf("PrefillLast (split=%d): %v", split, err)
	}
	out := make([][]float32, len(refTokens))
	out[0] = append([]float32(nil), seed...)
	pos := K - 1
	for i := 1; i < len(refTokens); i++ {
		pos++
		lg, err := rf.Forward(m.EmbedResidentForTest(refTokens[i-1]), pos)
		if err != nil {
			t.Fatalf("teacher-forced Forward pos=%d (split=%d): %v", pos, split, err)
		}
		out[i] = append([]float32(nil), lg...)
	}
	return out
}

func scoreVsumArm(refLogits, arm [][]float32) vsumArmScore {
	var s vsumArmScore
	for i := range arm {
		_, gap, hf := decoder.NearTieArgmaxForTest(refLogits[i], arm[i])
		if hf {
			s.hardFail++
		}
		s.worstGap = max(s.worstGap, gap)
		s.kl += decoder.KLDivergenceForTest(refLogits[i], arm[i])
	}
	s.kl /= float64(len(arm))
	return s
}

func sameLogits(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
