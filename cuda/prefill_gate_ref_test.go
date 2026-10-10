//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestPrefillGateVsReferenceCUDA is docs/completed/task-prefill-gap.md §3's fidelity gate, run on CUDA for
// the L2 (attn_fused) + L3 (gemm_w4a8_mma) fast prefill. It is the ONLY thing that can justify
// changing the default, and until it passes both levers stay opt-in however fast they are.
//
// THE ORACLE IS NOT THE EXACT PATH (§3.1). Scoring a fast path against an exact path and calling the
// exact path truth books the distance against the faster arm even where the exact path is itself a
// quantisation of the activations. Here BOTH arms are scored against a THIRD thing — the CPU backend's
// own forward with f32 activations and the exact f64-accumulating attention
// (decoder/prefill_ref_gen_test.go, Phase A, run in its own process). A missing reference file SKIPS
// the cell with a message; it never falls back to exact-as-oracle.
//
// THE TWO ARMS. The exact arm is PrefillLast with both levers off: the shipped CUDA default is already
// the BATCHED prefill with the exact kernels (attn_batched + gemv_w4a8_rn), bit-identical to the M=1
// decode kernels by construction, and §3 sets the bar at what ships today. Both arms are
// teacher-forced on the SAME reference-supplied tokens — not on their own greedy output and not on
// each other's — so a per-position difference is attributable to the arm.
//
// GATE, pre-registered in §3, per (model, K) DECISION cell (K from decisionKs; S at K=3900 is a
// confirmation cell, reported the same way but not part of the decision):
//
//	(a) fast's hard-flip count vs the reference <= exact's, over the same continuation positions
//	(b) fast's mean teacher-forced agreement >= exact's mean - 1.0 pt AND fast >= exact on >= half
//	    the prompts (the PAIRED comparison, not just the cell mean)
//	(c) fast's mean continuation KL(reference || arm) <= 1.1 x exact's mean
//
// If fast is measurably WORSE, the per-lever env split (=attn / =gemm) says which lever.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestPrefillGateVsReferenceCUDA -v -timeout 4h
func TestPrefillGateVsReferenceCUDA(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (real checkpoints; needs Phase A's reference files)")
	}
	if testing.Short() {
		t.Skip("long-running gate: skipped in -short")
	}
	// THE GATE MEASURES THE KERNELS; THE FLOOR IS A PRODUCTION POLICY BUILT ON WHAT THE GATE
	// MEASURES. Leaving the floor in force here would make every sub-floor cell silently score the
	// EXACT path in both arms and report a pass — the same "scored the exact path twice" failure
	// SetFastPrefillForTest guards against, arriving by a different route. Disabling it keeps the
	// K=256 cell a real measurement of what the fast kernels do there, which is precisely the
	// evidence the floor rests on.
	t.Setenv("GOINFER_CUDA_FAST_PREFILL_FLOOR", "0")
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
		{"S", "GOINFER_CUDA_GATE_MODEL", "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"},
		{"D7", "GOINFER_CUDA_GATE_MODEL_D7", "$HOME/models/qwen2.5-7b-instruct-q4_k_m.gguf"},
	}
	// The decision set is §3's as amended: {512, 1024}, because the fast path ships behind
	// fastPrefillFloor (512) and every cell in it ships on both models
	// (docs/measurements/prefill-l2l3-phase3-2026-09-05.md §2.6). GOINFER_CUDA_GATE_KS overrides it for a
	// run that needs a depth the standing set does not cover (placing a floor between measured depths
	// without measuring there interpolates a fidelity result nobody took); an override run labels itself
	// in the log and does not silently become the decision set.
	//
	// K=256 still runs (the floor is disabled above) as a REPORTED sub-floor cell: its DOES NOT SHIP is
	// why the floor is at 512, and a pass there is what moving the floor down would need
	// (fastPrefillFloor's comment).
	decisionKs := []int{512, 1024}
	subFloorKs := []int{256}
	confirmKsByModel := map[string][]int{"S": {3900}}
	if v := os.Getenv("GOINFER_CUDA_GATE_KS"); strings.TrimSpace(v) != "" {
		subFloorKs = nil
		decisionKs = decisionKs[:0]
		for f := range strings.SplitSeq(v, ",") {
			if k, err := strconv.Atoi(strings.TrimSpace(f)); err == nil && k > 0 {
				decisionKs = append(decisionKs, k)
			}
		}
		confirmKsByModel = map[string][]int{}
		t.Logf("GOINFER_CUDA_GATE_KS override: cells %v (NOT the pre-registered decision set)", decisionKs)
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
				t.Skip("cuda resident did not engage")
			}
			if !rf.prefillReady {
				t.Skip("batched prefill kernels did not load")
			}
			tk, err := tokenizer.LoadGGUF(path)
			if err != nil {
				t.Fatalf("load tokenizer: %v", err)
			}
			// Fail loudly rather than scoring the exact path against itself.
			if err := rf.SetFastPrefillForTest(true, true); err != nil {
				t.Fatalf("could not load the fast prefill kernels: %v — a gate that cannot load "+
					"them would score the exact path twice and call it a pass", err)
			}

			allKs := append(append(append([]int{}, subFloorKs...), decisionKs...), confirmKsByModel[mc.name]...)
			maxK := 0
			for _, k := range allKs {
				maxK = max(maxK, k)
			}
			// THE SNAPSHOTS, NOT THE LIVE DOCS. PrefillGateProseFiles names live repo documents
			// (../docs/QUEUE.md and friends) that are edited constantly; PrefillGatePromptSet
			// returns the frozen copies under testdata/prefill-gate-prose-<set>/ that Phase A
			// actually built the reference from. Prefixing a LIVE document teacher-forces both arms against a
			// reference built from the SNAPSHOT of an older revision, an input the reference never saw. It fails
			// silently and plausibly: both arms get the same wrong prompt, so the run still prints agreement
			// figures and a verdict.
			setLabel, promptFiles := decoder.PrefillGatePromptSet()
			t.Logf("prompt set %q (%d snapshot files)", setLabel, len(promptFiles))
			prompts := make([][]int, 0, len(promptFiles))
			for _, f := range promptFiles {
				prompts = append(prompts, decoder.PrefillGateProseIDsForTest(t, tk, f, maxK))
			}

			ship, anyDecision := true, false
			for _, K := range allKs {
				ran := runCUDARefGateCell(t, rf, m, mc.name, K, prompts, refDir)
				if ran == nil {
					continue
				}
				if slices.Contains(subFloorKs, K) {
					t.Logf("%s K=%d is below fastPrefillFloor (%d): reported, not graded — production never runs "+
						"the fast path at this depth, and this cell's result is the floor's evidence", mc.name, K, fastPrefillFloor)
				}
				if slices.Contains(decisionKs, K) {
					anyDecision = true
					if !*ran {
						ship = false
					}
				}
			}
			if !anyDecision {
				t.Skip("no decision-set reference cells under " + refDir +
					" — run TestPrefillGateReference (decoder package) first")
			}
			verdict := map[bool]string{true: "SHIPS", false: "DOES NOT SHIP"}[ship]
			fmt.Printf("=== CUDA %s: §3 DECISION-SET VERDICT = %s ===\n", mc.name, verdict)
			t.Logf("CUDA %s: §3 decision-set verdict = %s", mc.name, verdict)
			if !ship {
				t.Errorf("§3 gate: CUDA %s does not ship on the decision set — the default stays "+
					"exact and the fast path stays opt-in, which is a RESULT, not a failure of the run",
					mc.name)
			}
		})
	}
}

// runCUDARefGateCell runs one (model, K) cell over every prompt. Returns nil if the reference files
// for the cell are absent; otherwise whether the cell ships under §3.
func runCUDARefGateCell(t *testing.T, rf *cudaResident, m *decoder.Model, model string, K int,
	prompts [][]int, refDir string) *bool {
	t.Helper()
	for pi := range prompts {
		p := filepath.Join(refDir, fmt.Sprintf("%s-K%d-p%d.bin", model, K, pi))
		if _, err := os.Stat(p); err != nil {
			t.Logf("%s K=%d: reference file missing (%s) — SKIPPING the cell; there is no fallback "+
				"to the withdrawn exact-as-oracle scoring (§3.1)", model, K, p)
			return nil
		}
	}
	var (
		sumEA, sumFA, sumEKL, sumFKL float64
		eHF, fHF, contN, n, fWins    int
		worstEGap, worstFGap         float64
	)
	t0 := time.Now()
	var identityFail []int
	for pi, ids := range prompts {
		refPath := filepath.Join(refDir, fmt.Sprintf("%s-K%d-p%d.bin", model, K, pi))
		seedRef, refTokens, refLogits, err := decoder.ReadPrefillReferenceForTest(refPath)
		if err != nil {
			t.Fatalf("read reference: %v", err)
		}
		// TE6(a): a reference from the content-keyed cache carries its prompt's hash, so identity is checked exactly;
		// a file from before the cache has no sidecar and falls back to the seed-logit KL inference below.
		refID, refWhy := decoder.PrefillRefIdentityForTest(refPath, ids[:K])
		contN = len(refTokens)
		r := runCUDARefCell(t, rf, m, ids[:K], K, seedRef, refTokens, refLogits)
		n++
		sumEA += r.exactAgree
		sumFA += r.fastAgree
		sumEKL += r.exactKL
		sumFKL += r.fastKL
		eHF += r.exactHF
		fHF += r.fastHF
		worstEGap = max(worstEGap, r.exactWorstGap)
		worstFGap = max(worstFGap, r.fastWorstGap)
		if r.fastAgree >= r.exactAgree {
			fWins++
		}
		mismatch := ""
		if refID == decoder.RefIdentityMismatch || (refID == decoder.RefIdentityNoSidecar && r.seedKL > 1.0) {
			identityFail = append(identityFail, pi+1)
			mismatch = "  <-- REFERENCE/PROMPT MISMATCH"
			if refID == decoder.RefIdentityMismatch {
				mismatch += " (" + refWhy + ")"
			}
		}
		fmt.Printf("[cuda-gate] %s K=%d prompt %2d/%2d exact(agree=%.1f%% HF=%d/%d KL=%.4f) "+
			"fast(agree=%.1f%% HF=%d/%d KL=%.4f) diff(agree=%+.1fpt KL=%+.4f) seedKL=%.4f%s elapsed=%s\n",
			model, K, pi+1, len(prompts), r.exactAgree*100, r.exactHF, contN, r.exactKL,
			r.fastAgree*100, r.fastHF, contN, r.fastKL,
			(r.fastAgree-r.exactAgree)*100, r.fastKL-r.exactKL, r.seedKL, mismatch, time.Since(t0).Round(time.Second))
	}
	mEA, mFA := sumEA/float64(n)*100, sumFA/float64(n)*100
	mEKL, mFKL := sumEKL/float64(n), sumFKL/float64(n)
	critA := fHF <= eHF
	critB := mFA >= mEA-1.0 && fWins*2 >= n
	critC := mFKL <= 1.1*mEKL
	cell := critA && critB && critC
	verdict := map[bool]string{true: "SHIPS", false: "DOES NOT SHIP"}[cell]
	// Prompt identity (docs/measurements/prefill-ref-identity-2026-09-26.md): reference files without a content-key
	// sidecar carry no prompt ids, and older set-A files predate the prompt snapshot, so some of their prompts are
	// scored against logits for different text. A cell whose exact-arm prompt-final logits sit above KL 1.0 from the
	// reference's (valid prompts: <= 0.8) is VOID, not a verdict.
	if len(identityFail) > 0 {
		cell = false
		verdict = fmt.Sprintf("VOID (reference identity check failed for prompts %v)", identityFail)
		t.Errorf("CUDA %s K=%d: reference files do not match prompts %v — regenerate them from the snapshot "+
			"(TestPrefillGateReference) before trusting this cell", model, K, identityFail)
	}
	fmt.Printf("=== CUDA %s K=%d SUMMARY (n=%d prompts x %d positions): "+
		"exact(meanAgree=%.2f%% HF=%d/%d worstGap=%.3f%% meanKL=%.5f) "+
		"fast(meanAgree=%.2f%% HF=%d/%d worstGap=%.3f%% meanKL=%.5f) "+
		"critA(HF fast<=exact)=%v critB(agree>=exact-1pt & >=half)=%v critC(KL<=1.1x)=%v — CELL %s\n",
		model, K, n, contN, mEA, eHF, n*contN, worstEGap*100, mEKL,
		mFA, fHF, n*contN, worstFGap*100, mFKL, critA, critB, critC, verdict)
	t.Logf("CUDA %s K=%d: exact(agree=%.2f%% HF=%d KL=%.5f) fast(agree=%.2f%% HF=%d KL=%.5f) "+
		"A=%v B=%v C=%v cell=%s", model, K, mEA, eHF, mEKL, mFA, fHF, mFKL, critA, critB, critC, verdict)
	return &cell
}

type cudaRefCellResult struct {
	exactAgree, fastAgree       float64
	exactHF, fastHF             int
	exactWorstGap, fastWorstGap float64
	exactKL, fastKL             float64
	seedKL                      float64 // KL(reference prompt-final logits || exact arm's): the prompt-identity check
}

// runCUDARefCell runs both arms over one K-token prompt and scores each against the reference.
//
// EVERY LOGITS SLICE IS CLONED AT CAPTURE. The resident returns a view onto device-backed scratch
// that the next launch overwrites; keeping the slice instead of a copy would silently score the
// LAST call's logits for every position — a bug that produces plausible numbers, which is the kind
// this gate exists to avoid producing.
func runCUDARefCell(t *testing.T, rf *cudaResident, m *decoder.Model, ids []int, K int,
	seedRef []float32, refTokens []int, refLogits [][]float32) cudaRefCellResult {
	t.Helper()
	ctx := context.Background()
	embs := make([][]float32, K)
	for i, id := range ids {
		embs[i] = m.EmbedResidentForTest(id)
	}
	arm := func(attn, gemm bool) ([]float32, [][]float32) {
		if err := rf.SetFastPrefillForTest(attn, gemm); err != nil {
			t.Fatalf("set levers (%v,%v): %v", attn, gemm, err)
		}
		seed, err := rf.PrefillLast(ctx, embs, 0)
		if err != nil {
			t.Fatalf("PrefillLast (attn=%v gemm=%v): %v", attn, gemm, err)
		}
		out := make([][]float32, len(refTokens))
		out[0] = append([]float32(nil), seed...)
		pos := K - 1
		for i := 1; i < len(refTokens); i++ {
			pos++
			lg, err := rf.Forward(m.EmbedResidentForTest(refTokens[i-1]), pos)
			if err != nil {
				t.Fatalf("teacher-forced Forward pos=%d: %v", pos, err)
			}
			out[i] = append([]float32(nil), lg...)
		}
		return out[0], out
	}
	// EXACT first, then FAST: each PrefillLast(…, 0) rewrites positions 0..K-1 and the
	// continuation rewrites K.., so nothing from one arm survives into the other's reads.
	exactSeed, exactCont := arm(false, false)
	useAttn, useGemm := gateFastLevers()
	_, fastCont := arm(useAttn, useGemm)

	exactAgree, _ := decoder.TeacherForcedTop1AgreementForTest(exactCont, refTokens)
	fastAgree, _ := decoder.TeacherForcedTop1AgreementForTest(fastCont, refTokens)
	eHF, eGap, eKL := scoreCUDAContVsRef(refLogits, exactCont)
	fHF, fGap, fKL := scoreCUDAContVsRef(refLogits, fastCont)
	return cudaRefCellResult{exactAgree, fastAgree, eHF, fHF, eGap, fGap, eKL, fKL,
		decoder.KLDivergenceForTest(seedRef, exactSeed)}
}

// gateFastLevers selects WHICH levers the "fast" arm uses, so the L2-only and L3-only arms §3.1
// pre-registers can be run without editing the gate.
//
// §3.1: "If fast is measurably worse, run L2-only and L3-only arms — the f16 conversion of K/V is
// the first suspect." That instruction was written before any of these kernels existed, so using it
// after a cell fails is following the pre-registration, NOT rescuing a result: the arms it names
// were part of the design. What it does NOT license is moving a decision cell or a bar after seeing
// a number, and this knob cannot do that — it changes which kernels the fast arm runs, never what
// the fast arm must achieve.
//
//	GOINFER_GATE_FAST_LEVERS=both (default) | attn | gemm
func gateFastLevers() (attn, gemm bool) {
	switch os.Getenv("GOINFER_GATE_FAST_LEVERS") {
	case "attn":
		return true, false
	case "gemm":
		return false, true
	}
	return true, true
}

// scoreCUDAContVsRef applies the 3% near-tie rule per position against the reference's own logits,
// returning hard flips, the worst gap and the mean KL.
func scoreCUDAContVsRef(refLogits, arm [][]float32) (hardFails int, worstGap, meanKL float64) {
	var kl float64
	for i := range arm {
		_, gap, hf := decoder.NearTieArgmaxForTest(refLogits[i], arm[i])
		if hf {
			hardFails++
		}
		worstGap = max(worstGap, gap)
		kl += decoder.KLDivergenceForTest(refLogits[i], arm[i])
	}
	return hardFails, worstGap, kl / float64(len(arm))
}
