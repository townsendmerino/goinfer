//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestR1_decodeFidelityGate is R1's Build gate (3) (docs/tasks/red-october.md), pre-registered at
// docs/measurements/w4f16-decode-fidelity-PREREGISTERED.md before this test was written. It reuses
// two things already in the tree rather than re-deriving them: the S-K64 CPU f32-weight/
// f32-activation reference (decoder/prefill_ref_gen_test.go's TestPrefillGateReference, already
// built for R6's own gate — prompt-final logits + 64 teacher-forced DECODE continuation rows,
// exact attention), and the §3.2 pooled implementation (cellSummary/poolCells,
// metal/prefill_gate_ref_test.go), called directly rather than reimplemented. Only the two ARMS
// differ from that file's own gate: W4A8 (shipped) vs the f16 decode lane, toggled at runtime on
// ONE resident via r.decodeLaneW4F16 (the technique docs/measurements/r1-layer26-rootcause-2026-09-20.md
// validated), both built via sequential single-token Forward calls through the prefill positions
// too (not the batched PrefillLast fast-prefill path, which this lane does not touch).
//
//	GOINFER_HEAVY_TESTS=1 go test -tags "darwin goinfer_testhooks" ./metal/ -run TestR1_decodeFidelityGate -v -timeout 30m
func TestR1_decodeFidelityGate(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a real checkpoint; needs the S-K64 prefill-gate reference)")
	}
	if testing.Short() {
		t.Skip("long-running gate: skipped in -short")
	}
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("home dir: %v", err)
	}
	refDir := filepath.Join(home, "goinfer-logs", "prefill-ref")
	const K = 64
	label, promptFiles := decoder.PrefillGatePromptSet()

	// Fail fast, before loading anything, if the reference this gate depends on is missing — no
	// fallback to any withdrawn oracle.
	for pi := range promptFiles {
		p := filepath.Join(refDir, fmt.Sprintf("S-K%d-p%d.bin", K, pi))
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("reference file missing: %s — run TestPrefillGateReference (decoder package, "+
				"GOINFER_HEAVY_TESTS=1 go test -tags goinfer_testhooks ./decoder/ -run TestPrefillGateReference) first", p)
		}
	}

	t.Setenv("GOINFER_METAL_DECODE_LANE", "") // lane OFF at load; toggled at runtime below
	// ResidentContext pinned to what this run actually touches (K prefill + K teacher-forced
	// continuation = 128 positions), not metalCtxCapMax — the same "honest KV budget" reasoning
	// decoder/prefill_ref_gen_test.go's own ResidentContext: maxK+continuationN uses. Pricing the
	// fit guard against a 32768-position ceiling this test never approaches turns a load that
	// comfortably fits into one the guard declines.
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: 2 * K})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	rf, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("metal resident not built for this model")
	}
	if rf.r.decodeLaneW4F16 {
		t.Fatalf("lane ON at load despite GOINFER_METAL_DECODE_LANE=\"\"")
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	prompts := make([][]int, 0, len(promptFiles))
	for _, f := range promptFiles {
		prompts = append(prompts, decoder.PrefillGateProseIDsForTest(t, tk, f, K))
	}

	cs := &cellSummary{K: K}
	t0 := time.Now()
	for pi, ids := range prompts {
		refPath := filepath.Join(refDir, fmt.Sprintf("S-K%d-p%d.bin", K, pi))
		_, _, refLogitsRef, err := decoder.ReadPrefillReferenceForTest(refPath)
		if err != nil {
			t.Fatalf("read reference %s: %v", refPath, err)
		}
		res := runR1GateCell(t, rf, m, ids[:K], K, refLogitsRef)
		cs.n++
		cs.contN = res.contN
		cs.exactHF += res.exactHF // "exact" slot (cellSummary's naming) = W4A8, the shipped arm
		cs.fastHF += res.fastHF   // "fast" slot = the f16 decode lane
		cs.exactMatch += res.exactMatch
		cs.fastMatch += res.fastMatch
		cs.d += res.d
		cs.exactKLsum += res.exactKLsum
		cs.fastKLsum += res.fastKLsum
		cs.promptsCounted++
		if res.fastMeanKL <= res.exactMeanKL {
			cs.promptFastLowerKL++
		}
		fmt.Printf("[r1-gate3] set %q K=%d prompt %2d/%2d w4a8(agree=%.1f%% HF=%d/%d KL=%.4f) "+
			"f16(agree=%.1f%% HF=%d/%d KL=%.4f) diff(agree=%+.1fpt KL=%+.4f) elapsed=%s\n",
			label, K, pi+1, len(prompts),
			res.exactMatchRate()*100, res.exactHF, res.contN, res.exactMeanKL,
			res.fastMatchRate()*100, res.fastHF, res.contN, res.fastMeanKL,
			(res.fastMatchRate()-res.exactMatchRate())*100, res.fastMeanKL-res.exactMeanKL, time.Since(t0).Round(time.Second))
	}
	total := float64(cs.n * cs.contN)
	cs.exactMeanKL = cs.exactKLsum / total
	cs.fastMeanKL = cs.fastKLsum / total
	cs.exactAgreeRate = float64(cs.exactMatch) / total
	cs.fastAgreeRate = float64(cs.fastMatch) / total

	pooled := poolCells([]*cellSummary{cs})
	ships := pooled.critA && pooled.critB && pooled.critC
	verdict := map[bool]string{true: "PASSES", false: "DOES NOT PASS"}[ships]
	fmt.Printf("=== R1 gate (3), S, set %q, K=%d, %d prompts, %d positions: "+
		"critA(HF f16<=w4a8+2*sqrt(w4a8))=%v (w4a8=%d f16=%d) "+
		"critB(agree f16>=w4a8-2*sqrt(d)/N)=%v (w4a8Agree=%.2f%% f16Agree=%.2f%% d=%d N=%d) "+
		"critC(KL pooled f16<=w4a8 & f16 lower on >=half prompts & cell<=1.1x)=%v "+
		"(w4a8MeanKL=%.4f f16MeanKL=%.4f f16LowerPrompts=%d/%d ceilingOK=%v) — %s ===\n",
		label, K, cs.n, cs.contN,
		pooled.critA, pooled.exactHF, pooled.fastHF,
		pooled.critB, pooled.exactAgreeRate*100, pooled.fastAgreeRate*100, pooled.d, pooled.n,
		pooled.critC, pooled.exactMeanKL, pooled.fastMeanKL, pooled.promptFastLowerKL, pooled.promptsCounted, pooled.klCeilingOK,
		verdict)
	t.Logf("R1 gate (3): pooled verdict = %s (critA=%v critB=%v critC=%v)", verdict, pooled.critA, pooled.critB, pooled.critC)
}

// runR1GateCell runs the W4A8 and f16-decode-lane arms over one K-token prompt (both built via
// sequential single-token Forward calls, not batched prefill), teacher-forces both on the
// reference's own greedy continuation tokens, and scores both against the reference — mirroring
// metal/prefill_gate_ref_test.go's runPrefillRefCell exactly, with the arms swapped (W4A8/f16
// instead of exact/fast-prefill) and prefillRefCellResult's "exact"/"fast" field names reused
// unchanged so poolCells needs no modification.
func runR1GateCell(t *testing.T, rf *metalResident, m *decoder.Model, ids []int, K int, refLogitsRef [][]float32) prefillRefCellResult {
	t.Helper()
	r := rf.r
	embs := make([][]float32, K)
	for i, id := range ids {
		embs[i] = m.EmbedResidentForTest(id)
	}
	refTokens := make([]int, len(refLogitsRef))
	for i, lg := range refLogitsRef {
		refTokens[i] = argmaxF(lg)
	}

	runArm := func(f16 bool) [][]float32 {
		r.decodeLaneW4F16 = f16
		defer func() { r.decodeLaneW4F16 = false }()
		var seed []float32
		for i := range K {
			lg, err := rf.Forward(embs[i], i)
			if err != nil {
				t.Fatalf("prefill Forward pos=%d f16=%v: %v", i, f16, err)
			}
			seed = cloneF32(lg)
		}
		return teacherForceOnRef(t, rf, m, seed, refTokens, K)
	}

	w4a8Cont := runArm(false)
	f16Cont := runArm(true)

	w4a8HF, w4a8WorstGap, w4a8MeanKL, w4a8Matches := scoreContinuationVsRef(refLogitsRef, w4a8Cont)
	f16HF, f16WorstGap, f16MeanKL, f16Matches := scoreContinuationVsRef(refLogitsRef, f16Cont)

	d := 0
	for i := range w4a8Matches {
		if w4a8Matches[i] != f16Matches[i] {
			d++
		}
	}
	w4a8MatchCount, f16MatchCount := 0, 0
	for _, ok := range w4a8Matches {
		if ok {
			w4a8MatchCount++
		}
	}
	for _, ok := range f16Matches {
		if ok {
			f16MatchCount++
		}
	}

	return prefillRefCellResult{
		contN:         len(w4a8Cont),
		exactHF:       w4a8HF,
		fastHF:        f16HF,
		exactMatch:    w4a8MatchCount,
		fastMatch:     f16MatchCount,
		d:             d,
		exactKLsum:    w4a8MeanKL * float64(len(w4a8Cont)),
		fastKLsum:     f16MeanKL * float64(len(f16Cont)),
		exactMeanKL:   w4a8MeanKL,
		fastMeanKL:    f16MeanKL,
		exactWorstGap: w4a8WorstGap,
		fastWorstGap:  f16WorstGap,
	}
}
