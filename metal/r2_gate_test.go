//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestR2_decodeFidelityGate is R2's gate (3) (docs/tasks/red-october.md: "the same teacher-forced
// fidelity gate R1 uses, S at depth 3900 as the decision cell, W4A8 + shipped attention as the exact
// arm"). Same construction as metal/r1_gate3_test.go, two substitutions: the toggle is
// r.decodeAttnFA (attention_fa vs the shipped attention kernel; canUseAttnFA reads the field on
// every dispatch, pipelines and the partial buffer are always built), and the cell is K=3900 —
// above attnFADepthFloor (1536), so every one of the 64 teacher-forced continuation positions
// dispatches attention_fa in the candidate arm. Reference: the S-K3900 CPU f32-weight/
// f32-activation files decoder/prefill_ref_gen_test.go already built (prompt-final logits + 64
// teacher-forced decode rows, exact attention). Prefill goes through rf.PrefillLast (the batched
// path, identical for both arms — attention_fa cannot engage there, a structural fact from its
// single-query-position addressing) and the continuation through the per-token Forward path this
// kernel actually changes. Pooled §3.2 criteria via poolCells, called directly.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags "darwin goinfer_testhooks" ./metal/ -run 'TestR2_decodeFidelityGate$' -v -timeout 60m
func TestR2_decodeFidelityGate(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a real checkpoint; needs the S-K3900 prefill-gate reference)")
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
	const K = 3900
	const contN = 64
	label, promptFiles := decoder.PrefillGatePromptSet()
	for pi := range promptFiles {
		p := filepath.Join(refDir, fmt.Sprintf("S-K%d-p%d.bin", K, pi))
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("reference file missing: %s — run TestPrefillGateReference (decoder package) first", p)
		}
	}

	t.Setenv("GOINFER_METAL_ATTN_FA", "") // kernel OFF at build; toggled at runtime below
	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4", ResidentContext: K + contN + 8})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	rf, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Fatalf("metal resident not built for this model")
	}
	r := rf.r
	if r.decodeAttnFA {
		t.Fatalf("attention_fa ON at load despite GOINFER_METAL_ATTN_FA=\"\"")
	}
	if r.attnFAPartial == (Buffer{}) {
		t.Fatalf("attnFAPartial not allocated — attention_fa cannot engage on this model")
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
		res := runR2GateCell(t, rf, m, ids[:K], K, refLogitsRef)
		cs.n++
		cs.contN = res.contN
		cs.exactHF += res.exactHF // "exact" slot = shipped attention (W4A8 decode, shipped kernel)
		cs.fastHF += res.fastHF   // "fast" slot = attention_fa
		cs.exactMatch += res.exactMatch
		cs.fastMatch += res.fastMatch
		cs.d += res.d
		cs.exactKLsum += res.exactKLsum
		cs.fastKLsum += res.fastKLsum
		cs.promptsCounted++
		if res.fastMeanKL <= res.exactMeanKL {
			cs.promptFastLowerKL++
		}
		fmt.Printf("[r2-gate3] set %q K=%d prompt %2d/%2d shipped(agree=%.1f%% HF=%d/%d KL=%.4f) "+
			"attention_fa(agree=%.1f%% HF=%d/%d KL=%.4f) diff(agree=%+.1fpt KL=%+.4f) elapsed=%s\n",
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
	fmt.Printf("=== R2 gate (3), S, set %q, K=%d, %d prompts, %d positions: "+
		"critA(HF fa<=shipped+2*sqrt(shipped))=%v (shipped=%d fa=%d) "+
		"critB(agree fa>=shipped-2*sqrt(d)/N)=%v (shippedAgree=%.2f%% faAgree=%.2f%% d=%d N=%d) "+
		"critC(KL pooled fa<=shipped & fa lower on >=half prompts & cell<=1.1x)=%v "+
		"(shippedMeanKL=%.4f faMeanKL=%.4f faLowerPrompts=%d/%d ceilingOK=%v) — %s ===\n",
		label, K, cs.n, cs.contN,
		pooled.critA, pooled.exactHF, pooled.fastHF,
		pooled.critB, pooled.exactAgreeRate*100, pooled.fastAgreeRate*100, pooled.d, pooled.n,
		pooled.critC, pooled.exactMeanKL, pooled.fastMeanKL, pooled.promptFastLowerKL, pooled.promptsCounted, pooled.klCeilingOK,
		verdict)
	t.Logf("R2 gate (3): pooled verdict = %s (critA=%v critB=%v critC=%v)", verdict, pooled.critA, pooled.critB, pooled.critC)
}

// runR2GateCell: both arms prefill via the batched path (identical; attention_fa cannot engage
// there), then teacher-force the reference's own continuation through the per-token path with
// r.decodeAttnFA off (shipped) and on (attention_fa). The prefill is re-run per arm so each arm's
// KV state at positions >= K is written only by its own continuation.
func runR2GateCell(t *testing.T, rf *metalResident, m *decoder.Model, ids []int, K int, refLogitsRef [][]float32) prefillRefCellResult {
	t.Helper()
	r := rf.r
	ctx := context.Background()
	embs := make([][]float32, K)
	for i, id := range ids {
		embs[i] = m.EmbedResidentForTest(id)
	}
	refTokens := make([]int, len(refLogitsRef))
	for i, lg := range refLogitsRef {
		refTokens[i] = argmaxF(lg)
	}
	runArm := func(fa bool) [][]float32 {
		r.decodeAttnFA = false
		seed, err := rf.PrefillLast(ctx, embs, 0)
		if err != nil {
			t.Fatalf("PrefillLast: %v", err)
		}
		seed = cloneF32(seed)
		r.decodeAttnFA = fa
		defer func() { r.decodeAttnFA = false }()
		if fa {
			// every continuation position is past the floor; confirm the toggle engages
			r.setPos(K)
			for l := 0; l < r.nL; l++ {
				if !r.canUseAttnFA(l) {
					t.Fatalf("layer %d: canUseAttnFA false at curNKeys=%d with the toggle on", l, r.curNKeys)
				}
			}
		}
		return teacherForceOnRef(t, rf, m, seed, refTokens, K)
	}
	shCont := runArm(false)
	faCont := runArm(true)
	if err := r.takeExecErr(); err != nil {
		t.Fatalf("resident exec error: %v", err)
	}

	shHF, shWorstGap, shMeanKL, shMatches := scoreContinuationVsRef(refLogitsRef, shCont)
	faHF, faWorstGap, faMeanKL, faMatches := scoreContinuationVsRef(refLogitsRef, faCont)
	d := 0
	for i := range shMatches {
		if shMatches[i] != faMatches[i] {
			d++
		}
	}
	shMatchCount, faMatchCount := 0, 0
	for _, ok := range shMatches {
		if ok {
			shMatchCount++
		}
	}
	for _, ok := range faMatches {
		if ok {
			faMatchCount++
		}
	}
	return prefillRefCellResult{
		contN:         len(shCont),
		exactHF:       shHF,
		fastHF:        faHF,
		exactMatch:    shMatchCount,
		fastMatch:     faMatchCount,
		d:             d,
		exactKLsum:    shMeanKL * float64(len(shCont)),
		fastKLsum:     faMeanKL * float64(len(faCont)),
		exactMeanKL:   shMeanKL,
		fastMeanKL:    faMeanKL,
		exactWorstGap: shWorstGap,
		fastWorstGap:  faWorstGap,
	}
}
