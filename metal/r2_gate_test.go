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
	runDecodeFidelityGate(t, "r2-gate3", "attention_fa", nil)
}

// runDecodeFidelityGate is TestR2_decodeFidelityGate's body, shared with R17's gate
// (TestR17_decodeFidelityGate): prep, when non-nil, runs once on the built resident before the first cell —
// R17 uses it to install its prototype as r.pAttnFA, so the candidate arm ("fast" slot, r.decodeAttnFA on)
// dispatches the prototype while the exact arm is unchanged. tag and candName label the output lines.
func runDecodeFidelityGate(t *testing.T, tag, candName string, prep func(t *testing.T, r *resident)) {
	t.Helper()
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
	const K = 3900
	const contN = 64
	label, promptFiles := decoder.PrefillGatePromptSet()
	// The reference directory follows the prompt set (refDirFor, the generator's own rule). This was hardcoded to set
	// A's directory, so GOINFER_PREFILL_GATE_PROMPTS=b silently scored set-B prompts against set-A logits.
	refDir := refDirFor(home, label)
	for pi := range promptFiles {
		p := filepath.Join(refDir, fmt.Sprintf("S-K%d-p%d.bin", K, pi))
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("reference file missing: %s — run TestPrefillGateReference (decoder package) first", p)
		}
	}

	t.Setenv("GOINFER_METAL_ATTN_FA", "0") // kernel OFF at build; toggled at runtime below
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
	if prep != nil {
		prep(t, r)
	}
	prompts := make([][]int, 0, len(promptFiles))
	for _, f := range promptFiles {
		prompts = append(prompts, decoder.PrefillGateProseIDsForTest(t, tk, f, K))
	}

	cs := &cellSummary{K: K}
	t0 := time.Now()
	var suspect []int
	for pi, ids := range prompts {
		refPath := filepath.Join(refDir, fmt.Sprintf("S-K%d-p%d.bin", K, pi))
		_, _, refLogitsRef, err := decoder.ReadPrefillReferenceForTest(refPath)
		if err != nil {
			t.Fatalf("read reference %s: %v", refPath, err)
		}
		res := runR2GateCell(t, rf, m, ids[:K], K, refLogitsRef)
		// Prompt identity: the reference files carry no prompt ids, so check the one row both arms share — the
		// prompt-final logits (reference row 0 vs the batched prefill's seed). A reference generated from different
		// text lands far above the W4A8 level here. Found 2026-09-25: set A's S-K3900 files (generated 2026-09-05 from
		// the live docs) predate the 2026-09-09 snapshot the prompts now come from, and 4 of 10 prompts changed inside
		// the 3900-token window (metal-decode-attn-r17-2026-09-25.md).
		if res.seedKL > 1.0 {
			suspect = append(suspect, pi+1)
		}
		fmt.Printf("[%s] prompt %2d seed-row KL(ref || prefill) = %.4f%s\n", tag, pi+1, res.seedKL,
			map[bool]string{true: "  <-- REFERENCE/PROMPT MISMATCH SUSPECTED", false: ""}[res.seedKL > 1.0])
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
		fmt.Printf("[%s] set %q K=%d prompt %2d/%2d shipped(agree=%.1f%% HF=%d/%d KL=%.4f) "+
			"%s(agree=%.1f%% HF=%d/%d KL=%.4f) diff(agree=%+.1fpt KL=%+.4f) elapsed=%s\n",
			tag, label, K, pi+1, len(prompts),
			res.exactMatchRate()*100, res.exactHF, res.contN, res.exactMeanKL,
			candName, res.fastMatchRate()*100, res.fastHF, res.contN, res.fastMeanKL,
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
	fmt.Printf("=== %s (candidate %s), S, set %q, K=%d, %d prompts, %d positions: "+
		"critA(HF fa<=shipped+2*sqrt(shipped))=%v (shipped=%d fa=%d) "+
		"critB(agree fa>=shipped-2*sqrt(d)/N)=%v (shippedAgree=%.2f%% faAgree=%.2f%% d=%d N=%d) "+
		"critC(KL pooled fa<=shipped & fa lower on >=half prompts & cell<=1.1x)=%v "+
		"(shippedMeanKL=%.4f faMeanKL=%.4f faLowerPrompts=%d/%d ceilingOK=%v) — %s ===\n",
		tag, candName, label, K, cs.n, cs.contN,
		pooled.critA, pooled.exactHF, pooled.fastHF,
		pooled.critB, pooled.exactAgreeRate*100, pooled.fastAgreeRate*100, pooled.d, pooled.n,
		pooled.critC, pooled.exactMeanKL, pooled.fastMeanKL, pooled.promptFastLowerKL, pooled.promptsCounted, pooled.klCeilingOK,
		verdict)
	t.Logf("%s: pooled verdict = %s (critA=%v critB=%v critC=%v)", tag, verdict, pooled.critA, pooled.critB, pooled.critC)
	if len(suspect) > 0 {
		t.Errorf("%s: prompts %v fail the reference-identity check (seed-row KL > 1.0): the verdict above is not valid for this reference set", tag, suspect)
	}
}

// r2GateArmHook, when non-nil, runs at the start of each arm's continuation (cand=false: the exact arm, true: the
// candidate arm), after the attention_fa toggle is set — for candidates that are not "attention_fa on", e.g. R17's
// null controls that swap the exact kernel's pipeline. Test-only; set and cleared by the test that needs it.
var r2GateArmHook func(r *resident, cand bool)

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
	var seedKL float64
	runArm := func(fa bool) [][]float32 {
		r.decodeAttnFA = false
		seed, err := rf.PrefillLast(ctx, embs, 0)
		if err != nil {
			t.Fatalf("PrefillLast: %v", err)
		}
		seed = cloneF32(seed)
		seedKL = decoder.KLDivergenceForTest(refLogitsRef[0], seed)
		r.decodeAttnFA = fa
		defer func() { r.decodeAttnFA = false }()
		// Drop the command buffer the pipelined executor pre-encoded under the PREVIOUS arm's toggle: execLoop
		// encodes token t+1 right after committing t, from the state at that moment, and PrefillLast does not
		// go through it — so without this each arm's first continuation step ran the other arm's attention
		// kernel (found 2026-09-25, R17: metal-decode-attn-r17-2026-09-25.md). ensureExec restarts it.
		r.stopExec()
		if fa {
			// every continuation position is past the floor; confirm the toggle engages
			r.setPos(K)
			for l := 0; l < r.nL; l++ {
				if !r.canUseAttnFA(l) {
					t.Fatalf("layer %d: canUseAttnFA false at curNKeys=%d with the toggle on", l, r.curNKeys)
				}
			}
		}
		if r2GateArmHook != nil {
			r2GateArmHook(r, fa)
			r.stopExec() // the hook changed encode-time state too
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
		seedKL:        seedKL,
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
