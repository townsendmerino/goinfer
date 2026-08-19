//go:build realckpt

// DOES A ROUTER FLIP EXPLAIN THE DIP? — the mechanism experiment for B13's last standing red.
//
// The state of the argument before this test. TestQwen35GGUF_vsSafetensors reports min cosine
// 0.987835 at step 63 against a mean of 0.998114 and calls it a loader bug. Two probes contradict
// that label: weightDiff finds every transform-bearing tensor bit-exact or at a uniform
// relL2 ~0.0057 Q8_0 floor, and locateDivergence finds a smooth, NON-MONOTONIC decay across all 40
// layers with no step (a localized defect cannot recover, and that curve recovers repeatedly).
// From those two, "a ~0.5% weight delta flips borderline top-k router choices at a few positions"
// is an INFERENCE about the outlier steps. This test measures it instead.
//
// THE PREDICTION, stated before the run so it can fail. Both containers are teacher-forced through
// the same 80 steps; every moeMLP call's top-k selection is recorded on each side (moeSelTrace, the
// existing seam). If routing flips are the mechanism:
//
//  1. the steps with the LOWEST logit cosine carry the MOST flipped layers, and
//  2. step 63 — the 0.987835 outlier — is at or near the top of the flip ranking, and
//  3. the great majority of steps have ZERO flips (which is why the mean sits at 0.998).
//
// If instead flips are spread evenly across steps, or step 63 has none, the near-tie story is
// WRONG and the dip needs another explanation — which is a real finding, not a failed test. So
// this asserts only what any story must satisfy (the two runs are comparable) and prints the
// correlation for the recorded decision.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestQwen35GGUF_routeFlip -v -timeout 120m
package decoder

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"slices"
	"sort"
	"testing"
)

func TestQwen35GGUF_routeFlipAtOutlier(t *testing.T) {
	requireHeavyModel(t)
	// DIAGNOSTIC, NOT A GATE — and it must not sit in the release sweep's path (2026-08-19).
	// scripts/parity_sweep.sh phase 2 runs `-run 'Qwen35|Real_gate'` under a 120m timeout, and this
	// test's name matches. On the v0.14.0 prep sweep the two B13 diagnostics together burned ~41
	// minutes of that budget and the run TIMED OUT inside this one — which pushed
	// TestQwen35GGUF_weightDiff, a required 40-second gate, off the end of the run entirely. A gate
	// that DID NOT RUN is a blocker by the sweep's own rule, so an instrument that asserts almost
	// nothing took a real gate down with it.
	//
	// Gated by env rather than renamed on purpose: Go's -run matches substrings, so excluding it by
	// name would mean stripping "Qwen35" from a qwen3.5 test — worse discoverability to work around
	// a scheduling problem. This way the name stays, and the sweep reports an honest SKIP.
	if os.Getenv("GOINFER_DIAG") == "" {
		t.Skip("DIAGNOSTIC (set GOINFER_DIAG=1): prints evidence for a judgement, asserts only what " +
			"holds under either story. Not a gate — see B13 in docs/queue-release.md")
	}
	gguf := assetPath(t, "GOINFER_QWEN35_GGUF")
	dir := realQwen35Dir(t)
	goldenDir := assetPath(t, "GOINFER_QWEN35_GOLDEN")
	raw, err := os.ReadFile(filepath.Join(goldenDir, "manifest.json"))
	if err != nil {
		t.Skipf("no golden manifest: %v", err)
	}
	var man gate2Manifest
	if err := json.Unmarshal(raw, &man); err != nil {
		t.Fatalf("manifest: %v", err)
	}

	// Teacher-forced exactly as the oracle does it, so "step 63" means the same step there and
	// here: prefill prompt[:-1], then feed prompt[-1] and each golden token in turn. Returns the
	// per-step last-position logits AND, per step, the top-k selection of every layer.
	run := func(label, path string) ([][]float32, [][][]int, int) {
		prev := runtime.GOMAXPROCS(2)
		m, err := Load(path, Options{Quant: "int8int8"})
		runtime.GOMAXPROCS(prev)
		if err != nil {
			t.Fatalf("%s Load: %v", label, err)
		}
		defer func() {
			m.Close()
			debug.FreeOSMemory()
		}()
		nL := m.w.arch.NumLayers
		var logits [][]float32
		var sel [][][]int
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
				// Arm the trace for exactly ONE token, so the call→layer mapping is unambiguous:
				// every layer of this family carries a routed FFN, so one token is nL calls in
				// layer order. (Arming it across the prefill would make the index arithmetic
				// depend on prompt length, which is the kind of bookkeeping that quietly slips.)
				moeSelTrace = make([][]int, 0, nL)
				lg, ferr := m.forward(cur, cache)
				step := moeSelTrace
				moeSelTrace = nil
				if ferr != nil {
					t.Fatalf("%s prompt %d step %d: %v", label, pi, s, ferr)
				}
				if len(step) != nL {
					t.Fatalf("%s prompt %d step %d: %d router calls, want %d (one per layer)",
						label, pi, s, len(step), nL)
				}
				logits = append(logits, append([]float32(nil), lg...))
				sel = append(sel, step)
				cur = p.GenIDs[s]
			}
		}
		return logits, sel, nL
	}

	refL, refSel, nL := run("safetensors", dir)
	gotL, gotSel, nL2 := run("gguf", gguf)
	if len(refL) != len(gotL) || nL != nL2 {
		t.Fatalf("shape mismatch: %d/%d steps, %d/%d layers", len(refL), len(gotL), nL, nL2)
	}

	type stepStat struct {
		step, flips, firstFlipLayer int
		cos                         float64
	}
	stats := make([]stepStat, len(refL))
	zero, totalFlips := 0, 0
	for s := range refL {
		st := stepStat{step: s, cos: cosineFull(gotL[s], refL[s]), firstFlipLayer: -1}
		for l := 0; l < nL; l++ {
			a := append([]int(nil), refSel[s][l]...)
			b := append([]int(nil), gotSel[s][l]...)
			sort.Ints(a)
			sort.Ints(b) // the SET, not the order: a reordered top-k is the same experts
			if !slices.Equal(a, b) {
				st.flips++
				if st.firstFlipLayer < 0 {
					st.firstFlipLayer = l
				}
			}
		}
		if st.flips == 0 {
			zero++
		}
		totalFlips += st.flips
		stats[s] = st
	}

	worst := slices.Clone(stats)
	sort.Slice(worst, func(i, j int) bool { return worst[i].cos < worst[j].cos })
	t.Logf("=== %d steps, %d layers each; %d steps with ZERO flipped layers; %d flipped (step,layer) pairs total ===",
		len(stats), nL, zero, totalFlips)
	t.Logf("--- the 10 WORST steps by logit cosine ---")
	for i := 0; i < 10 && i < len(worst); i++ {
		w := worst[i]
		t.Logf("  step %2d: cosine %.6f | %2d/%d layers flipped | first flip at layer %d",
			w.step, w.cos, w.flips, nL, w.firstFlipLayer)
	}
	byFlips := slices.Clone(stats)
	sort.Slice(byFlips, func(i, j int) bool { return byFlips[i].flips > byFlips[j].flips })
	t.Logf("--- the 10 steps with the MOST flipped layers ---")
	for i := 0; i < 10 && i < len(byFlips); i++ {
		b := byFlips[i]
		t.Logf("  step %2d: %2d/%d layers flipped | cosine %.6f | first flip at layer %d",
			b.step, b.flips, nL, b.cos, b.firstFlipLayer)
	}
	// The mean cosine of zero-flip steps vs flipped steps is the whole claim in two numbers.
	var sumZero, sumFlip float64
	var nZero, nFlip int
	for _, s := range stats {
		if s.flips == 0 {
			sumZero += s.cos
			nZero++
		} else {
			sumFlip += s.cos
			nFlip++
		}
	}
	if nZero > 0 {
		t.Logf("mean cosine, steps with NO routing flip:  %.6f  (n=%d)", sumZero/float64(nZero), nZero)
	}
	if nFlip > 0 {
		t.Logf("mean cosine, steps WITH a routing flip:   %.6f  (n=%d)", sumFlip/float64(nFlip), nFlip)
	}
	t.Logf("PREDICTION UNDER THE NEAR-TIE STORY: the worst-cosine steps are the flipped ones, most " +
		"steps have zero flips, and the no-flip mean sits far above the flipped mean. Even spread, " +
		"or a zero-flip worst step, REFUTES it and the dip needs another explanation.")

	// Only comparability is asserted; the mechanism is reported, not thresholded (same discipline
	// as locateDivergence — a bar here would pre-empt the judgement this evidence informs).
	if len(stats) == 0 {
		t.Fatal("no steps compared")
	}
}
