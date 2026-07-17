//go:build darwin

package metal

import (
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// parityStats is what a resident-vs-CPU lockstep run measures. Reported by BOTH the subject and
// the control, because neither metric alone is trustworthy: argmax alone can pass a broken
// kernel on a near-tie-free trajectory, and a cosine has no absolute meaning at int4.
type parityStats struct {
	steps, exact, hard int
	worstTie, minCos   float64
}

// residentParity drives Metal resident decode against the CPU forward in greedy LOCKSTEP — the
// shipped metal convention (model_test.go): the CPU's argmax drives both sides, so they walk a
// coherent trajectory instead of an arbitrary id sequence full of near-ties.
//
// On the bar: Metal has NO like-for-like CPU reference. BuildResident requires an int8 load and
// re-quantizes to its own W4A8 (group=32, scale=max/7), which no CPU load reproduces — so this
// is always int4-GPU vs int8-CPU. CUDA's 3%-near-tie bar does NOT transfer; it compares
// int4-vs-int4. Measured here on the KNOWN-GOOD dense path, CUDA's bar fails. That is why the
// control is committed: the bar is read off it, not assumed — the same lesson CUDA learned when
// a cosine >= 0.999 draft failed its own shipped path. The threshold is the usual bug.
func residentParity(t *testing.T, path string, seed []int, steps int) parityStats {
	t.Helper()
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no checkpoint at %s", path)
	}
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	mg, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load (metal): %v", err)
	}
	rf := mg.ResidentForwardForTest()
	if rf == nil { // without this, a silent CPU fallback would pass every assertion trivially
		t.Fatal("metal resident DECLINED — admission says it should be admitted")
	}
	mcpu, err := decoder.Load(path, decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load (cpu): %v", err)
	}
	_, nL, _, nKV, hd, _, _ := mcpu.Dims()
	cache := decoder.NewKVCache(nL, nKV, hd, 0, 1024)

	st := parityStats{steps: steps, minCos: 1}
	tok := seed[0]
	for i := 0; i < steps; i++ {
		cpuL, err := mcpu.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu forward: %v", err)
		}
		// The GPU side goes through EmbedResidentForTest — that is what applies Gemma's ×√hidden
		// embed scale, so FeatEmbedScale is under test only via this call.
		gpuL, err := rf.Forward(mg.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("gpu forward: %v", err)
		}
		if c := cosF(cpuL, gpuL); c < st.minCos {
			st.minCos = c
		}
		ca, ga := argmaxF(cpuL), argmaxF(gpuL)
		if ca == ga {
			st.exact++
		} else {
			lo, hi := cpuL[0], cpuL[0]
			for _, v := range cpuL {
				if v < lo {
					lo = v
				}
				if v > hi {
					hi = v
				}
			}
			gap := float64(cpuL[ca]-cpuL[ga]) / (float64(hi-lo) + 1e-9)
			if gap > st.worstTie {
				st.worstTie = gap
			}
			if gap > 0.03 {
				st.hard++
			}
		}
		if i+1 < len(seed) { // seed the prompt, then greedy-continue off the CPU
			tok = seed[i+1]
		} else {
			tok = ca
		}
	}
	t.Logf("%s: %d/%d argmax-exact, worst near-tie %.3f%%, %d gaps >3%%, min cosine %.6f",
		path, st.exact, st.steps, st.worstTie*100, st.hard, st.minCos)
	return st
}

// Arbitrary valid seed ids; the run greedy-continues after them.
var denseControlIDs = []int{785, 3840, 315, 6137, 448, 264, 1467, 315, 264, 3070}
var gemma3IDs = []int{2, 651, 6037, 529, 6081, 603, 12545, 235265, 714, 6398}

// TestDenseResidentParity is the CONTROL: the same harness on the known-good SHIPPED dense Qwen
// path, so "is this cosine / this many near-ties good?" has an answer measured on THIS box
// rather than assumed. Without it a Gemma number cannot be judged.
func TestDenseResidentParity(t *testing.T) {
	st := residentParity(t, os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"), denseControlIDs, 24)
	assertParity(t, "dense control", st)
}

// TestGemma3ResidentParity is the Metal Gemma 3 gate — judged by the SAME bar the control meets.
// Needs the checkpoint (~4 GB at int8, loaded twice → budget ~10 GB); skips without it.
func TestGemma3ResidentParity(t *testing.T) {
	// Dormant until the kernels are validated: metal ships the Gemma kernels but does not yet
	// DECLARE the features, so gemma3 declines to CPU. Skip rather than fail — and the moment
	// the declaration lands this becomes a live gate (residentParity t.Fatals on a decline,
	// which is what catches a silent fallback).
	if !decoder.ResidentBackendFeatures["metal"][decoder.FeatSandwichNorm] {
		t.Skip("metal does not declare the Gemma features yet (kernels dormant) — see docs/task-metal-gemma.md")
	}
	st := residentParity(t, os.ExpandEnv("$HOME/models/gemma-3-4b-it-Q4_K_M.gguf"), gemma3IDs, 24)
	assertParity(t, "gemma3", st)
}

// assertParity applies bars READ OFF the dense control (measured on this box), not invented.
//
// Why these and not CUDA's: Metal's int4-GPU vs int8-CPU gap puts many positions near ties, so
// argmax parity here is TRAJECTORY-SENSITIVE — the same shipped dense path scores 20/24 on
// model_test.go's ids but 15/24 on this test's. So argmax is a loose sanity bound only, and
// COSINE is the stable signal: the control measures 0.962 (this harness) to 0.990 (shipped
// harness). A wrong kernel — sandwich norm misplaced, rope base swapped, GeGLU run as SwiGLU —
// collapses cosine far below 0.95; int4-vs-int8 noise does not come close.
func assertParity(t *testing.T, what string, st parityStats) {
	t.Helper()
	if st.minCos < 0.95 { // control: 0.962 → 0.95 with margin
		t.Errorf("%s: min logit cosine %.6f < 0.95 — gross breakage, not int4 noise (control measures ~0.96)", what, st.minCos)
	}
	if st.exact*2 < st.steps { // control: 15/24 = 62% → require >= 50%
		t.Errorf("%s: argmax parity %d/%d < 50%% — the control manages ~62%% on this harness", what, st.exact, st.steps)
	}
}
