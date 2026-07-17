//go:build darwin

package metal

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestQuantBar_CPUInt4VsInt8 answers the question every Metal Gemma parity number has been
// unable to: how much of the gap is QUANTIZATION and how much is METAL?
//
// The problem it solves: Metal has no like-for-like CPU reference by design. BuildResident
// requires an int8 load and re-quantizes to its own W4A8, so residentParity is always
// int4-GPU vs int8-CPU — a comparison that mixes two independent effects. Gemma measures
// 0.818 there and the control 0.990, and for months that delta was read as "Metal has a Gemma
// bug". It might instead be "int4 costs more at Gemma's shape", and the two are indistinguishable
// from a single number.
//
// This fork separates them on the CPU alone, no GPU involved. decoder's int4 mode is the near-
// exact quantization twin of Metal's resident W4A8: group-32 symmetric weights at scale=maxabs/7
// (int4GroupSize, weightmat.go) and an int8 LM head + embedding (quantMode.embedding() pins them —
// the same pin Metal ships). Same weights, same activations, same arithmetic class — the ONLY
// thing that changes between the two runs is int4-vs-int8. So:
//
//	cpu-int4 vs cpu-int8  =  the cost of the quantization class, at THIS model's shape
//	metal    vs cpu-int8  =  that cost + whatever Metal adds
//
// If Gemma's CPU fork lands near its Metal number (~0.82), then 0.818 IS the int4 bar at Gemma's
// shape and there is nothing left to fix — the model ships. If the fork comes back ~0.99 while
// Metal stays at 0.82, the quantization is exonerated and Metal really does have a Gemma-specific
// bug, which reopens the hunt with the suspect list cut in half either way.
//
// Both models run so the control calibrates the subject, per the lesson that a bar must be
// measured on this box rather than assumed.
func TestQuantBar_CPUInt4VsInt8(t *testing.T) {
	if testing.Short() {
		t.Skip("loads real models twice each")
	}
	for _, tc := range []struct {
		what string
		path string
	}{
		{"dense control (qwen2.5-1.5b)", os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")},
		{"gemma3-4b", os.ExpandEnv("$HOME/models/gemma-3-4b-it-Q4_K_M.gguf")},
	} {
		t.Run(tc.what, func(t *testing.T) {
			if _, err := os.Stat(tc.path); err != nil {
				t.Skipf("no checkpoint at %s", tc.path)
			}
			seed := seedPrompt(t, tc.path, probeText)
			st := cpuQuantFork(t, tc.path, seed, 24)
			t.Logf("%s: int4-vs-int8 min cosine %.6f, worst dNLL %.4f nats, %d/%d argmax-exact",
				tc.what, st.minCos, st.worstNLL, st.exact, st.steps)
		})
	}
}

type forkStats struct {
	steps, exact int
	minCos       float64
	worstNLL     float64
}

// cpuQuantFork runs the SAME trajectory through a cpu-int8int8 model and a cpu-int4 model in
// greedy lockstep (int8 drives, mirroring how residentParity lets the CPU drive Metal), and
// reports how far int4 drifts from int8.
//
// dNLL is reported alongside cosine because cosine has no absolute meaning at int4 — dNLL is in
// nats and says what the quantization actually costs the token the reference wanted.
func cpuQuantFork(t *testing.T, path string, seed []int, steps int) forkStats {
	t.Helper()
	m8, err := decoder.Load(path, decoder.Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load int8: %v", err)
	}
	m4, err := decoder.Load(path, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load int4: %v", err)
	}
	_, nL, _, nKV, hd, _, _ := m8.Dims()
	c8 := decoder.NewKVCache(nL, nKV, hd, 0, 1024)
	c4 := decoder.NewKVCache(nL, nKV, hd, 0, 1024)

	st := forkStats{steps: steps, minCos: 1}
	tok := seed[0]
	for i := 0; i < steps; i++ {
		l8, err := m8.ForwardForTest(tok, c8)
		if err != nil {
			t.Fatalf("int8 forward: %v", err)
		}
		l4, err := m4.ForwardForTest(tok, c4)
		if err != nil {
			t.Fatalf("int4 forward: %v", err)
		}
		a8 := argmaxF(l8)
		// Skip the two <bos> sink positions, exactly as residentParity does, so this number is
		// comparable to the Metal one it is meant to calibrate. Gemma's <bos> V is trained
		// near-zero, so a cosine there is rounding noise on both backends alike.
		if i >= 2 {
			if c := cosF(l8, l4); c < st.minCos {
				st.minCos = c
			}
			if d := nllDelta(l8, l4, a8); d > st.worstNLL {
				st.worstNLL = d
			}
		}
		if a8 == argmaxF(l4) {
			st.exact++
		}
		if i+1 < len(seed) {
			tok = seed[i+1]
		} else {
			tok = a8
		}
	}
	return st
}

// nllDelta is the extra negative log-likelihood int4 assigns to the token int8 chose: how much
// probability mass the quantization moved off the reference's own answer.
func nllDelta(ref, got []float32, id int) float64 {
	return logSoftmaxNLL(got, id) - logSoftmaxNLL(ref, id)
}

func logSoftmaxNLL(l []float32, id int) float64 {
	max := l[0]
	for _, v := range l {
		if v > max {
			max = v
		}
	}
	var sum float64
	for _, v := range l {
		sum += math.Exp(float64(v - max))
	}
	return -(float64(l[id]-max) - math.Log(sum))
}
