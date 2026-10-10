//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGLMResidentParity is the joint end-to-end gate for two features that only arrive together:
// PARTIAL ROTARY (rotary_dim < head_dim) and the UNGATED SHARED EXPERT (GLM/DeepSeek). Every
// committed partial-rotary fixture also has a shared expert, so neither is independently
// reachable through a model — glm-tiny is the one checkpoint that exercises both at once, plus
// sigmoid routing, a router selection bias, qk-norm, and a dense first_k_dense=1 prefix layer.
//
// A failure here is therefore not narrowly attributable, which is exactly why the kernel-level
// gates exist alongside it (TestRopePartial for the tail-caching; TestMoE* for routing and the
// indexed GEMV). This test is the integration: that the resident forward, with all of those
// composed, tracks the CPU forward on real weights under the repo's 3% near-tie rule.
func TestGLMResidentParity(t *testing.T) {
	for _, path := range []string{"../testdata/glm-tiny", "../testdata/glm-tiny-bias"} {
		t.Run(path, func(t *testing.T) {
			requireDeviceAndFixture(t, path)
			mc, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int4"})
			if err != nil {
				t.Fatalf("load (cuda): %v", err)
			}
			defer mc.Close()

			feats := mc.RequiredResidentFeatures()
			hasPR, hasMoE := false, false
			for _, f := range feats {
				hasPR = hasPR || f == decoder.FeatPartialRotary
				hasMoE = hasMoE || f == decoder.FeatMoE
			}
			if !hasPR || !hasMoE {
				t.Fatalf("fixture requires %v — this gate is only meaningful if it exercises BOTH "+
					"partial-rotary AND moe; if the fixture changed, this test no longer gates what it claims", feats)
			}
			rf := mc.ResidentForwardForTest()
			if rf == nil {
				t.Fatalf("cuda resident DECLINED %s — it requires %v, all of which CUDA now declares, "+
					"so admission and BuildResident disagree", path, feats)
			}

			mcpu, err := decoder.Load(path, decoder.Options{Quant: "int4"})
			if err != nil {
				t.Fatalf("load (cpu): %v", err)
			}
			defer mcpu.Close()

			_, _, _, _, _, _, vocab := mc.Dims()
			// Spread across the vocab so different tokens route to different experts. Kept < 256
			// (glm-tiny's vocab); the range check below is the backstop.
			prompt := []int{1, 40, 128, 233, 7, 190, 66, 201, 15, 250, 88, 170}
			for _, id := range prompt {
				if id < 0 || id >= vocab {
					t.Fatalf("prompt id %d out of range for vocab %d", id, vocab)
				}
			}

			cache := mcpu.NewCache(len(prompt) + 2)
			exact, hard := 0, 0
			worst, minCos := 0.0, 1.0
			for i, tok := range prompt {
				cpuL, err := mcpu.ForwardForTest(tok, cache)
				if err != nil {
					t.Fatalf("cpu pos %d: %v", i, err)
				}
				gpuL, err := rf.Forward(mc.EmbedResidentForTest(tok), i)
				if err != nil {
					t.Fatalf("cuda pos %d: %v", i, err)
				}
				var dot, na, nb float64
				for j := range cpuL {
					dot += float64(cpuL[j]) * float64(gpuL[j])
					na += float64(cpuL[j]) * float64(cpuL[j])
					nb += float64(gpuL[j]) * float64(gpuL[j])
				}
				c := dot / (math.Sqrt(na) * math.Sqrt(nb))
				if math.IsNaN(c) {
					t.Fatalf("pos %d: logit cosine is NaN — degenerate resident output (an unstored KV "+
						"tail reads back as poison/zero, or a routing NaN)", i)
				}
				if c < minCos {
					minCos = c
				}
				ca, ga := argmaxF(cpuL), argmaxF(gpuL)
				if ca == ga {
					exact++
					continue
				}
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
				if gap > worst {
					worst = gap
				}
				if gap > 0.03 {
					hard++
					t.Errorf("pos %d: CPU=%d CUDA=%d gap=%.3f%% > 3%%", i, ca, ga, gap*100)
				}
			}
			t.Logf("GLM resident vs CPU: %d/%d exact | worst near-tie %.3f%% | hard fails %d | min cosine %.6f",
				exact, len(prompt), worst*100, hard, minCos)

			// The 3% rule and this floor divide the work, and BOTH are needed. A tail cache that is not kept or a
			// garbled shared-expert combine is caught by the 3% rule; a shared expert that is SKIPPED or has its
			// gate/up swapped is what the argmax rule MISSES: at sharedInter=32 over four layers it is a small
			// perturbation of 256-dim logits, so the top token barely moves (the mixtral-tiny problem), and the
			// cosine floor catches it. 0.998 sits below every correct run and above the tightest real bug. It is
			// a NARROW gate (~0.0015 margin), the honest ceiling this tiny fixture affords for a component this
			// small; TestRopePartial gates the tail-caching independently and strongly. The measured table (each
			// composed piece broken in turn): docs/code-notes/cuda.md#TestGLMResidentParity.
			if minCos < 0.998 {
				t.Errorf("logit cosine %.6f < 0.998 on %s — below the measured correct run (~0.9997) and "+
					"into shared-expert-bug territory (skipped ~0.996, gate/up swapped ~0.97); the argmax "+
					"rule cannot see a shared-expert bug on a fixture this small, so this floor is the gate", minCos, path)
			}
		})
	}
}
