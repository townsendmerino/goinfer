//go:build cuda

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

			// The 3% rule and this floor divide the work, and BOTH are needed — proven by breaking
			// each composed piece and measuring (glm-tiny / glm-tiny-bias, this box):
			//
			//   broken piece                         exact       min cosine     caught by
			//   ----------------------------------  ----------  -------------  -----------
			//   correct                              12/12,12/12  0.9998,0.9999  —
			//   partial-rotary tail not cached       4/12, 3/12   0.498, 0.617   3% rule
			//   shared-expert combine garbage        0/12, 0/12   -0.08, -0.11   3% rule
			//   shared expert SKIPPED entirely      12/12,11/12   0.9966,0.9949  FLOOR only
			//   shared gate/up swapped in glu        9/12,10/12    0.968, 0.980   FLOOR (mostly)
			//
			// The last two are the ones the argmax rule MISSES: the shared expert at sharedInter=32
			// over four layers is a small perturbation of 256-dim logits, so dropping or mangling it
			// barely moves the top token — exactly the mixtral-tiny problem, and the reason a cosine
			// floor is not optional here. 0.998 sits below every correct run (min 0.9995 across three
			// prompts) and above the tightest real bug (shared skipped, 0.9966). It is a NARROW gate
			// (~0.0015 margin), which is the honest ceiling this tiny fixture affords for a component
			// this small; TestRopePartial gates the tail-caching independently and strongly.
			if minCos < 0.998 {
				t.Errorf("logit cosine %.6f < 0.998 on %s — below the measured correct run (~0.9997) and "+
					"into shared-expert-bug territory (skipped ~0.996, gate/up swapped ~0.97); the argmax "+
					"rule cannot see a shared-expert bug on a fixture this small, so this floor is the gate", minCos, path)
			}
		})
	}
}
