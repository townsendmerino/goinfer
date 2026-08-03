//go:build darwin

package metal

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

const moeTinyDir = "../testdata/gemma4-moe-tiny"

// buildMoeTiny loads the gemma4-moe-tiny fixture int4 (both sides) and builds the Metal resident.
// env on so the arch is bridge-eligible; BuildResident is the forward-numerics vehicle (admission is
// covered by TestGemma4Admission_envGated). A build error here means metal DECLINED the parallel
// dense‖MoE — which, before Step 5d, it did on purpose (the buildMoE guard).
func buildMoeTiny(t *testing.T) (*Resident, *decoder.Model, *decoder.Model) {
	t.Helper()
	if _, err := os.Stat(moeTinyDir); err != nil {
		t.Skipf("no fixture (%s) — scp testdata/gemma4-moe-tiny from the box", moeTinyDir)
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	mg, err := decoder.Load(moeTinyDir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (resident side): %v", err)
	}
	r, err := BuildResident(mg)
	if err != nil {
		mg.Close()
		t.Fatalf("BuildResident: %v — metal declined the gemma4 parallel dense‖MoE fixture", err)
	}
	mcpu, err := decoder.Load(moeTinyDir, decoder.Options{Quant: "int4"})
	if err != nil {
		r.Close()
		mg.Close()
		t.Fatalf("load (cpu side): %v", err)
	}
	return r, mg, mcpu
}

// TestGemma4MoE_localize is the per-layer localization wired BEFORE the whole-forward gate — the
// single thing that turned the CUDA dense divergence into a one-run answer. It diffs the resident
// residual stream after each layer against the CPU's (g4traceHidden) at pos 0, so a whole-forward
// miss is immediately attributable to a layer: a MoE layer red with the dense prefix clean isolates
// the fault to the parallel dense‖MoE block (router / dense / expert / join), which the Step-5a/5c
// isolation gates then further localize. The 0.90 floor is the crater backstop (this toy is
// quant-hostile — see the Step-5 noise-floor pre-flight, 0.79 logit cosine int4-vs-f32); the SHARP
// signal is a single layer sitting far below its neighbours.
func TestGemma4MoE_localize(t *testing.T) {
	r, mg, mcpu := buildMoeTiny(t)
	defer r.Close()
	defer mg.Close()
	defer mcpu.Close()

	decoder.SetGemma4HiddenCaptureForTest(true)
	defer decoder.SetGemma4HiddenCaptureForTest(false)
	tok := twoGeomPrompt[0]
	if _, err := mcpu.ForwardForTest(tok, mcpu.NewCache(len(twoGeomPrompt))); err != nil {
		t.Fatalf("cpu forward: %v", err)
	}
	cpuHidden := decoder.Gemma4HiddenCaptureForTest() // [post-embed, after-L0, after-L1, ...]
	if len(cpuHidden) < 2 {
		t.Fatalf("expected >=2 captured hidden states (post-embed + >=1 layer), got %d", len(cpuHidden))
	}
	emb := mg.EmbedResidentForTest(tok)
	nLayers := len(cpuHidden) - 1
	worst, worstLayer := 1.0, -1
	for L := 1; L <= nLayers; L++ {
		metalL := r.forwardTrunkForTest(emb, 0, L)
		c, m := cosMaxAbs(cpuHidden[L], metalL)
		t.Logf("after layer %d: cosine %.6f maxAbs %.4e", L-1, c, m)
		if c < worst {
			worst, worstLayer = c, L-1
		}
	}
	t.Logf("gemma4 MoE localize: worst layer %d cosine %.6f over %d layers", worstLayer, worst, nLayers)
	if worst < 0.90 {
		t.Errorf("layer %d cosine %.6f < 0.90 — a crater, not quant noise; the dense‖MoE block at that layer diverges", worstLayer, worst)
	}
}

// TestGemma4MoE_residentParity is the Step-5d whole-forward gate: the parallel dense‖MoE forward,
// resident vs CPU (int4 both sides), over the fixed prompt. ARGMAX-PRIMARY with the repo's 3%
// near-tie rule — the noise-floor pre-flight established this fixture is quant-hostile (0.79 logit
// cosine int4-vs-f32), so a logit cosine cannot gate it; the argmax + near-tie shape does. Routing
// is already bit-exact (Step 5a) and the expert gelu-tanh chain matches (Step 5c), so this gate's
// job is the COMPOSITION: the three-norms-of-h, the join order, and the value-independent dispatch.
func TestGemma4MoE_residentParity(t *testing.T) {
	r, mg, mcpu := buildMoeTiny(t)
	defer r.Close()
	defer mg.Close()
	defer mcpu.Close()

	cache := mcpu.NewCache(len(twoGeomPrompt))
	minCos, maxMaxAbs := 1.0, 0.0
	exact, gaps3 := 0, 0
	worstTie := 0.0
	for i, tok := range twoGeomPrompt {
		cpuL, err := mcpu.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu pos %d: %v", i, err)
		}
		gpuL := r.ForwardEmb(mg.EmbedResidentForTest(tok), i)
		c, m := cosMaxAbs(cpuL, gpuL)
		if c < minCos {
			minCos = c
		}
		if m > maxMaxAbs {
			maxMaxAbs = m
		}
		ca, ga := argmaxF(cpuL), argmaxF(gpuL)
		if ca == ga {
			exact++
		} else {
			top := math.Abs(float64(cpuL[ca]))
			gap := (float64(cpuL[ca]) - float64(cpuL[ga])) / (top + 1e-30)
			if gap > worstTie {
				worstTie = gap
			}
			if gap > 0.03 {
				gaps3++
			}
		}
		t.Logf("  pos %2d cosine %.6f maxAbs %.4e argmax cpu=%d metal=%d", i, c, m, ca, ga)
	}
	t.Logf("gemma4 dense‖MoE resident parity: exact-argmax %d/%d gaps>3%%=%d worstNearTie=%.2f%% minCosine=%.6f maxAbs=%.4e",
		exact, len(twoGeomPrompt), gaps3, worstTie*100, minCos, maxMaxAbs)
	// PRIMARY gate: argmax + 3% near-tie (§B2's "9/10 exact argmax, 0 hard fails" shape).
	if gaps3 > 0 {
		t.Errorf("%d position(s) diverge by >3%% (real argmax divergence, not a near-tie) — the dense‖MoE composition is broken", gaps3)
	}
	// SECONDARY crater backstop, deliberately LOOSE: this toy's int4 logit cosine floors ~0.79 even
	// CPU-vs-CPU (noise-floor pre-flight), so 0.60 catches a crater without encoding the quant floor
	// as a quality bar. Correctness rests on the argmax gate + Step-5a router idx parity + Step-5c
	// expert chain + the localization above.
	if minCos < 0.60 {
		t.Errorf("minCosine %.6f < 0.60 — a crater, not quant noise; the dense‖MoE forward is broken", minCos)
	}
}
