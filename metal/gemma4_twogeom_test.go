//go:build darwin

package metal

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// twoGeomPrompt is pin_gemma4_dense_twogeom's fixed prompt (the SAME ids the CUDA gate uses, so
// the two backends' results are directly comparable against the byte-identical reference fixture).
var twoGeomPrompt = []int{1, 7, 42, 100, 5, 200, 13, 88}

const twoGeomDir = "../testdata/gemma4-dense-twogeom-tiny"

func cosMaxAbs(a, b []float32) (cos, maxAbs float64) {
	var dot, na, nb float64
	for i := range a {
		if i >= len(b) {
			break
		}
		if d := math.Abs(float64(a[i]) - float64(b[i])); d > maxAbs {
			maxAbs = d
		}
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30), maxAbs
}

// argmaxF is defined in model_test.go (same package).

// buildTwoGeom loads the fixture int4 (both sides, as CUDA does) and builds the Metal resident
// directly. env on so the arch is bridge-eligible; BuildResident itself does not gate on the env
// (admission is covered by TestGemma4Admission_envGated), so a direct build is the forward-numerics
// vehicle here. Returns the resident, the resident-embedding model, and the CPU model.
func buildTwoGeom(t *testing.T) (*Resident, *decoder.Model, *decoder.Model) {
	t.Helper()
	if _, err := os.Stat(twoGeomDir); err != nil {
		t.Skipf("no fixture (%s) — scp testdata/gemma4-dense-twogeom-tiny from the box", twoGeomDir)
	}
	t.Setenv("GOINFER_GEMMA4_RESIDENT", "1")
	mg, err := decoder.Load(twoGeomDir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (resident side): %v", err)
	}
	r, err := BuildResident(mg)
	if err != nil {
		t.Fatalf("BuildResident: %v — metal declined the dense two-geometry fixture", err)
	}
	mcpu, err := decoder.Load(twoGeomDir, decoder.Options{Quant: "int4"})
	if err != nil {
		t.Fatalf("load (cpu side): %v", err)
	}
	return r, mg, mcpu
}

// TestGemma4TwoGeom_localize is the per-layer localization wired BEFORE the gate (the single thing
// that turned the CUDA divergence into a one-run answer). The fixture's two layers are the two
// geometries: layer 0 is LOCAL (head_dim=16, real V) — the per-layer geometry seam; layer 1 is
// GLOBAL (head_dim=512, K=V: V=v_norm(raw k)) — the Step-3 K=V forward. It diffs the resident
// residual stream after each layer against the CPU's (captured via g4traceHidden) at pos 0, so a
// whole-forward miss is immediately attributable: L0 red = geometry seam, L1 red (L0 clean) = K=V.
func TestGemma4TwoGeom_localize(t *testing.T) {
	r, mg, mcpu := buildTwoGeom(t)
	defer r.Close()
	defer mg.Close()
	defer mcpu.Close()

	decoder.SetGemma4HiddenCaptureForTest(true)
	defer decoder.SetGemma4HiddenCaptureForTest(false)
	tok := twoGeomPrompt[0]
	if _, err := mcpu.ForwardForTest(tok, mcpu.NewCache(8)); err != nil {
		t.Fatalf("cpu forward: %v", err)
	}
	cpuHidden := decoder.Gemma4HiddenCaptureForTest() // [post-embed, after-L0, after-L1]
	if len(cpuHidden) < 3 {
		t.Fatalf("expected >=3 captured hidden states (post-embed + 2 layers), got %d", len(cpuHidden))
	}
	emb := mg.EmbedResidentForTest(tok)
	// Resident residual stream after N layers (forwardTrunkForTest runs encodeTrunkInto with nL=N;
	// r.x holds the pre-final-norm residual, matching the CPU capture).
	metalL0 := r.forwardTrunkForTest(emb, 0, 1)
	metalL1 := r.forwardTrunkForTest(emb, 0, 2)

	c0, m0 := cosMaxAbs(cpuHidden[1], metalL0)
	c1, m1 := cosMaxAbs(cpuHidden[2], metalL1)
	t.Logf("layer 0 (LOCAL hd=16, real V — geometry seam): cosine %.6f maxAbs %.4e", c0, m0)
	t.Logf("layer 1 (GLOBAL hd=512, K=V — v_norm(raw k)) : cosine %.6f maxAbs %.4e", c1, m1)
	// Attribution asserts. The floor is 0.95, not ~0.99: this is int4-Metal (f16 group scales) vs
	// int4-CPU (f32 scales), whose per-layer cosine floors at ~0.98 even with an identical forward
	// (layer 0 here is 0.988 with NO K=V, the clean quant baseline). A REAL seam/K=V break — a
	// misthreaded geometry, or the 2× v_norm trap — craters a layer far below 0.95, well clear of
	// the quant floor. The KEY diagnostic is that layer 1 (K=V) tracks layer 0 (no K=V): if the K=V
	// forward were wrong, L1 would sit far below L0, not ~0.007 under it. So the relative check is
	// the sharp one; the absolute 0.95 is the crater backstop.
	if c0 < 0.95 {
		t.Errorf("layer 0 cosine %.6f < 0.95 — the per-layer GEOMETRY seam (local hd=16) diverges", c0)
	}
	if c1 < 0.95 {
		t.Errorf("layer 1 cosine %.6f < 0.95 — the K=V forward (global hd=512, v_norm(raw k)) diverges", c1)
	}
	if c1 < c0-0.05 {
		t.Errorf("layer 1 (K=V) cosine %.6f is >0.05 below layer 0 (%.6f) — the K=V forward adds error the geometry/quant baseline does not; K=V is the culprit, not the seam", c1, c0)
	}
}

// TestGemma4TwoGeom_residentParity is the Step-4 dense gate: the two-geometry K=V forward, resident
// vs CPU (int4 both sides), over the fixed prompt. Held to the repo's 3% near-tie rule + the Split-A
// cosine floor (0.979), NOT Metal's looser inherited gemma3 bar (0.88): a near-tie argmax mismatch
// is allowed, a >3% divergence is not.
func TestGemma4TwoGeom_residentParity(t *testing.T) {
	r, mg, mcpu := buildTwoGeom(t)
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
			// Near-tie rule: an argmax mismatch is benign iff CPU's top-1 and the GPU-chosen index
			// are within 3% in the CPU distribution (both essentially tied); a wider gap is a real
			// divergence the gate must reject.
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
	t.Logf("two-geometry K=V resident parity: minCosine=%.6f maxAbs=%.4e exact-argmax %d/%d worstNearTie=%.2f%% gaps>3%%=%d",
		minCos, maxMaxAbs, exact, len(twoGeomPrompt), worstTie*100, gaps3)
	if minCos < 0.979 {
		t.Errorf("minCosine %.6f < 0.979 (Split-A bar) — the resident two-geometry/K=V forward diverges from CPU", minCos)
	}
	if gaps3 > 0 {
		t.Errorf("%d position(s) diverge by >3%% (real argmax divergence, not a near-tie) — fails the near-tie rule", gaps3)
	}
}
