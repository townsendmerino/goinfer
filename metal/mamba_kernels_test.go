//go:build darwin && goinfer_testhooks

package metal

import (
	"math"
	"math/rand"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/metal/kernels"
)

// mambaFixture is one tiny SSM fixture's first Mamba-2 layer, its geometry, and its real per-token mixer inputs and outputs,
// captured from the CPU's own decoder.mamba2Step (decoder.SetMambaCapHook) over a prompt long enough to cross the conv window
// many times.
type mambaFixture struct {
	name                                  string
	nHeads, headDim, dState, nGroups, K   int
	normGroups                            int
	eps                                   float32
	convW, convB, headP, normW            []float32
	proj, gated                           [][]float32 // per step: the in_proj output [z | xBC | dt], and the gated-norm output
	dInner, gSize, convDim, repeat, steps int
}

const mambaGateSteps = 72 // at least 64 steps past the conv window (K-1 ≤ 3)

func loadMambaFixture(t *testing.T, dir string) mambaFixture {
	t.Helper()
	m, err := decoder.Load(filepath.Join("..", "testdata", dir), decoder.Options{Backend: "cpu"})
	if err != nil {
		t.Fatalf("load %s: %v", dir, err)
	}
	defer m.Close()
	f := mambaFixture{name: dir, eps: m.NormEps()}
	_, nLayers, _, _, _, _, vocab := m.Dims()
	isMamba := func(i int) bool { return m.GraniteMambaLayer(i) }
	if nh, hd, ds, ng, k, normG, _, ok := m.NemotronResidentParams(); ok {
		f.nHeads, f.headDim, f.dState, f.nGroups, f.K, f.normGroups = nh, hd, ds, ng, k, normG
		isMamba = func(i int) bool { return m.NemotronBlockKind(i) == 0 }
	} else if nh, hd, ds, ng, k, _, _, _, _, ok := m.GraniteResidentParams(); ok {
		f.nHeads, f.headDim, f.dState, f.nGroups, f.K, f.normGroups = nh, hd, ds, ng, k, 1
	} else {
		t.Fatalf("%s is neither Nemotron-H nor Granite-4.0-H", dir)
	}
	first, nMamba := -1, 0
	for i := range nLayers {
		if isMamba(i) {
			if first < 0 {
				first = i
			}
			nMamba++
		}
	}
	if first < 0 {
		t.Fatalf("%s has no Mamba layer", dir)
	}
	_, convW, convB, aLog, dW, dtBias, normW, _ := m.GraniteMambaWeights(first)
	// Copied: the slices alias the model's weights, which Close releases (the deferred Close above runs before the replay).
	f.convW, f.convB, f.normW = slices.Clone(convW), slices.Clone(convB), slices.Clone(normW)
	f.headP = make([]float32, f.nHeads*3)
	for h := range f.nHeads {
		f.headP[h*3] = float32(-math.Exp(float64(aLog[h])))
		f.headP[h*3+1], f.headP[h*3+2] = dtBias[h], dW[h]
	}
	f.dInner, f.gSize = f.nHeads*f.headDim, f.nGroups*f.dState
	f.convDim, f.repeat = f.dInner+2*f.gSize, f.nHeads/f.nGroups

	cnt := 0
	decoder.SetMambaCapHook(func(proj, gated []float32) {
		if cnt%nMamba == 0 { // the first Mamba layer of each token
			f.proj = append(f.proj, append([]float32(nil), proj...))
			f.gated = append(f.gated, append([]float32(nil), gated...))
		}
		cnt++
	})
	defer decoder.SetMambaCapHook(nil)
	rng := rand.New(rand.NewSource(5))
	prompt := make([]int, mambaGateSteps)
	for i := range prompt {
		prompt[i] = rng.Intn(vocab)
	}
	cache := m.NewCache(0)
	for _, tok := range prompt {
		if _, err := m.ForwardForTest(tok, cache); err != nil {
			t.Fatalf("%s forward: %v", dir, err)
		}
	}
	f.steps = len(f.proj)
	if f.steps < mambaGateSteps {
		t.Fatalf("%s: captured %d steps, want %d: the gate would not cross the conv window often enough", dir, f.steps, mambaGateSteps)
	}
	return f
}

// mambaChainWorst replays a fixture's captured inputs through the Metal conv → SSM → gated-norm chain (compiled from src,
// dispatched with normGroups groups), from a fresh state, and returns the worst per-step cosine and max |diff| of the gated
// output against the CPU's.
func mambaChainWorst(t *testing.T, f mambaFixture, src string, normGroups int) (worstCos, worstAbs float64) {
	t.Helper()
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(src, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe := func(name string) Pipeline {
		p, e := d.NewComputePipeline(lib, name)
		if e != nil {
			t.Fatalf("pipeline %s: %v", name, e)
		}
		return p
	}
	pConv, pSSM, pNorm := pipe("mamba_conv"), pipe("mamba_ssm"), pipe("mamba_gnorm")
	dProj := d.NewBufferLen(len(f.proj[0]))
	dConvW, dConvB := NewBufferFloats(d, f.convW), NewBufferFloats(d, f.convB)
	dHeadP, dNormW := NewBufferFloats(d, f.headP), NewBufferFloats(d, f.normW)
	dWin := NewBufferFloats(d, make([]float32, (f.K-1)*f.convDim))
	dState := NewBufferFloats(d, make([]float32, f.nHeads*f.headDim*f.dState))
	dConv, dY, dGated := d.NewBufferLen(f.convDim), d.NewBufferLen(f.dInner), d.NewBufferLen(f.dInner)
	u := func(v int) Buffer { return NewBufferU32(d, uint32(v)) }
	uEps := NewBufferFloats(d, []float32{f.eps})
	q := d.NewCommandQueue()
	worstCos = 1
	for s := range f.steps {
		uploadF32(dProj, f.proj[s])
		e := q.Begin()
		e.Dispatch(pConv, f.convDim, 64, dProj, dConvW, dConvB, dWin, dConv, u(f.convDim), u(f.K), u(f.dInner))
		e.Dispatch(pSSM, f.nHeads*f.headDim, 64, dConv, dProj, dHeadP, dState, dY,
			u(f.nHeads), u(f.headDim), u(f.dState), u(f.repeat), u(f.gSize), u(f.dInner), u(f.dInner+f.convDim))
		e.Dispatch(pNorm, normGroups*64, 64, dY, dProj, dNormW, dGated, u(f.dInner/normGroups), uEps, u(0))
		e.End()
		cos, maxAbs := mambaCos(f.gated[s], dGated.Floats())
		worstCos, worstAbs = min(worstCos, cos), max(worstAbs, maxAbs)
	}
	return worstCos, worstAbs
}

// mambaCos is the cosine and max |diff| of a against b, with an all-zero vector against a nonzero one scored 0, not 1: a chain
// that writes nothing must not read as a match.
func mambaCos(a, b []float32) (cos, maxAbs float64) {
	cos, maxAbs = cosF32(a, b)
	var na, nb float64
	for i := range a {
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if (na == 0) != (nb == 0) {
		cos = 0
	}
	return cos, maxAbs
}

// TestMambaKernels_cpuParity is OP2b's K gate on Metal (docs/tasks/task-forward-architecture-2026-10.md §4.7, "OP2b on Metal"):
// the conv, SSM and gated-norm kernels, chained as the resident runs them, against decoder.mamba2Step on the tiny Nemotron-H
// (normalized per group) and Granite-4.0-H (over the full width) fixtures' real per-token inputs, over 72 steps. Bar:
// cosine ≥ 0.99999 and max |diff| ≤ 1e-3 at every step.
func TestMambaKernels_cpuParity(t *testing.T) {
	for _, dir := range []string{"nemotron-tiny", "granite-tiny"} {
		f := loadMambaFixture(t, dir)
		c, a := mambaChainWorst(t, f, kernels.AllKernels, f.normGroups)
		t.Logf("%s: %d steps, nHeads=%d headDim=%d dState=%d nGroups=%d normGroups=%d K=%d: worst cosine %.9f, max |diff| %.3g",
			dir, f.steps, f.nHeads, f.headDim, f.dState, f.nGroups, f.normGroups, f.K, c, a)
		if c < 0.99999 || a > 1e-3 {
			t.Errorf("%s: the Metal Mamba-2 chain drifts from the CPU: worst cosine %.9f, max |diff| %.3g", dir, c, a)
		}
	}
}

// TestMambaKernels_plantsAreRed is OP2b's D gate for the conv and SSM plants: each must turn K red, which is K failing on at least
// one fixture. A plant a fixture's regime cannot show is logged for that fixture: granite-tiny's dt + dt_bias sits where softplus
// is nearly the identity, so dropping it moves its output by about 4e-5 there (CLAUDE.md, "a minimal repro can be minimal in
// exactly the dimension that hides the bug").
func TestMambaKernels_plantsAreRed(t *testing.T) {
	plants := []struct {
		name, from, to string
	}{
		{"conv taps shifted by one", "s = s + convW[c * K + j] * wv;", "s = s + convW[c * K + j + 1u] * wv;"},
		{"softplus on dt dropped", "float dth = mb_softplus(dtRaw[dtBase + head] + dtb);", "float dth = (dtRaw[dtBase + head] + dtb);"},
	}
	fixtures := []mambaFixture{loadMambaFixture(t, "nemotron-tiny"), loadMambaFixture(t, "granite-tiny")}
	for _, p := range plants {
		if strings.Count(kernels.AllKernels, p.from) != 1 {
			t.Fatalf("plant %q: the library must contain %q exactly once", p.name, p.from)
		}
		red := false
		for _, f := range fixtures {
			c, a := mambaChainWorst(t, f, strings.Replace(kernels.AllKernels, p.from, p.to, 1), f.normGroups)
			shows := !(c >= 0.99999 && a <= 1e-3)
			t.Logf("%s, %s: worst cosine %.6f, max |diff| %.3g, red %v", f.name, p.name, c, a, shows)
			red = red || shows
		}
		if !red {
			t.Errorf("the plant %q stayed inside the bar on every fixture", p.name)
		}
	}
}

// uploadF32 and cosF32 are copies of the test helpers in metal/kernels/deltanet_test.go, which moved there (SB1 metal batch 6).
func uploadF32(dst Buffer, src []float32) {
	copy(dst.Floats(), src)
}

func cosF32(a, b []float32) (cos, maxAbs float64) {
	var dot, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		na += x * x
		nb += y * y
		if d := math.Abs(x - y); d > maxAbs {
			maxAbs = d
		}
	}
	if na == 0 || nb == 0 {
		return 1, maxAbs
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb)), maxAbs
}
