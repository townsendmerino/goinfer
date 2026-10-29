//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// OP2b on CUDA, gate K (docs/tasks/task-forward-architecture-2026-10.md §4.7, "OP2b on CUDA"): the Mamba-2 kernels against the CPU's own computation
// (decoder/mamba2.go:mamba2Step) on the tiny fixtures' real per-token inputs, the squared-ReLU quantize against the CPU's rule. Every reading asserts both
// outputs are non-zero before it takes a cosine (mambaCosNonZero), because CUDA's cosF32 scores 1 for an all-zero side.

var (
	adMamba    = adMod{"mamba", mambaPTX}
	adNongated = adMod{"nongated_act", nongatedActPTX}
)

// mambaFixture is one tiny SSM fixture's first Mamba-2 layer, its geometry, and its real per-token mixer inputs and outputs, captured from the CPU's own
// mamba2Step (decoder.SetMambaCapHook) over a prompt long enough to cross the conv window many times.
type mambaFixture struct {
	name                                  string
	nHeads, headDim, dState, nGroups, K   int
	normGroups                            int
	eps                                   float32
	convW, convB, headP, normW            []float32
	proj, gated                           [][]float32 // per step: the in_proj output [z | xBC | dt], and the gated-norm output
	dInner, gSize, convDim, repeat, steps int
}

const mambaGateSteps = 72 // at least 64 steps past the conv window (K-1 <= 3)

func loadMambaFixture(t *testing.T, dir string) mambaFixture {
	t.Helper()
	path := filepath.Join("..", "testdata", dir)
	if _, err := os.Stat(filepath.Join(path, "config.json")); err != nil {
		t.Skipf("no fixture at %s (a gitignored local fixture: copy it from the main checkout, scripts/worktree_fixtures.sh)", path)
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "cpu"})
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

// allZero reports whether every element of x is exactly zero.
func allZero(x []float32) bool {
	for _, v := range x {
		if v != 0 {
			return false
		}
	}
	return true
}

// mambaCosNonZero is the cosine and max |diff| of a against b, after asserting that neither is all zeros. A kernel that writes nothing, or a replay of
// a model closed under it, must fail the reading by name and not agree with a zero reference (cosF32 returns 1 for that case).
func mambaCosNonZero(t *testing.T, what string, a, b []float32) (cos, maxAbs float64) {
	t.Helper()
	if allZero(a) {
		t.Fatalf("%s: the reference output is all zeros: the reading is void", what)
	}
	if allZero(b) {
		t.Fatalf("%s: the kernel's output is all zeros: the reading is void", what)
	}
	return cosF32(a, b)
}

// mambaChainWorst replays a fixture's captured inputs through the CUDA conv -> SSM -> gated-norm chain from a fresh state and returns the worst per-step
// cosine and max |diff| of the gated output against the CPU's.
func mambaChainWorst(t *testing.T, r *adRig, f mambaFixture, normGroups int) (worstCos, worstAbs float64) {
	t.Helper()
	dConvW, dConvB, dHeadP, dNormW := r.f32(f.convW), r.f32(f.convB), r.f32(f.headP), r.f32(f.normW)
	dWin := r.zeros((f.K - 1) * f.convDim)
	dState := r.zeros(f.nHeads * f.headDim * f.dState)
	dConv, dY, dGated := r.zeros(f.convDim), r.zeros(f.dInner), r.zeros(f.dInner)
	worstCos = 1
	for s := range f.steps {
		dProj := r.f32(f.proj[s])
		r.launch(adMamba, "mamba_conv", adCfg((f.convDim+127)/128, 1, 1, 128, 0), gc.Arg(dProj), gc.Arg(dConvW), gc.Arg(dConvB), gc.Arg(dWin), gc.Arg(dConv),
			adI(f.convDim), adI(f.K), adI(f.dInner))
		r.launch(adMamba, "mamba_ssm", adCfg((f.nHeads*f.headDim+127)/128, 1, 1, 128, 0), gc.Arg(dConv), gc.Arg(dProj), gc.Arg(dHeadP), gc.Arg(dState), gc.Arg(dY),
			adI(f.nHeads), adI(f.headDim), adI(f.dState), adI(f.repeat), adI(f.gSize), adI(f.dInner), adI(f.dInner+f.convDim))
		r.launch(adMamba, "mamba_gnorm", adCfg(normGroups, 1, 1, 256, 0), gc.Arg(dY), gc.Arg(dProj), gc.Arg(dNormW), gc.Arg(dGated),
			adI(f.dInner/normGroups), gc.ArgValue(f.eps), adI(0))
		cos, maxAbs := mambaCosNonZero(t, f.name+" step "+itoa(s), f.gated[s], r.read(dGated, f.dInner))
		worstCos, worstAbs = min(worstCos, cos), max(worstAbs, maxAbs)
	}
	return worstCos, worstAbs
}

// TestMambaKernels_cpuParity is K for the chain: the conv, SSM and gated-norm kernels, chained as the resident will run them, against mamba2Step on the tiny
// Nemotron-H (normalized per group) and Granite-4.0-H (over the full width) fixtures' real per-token inputs, 72 steps, from a fresh state. Bar (gpu/mamba_test.go's):
// cosine >= 0.99999 and max |diff| <= 1e-3 at every step.
func TestMambaKernels_cpuParity(t *testing.T) {
	r, _ := newAdRig(t)
	for _, dir := range []string{"nemotron-tiny", "granite-tiny"} {
		f := loadMambaFixture(t, dir)
		c, a := mambaChainWorst(t, r, f, f.normGroups)
		r.release()
		t.Logf("%s: %d steps, nHeads=%d headDim=%d dState=%d nGroups=%d normGroups=%d K=%d: worst cosine %.9f, max |diff| %.3g",
			dir, f.steps, f.nHeads, f.headDim, f.dState, f.nGroups, f.normGroups, f.K, c, a)
		if !(c >= 0.99999) || !(a <= 1e-3) {
			t.Errorf("%s: the CUDA Mamba-2 chain drifts from the CPU: worst cosine %.9f, max |diff| %.3g", dir, c, a)
		}
	}
}

// TestMambaGNorm_groups is K for the gated norm at a group count the tiny fixtures lack (both have one group): 4 groups of 32 against the CPU's grouped norm
// (decoder.GatedRMSNormGroupedForTest), and D's norm plant, the same input taken as one group over the full width, must fail it. It also covers a group size
// that is not a multiple of the block (the 4B's is 896 against a block of 256).
func TestMambaGNorm_groups(t *testing.T) {
	r, _ := newAdRig(t)
	const eps = 1e-5
	rng := rand.New(rand.NewSource(9))
	rnd := func(n int, scale float64) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64() * scale)
		}
		return s
	}
	for _, c := range []struct{ dInner, groups int }{{128, 4}, {896 * 2, 2}} {
		y, z, w := rnd(c.dInner, 1), rnd(c.dInner, 1), rnd(c.dInner, 0.5)
		size := c.dInner / c.groups
		for g := range c.groups { // groups at different scales, so a norm over the full width cannot pass for a per-group one
			for i := g * size; i < (g+1)*size; i++ {
				y[i] *= float32(1 + 3*g)
			}
		}
		want := make([]float32, c.dInner)
		for i := range want {
			want[i] = y[i] * (z[i] / (1 + float32(math.Exp(float64(-z[i])))))
		}
		decoder.GatedRMSNormGroupedForTest(want, w, c.groups, eps)
		run := func(nGroups int) []float32 {
			dY, dZ, dW, dOut := r.f32(y), r.f32(z), r.f32(w), r.zeros(c.dInner)
			r.launch(adMamba, "mamba_gnorm", adCfg(nGroups, 1, 1, 256, 0), gc.Arg(dY), gc.Arg(dZ), gc.Arg(dW), gc.Arg(dOut), adI(c.dInner/nGroups), gc.ArgValue(float32(eps)), adI(0))
			return r.read(dOut, c.dInner)
		}
		cos, a := mambaCosNonZero(t, "gated norm", want, run(c.groups))
		t.Logf("gated norm, %d groups of %d: cosine %.9f, max |diff| %.3g", c.groups, size, cos, a)
		if !(cos >= 0.99999) || !(a <= 1e-3) {
			t.Errorf("the gated norm at %d groups drifts from the CPU: cosine %.9f, max |diff| %.3g", c.groups, cos, a)
		}
		cos, a = mambaCosNonZero(t, "gated norm plant", want, run(1))
		t.Logf("plant, one group over the full width: cosine %.6f, max |diff| %.3g", cos, a)
		if cos >= 0.99999 && a <= 1e-3 { // NaN is not inside the bar, so it reads as red
			t.Errorf("the norm taken over the full width where %d groups are asked for stayed inside the bar", c.groups)
		}
		r.release()
	}
}

// TestRelu2Quant_exact is K for relu2_quant: its packed int8 codes and scale equal, exactly, relu(u)^2 quantized by the rule glu_quant states (scale = max/127,
// codes rounded to nearest even and clamped to +-127). Exact because the activation is one f32 multiply and the quantization is the same arithmetic on both
// sides. I is a multiple of 4, as every real intermediate width is (the kernel packs four codes a word).
func TestRelu2Quant_exact(t *testing.T) {
	r, _ := newAdRig(t)
	rng := rand.New(rand.NewSource(3))
	const I = 1028
	u := make([]float32, I)
	for i := range u {
		u[i] = float32(rng.NormFloat64() * 2)
	}
	sq := make([]float32, I)
	var mx float32
	for i, v := range u {
		v = max(v, 0)
		sq[i] = v * v
		mx = max(mx, sq[i])
	}
	wantScale := mx / 127
	inv := 1 / wantScale
	want := make([]int8, I)
	for i, d := range sq {
		want[i] = int8(max(-127, min(127, int(math.RoundToEven(float64(d*inv))))))
	}
	dU, dScale, dScratch := r.f32(u), r.zeros(1), r.zeros(I)
	dQ := mustAlloc[int32](t, r.cx, I/4)
	r.free = append(r.free, func() { dQ.Close() })
	r.launch(adNongated, "relu2_quant", adCfg(1, 1, 1, 256, 256*4), gc.Arg(dU), adI(0), adI(I), gc.Arg(dQ), gc.Arg(dScale), gc.Arg(dScratch))
	packed := make([]int32, I/4)
	if err := gc.CopyDtoH(r.bg, packed, dQ); err != nil {
		t.Fatal(err)
	}
	got := make([]int8, 0, I)
	for _, w := range packed {
		for b := range 4 {
			got = append(got, int8(w>>(8*b)))
		}
	}
	nonzero := false
	for _, c := range got {
		nonzero = nonzero || c != 0
	}
	if !nonzero {
		t.Fatal("relu2_quant wrote only zero codes: the reading is void")
	}
	if s := r.read(dScale, 1)[0]; s != wantScale {
		t.Errorf("scale %v, want %v", s, wantScale)
	}
	bad := 0
	for i := range want {
		if got[i] != want[i] {
			if bad < 3 {
				t.Errorf("code %d: %d, want %d (u %v)", i, got[i], want[i], u[i])
			}
			bad++
		}
	}
	if bad > 0 {
		t.Errorf("%d of %d codes differ", bad, I)
	}
	r.release()
}

// cosF32 is the cosine and max |diff| of a against b. It returns cosine 1 when either side is all zeros, so a caller that
// must not accept a zero output checks for it first (mambaCosNonZero).
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
