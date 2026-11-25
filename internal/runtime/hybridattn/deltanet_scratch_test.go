package hybridattn

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// scratchNet is a random f32 Gated DeltaNet layer for p at hidden width hidden, its decay negative so the state stays bounded.
func scratchNet(p DeltaNetParams, hidden int, rng *rand.Rand) *DeltaNet {
	vec := func(n int, scale float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32((rng.Float64()*2 - 1) * scale)
		}
		return v
	}
	nk, nv, hk, hv := p.NumKeyHeads, p.NumValueHeads, p.KeyHeadDim, p.ValueHeadDim
	convDim, valueDim := 2*hk*nk+hv*nv, hv*nv
	negExpA := vec(nv, 1)
	for i := range negExpA {
		negExpA[i] = -float32(math.Abs(float64(negExpA[i]))) - 0.1
	}
	dn, _ := NewDeltaNet(p.ConvKernel, hk, hv, nk, nv, hidden,
		vec(convDim*hidden, 0.1), vec(valueDim*hidden, 0.1), vec(nv*hidden, 0.5), vec(nv*hidden, 0.5),
		vec(convDim*p.ConvKernel, 0.5), vec(nv, 0.5), negExpA, vec(hv, 1), vec(hidden*valueDim, 0.1))
	dn.Params.NegEigval, dn.Params.ONormEps = p.NegEigval, p.ONormEps
	return dn
}

func bitsEqual(t *testing.T, what string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", what, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d] = %g (%#x), want %g (%#x)", what, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

// TestDeltaScratch_bitIdenticalToFreshBuffers: stepping a layer through one warm scratch (DeltaNetStep, and a DeltaScratch
// driven directly) returns the same bits and leaves the same matrix state and conv window, row for row and by identity, as
// GatedDeltaNetStep with fresh buffers every step. The shapes reach the stack buffers (keyDim at most 512, a value head at most
// 256), the scratch past both (keyDim 2048; a value head of 300), grouped value heads, the NegEigval gate, the ONormEps
// override, and a one-tap conv with no window. Each step's input is new, so a buffer read before it is rewritten carries the
// previous step's values and shows.
func TestDeltaScratch_bitIdenticalToFreshBuffers(t *testing.T) {
	const hidden, eps = 24, 1e-6
	for _, p := range []DeltaNetParams{
		{ConvKernel: 4, NumKeyHeads: 2, NumValueHeads: 4, KeyHeadDim: 3, ValueHeadDim: 5},
		{ConvKernel: 4, NumKeyHeads: 16, NumValueHeads: 32, KeyHeadDim: 128, ValueHeadDim: 8, NegEigval: true},
		{ConvKernel: 3, NumKeyHeads: 3, NumValueHeads: 3, KeyHeadDim: 200, ValueHeadDim: 300, ONormEps: 1e-5},
		{ConvKernel: 1, NumKeyHeads: 1, NumValueHeads: 2, KeyHeadDim: 4, ValueHeadDim: 6},
	} {
		t.Run(fmt.Sprintf("k%d_nk%d_nv%d_hk%d_hv%d", p.ConvKernel, p.NumKeyHeads, p.NumValueHeads, p.KeyHeadDim, p.ValueHeadDim), func(t *testing.T) {
			rng := rand.New(rand.NewPCG(21, uint64(p.KeyHeadDim*1000+p.ValueHeadDim)))
			dn := scratchNet(p, hidden, rng)
			w := &dn.Weights
			fresh, viaState, viaScratch := NewDeltaState(p), NewDeltaState(p), NewDeltaState(p)
			var sc DeltaScratch
			for step := range 8 {
				h := make([]float32, hidden)
				for i := range h {
					h[i] = float32(rng.Float64()*2 - 1)
				}
				want := GatedDeltaNetStep(nil, slices.Clone(h), &w.InProjQKV, &w.InProjZ, &w.OutProj,
					w.InProjB, w.InProjA, w.ConvW, w.DtBias, w.NegExpA, w.NormW, dn.Params, hidden, eps, fresh.S, &fresh.ConvWin, nil)
				gotState := DeltaNetStep(slices.Clone(h), dn, hidden, eps, viaState)
				gotScratch := sc.Step(nil, slices.Clone(h), &w.InProjQKV, &w.InProjZ, &w.OutProj,
					w.InProjB, w.InProjA, w.ConvW, w.DtBias, w.NegExpA, w.NormW, dn.Params, hidden, eps, viaScratch.S, &viaScratch.ConvWin, nil)
				for _, got := range []struct {
					arm string
					out []float32
					st  *DeltaState
				}{{"DeltaNetStep", gotState, viaState}, {"DeltaScratch.Step", gotScratch, viaScratch}} {
					bitsEqual(t, fmt.Sprintf("%s step %d output", got.arm, step), got.out, want)
					bitsEqual(t, fmt.Sprintf("%s step %d state", got.arm, step), got.st.S, fresh.S)
					if len(got.st.ConvWin) != len(fresh.ConvWin) {
						t.Fatalf("%s step %d: window holds %d rows, fresh buffers %d", got.arm, step, len(got.st.ConvWin), len(fresh.ConvWin))
					}
					for i := range fresh.ConvWin {
						bitsEqual(t, fmt.Sprintf("%s step %d window row %d", got.arm, step, i), got.st.ConvWin[i], fresh.ConvWin[i])
					}
				}
			}
		})
	}
}

// TestDeltaNetCore_windowKeepsTheLastMixedRows: once full, the conv window is the last K-1 mixed vectors DeltaNetCore was
// given, oldest first and by identity (the core retains mixed and copies nothing).
func TestDeltaNetCore_windowKeepsTheLastMixedRows(t *testing.T) {
	p := DeltaNetParams{ConvKernel: 4, NumKeyHeads: 2, NumValueHeads: 2, KeyHeadDim: 4, ValueHeadDim: 4}
	rng := rand.New(rand.NewPCG(5, 9))
	dn := scratchNet(p, 8, rng)
	w := &dn.Weights
	st := NewDeltaState(p)
	convDim, nv, valueDim := 2*p.KeyHeadDim*p.NumKeyHeads+p.ValueHeadDim*p.NumValueHeads, p.NumValueHeads, p.ValueHeadDim*p.NumValueHeads
	var given [][]float32
	var sc DeltaScratch
	for step := range 9 {
		mixed := make([]float32, convDim)
		for i := range mixed {
			mixed[i] = float32(rng.Float64()*2 - 1)
		}
		given = append(given, mixed)
		bt, at, z, core := make([]float32, nv), make([]float32, nv), make([]float32, valueDim), make([]float32, valueDim)
		sc.Core(core, mixed, bt, at, z, w.ConvW, w.DtBias, w.NegExpA, w.NormW, p, 1e-6, st.S, &st.ConvWin, nil)
		want := given[max(0, len(given)-(p.ConvKernel-1)):]
		if len(st.ConvWin) != len(want) {
			t.Fatalf("step %d: window holds %d rows, want %d", step, len(st.ConvWin), len(want))
		}
		for i := range want {
			if &st.ConvWin[i][0] != &want[i][0] {
				t.Fatalf("step %d: window row %d is not mixed vector %d", step, i, len(given)-len(want)+i)
			}
		}
	}
}

// TestDeltaScratch_noAllocsOnceWarm: at keyDim 2048 (past the stack buffers), once the conv window is full a DeltaScratch core
// step allocates nothing, and a DeltaNetStep allocates only the two vectors that outlive the step (in_proj_qkv's output, which
// the conv window keeps, and out_proj's output, which is returned) plus whatever linalg.MatmulBT allocates inside each of its
// five M=1 projections, measured here rather than assumed.
func TestDeltaScratch_noAllocsOnceWarm(t *testing.T) {
	const hidden, eps = 32, 1e-6
	p := DeltaNetParams{ConvKernel: 4, NumKeyHeads: 16, NumValueHeads: 32, KeyHeadDim: 128, ValueHeadDim: 128}
	rng := rand.New(rand.NewPCG(3, 4))
	dn := scratchNet(p, hidden, rng)
	w := &dn.Weights
	h := make([]float32, hidden)
	for i := range h {
		h[i] = float32(rng.Float64()*2 - 1)
	}

	st := NewDeltaState(p)
	for range p.ConvKernel {
		DeltaNetStep(h, dn, hidden, eps, st)
	}
	y := make([]float32, p.NumValueHeads)
	perMatmul := testing.AllocsPerRun(10, func() { linalg.MatmulBT(h, w.InProjB, y, 1, hidden, p.NumValueHeads) })
	if allocs, want := testing.AllocsPerRun(10, func() { DeltaNetStep(h, dn, hidden, eps, st) }), 2+5*perMatmul; allocs != want {
		t.Errorf("a warm DeltaNetStep allocates %v times, want %v (the window's mixed row, the returned output, and %v in each of "+
			"five MatmulBT calls)", allocs, want, perMatmul)
	}

	nv, valueDim := p.NumValueHeads, p.ValueHeadDim*p.NumValueHeads
	convDim := 2*p.KeyHeadDim*p.NumKeyHeads + valueDim
	mixed := make([][]float32, p.ConvKernel) // the window retains mixed, so steps cycle through K distinct rows
	for i := range mixed {
		mixed[i] = make([]float32, convDim)
		for j := range mixed[i] {
			mixed[i][j] = float32(rng.Float64()*2 - 1)
		}
	}
	bt, at, z, core := make([]float32, nv), make([]float32, nv), make([]float32, valueDim), make([]float32, valueDim)
	cst := NewDeltaState(p)
	var sc DeltaScratch
	n := 0
	coreStep := func() {
		sc.Core(core, mixed[n%len(mixed)], bt, at, z, w.ConvW, w.DtBias, w.NegExpA, w.NormW, p, eps, cst.S, &cst.ConvWin, nil)
		n++
	}
	for range p.ConvKernel {
		coreStep()
	}
	if allocs := testing.AllocsPerRun(20, coreStep); allocs != 0 {
		t.Errorf("a warm DeltaScratch core step allocates %v times, want 0", allocs)
	}
}
