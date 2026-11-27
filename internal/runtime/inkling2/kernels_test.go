package inkling2

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// opsGolden is testdata/ops_golden.json, written by scripts/inkling2/pin_ops.py from transformers' own Inkling2Loop modules.
type opsGolden struct {
	Clip struct {
		X, Scale, Y []float32
		D           int
		Thr         float64
	}
	Logscale struct {
		Floor int
		Alpha float64
		Y     []float32
	}
	Mix struct {
		Queries, States, Y []float32
		Nq                 int
	}
	Control struct {
		X, M, Down, Gain, Forget, Y []float32
		Rank                        int
	}
	Conv struct {
		X, Taps, Y []float32
		K, N       int
	}
	Dynconv struct {
		X, Taps, Dyn, Y []float32
		K, N            int
	}
	Situ struct {
		Gate, Up, Y []float32
		Bg, Bu      float64
	}
	Attend struct {
		Heads, Kv, Hd, Drel, N, Window, Extent int
		Scale                                  float64
		Q, K, V, Rel, Relbias, Sinks, Y        []float32
	}
}

func readOps(t *testing.T) *opsGolden {
	t.Helper()
	b, err := os.ReadFile("testdata/ops_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g opsGolden
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	return &g
}

func near(t *testing.T, what string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) || len(want) == 0 {
		t.Fatalf("%s: %d values, want %d", what, len(got), len(want))
	}
	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); d > tol*(1+math.Abs(float64(want[i]))) {
			t.Fatalf("%s[%d] = %v, want %v (|diff| %g)", what, i, got[i], want[i], d)
		}
	}
}

func TestClip_matchesTransformers(t *testing.T) {
	g := readOps(t).Clip
	x := append([]float32(nil), g.X...)
	Clip(x, g.Scale, g.D, g.Thr)
	near(t, "clip", x, g.Y, 1e-6)
	clipped := 0
	for r := 0; r < len(x); r += g.D {
		var s float64
		for _, v := range x[r : r+g.D] {
			s += float64(v) * float64(v)
		}
		if math.Abs(math.Sqrt(s)-g.Thr) < 1e-4 {
			clipped++
		}
	}
	if clipped == 0 || clipped == len(x)/g.D {
		t.Fatalf("%d of %d rows at the threshold: the fixture must hold both a clipped and an unclipped row", clipped, len(x)/g.D)
	}
}

func TestLogScale_matchesTransformers(t *testing.T) {
	g := readOps(t).Logscale
	got := make([]float32, len(g.Y))
	for p := range got {
		got[p] = float32(LogScale(p, g.Floor, g.Alpha))
	}
	near(t, "logscale", got, g.Y, 1e-6)
	if got[0] != 1 || got[len(got)-1] <= 1 {
		t.Fatalf("log scale %v: want 1 below the floor and > 1 past it", got)
	}
}

func TestMix_matchesTransformers(t *testing.T) {
	g := readOps(t).Mix
	hidden := len(g.Queries) / g.Nq
	var normed [][]float32
	for r := 0; r < len(g.States); r += hidden {
		n := make([]float32, hidden)
		UnitRMS(n, g.States[r:r+hidden], 1e-6)
		normed = append(normed, n)
	}
	out := make([]float32, g.Nq*hidden)
	Mix(out, g.Queries, g.Nq, normed, hidden)
	near(t, "mix", out, g.Y, 1e-5)
}

func TestControl_matchesTransformers(t *testing.T) {
	g := readOps(t).Control
	x := append([]float32(nil), g.X...)
	Control(x, g.M, g.Down, g.Gain, g.Forget, g.Rank, 1e-6)
	near(t, "control", x, g.Y, 1e-5)
}

// TestConv_matchesTransformers runs six positions so the window fills past the kernel width: one position would zero every
// tap but the current one.
func TestConv_matchesTransformers(t *testing.T) {
	g := readOps(t).Conv
	ch := len(g.X) / g.N
	var win [][]float32
	got := make([]float32, 0, len(g.X))
	for p := range g.N {
		x := append([]float32(nil), g.X[p*ch:(p+1)*ch]...)
		Conv(x, g.Taps, g.K, nil, &win)
		got = append(got, x...)
	}
	near(t, "conv", got, g.Y, 1e-5)
}

func TestDynConv_matchesTransformers(t *testing.T) {
	g := readOps(t).Dynconv
	ch := len(g.X) / g.N
	nd := len(g.Dyn) / g.N
	var win [][]float32
	got := make([]float32, 0, len(g.X))
	for p := range g.N {
		x := append([]float32(nil), g.X[p*ch:(p+1)*ch]...)
		Conv(x, g.Taps, g.K, g.Dyn[p*nd:(p+1)*nd], &win)
		got = append(got, x...)
	}
	near(t, "dynconv", got, g.Y, 1e-5)
}

func TestSitu_matchesTransformers(t *testing.T) {
	g := readOps(t).Situ
	gate := append([]float32(nil), g.Gate...)
	Situ(gate, g.Up, g.Bg, g.Bu)
	near(t, "situ", gate, g.Y, 1e-5)
}

// TestAttend_matchesTransformers covers seven positions against a window of 4 and a bias extent of 5, so some keys are
// outside the window and some inside it are past the extent.
func TestAttend_matchesTransformers(t *testing.T) {
	g := readOps(t).Attend
	p := AttendParams{Heads: g.Heads, KVHeads: g.Kv, HeadDim: g.Hd, DRel: g.Drel, Extent: g.Extent, Window: g.Window, Scale: g.Scale}
	qd, kd := g.Heads*g.Hd, g.Kv*g.Hd
	got := make([]float32, 0, len(g.Y))
	for pos := range g.N {
		ctx := make([]float32, qd)
		Attend(p, ctx, g.Q[pos*qd:(pos+1)*qd], g.K, g.V, kd, pos+1, 0, pos,
			g.Rel[pos*g.Heads*g.Drel:(pos+1)*g.Heads*g.Drel], g.Relbias, g.Sinks[pos*g.Heads:(pos+1)*g.Heads])
		got = append(got, ctx...)
	}
	near(t, "attend", got, g.Y, 1e-5)
}
