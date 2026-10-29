//go:build darwin

package metal

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/townsendmerino/goinfer/internal/vision"
)

// g4vTinyTower loads the committed tiny Gemma 4 tower (finite ClippableLinear bounds) with random position tables and
// norm weights written through the export's aliasing slices: the fixture's own are all ones, under which a swapped
// table or a norm given the wrong weight is invisible.
func g4vTinyTower(t *testing.T) *vision.Gemma4Encoder {
	t.Helper()
	enc, err := vision.LoadGemma4Encoder("../testdata/gemma4-vision-tiny", false)
	if err != nil {
		t.Fatal(err)
	}
	w, err := enc.Weights()
	if err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewPCG(7, 8))
	for _, tb := range [][]float32{w.PosEmbX, w.PosEmbY} {
		for i := range tb {
			tb[i] = r.Float32()*2 - 1
		}
	}
	for _, l := range w.Layers {
		for _, nw := range [][]float32{l.InputNorm, l.PostAttnNorm, l.PreFFNNorm, l.PostFFNNorm, l.QNorm, l.KNorm} {
			for i := range nw {
				nw[i] = 0.5 + r.Float32()
			}
		}
	}
	clipped := 0
	for _, l := range w.Layers {
		for _, p := range []vision.Gemma4Proj{l.Q, l.K, l.V, l.O, l.Gate, l.Up, l.Down} {
			if p.Clipped() {
				clipped++
			}
		}
	}
	if clipped == 0 {
		t.Fatal("the tiny tower has no finite clip bound; the clamps would go unchecked")
	}
	return enc
}

// g4vCase is a patch grid of gw x gh pooled cells (each k x k patches), its patches random, its positions in raster
// order or shuffled (the tower must not depend on the order patches arrive in).
func g4vCase(enc *vision.Gemma4Encoder, gw, gh int, shuffle bool, seed uint64) ([]float32, [][2]int) {
	c := enc.Cfg
	k := c.PoolingKernelSize
	var pos [][2]int
	for y := range gh * k {
		for x := range gw * k {
			pos = append(pos, [2]int{x, y})
		}
	}
	r := rand.New(rand.NewPCG(seed, 1))
	if shuffle {
		r.Shuffle(len(pos), func(i, j int) { pos[i], pos[j] = pos[j], pos[i] })
	}
	pd := 3 * c.PatchSize * c.PatchSize
	p := make([]float32, len(pos)*pd)
	for i := range p {
		p[i] = r.Float32()
	}
	return p, pos
}

// g4vWorst is the worst per-soft-token cosine of got against want, [n, d] each.
func g4vWorst(got, want []float32, d int) float64 {
	worst := 1.0
	for r := range len(want) / d {
		var dot, na, nb float64
		for j := range d {
			x, y := float64(got[r*d+j]), float64(want[r*d+j])
			dot, na, nb = dot+x*y, na+x*x, nb+y*y
		}
		worst = math.Min(worst, dot/math.Sqrt(na*nb))
	}
	return worst
}

// TestG4VMetal_matchesCPU is Phase VM's gate VM1 (docs/tasks/task-embeddinggemma2.md): on the tiny tower, with its
// finite clip bounds and random position tables, the Metal tower followed by aikit's tail (FinishHidden) matches
// aikit's CPU Forward to cosine 0.9999 on every soft token: grids of 18, 54 and 135 patches, raster and shuffled
// order, at attention blocks of 256 (one block), 6 (divides 18 and 54) and 5 (divides none of 18 or 54). The clamps'
// share is checked by the seam that skips them: with it set, the same comparison must fail.
func TestG4VMetal_matchesCPU(t *testing.T) {
	enc := g4vTinyTower(t)
	a, err := newG4VAccel(enc)
	if err != nil {
		t.Fatal(err)
	}
	c := enc.Cfg
	t.Logf("tiny tower: hidden %d, %d layers, %d heads of %d, MLP %d, pool %d", c.HiddenSize, c.NumHiddenLayers, c.NumAttentionHeads, c.HeadDim, c.IntermediateSize, c.PoolingKernelSize)
	defer func(b int) { eg2Block = b }(eg2Block)
	type grid struct {
		gw, gh  int
		shuffle bool
	}
	grids := []grid{{2, 1, false}, {3, 2, true}, {5, 3, false}, {5, 3, true}}
	for _, B := range []int{256, 6, 5} {
		eg2Block = B
		a.cap = 0 // the scores buffer is sized by the block
		for gi, g := range grids {
			p, pos := g4vCase(enc, g.gw, g.gh, g.shuffle, uint64(10+gi))
			want, err := enc.Forward(p, pos)
			if err != nil {
				t.Fatal(err)
			}
			h, err := a.Hidden(p, pos)
			if err != nil {
				t.Fatal(err)
			}
			got, err := enc.FinishHidden(h, pos)
			if err != nil {
				t.Fatal(err)
			}
			w := g4vWorst(got, want, enc.TextHiddenSize)
			t.Logf("block %3d, %3d patches (shuffled %v): worst soft-token cosine %.9f over %d", B, len(pos), g.shuffle, w, len(want)/enc.TextHiddenSize)
			if w < 0.9999 {
				t.Errorf("block %d, %d patches: worst soft-token cosine %.9f, under 0.9999", B, len(pos), w)
			}
		}
	}
	eg2Block = 256
	a.cap = 0
	a.clampOff = true
	p, pos := g4vCase(enc, 5, 3, false, 99)
	want, _ := enc.Forward(p, pos)
	h, err := a.Hidden(p, pos)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := enc.FinishHidden(h, pos)
	w := g4vWorst(got, want, enc.TextHiddenSize)
	t.Logf("control, clamps skipped: worst soft-token cosine %.9f", w)
	if w >= 0.9999 {
		t.Errorf("with the clamps skipped the tower still matches (%.9f): the comparison does not see them", w)
	}
}

// TestG4VWalkBlock_refusesMisroutedOps holds the Gemma 4 tower's walker to the registers and forms its dispatches take:
// an op that does not fit is an error before anything is queued, never a dispatch on another buffer or at another scale.
func TestG4VWalkBlock_refusesMisroutedOps(t *testing.T) {
	const hidden, inter = 8, 32
	a := &g4vAccel{cfg: vision.Gemma4EncoderConfig{HiddenSize: hidden, IntermediateSize: inter, NumAttentionHeads: 2, HeadDim: 4}}
	hp := g4vProj{out: hidden, in: hidden}
	ly := &g4vLayer{q: hp, k: hp, v: hp, o: hp, gate: g4vProj{out: inter, in: hidden}, up: g4vProj{out: inter, in: hidden},
		down: g4vProj{out: hidden, in: inter}}
	for _, c := range []struct {
		name string
		op   vision.TowerOp
	}{
		{"a residual add from mid", vision.TowerOp{Kind: vision.TowerAdd, In: vision.TowerMid, Out: vision.TowerX}},
		{"the GELU product of mid with itself", vision.TowerOp{Kind: vision.TowerGELUMul, In: vision.TowerMid, Out: vision.TowerMid}},
		{"axial RoPE on v", vision.TowerOp{Kind: vision.TowerRoPEAxial, Out: vision.TowerV}},
		{"a per-head norm on the residual", vision.TowerOp{Kind: vision.TowerHeadNorm, Out: vision.TowerX, W: vision.TowerQNorm, Form: vision.TowerRMSNorm}},
		{"an unscaled norm naming a weight", vision.TowerOp{Kind: vision.TowerHeadNorm, Out: vision.TowerV, W: vision.TowerQNorm, Form: vision.TowerRMSNormUnscaled}},
		{"a per-head norm of no weight", vision.TowerOp{Kind: vision.TowerHeadNorm, Out: vision.TowerQ, Form: vision.TowerRMSNorm}},
		{"a layer norm", vision.TowerOp{Kind: vision.TowerNorm, In: vision.TowerX, Out: vision.TowerXN, W: vision.TowerNorm1, Form: vision.TowerLayerNorm}},
		{"the gate into a hidden-width register", vision.TowerOp{Kind: vision.TowerLinear, In: vision.TowerXN, Out: vision.TowerQ, W: vision.TowerWGate}},
		{"the down linear from a hidden-width register", vision.TowerOp{Kind: vision.TowerLinear, In: vision.TowerXN, Out: vision.TowerX, W: vision.TowerWDown}},
		{"attention at another scale", vision.TowerOp{Kind: vision.TowerAttention, Out: vision.TowerCtx, Scale: 0.5}},
	} {
		err := func() (err error) {
			defer func() {
				if recover() != nil {
					t.Errorf("%s: queued instead of refused", c.name)
				}
			}()
			return a.walkBlock(nil, vision.TowerBlock{c.op}, ly, 4) // a nil encoder: any dispatch panics
		}()
		if err == nil {
			t.Errorf("%s: accepted", c.name)
		} else {
			t.Logf("%s: %v", c.name, err)
		}
	}
}
