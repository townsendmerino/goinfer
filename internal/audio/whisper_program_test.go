package audio

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/townsendmerino/goinfer/internal/vision"
)

// errRan marks an op the walker ran (and panicked in) instead of refusing.
var errRan = errors.New("ran")

// TestWhisperWalkBlock_refusesMisroutedOps: the encoder's walker refuses each misrouted op with an error, on a layer whose
// weights all have their right lengths, so what refuses an op is its routing or its shape.
func TestWhisperWalkBlock_refusesMisroutedOps(t *testing.T) {
	const hidden, inter = 8, 32
	e := &WhisperEncoder{Cfg: WhisperEncoderConfig{DModel: hidden, FFN: inter, Heads: 2}}
	sq, hb := make([]float32, hidden*hidden), make([]float32, hidden)
	ly := &whisperLayer{ln1W: hb, ln1B: hb, ln2W: hb, ln2B: hb, qW: sq, qB: hb, kW: sq, vW: sq, vB: hb, oW: sq, oB: hb,
		fc1W: make([]float32, inter*hidden), fc1B: make([]float32, inter), fc2W: make([]float32, hidden*inter), fc2B: hb}
	for _, c := range vision.MisroutedTowerOpsForTest() {
		err := func() (err error) {
			defer func() {
				if recover() != nil {
					err = errRan
				}
			}()
			return e.walkBlock(vision.TowerBlock{c.Op}, ly, make([]float32, 4*hidden), 4, 4)
		}()
		switch {
		case errors.Is(err, errRan):
			t.Errorf("%s: ran instead of refused", c.Name)
		case err == nil:
			t.Errorf("%s: accepted", c.Name)
		}
	}
}

// TestWhisperWalkBlock_postNormMatchesReference: the post-norm defect's program (SetDefectForTest 7), walked over one
// layer, equals a post-norm layer written out by hand from the encoder's own operations, bit for bit. Its last norm
// writes the residual x, so this also holds the walker to writing x in place.
func TestWhisperWalkBlock_postNormMatchesReference(t *testing.T) {
	const n, D, F, heads = 6, 8, 16, 2
	e := &WhisperEncoder{Cfg: WhisperEncoderConfig{DModel: D, FFN: F, Heads: heads}}
	r := rand.New(rand.NewPCG(7, 11))
	rnd := func(k int, scale float32) []float32 {
		v := make([]float32, k)
		for i := range v {
			v[i] = (r.Float32()*2 - 1) * scale
		}
		return v
	}
	ly := &whisperLayer{ln1W: rnd(D, 1), ln1B: rnd(D, 0.1), ln2W: rnd(D, 1), ln2B: rnd(D, 0.1),
		qW: rnd(D*D, 0.5), qB: rnd(D, 0.1), kW: rnd(D*D, 0.5), vW: rnd(D*D, 0.5), vB: rnd(D, 0.1), oW: rnd(D*D, 0.5), oB: rnd(D, 0.1),
		fc1W: rnd(F*D, 0.5), fc1B: rnd(F, 0.1), fc2W: rnd(D*F, 0.5), fc2B: rnd(D, 0.1)}
	h0 := rnd(n*D, 1)

	want := append([]float32(nil), h0...)
	q := linear(want, ly.qW, ly.qB, n, D, D)
	k := linear(want, ly.kW, make([]float32, D), n, D, D)
	v := linear(want, ly.vW, ly.vB, n, D, D)
	att := e.attend(q, k, v, n, n, float32(1/math.Sqrt(float64(D/heads))))
	a := linear(att, ly.oW, ly.oB, n, D, D)
	for i := range want {
		want[i] += a[i]
	}
	layerNorm(want, ly.ln1W, ly.ln1B, n, D)
	f := linear(want, ly.fc1W, ly.fc1B, n, D, F)
	e.geluInto(f)
	g := linear(f, ly.fc2W, ly.fc2B, n, F, D)
	for i := range want {
		want[i] += g[i]
	}
	layerNorm(want, ly.ln2W, ly.ln2B, n, D)

	got := append([]float32(nil), h0...)
	if err := e.walkBlock(whisperPostNormProgram(e.Cfg), ly, got, n, n); err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("value %d: walked %v, reference %v (and the rest may differ)", i, got[i], want[i])
		}
	}
}
