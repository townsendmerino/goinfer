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

// TestWhisperCheckOp_refusesWrongLengths: a layer weight of the wrong length is refused before the op runs: a norm's
// weight or bias, a linear's bias, and a linear's matrix, each one short.
func TestWhisperCheckOp_refusesWrongLengths(t *testing.T) {
	const D, F = 8, 32
	full := func() *whisperLayer {
		sq, hb := make([]float32, D*D), make([]float32, D)
		return &whisperLayer{ln1W: hb, ln1B: hb, ln2W: hb, ln2B: hb, qW: sq, qB: hb, kW: sq, vW: sq, vB: hb, oW: sq, oB: hb,
			fc1W: make([]float32, F*D), fc1B: make([]float32, F), fc2W: make([]float32, D*F), fc2B: hb}
	}
	short := make([]float32, D-1)
	norm := vision.TowerOp{Kind: vision.TowerNorm, In: vision.TowerX, Out: vision.TowerXN, W: vision.TowerNorm1, Form: vision.TowerLayerNorm}
	fc1 := vision.TowerOp{Kind: vision.TowerLinear, In: vision.TowerXN, Out: vision.TowerMid, W: vision.TowerWFC1}
	q := vision.TowerOp{Kind: vision.TowerLinear, In: vision.TowerXN, Out: vision.TowerQ, W: vision.TowerWQ}
	for _, c := range []struct {
		name string
		op   vision.TowerOp
		cut  func(*whisperLayer)
	}{
		{"a short norm weight", norm, func(l *whisperLayer) { l.ln1W = short }},
		{"a short norm bias", norm, func(l *whisperLayer) { l.ln1B = short }},
		{"a short linear bias", q, func(l *whisperLayer) { l.qB = short }},
		{"a short fc1 bias", fc1, func(l *whisperLayer) { l.fc1B = make([]float32, F-1) }},
		{"a short linear matrix", q, func(l *whisperLayer) { l.qW = make([]float32, D*D-1) }},
	} {
		if err := full().checkOp(c.op, D, F); err != nil {
			t.Fatalf("%s: the full layer refused the op: %v", c.name, err)
		}
		ly := full()
		c.cut(ly)
		if err := ly.checkOp(c.op, D, F); err == nil {
			t.Errorf("%s: accepted", c.name)
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
