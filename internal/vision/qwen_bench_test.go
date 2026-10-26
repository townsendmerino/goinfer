package vision

import (
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// newQMat32 wraps random f32 weights as an unquantized WeightMat.
func newQMat32(w []float32, rows, cols int) linalg.WeightMat {
	return linalg.WrapF32(w, rows, cols)
}

// benchQwenEncoder builds a REAL-SIZED Qwen2.5-VL ViT in process with random
// weights, the same way benchEncoder does for SigLIP. No checkpoint is needed:
// this measures allocation and memset behaviour, which does not depend on weight
// values, and the parity gate against the HF golden runs separately on the tiny
// fixture.
//
// Dims follow Qwen2.5-VL-7B's vision tower, scaled DOWN in depth so a benchmark
// iteration is seconds rather than minutes: hidden 1280, inter 3420, 16 heads,
// merge 2. depth is a parameter.
func benchQwenEncoder(depth, gridH, gridW int) (*QwenVisionEncoder, []float32, [][3]int) {
	rng := rand.New(rand.NewSource(11))
	rnd := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64()) * 0.02
		}
		return s
	}
	const hidden, nH, inter, merge = 1280, 16, 3420, 2
	const inChans, patch, tPatch = 3, 14, 2
	patchDim := inChans * tPatch * patch * patch

	cfg := QwenEncoderConfig{
		Depth: depth, HiddenSize: hidden, IntermediateSize: inter, NumHeads: nH,
		InChans: inChans, PatchSize: patch, SpatialMergeSize: merge,
		TemporalPatchSize: tPatch, OutHiddenSize: hidden, WindowSize: 112,
		FullattBlockIndexes: []int{depth - 1}, HiddenAct: "silu",
	}
	e := &QwenVisionEncoder{
		Cfg:       cfg,
		patchW:    rnd(hidden * patchDim),
		mergerLNw: rnd(hidden),
		merger0w:  newQMat32(rnd(hidden*merge*merge*hidden*merge*merge), hidden*merge*merge, hidden*merge*merge),
		merger0b:  rnd(hidden * merge * merge),
		merger2w:  newQMat32(rnd(hidden*hidden*merge*merge), hidden, hidden*merge*merge),
		merger2b:  rnd(hidden),
	}
	e.rotInvFreq = rnd(hidden / nH / 4)
	for range depth {
		e.blocks = append(e.blocks, qwenBlock{
			norm1w: rnd(hidden),
			qkvw:   newQMat32(rnd(3*hidden*hidden), 3*hidden, hidden),
			qkvb:   rnd(3 * hidden),
			projw:  newQMat32(rnd(hidden*hidden), hidden, hidden),
			projb:  rnd(hidden),
			norm2w: rnd(hidden),
			gatew:  newQMat32(rnd(inter*hidden), inter, hidden),
			upw:    newQMat32(rnd(inter*hidden), inter, hidden),
			gateb:  rnd(inter), upb: rnd(inter),
			downw: newQMat32(rnd(hidden*inter), hidden, inter),
			downb: rnd(hidden),
		})
	}
	grid := [][3]int{{1, gridH, gridW}}
	nPatches := gridH * gridW
	return e, rnd(nPatches * patchDim), grid
}

// BenchmarkQwenViT is the arbiter for perf-campaign item 18. The per-layer
// allocations it targets are `make([]float32, …)` — which ZEROES — so the cost
// is a memset of every byte per layer, and B/op should fall by roughly the
// per-layer working set times the depth.
func BenchmarkQwenViT(b *testing.B) {
	for _, tc := range []struct {
		name                string
		depth, gridH, gridW int
	}{
		{"d4_1024patches", 4, 32, 32},
		{"d8_1024patches", 8, 32, 32},
	} {
		e, pixels, grid := benchQwenEncoder(tc.depth, tc.gridH, tc.gridW)
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				out, err := e.ForwardViT(pixels, grid)
				if err != nil {
					b.Fatal(err)
				}
				sinkVision = out
			}
		})
	}
}

// TestQwenScratch_poisonedArenaIsInert guards the hazard item 18 introduces: the
// block buffers were fresh (hence ZEROED) allocations per layer and are now one
// arena reused across every layer, so a buffer read before it is fully written
// would silently carry the previous layer's values.
//
// It poisons every arena buffer with NaN and requires the output to be
// bit-identical to a run on a fresh arena. Comparing two runs of the SAME code
// would not do — stale-arena data is deterministic, so both runs would agree and
// agree wrongly. Only a known-good reference catches it.
func TestQwenScratch_poisonedArenaIsInert(t *testing.T) {
	e, pixels, grid := benchQwenEncoder(2, 8, 8)
	c := e.Cfg
	hidden, hd := c.HiddenSize, c.HiddenSize/c.NumHeads

	// Drive one real block the way computeViTHidden does.
	_, cuWin := e.windowIndex(grid)
	seq := cuWin[len(cuWin)-1]
	maxSeg := maxSegment(cuWin)
	rng := rand.New(rand.NewSource(5))
	x := make([]float32, seq*hidden)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	cos := make([]float32, seq*hd)
	sin := make([]float32, seq*hd)
	for i := range cos {
		cos[i], sin[i] = float32(rng.NormFloat64()), float32(rng.NormFloat64())
	}
	b := &e.blocks[0]

	run := func(poison bool) ([]float32, []float32) {
		s := newQwenScratch(seq, hidden, c.IntermediateSize, hd, maxSeg, c.NumHeads)
		if poison {
			for _, buf := range [][]float32{s.n1, s.n2, s.o, s.att, s.mlpOut, s.qkv,
				s.q, s.k, s.v, s.gate, s.up} {
				for i := range buf {
					buf[i] = float32(math.NaN())
				}
			}
			for w := range s.headPool {
				ws := &s.headPool[w]
				for _, buf := range [][]float32{ws.qh, ws.kh, ws.vBlk, ws.ch,
					ws.fused.SBlk, ws.fused.Tmp, ws.fused.Acc, ws.fused.MRun, ws.fused.LRun} {
					for i := range buf {
						buf[i] = float32(math.NaN())
					}
				}
			}
		}
		// Destinations come from the ARENA, exactly as computeViTHidden passes them —
		// s.att and s.mlpOut are themselves reused across layers, so a
		// partially-written output is part of what this must catch.
		att := s.att[:seq*hidden]
		if err := e.attentionInto(att, x, b, seq, cos, sin, cuWin, s); err != nil {
			t.Fatal(err)
		}
		mlpOut := s.mlpOut[:seq*hidden]
		e.mlpInto(mlpOut, x, b, seq, s)
		return append([]float32(nil), att...), append([]float32(nil), mlpOut...)
	}

	wantAtt, wantMLP := run(false)
	gotAtt, gotMLP := run(true)

	for i := range wantAtt {
		if math.IsNaN(float64(gotAtt[i])) || gotAtt[i] != wantAtt[i] {
			t.Fatalf("attention element %d: fresh arena %v, poisoned arena %v — "+
				"a scratch buffer is read before it is written", i, wantAtt[i], gotAtt[i])
		}
	}
	for i := range wantMLP {
		if math.IsNaN(float64(gotMLP[i])) || gotMLP[i] != wantMLP[i] {
			t.Fatalf("mlp element %d: fresh arena %v, poisoned arena %v — "+
				"a scratch buffer is read before it is written", i, wantMLP[i], gotMLP[i])
		}
	}
	_ = pixels
}
