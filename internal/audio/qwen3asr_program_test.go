package audio

import (
	"testing"

	"github.com/townsendmerino/goinfer/internal/vision"
)

// tinyQwenASRLayer is one layer of d_model 8, two heads and an FFN of 32, every weight shaped as the loader shapes it
// (values zero), for the walker's refusal tests.
func tinyQwenASRLayer() (*QwenASREncoder, *qwenASRLayer) {
	const D, F = 8, 32
	z := func(n int) []float32 { return make([]float32, n) }
	ly := &qwenASRLayer{ln1W: z(D), ln1B: z(D), ln2W: z(D), ln2B: z(D), qW: z(D * D), qB: z(D), kW: z(D * D), kB: z(D),
		vW: z(D * D), vB: z(D), oW: z(D * D), oB: z(D), fc1W: z(F * D), fc1B: z(F), fc2W: z(D * F), fc2B: z(D)}
	return &QwenASREncoder{Cfg: QwenASRConfig{DModel: D, Heads: 2, FFN: F}}, ly
}

func tinyQwenASRRegs(n int) *qwenASRRegs {
	z := func(w int) []float32 { return make([]float32, n*w) }
	return &qwenASRRegs{x: z(8), xn: z(8), q: z(8), k: z(8), v: z(8), ctx: z(8), mid: z(32)}
}

// TestQwenASRWalkBlock_refusesMisroutedOps holds the Qwen3-ASR walker to the registers, weights and forms its
// operations take (vision.MisroutedTowerOpsForTest, then the cases only this tower has): an op that does not fit is an
// error before it runs. Its own program runs.
func TestQwenASRWalkBlock_refusesMisroutedOps(t *testing.T) {
	const n = 4
	if e, ly := tinyQwenASRLayer(); e.walkBlock(QwenASRProgram(e.Cfg), ly, tinyQwenASRRegs(n), n, 2) != nil {
		t.Fatal("the tower's own program was refused")
	}
	type tc = struct {
		name string
		op   vision.TowerOp
	}
	var cases []tc
	for _, m := range vision.MisroutedTowerOpsForTest() {
		cases = append(cases, tc{m.Name, m.Op})
	}
	cases = append(cases, []tc{
		{"an RMSNorm", vision.TowerOp{Kind: vision.TowerNorm, In: vision.TowerX, Out: vision.TowerXN, W: vision.TowerNorm1, Form: vision.TowerRMSNorm}},
		{"a norm of a weight the tower lacks", vision.TowerOp{Kind: vision.TowerNorm, In: vision.TowerX, Out: vision.TowerXN, W: vision.TowerNormPostAttn, Form: vision.TowerLayerNorm}},
		{"a linear of a weight the tower lacks", vision.TowerOp{Kind: vision.TowerLinear, In: vision.TowerXN, Out: vision.TowerMid, W: vision.TowerWGate}},
		{"fc1 into a hidden-width register", vision.TowerOp{Kind: vision.TowerLinear, In: vision.TowerXN, Out: vision.TowerQ, W: vision.TowerWFC1}},
		{"a linear into mid2, which the tower lacks", vision.TowerOp{Kind: vision.TowerLinear, In: vision.TowerXN, Out: vision.TowerMid2, W: vision.TowerWFC1}},
		{"fc2 from mid2, which the tower lacks", vision.TowerOp{Kind: vision.TowerLinearAdd, In: vision.TowerMid2, Out: vision.TowerX, W: vision.TowerWFC2}},
		{"a kind the tower lacks", vision.TowerOp{Kind: vision.TowerRoPE, Out: vision.TowerQ}},
	}...)
	for _, c := range cases {
		e, ly := tinyQwenASRLayer()
		err := func() (err error) {
			defer func() {
				if recover() != nil {
					t.Errorf("%s: ran instead of refused", c.name)
				}
			}()
			return e.walkBlock(vision.TowerBlock{c.op}, ly, tinyQwenASRRegs(n), n, 2)
		}()
		if err == nil {
			t.Errorf("%s: accepted", c.name)
		}
	}
	e, ly := tinyQwenASRLayer()
	ly.qB = ly.qB[:4]
	q := vision.TowerOp{Kind: vision.TowerLinear, In: vision.TowerXN, Out: vision.TowerQ, W: vision.TowerWQ}
	if err := e.walkBlock(vision.TowerBlock{q}, ly, tinyQwenASRRegs(n), n, 2); err == nil {
		t.Error("a bias of another length: accepted")
	}
}
