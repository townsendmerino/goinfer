package audio

import (
	"math"
	"testing"

	"github.com/townsendmerino/goinfer/internal/vision"
)

// tinyGemma4AudioBlock is one block of hidden 8, two heads, a window of 2 and a depthwise kernel of 3, with every
// weight shaped as the loader shapes it (values zero), for the walker's refusal tests.
func tinyGemma4AudioBlock() (*Gemma4AudioEncoder, *Gemma4AudioLayer) {
	const H = 8
	c := Gemma4AudioConfig{HiddenSize: H, NumHiddenLayers: 1, NumAttentionHeads: 2, ConvKernelSize: 3, ContextLeft: 3,
		ResidualWeight: 0.5, LogitCap: 50, RMSNormEps: 1e-6}
	inf := float32(math.Inf(1))
	proj := func(out, in int) Gemma4Proj {
		return Gemma4Proj{W: make([]float32, out*in), Out: out, In: in, InMin: -inf, InMax: inf, OutMin: -inf, OutMax: inf}
	}
	norm := func() []float32 { return make([]float32, H) }
	ff := func() Gemma4AudioFF {
		return Gemma4AudioFF{PreNorm: norm(), PostNorm: norm(), Up: proj(4*H, H), Down: proj(H, 4*H)}
	}
	ly := &Gemma4AudioLayer{FF1: ff(), FF2: ff(), NormPreAttn: norm(), NormPostAttn: norm(), NormOut: norm(),
		Q: proj(H, H), K: proj(H, H), V: proj(H, H), Post: proj(H, H), RelK: make([]float32, c.Window()*H),
		PerDimScale: make([]float32, H/2), ConvPreNorm: norm(), ConvStart: proj(2*H, H), ConvW: make([]float32, H*3),
		ConvNorm: norm(), ConvEnd: proj(H, H)}
	return &Gemma4AudioEncoder{Cfg: c}, ly
}

// TestGemma4AudioWalkBlock_runsItsProgram: the walker takes every op of Gemma4AudioProgram, so a refusal below is the
// misrouting's, not the tower's.
func TestGemma4AudioWalkBlock_runsItsProgram(t *testing.T) {
	e, ly := tinyGemma4AudioBlock()
	const n = 4
	if err := e.walkBlock(Gemma4AudioProgram(e.Cfg), ly, newGemma4Regs(make([]float32, n*8), n, 8), n); err != nil {
		t.Fatal(err)
	}
	for i, op := range Gemma4AudioProgram(e.Cfg) {
		if err := op.CheckRegs(); err != nil {
			t.Errorf("op %d (%s): %v", i, op.Kind, err)
		}
	}
}

// TestGemma4AudioWalkBlock_refusesMisroutedOps holds the CPU audio walker to the registers, weights and forms its
// operations take (vision.MisroutedTowerOpsForTest, then the cases only this tower has): an op that does not fit is an
// error before it runs.
func TestGemma4AudioWalkBlock_refusesMisroutedOps(t *testing.T) {
	type tc = struct {
		name string
		op   vision.TowerOp
	}
	var cases []tc
	for _, m := range vision.MisroutedTowerOpsForTest() {
		cases = append(cases, tc{m.Name, m.Op})
	}
	cases = append(cases, []tc{
		{"a layer norm", vision.TowerOp{Kind: vision.TowerNorm, In: vision.TowerX, Out: vision.TowerXN, W: vision.TowerNormFF1Pre, Form: vision.TowerLayerNorm}},
		{"a norm of a weight the tower lacks", vision.TowerOp{Kind: vision.TowerNorm, In: vision.TowerX, Out: vision.TowerXN, W: vision.TowerNorm2, Form: vision.TowerRMSNorm}},
		{"a linear of a weight the tower lacks", vision.TowerOp{Kind: vision.TowerLinear, In: vision.TowerXN, Out: vision.TowerMid, W: vision.TowerWFC1}},
		{"the up linear into a hidden-width register", vision.TowerOp{Kind: vision.TowerLinear, In: vision.TowerXN, Out: vision.TowerQ, W: vision.TowerWFF1Up}},
		{"the conv start into a hidden-width register", vision.TowerOp{Kind: vision.TowerLinear, In: vision.TowerXN, Out: vision.TowerCtx, W: vision.TowerWConvStart}},
		{"a linear into mid2, which the tower lacks", vision.TowerOp{Kind: vision.TowerLinear, In: vision.TowerXN, Out: vision.TowerMid2, W: vision.TowerWFF1Up}},
		{"the down linear from mid2, which the tower lacks", vision.TowerOp{Kind: vision.TowerLinear, In: vision.TowerMid2, Out: vision.TowerXN, W: vision.TowerWFF1Down}},
		{"local attention over no keys", vision.TowerOp{Kind: vision.TowerLocalAttention, Out: vision.TowerCtx, Window: 0}},
		{"local attention past the relative keys", vision.TowerOp{Kind: vision.TowerLocalAttention, Out: vision.TowerCtx, Window: 3}},
		{"a kind the tower lacks", vision.TowerOp{Kind: vision.TowerRoPE, Out: vision.TowerQ}},
	}...)
	const n = 4
	for _, c := range cases {
		e, ly := tinyGemma4AudioBlock()
		err := func() (err error) {
			defer func() {
				if recover() != nil {
					t.Errorf("%s: ran instead of refused", c.name)
				}
			}()
			return e.walkBlock(vision.TowerBlock{c.op}, ly, newGemma4Regs(make([]float32, n*8), n, 8), n)
		}()
		if err == nil {
			t.Errorf("%s: accepted", c.name)
		}
	}
	e, ly := tinyGemma4AudioBlock()
	ly.ConvW = ly.ConvW[:8*2]
	conv := vision.TowerOp{Kind: vision.TowerDWConv, In: vision.TowerCtx, Out: vision.TowerXN, W: vision.TowerConvW}
	if err := e.walkBlock(vision.TowerBlock{conv}, ly, newGemma4Regs(make([]float32, n*8), n, 8), n); err == nil {
		t.Error("a depthwise kernel of another size: accepted")
	}
}
