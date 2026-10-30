//go:build darwin

package metal

import (
	"compress/gzip"
	"encoding/json"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/internal/audio"
	"github.com/townsendmerino/goinfer/internal/vision"
)

// TestG4AMetal_matchesCPU is Phase A's gate AM1 on the tiny tower (docs/tasks/task-embeddinggemma2.md): the Metal
// blocks between aikit's subsampler and tail match aikit's CPU Forward to cosine 0.9999 on every soft token, and
// transformers' golden too, on the committed tiny random tower (testdata/gemma4-audio-tiny: finite asymmetric clip
// bounds binding in 15 of 20 projections, random norms and per_dim_scale, 33 soft tokens so the 12-key window binds).
func TestG4AMetal_matchesCPU(t *testing.T) {
	f, err := os.Open("../testdata/gemma4-audio-tiny/golden.json.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		TValid   int       `json:"t_valid"`
		Features []float32 `json:"features"`
		Embed    []float32 `json:"embed"`
	}
	if err := json.NewDecoder(zr).Decode(&g); err != nil {
		t.Fatal(err)
	}
	enc, err := audio.LoadGemma4AudioEncoder("../testdata/gemma4-audio-tiny")
	if err != nil {
		t.Fatal(err)
	}
	a, err := newG4AAccel(enc)
	if err != nil {
		t.Fatal(err)
	}
	cpu, err := enc.Forward(g.Features, g.TValid)
	if err != nil {
		t.Fatal(err)
	}
	h, n, err := enc.Subsample(g.Features, g.TValid)
	if err != nil {
		t.Fatal(err)
	}
	hb, err := a.Blocks(h, n)
	if err != nil {
		t.Fatal(err)
	}
	got, err := enc.FinishBlocks(hb, n)
	if err != nil {
		t.Fatal(err)
	}
	wc, wh := g4vWorst(got, cpu, enc.TextHiddenSize), g4vWorst(got, g.Embed, enc.TextHiddenSize)
	t.Logf("tiny audio tower, %d soft tokens: Metal blocks against the CPU tower worst soft-token cosine %.9f, against transformers %.9f", n, wc, wh)
	if wc < 0.9999 || wh < 0.9999 {
		t.Errorf("worst soft-token cosine %.9f (CPU) / %.9f (transformers), under 0.9999", wc, wh)
	}
}

// TestG4AWalkBlock_refusesMisroutedOps holds the Gemma 4 audio tower's walker to the registers, weights and forms its
// dispatches take (vision.MisroutedTowerOpsForTest, then the cases only this tower has): an op that does not fit is an
// error before anything is queued.
func TestG4AWalkBlock_refusesMisroutedOps(t *testing.T) {
	const H = 8
	a := &g4aAccel{cfg: audio.Gemma4AudioConfig{HiddenSize: H, NumAttentionHeads: 2, ConvKernelSize: 3, ContextLeft: 3}}
	hp := g4vProj{out: H, in: H}
	ff := g4aFF{up: g4vProj{out: 4 * H, in: H}, down: g4vProj{out: H, in: 4 * H}}
	ly := &g4aLayer{ff1: ff, ff2: ff, q: hp, k: hp, v: hp, post: hp, convStart: g4vProj{out: 2 * H, in: H}, convEnd: hp}
	type tc = struct {
		name string
		op   vision.TowerOp
	}
	var cases []tc
	for _, m := range vision.MisroutedTowerOpsForTest() {
		cases = append(cases, tc{m.Name, m.Op})
	}
	for _, c := range append(cases, []tc{
		{"a layer norm", vision.TowerOp{Kind: vision.TowerNorm, In: vision.TowerX, Out: vision.TowerXN, W: vision.TowerNormFF1Pre, Form: vision.TowerLayerNorm}},
		{"a norm of a weight the tower lacks", vision.TowerOp{Kind: vision.TowerNorm, In: vision.TowerX, Out: vision.TowerXN, W: vision.TowerNorm2, Form: vision.TowerRMSNorm}},
		{"a linear of a weight the tower lacks", vision.TowerOp{Kind: vision.TowerLinear, In: vision.TowerXN, Out: vision.TowerMid, W: vision.TowerWFC1}},
		{"the up linear into a hidden-width register", vision.TowerOp{Kind: vision.TowerLinear, In: vision.TowerXN, Out: vision.TowerQ, W: vision.TowerWFF1Up}},
		{"the conv start into a hidden-width register", vision.TowerOp{Kind: vision.TowerLinear, In: vision.TowerXN, Out: vision.TowerCtx, W: vision.TowerWConvStart}},
		{"a linear into mid2, which the tower lacks", vision.TowerOp{Kind: vision.TowerLinear, In: vision.TowerXN, Out: vision.TowerMid2, W: vision.TowerWFF1Up}},
		{"the down linear from mid2, which the tower lacks", vision.TowerOp{Kind: vision.TowerLinear, In: vision.TowerMid2, Out: vision.TowerXN, W: vision.TowerWFF1Down}},
		{"local attention over no keys", vision.TowerOp{Kind: vision.TowerLocalAttention, Out: vision.TowerCtx, Window: 0, Scale: 50}},
		{"local attention past the relative keys", vision.TowerOp{Kind: vision.TowerLocalAttention, Out: vision.TowerCtx, Window: 3, Scale: 50}},
		{"local attention with no softcap", vision.TowerOp{Kind: vision.TowerLocalAttention, Out: vision.TowerCtx, Window: 2}},
		{"a kind the tower lacks", vision.TowerOp{Kind: vision.TowerRoPE, Out: vision.TowerQ}},
	}...) {
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
		}
	}
	// Cases where the op fits but the tower does not: a window past the kernel's 64 keys (in a tower whose relative keys
	// reach that far), a depthwise kernel of no size.
	for _, c := range []struct {
		name string
		cfg  audio.Gemma4AudioConfig
		op   vision.TowerOp
	}{
		{"local attention past the kernel's 64 keys", audio.Gemma4AudioConfig{HiddenSize: H, NumAttentionHeads: 2, ConvKernelSize: 3, ContextLeft: 70},
			vision.TowerOp{Kind: vision.TowerLocalAttention, Out: vision.TowerCtx, Window: 65, Scale: 50}},
		{"a depthwise kernel of no size", audio.Gemma4AudioConfig{HiddenSize: H, NumAttentionHeads: 2, ContextLeft: 3},
			vision.TowerOp{Kind: vision.TowerDWConv, In: vision.TowerCtx, Out: vision.TowerXN, W: vision.TowerConvW}},
	} {
		a := &g4aAccel{cfg: c.cfg}
		err := func() (err error) {
			defer func() {
				if recover() != nil {
					t.Errorf("%s: queued instead of refused", c.name)
				}
			}()
			return a.walkBlock(nil, vision.TowerBlock{c.op}, ly, 4)
		}()
		if err == nil {
			t.Errorf("%s: accepted", c.name)
		}
	}
}
