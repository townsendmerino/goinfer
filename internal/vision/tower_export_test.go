package vision

import (
	"math"
	"math/rand"
	"os"
	"slices"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// G-S2a of goinfer's docs/tasks/task-multimodal-support-2026-10.md: a device tower built from these exports runs the
// patch embed and the blocks itself and calls FinishHidden for the tail, so the exported pieces, recomposed the way a
// device tower recomposes them, must reproduce Forward bit for bit (it is the same CPU code, called in a different
// order). The tiny towers' norms and Qwen3's position table are randomised first: an all-ones fixture would hide a
// dropped or swapped norm (goinfer's gemma4-vision-tiny lesson).

func randomise(rng *rand.Rand, xs ...[]float32) {
	for _, x := range xs {
		for i := range x {
			x[i] = float32(1 + 0.3*rng.NormFloat64())
		}
	}
}

func bitEqual(t *testing.T, what string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", what, len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s: value %d is %v, want %v (bit-exact)", what, i, got[i], want[i])
		}
	}
}

func TestQwen3Export_recomposesForward(t *testing.T) {
	const ckpt = "testdata/qwen35vl-vision-tiny"
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no %s", ckpt)
	}
	g := readQwen3Golden(t, "testdata/qwen35vl_vision_golden.json")
	e, err := LoadQwen3VisionEncoder(ckpt, false)
	if err != nil {
		t.Fatal(err)
	}
	w, err := e.Weights()
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(7))
	randomise(rng, e.posEmbed, e.mergerLNw)
	for _, b := range w.Blocks {
		randomise(rng, b.Norm1W, b.Norm1B, b.Norm2W, b.Norm2B)
	}
	if &w.Blocks[0].Norm1W[0] != &e.blocks[0].norm1w[0] || &w.PatchW[0] != &e.patchW[0] {
		t.Fatal("Weights must alias the encoder's slices")
	}
	c := e.Cfg
	hidden := c.HiddenSize
	np := len(g.PixelValues) / (c.InChannels * c.TemporalPatchSize * c.PatchSize * c.PatchSize)

	// Embed = patch matmul + bias + PositionEmbeds, in that order.
	h := make([]float32, np*hidden)
	linalg.MatmulBT(g.PixelValues, w.PatchW, h, np, len(w.PatchW)/hidden, hidden)
	addBias(h, w.PatchB, np, hidden)
	pos := e.PositionEmbeds(g.GridTHW)
	for i := range h {
		h[i] += pos[i]
	}
	want, err := e.Embed(g.PixelValues, g.GridTHW)
	if err != nil {
		t.Fatal(err)
	}
	bitEqual(t, "Qwen3 embed", h, want)

	// The tail: FinishHidden(last block's output) = Forward.
	vit, err := e.ForwardViT(g.PixelValues, g.GridTHW)
	if err != nil {
		t.Fatal(err)
	}
	got, err := e.FinishHidden(vit, g.GridTHW)
	if err != nil {
		t.Fatal(err)
	}
	fwd, err := e.Forward(g.PixelValues, g.GridTHW)
	if err != nil {
		t.Fatal(err)
	}
	bitEqual(t, "Qwen3 FinishHidden", got, fwd)

	cos, sin := e.RopeTables(g.GridTHW)
	if hd := hidden / c.NumHeads; len(cos) != np*hd || len(sin) != np*hd {
		t.Fatalf("rope tables %d/%d, want %d", len(cos), len(sin), np*hd)
	}
	if seg := VisionSegments(g.GridTHW); seg[0] != 0 || seg[len(seg)-1] != np {
		t.Fatalf("segments %v for %d patches", seg, np)
	}
	if _, err := e.FinishHidden(vit[:len(vit)-1], g.GridTHW); err == nil {
		t.Fatal("FinishHidden must refuse a short hidden")
	}
	q, err := LoadQwen3VisionEncoder(ckpt, true)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := q.Weights(); err == nil {
		t.Fatal("a quantized tower's export must error")
	}
}

func TestGlmOcrExport_recomposesForward(t *testing.T) {
	if _, err := os.Stat(glmTinyCkpt); err != nil {
		t.Skipf("no %s", glmTinyCkpt)
	}
	g := readGlmGolden(t)
	e, err := LoadGlmOcrVisionEncoder(glmTinyCkpt, false)
	if err != nil {
		t.Fatal(err)
	}
	w, err := e.Weights()
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(11))
	randomise(rng, e.postNormW, e.mLNw)
	for _, b := range w.Blocks {
		randomise(rng, b.Norm1W, b.Norm2W, b.QNorm, b.KNorm)
	}
	if &w.Blocks[0].QNorm[0] != &e.blocks[0].qNormW[0] {
		t.Fatal("Weights must alias the encoder's slices")
	}
	hidden := e.Cfg.HiddenSize
	np := g.NPatches

	h := make([]float32, np*hidden)
	linalg.MatmulBT(g.PixelValues, w.PatchW, h, np, len(w.PatchW)/hidden, hidden)
	addBias(h, w.PatchB, np, hidden)
	want, err := e.Embed(g.PixelValues, g.GridTHW)
	if err != nil {
		t.Fatal(err)
	}
	bitEqual(t, "GLM embed", h, want)

	blocks, err := e.forwardBlocks(g.PixelValues, g.GridTHW)
	if err != nil {
		t.Fatal(err)
	}
	blocksCopy := slices.Clone(blocks)
	got, err := e.FinishHidden(blocks, g.GridTHW)
	if err != nil {
		t.Fatal(err)
	}
	bitEqual(t, "FinishHidden leaves its input alone", blocks, blocksCopy)
	fwd, err := e.Forward(g.PixelValues, g.GridTHW)
	if err != nil {
		t.Fatal(err)
	}
	bitEqual(t, "GLM FinishHidden", got, fwd)

	q, err := LoadGlmOcrVisionEncoder(glmTinyCkpt, true)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := q.Weights(); err == nil {
		t.Fatal("a quantized tower's export must error")
	}
}
