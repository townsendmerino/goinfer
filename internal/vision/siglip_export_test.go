package vision

import (
	"math/rand"
	"os"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
)

// G-S3's export check (goinfer's docs/tasks/task-multimodal-support-2026-10.md, the S2 G-S2a rule applied to SigLIP):
// the exported pieces, recomposed the way a device tower recomposes them, reproduce Forward bit for bit. Norms are
// randomised first (an all-ones fixture hides a dropped norm).
func TestSiglipExport_recomposesForward(t *testing.T) {
	const ckpt = "testdata/siglip-tiny"
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no %s", ckpt)
	}
	e, err := LoadEncoder(ckpt, false)
	if err != nil {
		t.Fatal(err)
	}
	w, err := e.Weights()
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(13))
	randomise(rng, e.postLNw, e.postLNb)
	for _, b := range w.Blocks {
		randomise(rng, b.LN1W, b.LN1B, b.LN2W, b.LN2B)
	}
	if &w.Blocks[0].LN1W[0] != &e.layers[0].ln1w[0] || &w.PatchW[0] != &e.patchW[0] || &w.PosEmb[0] != &e.posEmb[0] {
		t.Fatal("Weights must alias the encoder's slices")
	}
	c := e.Cfg
	px := make([]float32, c.NumChannels*c.ImageSize*c.ImageSize)
	for i := range px {
		px[i] = float32(rng.NormFloat64())
	}
	// The patch embed a device tower runs: GridPatches · PatchWᵀ + PatchB + PosEmb, in that order.
	patches, err := e.GridPatches(px)
	if err != nil {
		t.Fatal(err)
	}
	H, np := c.HiddenSize, w.NumPatches
	h := make([]float32, np*H)
	linalg.MatmulBT(patches, w.PatchW, h, np, len(w.PatchW)/H, H)
	addBias(h, w.PatchB, np, H)
	addResidual(h, w.PosEmb)
	bitEqual(t, "SigLIP embed", h, e.embedPatches(patches))
	if len(w.Blocks) == 0 {
		t.Fatal("the tiny tower has no layers")
	}
	// The tail: FinishHidden(the last block's output) = Forward.
	blocks, err := e.forwardBlocks(px)
	if err != nil {
		t.Fatal(err)
	}
	got, err := e.FinishHidden(blocks)
	if err != nil {
		t.Fatal(err)
	}
	fwd, err := e.Forward(px)
	if err != nil {
		t.Fatal(err)
	}
	bitEqual(t, "SigLIP FinishHidden", got, fwd)
	// A dropped post-norm shows: FinishHidden must not be the identity on this fixture.
	same := true
	for i := range blocks {
		if blocks[i] != fwd[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("the post-layernorm is the identity on this fixture; the check cannot see it dropped")
	}
	if _, err := e.FinishHidden(blocks[:len(blocks)-1]); err == nil {
		t.Fatal("FinishHidden must refuse a short hidden")
	}
	q, err := LoadEncoder(ckpt, true)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := q.Weights(); err == nil {
		t.Fatal("a quantized tower's export must error")
	}
}
