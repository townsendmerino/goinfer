//go:build realckpt

package decoder

import (
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestDFlashGemma4_diag localizes the Gemma-4 pairing's 0.00 acceptance.
//
// 477 rounds, 480 tokens, mean accepted EXACTLY 0.00. That number is not "the drafter
// transfers badly" — a merely mismatched drafter still lands the occasional newline or
// closing brace. Zero across 477 first-position proposals says the drafter is being fed
// something it cannot use, and the pairing has no gate-1 reference dump to localize it,
// which is why this prints intermediate quantities instead of asserting a threshold.
//
// Already RULED OUT, so they are not re-checked here:
//   - the prompt: goinfer's Gemma4 template renders id-identical to HF's canonical one
//     (24/24 ids), and its trailing `<|channel>thought\n<channel|>` is the EMPTY thought
//     block, i.e. Gemma 4's NON-thinking prompt — the analogue of Qwen3's suppressor.
//   - the embedding scale: DrafterEmbedBlock goes through embedToken, which applies
//     arch.EmbedScale, so the block embedding matches the target's own convention.
//   - the LM head: DrafterHeadLogits handles tied heads, softcap and logit scale, and
//     argmax is invariant to the monotonic parts regardless.
//
// What is left is the hidden states, so that is what this measures. The tell is SCALE: the
// drafter's fc maps taps*hidden -> hidden with weights trained on the reference's residual
// magnitudes, and Gemma 4's residual stream is unusually large (embedding scale sqrt(2816)
// ~ 53). If our captured norms are far from the drafter's expectation, everything downstream
// is saturated noise and 0.00 is the expected consequence rather than a mystery.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run DFlashGemma4_diag -v
func TestDFlashGemma4_diag(t *testing.T) {
	requireHeavyModel(t)
	ddir := os.Getenv("GOINFER_DFLASH_DRAFTER")
	tdir := os.Getenv("GOINFER_DFLASH_TARGET")
	tokDir := os.Getenv("GOINFER_DFLASH_TOKENIZER")
	if ddir == "" || tdir == "" || tokDir == "" {
		t.Skip("set GOINFER_DFLASH_DRAFTER / _TARGET / _TOKENIZER")
	}
	d, err := LoadDFlashDrafter(ddir)
	if err != nil {
		t.Fatalf("LoadDFlashDrafter: %v", err)
	}
	defer d.Close()
	m, err := Load(tdir, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	tk, err := tokenizer.Load(tokDir)
	if err != nil {
		t.Fatalf("tokenizer.Load: %v", err)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatalf("chat.Detect: %v", err)
	}
	ids, err := tk.EncodeSegments(tmpl.RenderSegments("", []chat.Turn{
		{Role: "user", Content: "Write a Python function that returns the nth Fibonacci number."},
	}), false)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	t.Logf("prompt: %d ids", len(ids))

	rms := func(v []float32) float64 {
		var s float64
		for _, x := range v {
			s += float64(x) * float64(x)
		}
		return math.Sqrt(s / float64(len(v)))
	}

	B := d.BlockSize()
	cache := m.NewCache(len(ids) + B + 8)
	layers := d.TargetLayerIDs()
	var ctxCat [][]float32
	var logits []float32
	for _, id := range ids {
		lg, hidden, err := m.ForwardCapture(id, cache, layers)
		if err != nil {
			t.Fatalf("ForwardCapture: %v", err)
		}
		row := make([]float32, 0, len(hidden)*d.hidden)
		for _, h := range hidden {
			row = append(row, h...)
		}
		ctxCat = append(ctxCat, row)
		logits = lg
	}
	// Per-tap RMS of the LAST position: the drafter's fc consumes exactly this vector.
	last := ctxCat[len(ctxCat)-1]
	for i, l := range layers {
		t.Logf("  tap layer %2d  rms %.4f", l, rms(last[i*d.hidden:(i+1)*d.hidden]))
	}

	anchor := argmax(logits)
	txt, _ := tk.DecodePiece(anchor)
	t.Logf("target anchor = %d (%q)", anchor, txt)

	fused, err := d.FuseContext(m.be, ctxCat)
	if err != nil {
		t.Fatalf("FuseContext: %v", err)
	}
	t.Logf("fused[last] rms %.4f  (fc output — compare against the tap rms above)", rms(fused[len(fused)-1]))

	blk := make([]int, B)
	for i := range blk {
		blk[i] = d.MaskTokenID()
	}
	blk[0] = anchor
	trunk, err := d.DraftBlock(m.be, fused, m.DrafterEmbedBlock(blk))
	if err != nil {
		t.Fatalf("DraftBlock: %v", err)
	}
	t.Logf("trunk[1] rms %.4f", rms(trunk[1]))

	// The drafted block, with text. DEGENERATE output (the same id repeated) points at
	// saturated inputs; varied-but-wrong points at a convention mismatch instead.
	var drafted []int
	for _, h := range trunk[1:] {
		drafted = append(drafted, argmax(m.DrafterHeadLogits(h)))
	}
	t.Logf("drafted ids = %v", drafted)
	uniq := map[int]bool{}
	for _, id := range drafted {
		uniq[id] = true
	}
	for i, id := range drafted[:min(5, len(drafted))] {
		s, _ := tk.DecodePiece(id)
		t.Logf("  drafted[%d] = %d %q", i, id, s)
	}
	t.Logf("DISTINCT drafted ids: %d of %d — 1 means degenerate (saturated input), "+
		"many means a convention mismatch", len(uniq), len(drafted))
}
