//go:build realckpt

// P10 kill-gate 2: measured acceptance for the DFlash block drafter.
//
// Gate 1 (TestDFlash_referenceParity / TestDFlash_targetEndToEnd) proved the forward
// matches the reference. Only then is this legitimate to run — the pre-registered order,
// and the one 05 got backwards.
//
// WHAT THIS MEASURES, AND WHAT IT DOES NOT. Acceptance is a property of the drafter's
// distribution against the target's, i.e. NUMERICS — it does not depend on the backend, so
// it is measured here on CPU and transfers to the GPU paths unchanged (at equal precision).
// Wall-clock does NOT transfer and is NOT measured here: this loop verifies the block with
// 16 sequential single-token forwards rather than one batched M=16 pass, because sequential
// forwards are what ForwardCapture exposes and acceptance is indifferent to the difference.
// Reading a speed number off this harness would be wrong; that is gate 3, on the GPU.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run TestDFlashAcceptance -v -timeout 4h
package decoder

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// dflashSuite is a fixed, recorded prompt set per traffic class. Small and deterministic
// on purpose: the point is a defensible tok/verify per class, not a benchmark sweep. The
// prompts are chat-templated at render time — the raw-vs-chat gap measured in increment 2
// was 0/15 vs 10/15 accepted, so an untemplated suite would measure the template, not the
// drafter.
var dflashSuites = map[string][]string{
	"code": {
		"Write a Python function that returns the nth Fibonacci number.",
		"Write a Go function that reverses a slice of ints in place.",
		"Write a SQL query that selects the top 5 customers by total order value.",
	},
	"math": {
		"What is 17 * 23? Show your working.",
		"A train travels 120 km in 1.5 hours. What is its average speed in km/h?",
	},
	"chat": {
		"Explain what a hash table is, in two sentences.",
		"Give me three tips for keeping houseplants alive.",
	},
}

// TestDFlashAcceptance runs the DFlash block-verify loop over each suite and reports
// tok/verify. The bar (docs/spec/08 kill-gate 2) is >= 3.0 on at least one suite.
func TestDFlashAcceptance(t *testing.T) {
	requireHeavyModel(t)
	ddir := assetPath(t, "GOINFER_DFLASH_F32")
	tdir := assetPath(t, "GOINFER_QWEN3_4B")

	d, err := LoadDFlashDrafter(ddir)
	if err != nil {
		t.Fatalf("LoadDFlashDrafter: %v", err)
	}
	defer d.Close()
	// int8 target: this is the precision the resident GPU paths would actually run, and
	// an f32 4B forward per verify position makes the sweep hours instead of minutes.
	// TestDFlash_targetEndToEnd already pins the f32 numerics against the reference.
	m, err := Load(tdir, Options{Quant: "int8int8"})
	if err != nil {
		t.Fatalf("Load(%s): %v", tdir, err)
	}
	defer m.Close()

	tk, err := tokenizer.Load(tdir)
	if err != nil {
		t.Fatalf("tokenizer.Load(%s): %v", tdir, err)
	}

	maxNew := 48
	if os.Getenv("GOINFER_DFLASH_MAXNEW") != "" {
		if v, err := atoiPositive(os.Getenv("GOINFER_DFLASH_MAXNEW")); err == nil {
			maxNew = v
		}
	}

	type result struct{ rounds, accepted, generated int }
	overall := map[string]*result{}
	for _, suite := range []string{"code", "math", "chat"} {
		res := &result{}
		overall[suite] = res
		for _, prompt := range dflashSuites[suite] {
			r := dflashRun(t, m, d, tk, prompt, maxNew)
			res.rounds += r.rounds
			res.accepted += r.accepted
			res.generated += r.generated
		}
		tpv := float64(res.generated) / float64(res.rounds)
		t.Logf("%-5s  %2d rounds  %3d tokens  mean accepted %.2f/%d  => %.2f tok/verify",
			suite, res.rounds, res.generated, float64(res.accepted)/float64(res.rounds),
			d.BlockSize()-1, tpv)
	}

	best, bestSuite := 0.0, ""
	for s, r := range overall {
		if tpv := float64(r.generated) / float64(r.rounds); tpv > best {
			best, bestSuite = tpv, s
		}
	}
	t.Logf("GATE 2: best suite %q at %.2f tok/verify (bar >= 3.0)", bestSuite, best)
	if best < 3.0 {
		t.Errorf("kill-gate 2 MISSED: best %.2f tok/verify < 3.0 — protocol wrong (back to gate 1) or the claims do not transfer (stop, record)", best)
	}
}

type dflashRunResult struct{ rounds, accepted, generated int }

// dflashRun greedily generates from one prompt with the DFlash block-verify loop and
// returns how many rounds it took. Lossless by construction: every emitted token is one
// the TARGET's own argmax produced — the drafter only ever proposes, and a rejected
// proposal is rolled out of the cache.
func dflashRun(t *testing.T, m *Model, d *DFlashDrafter, tk *tokenizer.Tokenizer, prompt string, maxNew int) dflashRunResult {
	t.Helper()
	turns := []chat.Turn{{Role: "user", Content: prompt}}
	ids, err := tk.EncodeSegments(chat.ChatML().RenderSegments("", turns), false)
	if err != nil {
		t.Fatalf("encode segments: %v", err)
	}

	B := d.BlockSize()
	cache := m.NewCache(len(ids) + maxNew + B + 2)
	layers := d.TargetLayerIDs()
	var ctxCat [][]float32
	var logits []float32
	feed := func(id int) {
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
	for _, id := range ids {
		feed(id)
	}

	eos := map[int]bool{}
	for _, e := range m.w.Cfg.EOSIDs() {
		eos[e] = true
	}
	var out dflashRunResult
	anchor := argmax(logits)
	generated := 1 // the anchor is a real emitted token
	for generated < maxNew {
		fused, err := d.FuseContext(m.be, ctxCat)
		if err != nil {
			t.Fatalf("FuseContext: %v", err)
		}
		blk := make([]int, B)
		for i := range blk {
			blk[i] = d.MaskTokenID()
		}
		blk[0] = anchor
		trunk, err := d.DraftBlock(m.be, fused, m.DrafterEmbedBlock(blk))
		if err != nil {
			t.Fatalf("DraftBlock: %v", err)
		}
		drafted := make([]int, 0, B-1)
		for _, h := range trunk[1:] {
			drafted = append(drafted, argmax(m.DrafterHeadLogits(h)))
		}

		// Verify. Feed the anchor, then each drafted token, keeping the target's own
		// argmax after every position. accepted = the longest prefix the target agrees
		// with; the first disagreement is where the target's token wins.
		mark := cache.Pos()
		markCtx := len(ctxCat)
		feed(anchor) // the anchor is already committed; this is its real cache entry
		accepted := 0
		next := argmax(logits)
		for i, tok := range drafted {
			if tok != next {
				break
			}
			feed(tok)
			accepted = i + 1
			next = argmax(logits)
		}
		// Roll the cache back to exactly what was accepted: anchor + accepted drafts.
		keep := mark + 1 + accepted
		cache.TruncateTo(keep)
		ctxCat = ctxCat[:markCtx+1+accepted]

		out.rounds++
		out.accepted += accepted
		generated += accepted + 1 // the accepted drafts plus the target's own next token
		anchor = next             // always a TARGET-produced token — this is the losslessness
		if eos[anchor] {
			break
		}
	}
	out.generated = generated
	if out.rounds == 0 {
		t.Fatalf("no rounds ran for %q", prompt)
	}
	return out
}

// dflashMeanStd is a tiny helper kept for the log line's readability.
func dflashMeanStd(xs []int) (mean, std float64) {
	if len(xs) == 0 {
		return 0, 0
	}
	for _, x := range xs {
		mean += float64(x)
	}
	mean /= float64(len(xs))
	for _, x := range xs {
		d := float64(x) - mean
		std += d * d
	}
	return mean, math.Sqrt(std / float64(len(xs)))
}

var _ = fmt.Sprintf
var _ = dflashMeanStd
