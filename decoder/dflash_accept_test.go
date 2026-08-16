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

// envOr returns v when set, else the fallback — deferred so assetPath (which can Skip) only
// runs when the override is absent.
func envOr(v string, fallback func() string) string {
	if v != "" {
		return v
	}
	return fallback()
}

// TestDFlashAcceptance runs the DFlash block-verify loop over each suite and reports
// tok/verify. The bar (docs/spec/08 kill-gate 2) is >= 3.0 on at least one suite.
func TestDFlashAcceptance(t *testing.T) {
	requireHeavyModel(t)
	// PAIRING-PARAMETERIZED. The default is the Qwen3-4B pairing all the recorded numbers
	// were measured on; the overrides let the SAME harness measure a second pairing without
	// forking it, which is what keeps two acceptance numbers comparable.
	//
	// GOINFER_DFLASH_DRAFTER / _TARGET / _TOKENIZER are PATHS, not asset names — the second
	// pairing's target is a 36 GB .gguf whose tokenizer lives in the separate safetensors
	// directory, a split the asset registry has no entry shape for.
	ddir := envOr(os.Getenv("GOINFER_DFLASH_DRAFTER"), func() string { return assetPath(t, "GOINFER_DFLASH_F32") })
	tdir := envOr(os.Getenv("GOINFER_DFLASH_TARGET"), func() string { return assetPath(t, "GOINFER_QWEN3_4B") })
	tokDir := envOr(os.Getenv("GOINFER_DFLASH_TOKENIZER"), func() string { return tdir })

	d, err := LoadDFlashDrafter(ddir)
	if err != nil {
		t.Fatalf("LoadDFlashDrafter: %v", err)
	}
	defer d.Close()
	// int8 target: this is the precision the resident GPU paths would actually run, and
	// an f32 4B forward per verify position makes the sweep hours instead of minutes.
	// TestDFlash_targetEndToEnd already pins the f32 numerics against the reference.
	quant := "int8int8"
	if q := os.Getenv("GOINFER_DFLASH_QUANT"); q != "" {
		quant = q // "" selects f32 — the attribution knob for the int8-vs-bf16 question
	}
	if quant == "f32" {
		quant = ""
	}
	m, err := Load(tdir, Options{Quant: quant})
	if err != nil {
		t.Fatalf("Load(%s): %v", tdir, err)
	}
	defer m.Close()

	tk, err := tokenizer.Load(tokDir)
	if err != nil {
		t.Fatalf("tokenizer.Load(%s): %v", tokDir, err)
	}
	if m.w.arch.HiddenDim != d.hidden {
		t.Fatalf("target hidden %d != drafter hidden %d — wrong pairing", m.w.arch.HiddenDim, d.hidden)
	}
	for _, l := range d.TargetLayerIDs() {
		if l >= m.w.arch.NumLayers {
			t.Fatalf("drafter taps layer %d but the target has %d — wrong pairing", l, m.w.arch.NumLayers)
		}
	}
	t.Logf("pairing: drafter=%s target=%s (hidden %d, %d target layers, %d taps, block %d)",
		ddir, tdir, d.hidden, m.w.arch.NumLayers, len(d.TargetLayerIDs()), d.BlockSize())

	maxNew := 48
	if os.Getenv("GOINFER_DFLASH_MAXNEW") != "" {
		if v, err := atoiPositive(os.Getenv("GOINFER_DFLASH_MAXNEW")); err == nil {
			maxNew = v
		}
	}

	type result struct{ rounds, accepted, generated int }
	overall := map[string]*result{}
	suites := []string{"code", "math", "chat"}
	if s := os.Getenv("GOINFER_DFLASH_SUITE"); s != "" {
		suites = []string{s}
	}
	for _, suite := range suites {
		res := &result{}
		overall[suite] = res
		for _, prompt := range dflashSuites[suite] {
			r := dflashRun(t, m, d, tk, prompt, maxNew)
			res.rounds += r.rounds
			res.accepted += r.accepted
			res.generated += r.generated
		}
		tpv := float64(res.generated) / float64(res.rounds)
		t.Logf("[quant=%q] %-5s  %2d rounds  %3d tokens  mean accepted %.2f/%d  => %.2f tok/verify",
			quant, suite, res.rounds, res.generated, float64(res.accepted)/float64(res.rounds),
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

// noThinkSuffix returns the ids for "<think>\n\n</think>\n\n" — what Qwen3's template emits
// for enable_thinking=False.
//
// THIS USED TO BE A LITERAL []int{151667, 271, 151668, 271}, pinned from Qwen3-4B, and that was
// a bug waiting for the second pairing. Qwen3.6-35B-A3B has a 248320-token vocab in which
// <think> is 248068 and 151667 is an unrelated token — so the literal would have quietly fed
// the 35B four wrong tokens, depressing acceptance in a way that looks exactly like "the
// drafter transfers badly to this target". Resolve it through the tokenizer that ships with
// the target, and verify rather than trust: the encode must produce the <think>/</think> ids
// the tokenizer itself reports.
func noThinkSuffix(t *testing.T, tk *tokenizer.Tokenizer) []int {
	t.Helper()
	ids, err := tk.Encode("<think>\n\n</think>\n\n", false)
	if err != nil {
		t.Fatalf("encode no-think suffix: %v", err)
	}
	open, oOK := tk.TokenID("<think>")
	close, cOK := tk.TokenID("</think>")
	if !oOK || !cOK {
		t.Fatalf("target tokenizer has no <think>/</think> — this suite assumes a Qwen3-style template")
	}
	if len(ids) != 4 || ids[0] != open || ids[2] != close {
		t.Fatalf("no-think suffix encoded to %v; want [%d _ %d _] — the template assumption does not hold for this target",
			ids, open, close)
	}
	return ids
}

// skipNoThink reproduces the ORIGINAL (thinking-mode) measurement for comparison.
var skipNoThink = os.Getenv("GOINFER_DFLASH_THINKING") != ""

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
	// NON-THINKING MODE, and it is not optional. Qwen3's own template with
	// enable_thinking=False appends "<think>\n\n</think>\n\n" after the assistant tag;
	// chat.ChatML() stops at the tag, which leaves Qwen3-4B in THINKING mode. The DFlash
	// drafter was trained on non-thinking output (DeepSpec README: "generated by its
	// corresponding target model in non-thinking mode", and it warns explicitly about
	// thinking-mode targets), so omitting this measures the drafter against a
	// distribution it never saw. Cost, measured: code 2.90 -> see the doc.
	// Verified id-exact: ChatML ids + these four == HF apply_chat_template's ids.
	if !skipNoThink {
		ids = append(ids, noThinkSuffix(t, tk)...)
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

// BenchmarkDFlashTrunk times ONE block draft — the cost that decides increment 4's
// architecture. If the CPU trunk is cheap relative to a resident GPU target step, the
// target can go resident while the drafter stays on CPU. If it is not, the drafter has to
// be ported to the GPU too, and increment 4 is a much bigger build.
//
// This is the question Lever 2 already answered once the hard way: the DRAFT was the wall,
// not the verify — a CPU draft against a GPU target measured 0.11×. Measure before
// building, not after.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags realckpt ./decoder/ -run '^$' -bench DFlashTrunk -benchtime 10x
func BenchmarkDFlashTrunk(b *testing.B) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		b.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	ddir, err := lookupAsset("GOINFER_DFLASH_F32")
	if err != nil {
		b.Skip(err)
	}
	d, err := LoadDFlashDrafter(ddir)
	if err != nil {
		b.Fatalf("load: %v", err)
	}
	defer d.Close()
	be := &cpuBackend{}

	for _, ctxLen := range []int{64, 512, 2048} {
		b.Run(fmt.Sprintf("ctx%d", ctxLen), func(b *testing.B) {
			fused := make([][]float32, ctxLen)
			for i := range fused {
				fused[i] = make([]float32, d.hidden)
				for j := range fused[i] {
					fused[i][j] = float32((i*31+j*7)%97) / 97
				}
			}
			blockIn := make([][]float32, d.BlockSize())
			for i := range blockIn {
				blockIn[i] = make([]float32, d.hidden)
				for j := range blockIn[i] {
					blockIn[i][j] = float32((i*13+j*3)%89) / 89
				}
			}
			// The per-ROUND cost a generation loop pays: the context is projected once
			// when its positions commit (ExtendContext, amortized over the whole run),
			// and each round only re-runs the block. Timing DraftBlock instead would
			// re-project the whole context every round and measure work the reference
			// implementations never do.
			cctx := d.NewContext()
			d.ExtendContext(be, cctx, fused)
			b.ResetTimer()
			for range b.N {
				if _, err := d.DraftBlockCtx(be, cctx, blockIn); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(b.Elapsed().Milliseconds())/float64(b.N), "ms/block")
		})
	}
}
