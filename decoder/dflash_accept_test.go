//go:build realckpt

// P10 kill-gate 2: measured acceptance for the DFlash block drafter.
//
// Gate 1 (TestDFlash_referenceParity / TestDFlash_targetEndToEnd) proved the forward matches the reference; only then
// is this legitimate to run (the pre-registered order).
//
// WHAT THIS MEASURES, AND WHAT IT DOES NOT. Acceptance is a property of the drafter's distribution against the
// target's, i.e. NUMERICS: it does not depend on the backend, so it is measured here on CPU and transfers to the GPU
// paths unchanged (at equal precision). Wall-clock does NOT transfer and is NOT measured here: this loop verifies the
// block with 16 sequential single-token forwards rather than one batched M=16 pass, because sequential forwards are
// what ForwardCapture exposes and acceptance is indifferent to the difference. Reading a speed number off this harness
// would be wrong; that is gate 3, on the GPU.
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

// overrideOr returns the explicit override v when set, else the fallback — deferred so assetPath (which can Skip) only
// runs when the override is absent.
func overrideOr(v string, fallback func() string) string {
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
	ddir := overrideOr(os.Getenv("GOINFER_DFLASH_DRAFTER"), func() string { return assetPath(t, "GOINFER_DFLASH_F32") })
	tdir := overrideOr(os.Getenv("GOINFER_DFLASH_TARGET"), func() string { return assetPath(t, "GOINFER_QWEN3_4B") })
	tokDir := overrideOr(os.Getenv("GOINFER_DFLASH_TOKENIZER"), func() string { return tdir })

	d, err := LoadDFlashDrafter(ddir)
	if err != nil {
		t.Fatalf("LoadDFlashDrafter: %v", err)
	}
	defer d.Close()
	// Checked BEFORE the target loads: it needs only the drafter's block width, and the
	// target is a 36 GB load on the second pairing. A guard that fires after the expensive
	// step is a guard people route around.
	maxNew := 48
	if os.Getenv("GOINFER_DFLASH_MAXNEW") != "" {
		if v, err := atoiPositive(os.Getenv("GOINFER_DFLASH_MAXNEW")); err == nil {
			maxNew = v
		}
	}
	// maxNew MUST be several blocks, and this is a hard error rather than a note because a short run does not produce a
	// noisy tok/verify, it produces a systematically distorted one, from two independent mechanisms:
	//
	//  1. END-OF-RUN OVERSHOOT. The loop runs while generated < maxNew, so the final round is counted in FULL even though
	//     it overshoots: up to block-1 extra tokens credited against one verify. The smaller maxNew/block is, the less
	//     that overshoot is amortized.
	//  2. AN UNREPRESENTATIVE SLICE OF THE ANSWER. A truncated run measures whatever part of the output it reaches, and
	//     that part is not the workload. The DIRECTION is model-dependent: a coding answer's near-deterministic
	//     boilerplate prefix inflates one pairing, while another pairing's easy region, the code block, arrives AFTER a
	//     prose preamble a short run never gets past.
	//
	// So the bias is not "short runs read high" but "short runs read UNPREDICTABLY", and two pairings can rank in the
	// opposite order at different maxNew. Any two tok/verify numbers being compared must share this setting; prefer 160,
	// which is what gate 2 is recorded at. Measurements and the correction history:
	// docs/code-notes/decoder.md#TestDFlashAcceptance.maxNew.
	effBlock := d.BlockSize()
	if vw := verifyWidth(); vw > 0 && vw < effBlock {
		effBlock = vw
	}
	if minNew := 3 * effBlock; maxNew < minNew {
		t.Fatalf("GOINFER_DFLASH_MAXNEW=%d is under %d (3 blocks of %d): tok/verify is distorted at that "+
			"length by end-of-run overshoot and by sampling an unrepresentative slice of the answer, in a "+
			"MODEL-DEPENDENT direction — the 4B reads 7.11 at 16 vs 6.14 at 160, the 35B 4.77 vs 6.78, and "+
			"the two rank in opposite orders at 16 vs 48. Use 160, or the number is not comparable to any "+
			"other.", maxNew, minNew, d.BlockSize())
	}

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

	type result struct{ rounds, accepted, generated int }
	overall := map[string]*result{}
	suites := []string{"code", "math", "chat"}
	if s := os.Getenv("GOINFER_DFLASH_SUITE"); s != "" {
		suites = []string{s}
	}
	for _, suite := range suites {
		res := &result{}
		overall[suite] = res
		for i, prompt := range dflashSuites[suite] {
			r := dflashRun(t, m, d, tk, prompt, maxNew)
			res.rounds += r.rounds
			res.accepted += r.accepted
			res.generated += r.generated
			// PER-PROMPT PROGRESS: one line per prompt, so a half-hour run on a big target does not look like a hang, with a
			// running estimate of the final figure.
			// WHAT THE TARGET ACTUALLY PRODUCED, not what the suite label says it was asked for. Suite names describe the PROMPT;
			// acceptance is a property of the OUTPUT, and on a reasoning-by-default model the two diverge silently (gpt-oss has no
			// non-thinking harmony form, so its `code` run measured the analysis channel, reasoning prose, and gave a number that
			// looked comparable to the code numbers and was not). Printing a preview makes that visible in the run that produced
			// the number.
			preview, perr := tk.Decode(r.emitted[:min(24, len(r.emitted))])
			if perr != nil {
				preview = "<decode failed: " + perr.Error() + ">"
			}
			t.Logf("  [%s %d/%d] %d rounds, %d tokens, mean accepted %.2f  (running %.2f tok/verify)\n"+
				"      target produced: %q",
				suite, i+1, len(dflashSuites[suite]), r.rounds, r.generated,
				float64(r.accepted)/float64(r.rounds),
				float64(res.generated)/float64(res.rounds), preview)
		}
		tpv := float64(res.generated) / float64(res.rounds)
		// STEADY STATE, reported alongside the raw ratio because the raw one has a third small upward bias on top of the two
		// the maxNew guard covers.
		//
		// `generated` is seeded at 1 per PROMPT (the anchor, which prefill produced and no verify round paid for). Over R rounds
		// with P prompts:
		//     generated = P + R + sum(accepted)
		//     tok/verify = 1 + P/R + mean(accepted)
		// so the raw ratio carries a +P/R term that has nothing to do with the drafter and shrinks as the run lengthens.
		// 1 + mean(accepted) is the prompt-count-independent figure, and is what two pairings should be compared on.
		//
		// The gate below still uses the raw ratio, deliberately: it is the definition every recorded number in docs/spec/08 was
		// measured under, and silently redefining a metric to move a number past its own bar is the move this whole file exists
		// to prevent.
		meanAcc := float64(res.accepted) / float64(res.rounds)
		t.Logf("[quant=%q maxNew=%d vw=%d] %-5s  %2d rounds (%.1f/prompt)  %3d tokens  mean accepted %.2f/%d  => %.2f tok/verify (steady state %.2f)",
			quant, maxNew, verifyWidth(), suite, res.rounds, float64(res.rounds)/float64(len(dflashSuites[suite])),
			res.generated, meanAcc, d.BlockSize()-1, tpv, 1+meanAcc)
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

// noThinkSuffix returns the ids for "<think>\n\n</think>\n\n", what Qwen3's template emits for
// enable_thinking=False. It is resolved through the tokenizer that ships with the target, not a literal id list pinned
// from one pairing: another pairing's vocab puts <think> at a different id (Qwen3.6-35B-A3B: 248068, where 151667 is
// an unrelated token), and four wrong tokens would depress acceptance in a way that looks exactly like "the drafter
// transfers badly to this target". Verify rather than trust: the encode must produce the <think>/</think> ids the
// tokenizer itself reports.
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

// verifyWidth reads GOINFER_DFLASH_VERIFY_WIDTH — how many block positions the target
// verifies per round (anchor + width-1 drafts). 0 or unset means the drafter's full block.
func verifyWidth() int {
	if v := os.Getenv("GOINFER_DFLASH_VERIFY_WIDTH"); v != "" {
		if n, err := atoiPositive(v); err == nil {
			return n
		}
	}
	return 0
}

// skipNoThink reproduces the ORIGINAL (thinking-mode) measurement for comparison.
var skipNoThink = os.Getenv("GOINFER_DFLASH_THINKING") != ""

type dflashRunResult struct {
	rounds, accepted, generated int
	emitted                     []int // every token the TARGET committed, for the content preview
}

// dflashRun greedily generates from one prompt with the DFlash block-verify loop and
// returns how many rounds it took. Lossless by construction: every emitted token is one
// the TARGET's own argmax produced — the drafter only ever proposes, and a rejected
// proposal is rolled out of the cache.
func dflashRun(t *testing.T, m *Model, d *DFlashDrafter, tk *tokenizer.Tokenizer, prompt string, maxNew int) dflashRunResult {
	t.Helper()
	vw := verifyWidth()
	turns := []chat.Turn{{Role: "user", Content: prompt}}
	// THE TARGET'S OWN TEMPLATE, detected, not ChatML assumed. ChatML is right for both Qwen3 pairings and wrong for
	// Gemma-4 (<|turn>/<|channel> markers) and gpt-oss (harmony): feeding a Gemma target ChatML would not error, it
	// would measure the drafter against a prompt format the target never sees (raw vs chat accepted 0/15 vs 10/15).
	//
	// An unrecognized template is a HARD ERROR rather than the library's raw-completion fallback: falling back would
	// produce a number rather than a failure, and a plausible acceptance figure measured off a malformed prompt is the
	// most expensive failure mode this harness has.
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatalf("chat.Detect: %v — refusing the raw-completion fallback, which would measure "+
			"the drafter against a prompt format the target never sees and still print a number", err)
	}
	ids, err := tk.EncodeSegments(tmpl.RenderSegments("", turns), false)
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
	// Qwen3-family only: a target with no <think> in its vocab has no thinking mode to
	// suppress, and appending the suffix would inject literal text into the prompt.
	if !skipNoThink {
		if _, ok := tk.TokenID("<think>"); ok {
			ids = append(ids, noThinkSuffix(t, tk)...)
		}
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
		// VERIFY-WIDTH CAP. The drafter still drafts its full trained block; this bounds how
		// many of those positions the target verifies. Accepted length is CONCAVE in block
		// width — the tail positions rarely land, yet a k-wide batched verify costs
		// W + k*C regardless — so the widest block is not obviously the fastest one, and the
		// doc's own model puts the optimum near 7-8 rather than 16.
		//
		// Capping rather than drafting a narrower block is deliberate: DFlash was trained
		// with a fixed number of mask tokens, so feeding fewer changes the non-causal
		// attention pattern over [ctx‖block] and takes the drafter off-distribution. That is
		// a different experiment. This one holds the draft fixed and varies only the verify.
		if vw > 0 && len(drafted) > vw-1 {
			drafted = drafted[:vw-1]
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
		out.emitted = append(out.emitted, anchor)
		out.emitted = append(out.emitted, drafted[:accepted]...)
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

// BenchmarkDFlashTrunk times ONE block draft, the cost that decides increment 4's architecture. If the CPU trunk is
// cheap relative to a resident GPU target step, the target can go resident while the drafter stays on CPU; if not, the
// drafter has to be ported to the GPU too, and increment 4 is a much bigger build. Measure before building (a CPU draft
// against a GPU target was the wall in Lever 2, not the verify).
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
