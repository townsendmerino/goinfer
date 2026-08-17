//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// dflashLoop runs the full resident block-drafting round loop and returns the tokens it emitted
// plus the round count. It is the composition every measurement so far has stood in for.
//
// LOSSLESS BY CONSTRUCTION, and that is the whole design: every emitted token is one the
// TARGET's own argmax produced. The drafter only ever proposes; a proposal that the target
// disagrees with is discarded along with everything after it, and the target's own token is
// emitted instead. So the output is token-identical to plain greedy decoding whatever the
// drafter does — a broken drafter costs speed, never correctness.
func dflashLoop(t *testing.T, mc *decoder.Model, r *cudaResident, rd *residentDrafter,
	taps []int, prompt []int, maxNew, verifyWidth, maskTok int) (out []int, rounds int, ms float64) {
	t.Helper()
	hidden := rd.geo.Hidden
	blockW := rd.geo.Layers // placeholder; replaced below
	_ = blockW

	t0 := time.Now()
	// --- prefill: seed the target's KV and collect the drafter's context rows ---
	if e := r.SetBatchedCapture(taps); e != nil {
		t.Fatalf("SetBatchedCapture: %v", e)
	}
	embs := make([][]float32, len(prompt))
	for i, id := range prompt {
		embs[i] = mc.EmbedResidentForTest(id)
	}
	anchorIDs, err := r.PrefillLastNArgmax(embs, 0)
	if err != nil {
		t.Fatalf("prefill: %v", err)
	}
	cap0 := r.BatchedCapture()
	pos := len(prompt)
	anchor := anchorIDs[len(anchorIDs)-1]
	out = append(out, anchor)

	// fuse the prompt's tap rows into drafter context
	fuse := func(capt [][]float32, n int) {
		cat := make([][]float32, n)
		for m := 0; m < n; m++ {
			row := make([]float32, 0, len(taps)*hidden)
			for _, tp := range capt {
				row = append(row, tp[m*hidden:(m+1)*hidden]...)
			}
			cat[m] = row
		}
		fused, e := rd.FuseContext(cat)
		if e != nil {
			t.Fatalf("FuseContext: %v", e)
		}
		if e := rd.ExtendContext(fused); e != nil {
			t.Fatalf("ExtendContext: %v", e)
		}
	}
	fuse(cap0, len(prompt))

	// THE MASK TOKEN IS TRAINED, not a placeholder. DFlash learned to see this specific
	// embedding at unfilled block positions; feeding any other id puts the drafter
	// off-distribution and it drafts badly while everything still runs. Measured cost of
	// getting this wrong: 1.77 tok/round against the CPU sweep's 4.97 at the same width.
	maskID := maskTok
	for len(out) < maxNew {
		// --- draft ---
		ids := make([]int, verifyWidth)
		ids[0] = anchor
		for i := 1; i < verifyWidth; i++ {
			ids[i] = maskID
		}
		blockIn := make([][]float32, verifyWidth)
		for i, id := range ids {
			blockIn[i] = mc.EmbedResidentForTest(id)
		}
		trunk, e := rd.DraftBlock(blockIn)
		if e != nil {
			t.Fatalf("DraftBlock: %v", e)
		}
		drafted, e := rd.DraftTokens(trunk[1:])
		if e != nil {
			t.Fatalf("DraftTokens: %v", e)
		}

		// --- verify: the target scores anchor + drafts in ONE batched pass, capturing taps ---
		vin := make([][]float32, 0, 1+len(drafted))
		vin = append(vin, mc.EmbedResidentForTest(anchor))
		for _, id := range drafted {
			vin = append(vin, mc.EmbedResidentForTest(id))
		}
		tgt, e := r.PrefillLastNArgmax(vin, pos)
		if e != nil {
			t.Fatalf("verify: %v", e)
		}
		capt := r.BatchedCapture()
		rounds++

		// --- accept the longest prefix the target agrees with ---
		accepted := 0
		for i, d := range drafted {
			if tgt[i] != d {
				break
			}
			accepted = i + 1
		}
		next := tgt[accepted] // the target's own token at the first disagreement
		for i := 0; i < accepted; i++ {
			out = append(out, drafted[i])
		}
		out = append(out, next)

		// --- commit: target KV keeps anchor + accepted; drafter context takes their taps ---
		pos += 1 + accepted
		fuse(capt, 1+accepted)
		anchor = next
		if len(out) >= maxNew {
			break
		}
	}
	_ = r.SetBatchedCapture(nil)
	return out, rounds, float64(time.Since(t0).Milliseconds())
}

// TestDFlashLoop_lossless is gate 3's correctness half: the resident block-drafting loop must
// emit EXACTLY what plain greedy decoding emits.
//
// This is the property the whole design rests on, and it is checkable without trusting anything
// about the drafter: run the loop, run plain greedy on the same prompt, require the token
// sequences to be identical. A drafter that proposes badly changes the round count and the
// speed; if it changes the OUTPUT, the accept logic is wrong.
func TestDFlashLoop_lossless(t *testing.T) {
	requireHeavyModel(t)
	tgt := os.Getenv("GOINFER_CUDA_MODEL")
	if tgt == "" {
		tgt = os.ExpandEnv("$HOME/models/qwen3-4b")
	}
	ddir := decoder.AssetPathForTest(t, "GOINFER_DFLASH_F32")
	mc, err := decoder.Load(tgt, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	r := mc.ResidentForwardForTest().(*cudaResident)
	dr, err := decoder.LoadDFlashDrafter(ddir)
	if err != nil {
		t.Fatalf("load drafter: %v", err)
	}
	defer dr.Close()
	rd, err := r.AttachDrafter(dr)
	if err != nil {
		t.Fatalf("AttachDrafter: %v", err)
	}
	taps := dr.TargetLayerIDs()
	_, _, _, _, _, _, vocab := mc.Dims()
	prompt := make([]int, 24)
	for i := range prompt {
		prompt[i] = (i*7919 + 101) % (vocab - 1)
	}
	const maxNew = 24

	got, rounds, ms := dflashLoop(t, mc, r, rd, taps, prompt, maxNew, 7, dr.MaskTokenID())
	t.Logf("spec loop: %d tokens in %d rounds, %.0f ms (%.2f tok/round)", len(got), rounds, ms,
		float64(len(got))/float64(rounds))

	// plain greedy on the same prompt, same resident
	want := make([]int, 0, maxNew)
	embs := make([][]float32, len(prompt))
	for i, id := range prompt {
		embs[i] = mc.EmbedResidentForTest(id)
	}
	ids, e := r.PrefillLastNArgmax(embs, 0)
	if e != nil {
		t.Fatalf("greedy prefill: %v", e)
	}
	tok := ids[len(ids)-1]
	want = append(want, tok)
	for p := len(prompt); len(want) < len(got); p++ {
		one, e := r.PrefillLastNArgmax([][]float32{mc.EmbedResidentForTest(tok)}, p)
		if e != nil {
			t.Fatalf("greedy step: %v", e)
		}
		tok = one[0]
		want = append(want, tok)
	}
	if len(got) != len(want) {
		t.Fatalf("spec emitted %d tokens, greedy %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("LOSSLESS VIOLATED at token %d: spec %d, greedy %d\n  spec  %v\n  greedy %v",
				i, got[i], want[i], got, want)
		}
	}
	t.Logf("LOSSLESS: %d tokens identical to plain greedy", len(want))
}

// TestDFlashLoop_gate3 is GATE 3: the end-to-end wall-clock the whole projection has been
// standing in for.
//
// docs/spec/08 projects code 1.52x / math 1.96x at verify widths 7/8, composed from separately
// measured terms — draft 8.82 ms, the batched-head verify curve, a 1.09 ms batched seam,
// acceptance from a 7-width CPU sweep. Every term is measured; the COMPOSITION was arithmetic.
// This runs the real loop against plain greedy on the same resident and divides.
//
// REAL PROMPTS, chat-templated, because acceptance is a property of real text: the lossless gate
// above reads 1.56 tok/round on random ids, which says nothing about anything. The suite is the
// same one the CPU acceptance sweep used, so the numbers are comparable.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_CUDA_MODEL=$HOME/models/qwen3-4b \
//	  go test -tags 'cuda goinfer_testhooks' -run TestDFlashLoop_gate3 -v -timeout 2h
func TestDFlashLoop_gate3(t *testing.T) {
	requireHeavyModel(t)
	tgt := os.Getenv("GOINFER_CUDA_MODEL")
	if tgt == "" {
		tgt = os.ExpandEnv("$HOME/models/qwen3-4b")
	}
	ddir := decoder.AssetPathForTest(t, "GOINFER_DFLASH_F32")
	mc, err := decoder.Load(tgt, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	r := mc.ResidentForwardForTest().(*cudaResident)
	dr, err := decoder.LoadDFlashDrafter(ddir)
	if err != nil {
		t.Fatalf("load drafter: %v", err)
	}
	defer dr.Close()
	rd, err := r.AttachDrafter(dr)
	if err != nil {
		t.Fatalf("AttachDrafter: %v", err)
	}
	taps := dr.TargetLayerIDs()

	tk, err := decoder.LoadTokenizerForTest(tgt)
	if err != nil {
		t.Skipf("tokenizer: %v", err)
	}
	// The same prompts and the same non-thinking template the CPU acceptance sweep used, so
	// tok/round here is comparable to the 4.97 that sweep measured at width 7.
	prompts := map[string][]string{
		"code": {
			"Write a Python function that returns the nth Fibonacci number.",
			"Write a Go function that reverses a slice of ints in place.",
		},
		"math": {
			"What is 17 * 23? Show your working.",
			"A train travels 120 km in 1.5 hours. What is its average speed in km/h?",
		},
	}
	widths := map[string]int{"code": 7, "math": 8}
	const maxNew = 96

	for _, suite := range []string{"code", "math"} {
		w := widths[suite]
		var specMs, greedyMs float64
		var specToks, rounds int
		for _, p := range prompts[suite] {
			ids, e := decoder.EncodeChatForTest(tk, p)
			if e != nil {
				t.Fatalf("encode: %v", e)
			}
			got, rd2, ms := dflashLoop(t, mc, r, rd, taps, ids, maxNew, w, dr.MaskTokenID())
			specMs += ms
			specToks += len(got)
			rounds += rd2
			rd.TruncateContext(0)

			// plain greedy, same resident, same token count — the baseline this must beat.
			t0 := time.Now()
			embs := make([][]float32, len(ids))
			for i, id := range ids {
				embs[i] = mc.EmbedResidentForTest(id)
			}
			out, e := r.PrefillLastNArgmax(embs, 0)
			if e != nil {
				t.Fatalf("greedy prefill: %v", e)
			}
			tok := out[len(out)-1]
			for n, pos := 1, len(ids); n < len(got); n, pos = n+1, pos+1 {
				one, e := r.PrefillLastNArgmax([][]float32{mc.EmbedResidentForTest(tok)}, pos)
				if e != nil {
					t.Fatalf("greedy step: %v", e)
				}
				tok = one[0]
			}
			greedyMs += float64(time.Since(t0).Milliseconds())
		}
		tpr := float64(specToks) / float64(rounds)
		speed := greedyMs / specMs
		t.Logf("%-4s k=%d: %d tokens, %d rounds (%.2f tok/round) | spec %.0f ms vs greedy %.0f ms => %.2fx",
			suite, w, specToks, rounds, tpr, specMs, greedyMs, speed)
	}
}
