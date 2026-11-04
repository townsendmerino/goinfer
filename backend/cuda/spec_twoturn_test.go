//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"os"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// spec_twoturn_test.go is item 30 (docs/prompts/nobara-cuda-spec-trailing-token-2026-09.md §2): the CUDA check of two
// trailing-token fixes whose real drafter and kernels exist only on CUDA —
//
//   - the block drafter (--drafter, CUDA only): at a completed exit (EOS or max_tokens) the resident commit must record the
//     LAST emitted token (the next round's anchor) as forwarded through the target, or a second turn reuses that position
//     as it stood: a rejected draft's K/V, or nothing at max_tokens 1.
//   - the n-gram loop (serve --spec ngram): the trailing token must be forwarded, not only as the next round's seq[0], or a
//     generation ending at max_tokens leaves the cache one token short.
//
// It ports spec_multiturn_test.go's shape (fresh load per arm, turn 1 through the speculative path, turn 2 a strict ChatML
// extension through plain Generate) with three assertions: turn-1 ids equal (losslessness), turn-2 PrefillReused equal AND
// equal to len(prompt1)+len(out1) (the property the fixes are for), turn-2 ids equal. Reverting decoder/blockspec.go and
// decoder/spec_ngram.go to before those fixes makes turn 2 diverge; both runs are recorded in
// docs/measurements/spec-vs-batching-metal-2026-09-27.md's "CUDA check" Update section.

// twoTurnAssets resolves the target checkpoint and checks a tokenizer is actually loadable — shared by both new
// tests, mirroring blockspec_test.go's own asset resolution exactly so a real run exercises the identical
// checkpoint the existing CUDA gates already do. firstTurn/extendTurn each load their own tokenizer instance
// (cheap; no weights) rather than threading one through every call site.
func twoTurnAssets(t *testing.T) (tgt string) {
	t.Helper()
	requireHeavyModel(t)
	tgt = os.Getenv("GOINFER_CUDA_MODEL")
	if tgt == "" {
		tgt = os.ExpandEnv("$HOME/models/qwen3-4b")
	}
	if _, err := decoder.LoadTokenizerForTest(tgt); err != nil {
		t.Skipf("tokenizer: %v", err)
	}
	return tgt
}

// firstTurn encodes turn 1 through the real chat template (handles the family's own thinking/no-think suffix),
// exactly as the existing block-spec CUDA tests do.
func firstTurn(t *testing.T, tgt, prompt string) []int {
	t.Helper()
	tkr, err := decoder.LoadTokenizerForTest(tgt)
	if err != nil {
		t.Skipf("tokenizer: %v", err)
	}
	ids, err := decoder.EncodeChatForTest(tkr, prompt)
	if err != nil {
		t.Fatalf("encode turn 1: %v", err)
	}
	return ids
}

// extendTurn appends a second ChatML user turn (plus the assistant preamble) onto a full first-turn transcript
// (prompt1 + its emitted output), the same raw-segment shape spec_multiturn_test.go uses on Metal.
func extendTurn(t *testing.T, tgt string, transcript []int, userText string) []int {
	t.Helper()
	tkr, err := decoder.LoadTokenizerForTest(tgt)
	if err != nil {
		t.Skipf("tokenizer: %v", err)
	}
	tail, err := tkr.Encode("<|im_end|>\n<|im_start|>user\n"+userText+"<|im_end|>\n<|im_start|>assistant\n", false)
	if err != nil {
		t.Fatalf("encode turn 2 tail: %v", err)
	}
	return append(slices.Clone(transcript), tail...)
}

func firstDiffTwoTurn(a, b []int) int {
	for i := range min(len(a), len(b)) {
		if a[i] != b[i] {
			return i
		}
	}
	if len(a) != len(b) {
		return min(len(a), len(b))
	}
	return -1
}

const (
	twoTurnPrompt1 = "Write a Python function that returns the nth Fibonacci number."
	twoTurnPrompt2 = "Now write the same thing in Go."
)

// twoTurnResult is one arm's two turns: ids and the resident's own PrefillReused for each.
type twoTurnResult struct {
	out1, out2       []int
	reused1, reused2 int
}

// runPlainTwoTurn is the reference arm: two turns through Model.Generate on one fresh CUDA resident. No
// speculation at all — this is what "prompt + every emitted token held in KV" looks like when nothing can get it
// wrong, so it's the baseline every speculative arm below is compared against.
func runPlainTwoTurn(t *testing.T, tgt string, n int) twoTurnResult {
	t.Helper()
	mc, err := decoder.Load(tgt, decoder.Options{Backend: "cuda", Quant: "int4"})
	if err != nil {
		t.Fatalf("load (plain): %v", err)
	}
	defer mc.Close()
	prompt1 := firstTurn(t, tgt, twoTurnPrompt1)
	ch1, g1 := mc.Generate(context.Background(), prompt1, n, decoder.SamplingParams{})
	var out1 []int
	for id := range ch1 {
		out1 = append(out1, id)
	}
	if err := g1.Err(); err != nil {
		t.Fatalf("plain turn 1: %v", err)
	}
	if len(out1) != n {
		t.Fatalf("plain turn 1 emitted %d tokens, want %d (max_tokens exit, not EOS) — this case tests nothing", len(out1), n)
	}
	prompt2 := extendTurn(t, tgt, append(slices.Clone(prompt1), out1...), twoTurnPrompt2)
	ch2, g2 := mc.Generate(context.Background(), prompt2, n, decoder.SamplingParams{})
	var out2 []int
	for id := range ch2 {
		out2 = append(out2, id)
	}
	if err := g2.Err(); err != nil {
		t.Fatalf("plain turn 2: %v", err)
	}
	return twoTurnResult{out1: out1, out2: out2, reused1: g1.PrefillReused, reused2: g2.PrefillReused}
}

// checkTwoTurn asserts the three properties the prompt names, against the plain reference: turn-1 losslessness,
// turn-2 PrefillReused equal to plain's AND to len(prompt1)+len(out1) (the fixed property itself), and turn-2 ids
// equal. wantPrompt1Len is len(prompt1) as encoded by firstTurn — passed in rather than recomputed so a caller
// that already has it (every one here does) can't accidentally encode a second, possibly-nondeterministic copy.
//
// checkReused is false only for the turn2-via-GenerateStream variant: decoder/blockspec.go's GenerateStream never sets
// Generation.PrefillReused (BlockSpec has no separate prefill phase to report reuse for), so it always reads 0 there
// regardless of whether the trailing-token fix is doing its job. That is a reporting gap, not evidence about the fix: the
// turn-2 ids check is what proves the KV is correct for that variant, and it still runs unconditionally.
func checkTwoTurn(t *testing.T, label string, wantPrompt1Len int, plain, got twoTurnResult, checkReused bool) {
	t.Helper()
	if d := firstDiffTwoTurn(plain.out1, got.out1); d >= 0 {
		t.Errorf("%s: turn 1 differs from plain at token %d (plain %d, got %d) — not lossless", label, d, len(plain.out1), len(got.out1))
	}
	wantReused := wantPrompt1Len + len(got.out1)
	if checkReused {
		if got.reused2 != wantReused {
			t.Errorf("%s: turn 2 PrefillReused = %d, want %d (len(prompt1)=%d + len(out1)=%d) — the trailing token was not committed",
				label, got.reused2, wantReused, wantPrompt1Len, len(got.out1))
		}
		if got.reused2 != plain.reused2 {
			t.Errorf("%s: turn 2 PrefillReused = %d, plain's is %d — should be identical", label, got.reused2, plain.reused2)
		}
	}
	if d := firstDiffTwoTurn(plain.out2, got.out2); d >= 0 {
		t.Errorf("%s: turn 2 differs from plain at token %d (plain %d, got %d)", label, d, len(plain.out2), len(got.out2))
	}
	t.Logf("%s: turn1 %d tok (reused %d), turn2 %d tok (reused %d, want %d, checked=%v)", label, len(got.out1), got.reused1, len(got.out2), got.reused2, wantReused, checkReused)
}

// TestBlockSpec_twoTurnsMatchPlain is the block drafter fix's CUDA check: qwen3-4b int4 plus the DFlash block drafter, N in
// {1, 17, 48} (1 is the seed-only exit, no draft round at all, the shape most likely to expose a stale anchor). Two spec
// variants share turn 1 (the drafter's GenerateStream) and differ only in how turn 2 continues: through plain Generate (the
// property serve depends on: a spec turn followed by an ordinary one), and through the drafter's own GenerateStream again
// (spec-into-spec).
func TestBlockSpec_twoTurnsMatchPlain(t *testing.T) {
	tgt := twoTurnAssets(t)
	ddir := decoder.AssetPathForTest(t, "GOINFER_DFLASH_F32")
	prompt1Len := len(firstTurn(t, tgt, twoTurnPrompt1))

	runSpec := func(t *testing.T, n int, turn2ViaSpec bool) twoTurnResult {
		t.Helper()
		dr, err := decoder.LoadDFlashDrafter(ddir) // before the target: its device bytes are priced into the target's plan (withDrafterReserve)
		if err != nil {
			t.Fatalf("load drafter: %v", err)
		}
		defer dr.Close()
		mc, err := decoder.Load(tgt, withDrafterReserve(decoder.Options{Backend: "cuda", Quant: "int4"}, dr))
		if err != nil {
			t.Fatalf("load (spec): %v", err)
		}
		defer mc.Close()
		spec, err := mc.NewBlockSpec(dr, dr.TargetLayerIDs())
		if err != nil {
			t.Fatalf("NewBlockSpec: %v", err)
		}
		prompt1 := firstTurn(t, tgt, twoTurnPrompt1)
		ch1, g1, err := spec.GenerateStream(context.Background(), prompt1, n, decoder.SamplingParams{})
		if err != nil {
			t.Fatalf("spec turn 1: %v", err)
		}
		var out1 []int
		for id := range ch1 {
			out1 = append(out1, id)
		}
		if err := g1.Err(); err != nil {
			t.Fatalf("spec turn 1: %v", err)
		}
		if len(out1) != n {
			t.Fatalf("spec turn 1 emitted %d tokens, want %d (max_tokens exit, not EOS) — this case tests nothing", len(out1), n)
		}
		prompt2 := extendTurn(t, tgt, append(slices.Clone(prompt1), out1...), twoTurnPrompt2)
		var out2 []int
		var reused2 int
		if turn2ViaSpec {
			ch2, g2, err := spec.GenerateStream(context.Background(), prompt2, n, decoder.SamplingParams{})
			if err != nil {
				t.Fatalf("spec turn 2: %v", err)
			}
			for id := range ch2 {
				out2 = append(out2, id)
			}
			if err := g2.Err(); err != nil {
				t.Fatalf("spec turn 2: %v", err)
			}
			reused2 = g2.PrefillReused
		} else {
			ch2, g2 := mc.Generate(context.Background(), prompt2, n, decoder.SamplingParams{})
			for id := range ch2 {
				out2 = append(out2, id)
			}
			if err := g2.Err(); err != nil {
				t.Fatalf("plain-continuation turn 2: %v", err)
			}
			reused2 = g2.PrefillReused
		}
		return twoTurnResult{out1: out1, out2: out2, reused1: g1.PrefillReused, reused2: reused2}
	}

	for _, n := range []int{1, 17, 48} {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			plain := runPlainTwoTurn(t, tgt, n)
			t.Run("turn2-via-Generate", func(t *testing.T) {
				got := runSpec(t, n, false)
				checkTwoTurn(t, "block-spec, turn2=Generate", prompt1Len, plain, got, true)
			})
			t.Run("turn2-via-GenerateStream", func(t *testing.T) {
				got := runSpec(t, n, true)
				checkTwoTurn(t, "block-spec, turn2=GenerateStream", prompt1Len, plain, got, false)
			})
		})
	}
}

// TestNgramSpec_twoTurnsMatchPlain is the n-gram fix's CUDA check, the same shape on the target alone (no drafter object to
// attach: decoder.NgramDrafter{} is stateless) via GenerateNgramSpeculativeAdaptive, serve's own resident call for
// `--spec ngram`.
func TestNgramSpec_twoTurnsMatchPlain(t *testing.T) {
	tgt := twoTurnAssets(t)
	prompt1Len := len(firstTurn(t, tgt, twoTurnPrompt1))

	runNgram := func(t *testing.T, n int) twoTurnResult {
		t.Helper()
		mc, err := decoder.Load(tgt, decoder.Options{Backend: "cuda", Quant: "int4"})
		if err != nil {
			t.Fatalf("load (ngram): %v", err)
		}
		defer mc.Close()
		prompt1 := firstTurn(t, tgt, twoTurnPrompt1)
		ch1, g1, err := mc.GenerateNgramSpeculativeAdaptive(context.Background(), prompt1, n,
			&decoder.NgramDrafter{}, &decoder.AdaptiveDepth{MaxDraft: 8}, decoder.SamplingParams{})
		if err != nil {
			t.Fatalf("ngram turn 1: %v", err)
		}
		var out1 []int
		for id := range ch1 {
			out1 = append(out1, id)
		}
		if err := g1.Err(); err != nil {
			t.Fatalf("ngram turn 1: %v", err)
		}
		if len(out1) != n {
			t.Fatalf("ngram turn 1 emitted %d tokens, want %d (max_tokens exit, not EOS) — this case tests nothing", len(out1), n)
		}
		prompt2 := extendTurn(t, tgt, append(slices.Clone(prompt1), out1...), twoTurnPrompt2)
		ch2, g2 := mc.Generate(context.Background(), prompt2, n, decoder.SamplingParams{})
		var out2 []int
		for id := range ch2 {
			out2 = append(out2, id)
		}
		if err := g2.Err(); err != nil {
			t.Fatalf("ngram-continuation turn 2: %v", err)
		}
		return twoTurnResult{out1: out1, out2: out2, reused1: g1.PrefillReused, reused2: g2.PrefillReused}
	}

	for _, n := range []int{1, 17, 48} {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			plain := runPlainTwoTurn(t, tgt, n)
			got := runNgram(t, n)
			checkTwoTurn(t, "n-gram spec", prompt1Len, plain, got, true)
		})
	}
}
