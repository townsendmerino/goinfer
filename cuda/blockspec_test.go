//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGenerateBlockSpec_production gates the PRODUCTION loop — decoder.GenerateBlockSpec driving
// the CUDA backend through the ResidentDrafterHost / ResidentBlockDrafter interfaces.
//
// The earlier loop gate lived in this package and reached into cudaResident directly. This one
// goes through the interfaces a serving path would use, which is the difference between "the
// pieces work" and "the API works". It checks the same two things:
//
//	LOSSLESS — identical tokens to plain greedy. The property the whole design rests on, and
//	the one that holds however badly the drafter performs.
//	FASTER — the speedup, so a regression in the wiring shows up as a number rather than as a
//	still-correct slowdown nobody notices.
func TestGenerateBlockSpec_production(t *testing.T) {
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
	if !mc.BlockSpecCapable() {
		t.Fatal("BlockSpecCapable() = false on a cuda resident — the host interface is not wired")
	}
	dr, err := decoder.LoadDFlashDrafter(ddir)
	if err != nil {
		t.Fatalf("load drafter: %v", err)
	}
	defer dr.Close()

	tk, err := decoder.LoadTokenizerForTest(tgt)
	if err != nil {
		t.Skipf("tokenizer: %v", err)
	}
	// GOINFER_TEST_PROMPT selects the workload. Chat is the case the acceptance guard EXISTS
	// for: unguarded it measures 0.61x (1.96 accepted/round against a ~3.0 break-even), and
	// "the guard makes that safe" has so far been an inference from a different losing case
	// (thinking mode) rather than a measurement of this one.
	promptText := "Write a Python function that returns the nth Fibonacci number."
	if v := os.Getenv("GOINFER_TEST_PROMPT"); v != "" {
		promptText = v
	}
	prompt, err := decoder.EncodeChatForTest(tk, promptText)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	maxNew := 96
	if v := os.Getenv("GOINFER_TEST_MAXNEW"); v != "" {
		if n, e := strconv.Atoi(v); e == nil {
			maxNew = n
		}
	}

	// Attach ONCE — the weight upload is a per-process cost, not a per-request one. Timing the
	// attach inside the generation is what made the first version of this path measure 0.17x.
	spec, err := mc.NewBlockSpec(dr, dr.TargetLayerIDs())
	if err != nil {
		t.Fatalf("NewBlockSpec: %v", err)
	}
	t0 := time.Now()
	got, rounds, err := spec.Generate(prompt, decoder.BlockSpecOptions{MaxTokens: maxNew})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	specMs := float64(time.Since(t0).Milliseconds())

	// THE BASELINE IS Model.Generate — the path a server actually takes.
	//
	// An earlier version used a PrefillLastNArgmax(M=1) loop, which downloads the full 608 KB
	// logit row per token; Generate uses launchToken with the GPU argmax fast-path (a 4-byte
	// readback). That made the baseline slower than production and flattered every speedup
	// measured against it. Comparing against anything but the real path is measuring the wrong
	// thing.
	t1 := time.Now()
	ch, gen := mc.Generate(context.Background(), prompt, len(got), decoder.SamplingParams{})
	var want []int
	for id := range ch {
		want = append(want, id)
	}
	if e := gen.Err(); e != nil {
		t.Fatalf("greedy: %v", e)
	}
	greedyMs := float64(time.Since(t1).Milliseconds())

	if len(got) != len(want) {
		t.Fatalf("spec %d tokens, greedy %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("LOSSLESS VIOLATED at %d: spec %d, greedy %d", i, got[i], want[i])
		}
	}
	t.Logf("LOSSLESS: %d tokens identical to plain greedy", len(want))
	t.Logf("%d rounds (%.2f tok/round) | spec %.0f ms vs greedy %.0f ms => %.2fx",
		rounds, float64(len(got))/float64(rounds), specMs, greedyMs, greedyMs/specMs)
	if greedyMs/specMs < 1.2 {
		t.Errorf("production path only %.2fx — the wiring lost the speedup the loop measures",
			greedyMs/specMs)
	}
}

// TestBlockSpecStream gates the serving-shaped entry point: tokens arrive on a channel, in
// order, identical to what Generate returns, and cancellation stops the loop.
//
// A server consumes this, not Generate — so the streaming wrapper is where a wiring bug would
// live, and it is worth its own gate rather than trusting that it wraps the loop correctly.
func TestBlockSpecStream(t *testing.T) {
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
	dr, err := decoder.LoadDFlashDrafter(ddir)
	if err != nil {
		t.Fatalf("drafter: %v", err)
	}
	defer dr.Close()
	spec, err := mc.NewBlockSpec(dr, dr.TargetLayerIDs())
	if err != nil {
		t.Fatalf("NewBlockSpec: %v", err)
	}
	tk, err := decoder.LoadTokenizerForTest(tgt)
	if err != nil {
		t.Skipf("tokenizer: %v", err)
	}
	prompt, err := decoder.EncodeChatForTest(tk, "Write a Go function that reverses a slice of ints in place.")
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	const maxNew = 48

	// sampling must be refused, not silently ignored: acceptance compares against the target's
	// ARGMAX, so a temperature would break the losslessness the whole design rests on.
	if _, _, e := spec.GenerateStream(context.Background(), prompt, maxNew,
		decoder.SamplingParams{Temperature: 0.7}); e == nil {
		t.Error("GenerateStream accepted a temperature — greedy-only must be refused, not ignored")
	}

	ch, g, err := spec.GenerateStream(context.Background(), prompt, maxNew, decoder.SamplingParams{})
	if err != nil {
		t.Fatalf("GenerateStream: %v", err)
	}
	var streamed []int
	for id := range ch {
		streamed = append(streamed, id)
	}
	if g.Err() != nil {
		t.Fatalf("generation: %v", g.Err())
	}
	direct, rounds, err := spec.Generate(prompt, decoder.BlockSpecOptions{MaxTokens: maxNew})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(streamed) != len(direct) {
		t.Fatalf("streamed %d tokens, direct %d", len(streamed), len(direct))
	}
	for i := range direct {
		if streamed[i] != direct[i] {
			t.Fatalf("token %d: streamed %d, direct %d", i, streamed[i], direct[i])
		}
	}
	t.Logf("stream == direct: %d tokens, %d rounds, spec stats rounds=%d accepted=%d",
		len(streamed), rounds, g.Spec.Rounds, g.Spec.Accepted)

	// cancellation must stop the loop rather than run to completion
	cctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ch2, _, err := spec.GenerateStream(cctx, prompt, maxNew, decoder.SamplingParams{})
	if err != nil {
		t.Fatalf("GenerateStream(cancel): %v", err)
	}
	n := 0
	for range ch2 {
		n++
		if n == 3 {
			cancel()
		}
	}
	if n >= len(direct) {
		t.Errorf("cancellation did not stop the loop: got %d tokens of %d", n, len(direct))
	}
	t.Logf("cancelled after 3 tokens, stream closed at %d (full run is %d)", n, len(direct))
}
