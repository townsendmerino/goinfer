//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestR14DrafterArgmaxProfile is R14's step 0 (docs/tasks/red-october.md): what does the spec
// loop's sync → M×vocab D2H → host-argmax tail cost, at a real block width on a real model, as a
// share of the spec round it sits in? Both sites are profiled — the drafter's DraftTokens and the
// verify's batchedHeadArgmax, which has the identical shape — because a round pays both.
//
// Pre-registered rule: docs/measurements/r14-drafter-argmax-2026-09-22.md.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestR14DrafterArgmaxProfile -v -timeout 30m
func TestR14DrafterArgmaxProfile(t *testing.T) {
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
	// The gate-3 suites and widths, so the round shape is the one the shipped numbers describe.
	suites := []struct {
		name    string
		w       int
		prompts []string
	}{
		{"code", 7, []string{
			"Write a Python function that returns the nth Fibonacci number.",
			"Write a Go function that reverses a slice of ints in place.",
		}},
		{"math", 8, []string{
			"What is 17 * 23? Show your working.",
			"A train travels 120 km in 1.5 hours. What is its average speed in km/h?",
		}},
	}
	const maxNew = 96
	_, _, _, _, _, _, vocab := mc.Dims()

	// Warm-up: one short loop, discarded (first-touch JIT and buffer growth are not the tail's cost).
	{
		ids, e := decoder.EncodeChatForTest(tk, suites[0].prompts[0])
		if e != nil {
			t.Fatalf("encode: %v", e)
		}
		dflashLoop(t, mc, r, rd, taps, ids, 16, suites[0].w, dr.MaskTokenID())
		rd.TruncateContext(0)
	}

	for _, s := range suites {
		r.SetHeadArgProfForTest(true)
		var specMs float64
		var rounds, toks int
		for _, p := range s.prompts {
			ids, e := decoder.EncodeChatForTest(tk, p)
			if e != nil {
				t.Fatalf("encode: %v", e)
			}
			got, rd2, ms := dflashLoop(t, mc, r, rd, taps, ids, maxNew, s.w, dr.MaskTokenID())
			specMs += ms
			rounds += rd2
			toks += len(got)
			rd.TruncateContext(0)
		}
		sync, dl, host, calls, rows := r.HeadArgProfForTest()
		r.SetHeadArgProfForTest(false)
		tail := sync + dl + host
		wall := time.Duration(specMs * float64(time.Millisecond))
		t.Logf("%-4s w=%d: %d tokens, %d rounds, spec wall %.0f ms (%.2f ms/round)", s.name, s.w, toks, rounds, specMs, specMs/float64(rounds))
		t.Logf("  head→argmax tails: %d calls (%d rows, %.1f rows/call, vocab %d): sync %s | D2H %s | host argmax %s | total %s",
			calls, rows, float64(rows)/float64(calls), vocab,
			sync.Round(time.Microsecond), dl.Round(time.Microsecond), host.Round(time.Microsecond), tail.Round(time.Microsecond))
		t.Logf("  per round: sync %.3f ms | D2H %.3f ms | host %.3f ms | D2H+host %.3f ms = %.1f%% of the round (sync %.1f%%)",
			ms(sync)/float64(rounds), ms(dl)/float64(rounds), ms(host)/float64(rounds), ms(dl+host)/float64(rounds),
			100*float64(dl+host)/float64(wall), 100*float64(sync)/float64(wall))
		t.Logf("  D2H bandwidth %.2f GB/s; host argmax %.1f ns/element",
			float64(rows*vocab*4)/dl.Seconds()/1e9, float64(host.Nanoseconds())/float64(rows*vocab))
	}
}

func ms(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }
