//go:build darwin

package metal

import (
	"context"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestMetalThetaAB — does wiring the MEASURED Theta actually make Metal faster?
//
// Theta was reachable only as 0.5 on every backend, because AdaptiveDepth's
// domain was [0,1) and Metal measures 1.006-1.048. Under 0.5 the controller
// drafts; under the measured value it declines to draft, because a Metal verify
// node costs a full target step (ForwardN is a loop of single-token Forwards, so
// T(n) = n*T(1) — measured linear to n=16).
//
// That predicts the speculative path under Theta=0.5 is SLOWER than not
// speculating at all on Metal, and that the wired default recovers it by
// declining. This measures that rather than asserting it. Three arms, one
// prompt, interleaved:
//
//	off        plain Generate, no speculation — the do-nothing arm, which is
//	           the whole point: "beats every configuration" means nothing if
//	           off wins, and here off is EXPECTED to win against Theta=0.5
//	theta=0.5  the shipped-until-now behaviour, forced explicitly
//	wired      Theta unset, so verifyTheta() supplies the measured 1.02
//
// The assertion is deliberately weak in one direction and strong in the other:
// `wired` must not be materially slower than `off` (it should be within noise of
// it, since it declines to draft), and it must beat `theta=0.5`. Nothing here
// claims speculation is bad in general — it claims this backend's verify is not
// batched, which is exactly what item (2) would change.
func TestMetalThetaAB(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" || os.Getenv("GOINFER_THETA_AB") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 GOINFER_THETA_AB=1")
	}
	if testing.Short() {
		t.Skip("timing measurement: skipped in -short")
	}
	tpath := os.Getenv("GOINFER_SPEC_TARGET")
	if tpath == "" {
		home, _ := os.UserHomeDir()
		tpath = filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(tpath); err != nil {
		t.Skipf("missing model %s: %v", tpath, err)
	}
	target, err := decoder.Load(tpath, decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatalf("load target: %v", err)
	}
	defer target.Close()
	if !target.ResidentActive() {
		t.Skip("target not Metal-resident")
	}
	tk, err := tokenizer.LoadGGUF(tpath)
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	allToks, err := tk.Encode(readRepoCorpus(t), true)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if len(allToks) < 256 {
		t.Fatalf("corpus too short: %d tokens", len(allToks))
	}
	prompt := allToks[:256]

	ctx := context.Background()
	greedy := decoder.SamplingParams{Temperature: 0}
	const n = 48
	const reps = 3

	drain := func(ch <-chan int) int {
		c := 0
		for range ch {
			c++
		}
		return c
	}
	runOff := func() time.Duration {
		t0 := time.Now()
		ch, _ := target.Generate(ctx, prompt, n, greedy)
		drain(ch)
		return time.Since(t0)
	}
	runSpec := func(theta float64) func() time.Duration {
		return func() time.Duration {
			ad := &decoder.AdaptiveDepth{MaxDraft: 8}
			ad.Theta = theta // 0 => unset => verifyTheta() supplies the measured value
			t0 := time.Now()
			ch, _, err := target.GenerateNgramSpeculativeAdaptive(ctx, prompt, n, &decoder.NgramDrafter{}, ad, greedy)
			if err != nil {
				t.Fatalf("spec generate: %v", err)
			}
			drain(ch)
			return time.Since(t0)
		}
	}
	med := func(f func() time.Duration) float64 {
		f() // warm, discarded
		ms := make([]float64, 0, reps)
		for range reps {
			ms = append(ms, float64(f().Microseconds())/1000)
		}
		sort.Float64s(ms)
		return ms[len(ms)/2]
	}

	// Interleave the arms rather than blocking them, so drift cannot line up
	// with one of them.
	offMs := med(runOff)
	oldMs := med(runSpec(0.5))
	newMs := med(runSpec(0))
	offMs2 := med(runOff)

	t.Logf("Metal Theta A/B, prompt=%d gen=%d, median of %d:", len(prompt), n, reps)
	t.Logf("  off (no speculation)      %8.1f ms   %8.1f ms (repeat)", offMs, offMs2)
	t.Logf("  spec Theta=0.5 (was)      %8.1f ms   %.2fx vs off", oldMs, oldMs/offMs)
	t.Logf("  spec Theta wired (1.02)   %8.1f ms   %.2fx vs off   %.2fx vs Theta=0.5", newMs, newMs/offMs, oldMs/newMs)

	if newMs > oldMs {
		t.Fatalf("the wired Theta is SLOWER than the 0.5 it replaces (%.1f ms vs %.1f ms) — "+
			"the premise that Metal's verify node costs a full step does not hold here", newMs, oldMs)
	}
	// Declining to draft should land within noise of not speculating at all.
	if newMs > offMs*1.15 {
		t.Fatalf("wired Theta is %.2fx of plain generate (%.1f vs %.1f ms) — it is still paying "+
			"speculative overhead it should have declined", newMs/offMs, newMs, offMs)
	}
}
