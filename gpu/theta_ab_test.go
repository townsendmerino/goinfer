//go:build gpu

package gpu

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

// TestWebGPUThetaAB — does wiring the MEASURED Theta actually make WebGPU speculative decode
// faster (or at least no worse)?
//
// Theta was reachable only as 0.5 on this backend (P22, docs/queue-performance.md): WebGPU's
// residentDecoder implemented neither VerifyPathReporter nor PrefillPathReporter, so
// decoder.verifyTheta() fell through to the unmeasured default. Theta actually measures
// 0.978-1.028 (gpu/theta_probe_test.go) — ForwardN's single-submit structure does not make the
// marginal row cheap, it just removes Go-side dispatch overhead between rows. Under 0.5 the
// controller drafts several nodes deep; under the measured value it should decline almost
// entirely, because a WebGPU verify node costs very close to a full target step.
//
// That predicts the speculative path under Theta=0.5 is SLOWER than not speculating at all on
// this backend, and that VerifyPath's fix recovers it by declining. This measures that rather
// than asserting it. Three arms, one prompt, interleaved (mirrors metal/theta_ab_test.go exactly,
// same reasoning, same backend-agnostic decoder API):
//
//	off        plain Generate, no speculation — the do-nothing arm, which is the whole point:
//	           "beats every configuration" means nothing if off wins, and here off is EXPECTED
//	           to win against Theta=0.5
//	theta=0.5  the shipped-until-now behaviour, forced explicitly
//	wired      Theta unset, so verifyTheta() supplies the measured ~1.02 via VerifyPath
//
// The assertion is deliberately weak in one direction and strong in the other: `wired` must not
// be materially slower than `off` (it should be within noise of it, since it declines to draft),
// and it must beat `theta=0.5`.
func TestWebGPUThetaAB(t *testing.T) {
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
	newOrSkipHW(t).Close() // real-HW gate — a software adapter's timings would not answer P22
	// Quant int8int8, not int4: M-09 (decoder/spec_verify_guard.go) refuses ALL speculative
	// decoding on a staged webgpu-int4 model outright (M=1 decode and M>1 verify are two
	// different kernels with no measured tolerance) — a correctness guard this test must not
	// route around. int8int8 is not covered by that guard and still resides on WebGPU.
	target, err := decoder.Load(tpath, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load target: %v", err)
	}
	defer target.Close()
	if !target.ResidentActive() {
		t.Skip("target not WebGPU-resident")
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

	// Interleave the arms rather than blocking them, so drift cannot line up with one of them.
	offMs := med(runOff)
	oldMs := med(runSpec(0.5))
	newMs := med(runSpec(0))
	offMs2 := med(runOff)

	t.Logf("WebGPU Theta A/B, prompt=%d gen=%d, median of %d:", len(prompt), n, reps)
	t.Logf("  off (no speculation)      %8.1f ms   %8.1f ms (repeat)", offMs, offMs2)
	t.Logf("  spec Theta=0.5 (was)      %8.1f ms   %.2fx vs off", oldMs, oldMs/offMs)
	t.Logf("  spec Theta wired (1.02)   %8.1f ms   %.2fx vs off   %.2fx vs Theta=0.5", newMs, newMs/offMs, oldMs/newMs)

	if newMs > oldMs {
		t.Fatalf("the wired Theta is SLOWER than the 0.5 it replaces (%.1f ms vs %.1f ms) — "+
			"the premise that WebGPU's verify node costs a full step does not hold here", newMs, oldMs)
	}
	// Declining to draft should land within noise of not speculating at all.
	if newMs > offMs*1.15 {
		t.Fatalf("wired Theta is %.2fx of plain generate (%.1f vs %.1f ms) — it is still paying "+
			"speculative overhead it should have declined", newMs/offMs, newMs, offMs)
	}
}

// readRepoCorpus concatenates real repository source files. Real code, not a hand-written
// repetition-heavy fixture: a synthetic workload runs several times the copy density of real
// code, which flatters a drafter and hides a length-dependent cost behind a too-short prompt.
func readRepoCorpus(t *testing.T) string {
	t.Helper()
	// Files chosen only for being real, sizeable, and stable — not for their content.
	rel := []string{
		"../decoder/spec_ngram.go",
		"../decoder/model.go",
		"../decoder/attention.go",
	}
	var buf []byte
	for _, r := range rel {
		b, err := os.ReadFile(r)
		if err != nil {
			t.Fatalf("read corpus file %s: %v", r, err)
		}
		buf = append(buf, b...)
		buf = append(buf, '\n')
	}
	return string(buf)
}
