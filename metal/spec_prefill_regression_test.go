//go:build darwin

package metal

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// TestMetalSpecPrefillRegression is the Metal half of the 2026-08-31 speculative-prefill
// regression that was measured on CUDA and asserted on Metal by interface only.
//
// THE DEFECT: decoder/model.go's generateInto ingests a prompt through the batched Prefiller
// seam (m.resident.(Prefiller).PrefillLast); decoder/spec_ngram.go's genNgramInto instead
// loops target.resident.Forward(...) one token at a time. Speculative decode therefore pays a
// per-prompt-token cost that plain generation does not, and the gap grows LINEARLY in prompt
// length — on CUDA, 2.66 ms/prompt-token, R² 0.9977, which at 839 tokens made the speculative
// path 4.1x SLOWER than not speculating at all.
//
// THE GATE IS THE SLOPE, NOT THE RATIO. The spec-vs-off ratio moves with draft acceptance,
// which moves with the corpus, so it flaps; the slope of (spec - off) against prompt length is
// the defect's signature and is nearly acceptance-independent. Bar: 0.50 ms/prompt-token
// (CUDA measured 2.66 with the bug, 0.12 without).
//
// WHY THIS ASSERTS RATHER THAN LOGS: gpu/spec_ngram_resident_test.go measures the right
// quantity and only t.Logf's it — its own header says "speedup is logged per workload" — so a
// 0.3x printed and failed nothing for six weeks. The slope check below is t.Fatalf.
//
// WHY THE CORPUS IS READ FROM DISK: specWorkloads/ngramWorkloads are hand-written and
// deliberately copy-heavy (4-7x the copy density of real code) and only 36-74 tokens long,
// which is precisely why a regression that needs LENGTH to show went unseen. This reads real
// repository source at run time instead.
//
// METAL-SPECIFIC PRECONDITION — GOINFER_METAL_BATCHED_PREFILL=1 IS MANDATORY HERE, and a green
// without it is meaningless. Metal's PrefillLast (metal/backend.go) DECLINES by default,
// because Metal's batched prefill is not bit-identical to its decode path (54% stream
// divergence, TestMetalPrefillDivergenceRate). When it declines, generateInto falls through to
// the same per-token loop the speculative path already uses, so there is NO asymmetry to
// measure and the slope would come back ~0 for a reason that has nothing to do with the fix.
// Metal also does NOT implement ResidentPrefillKV (only CUDA does), so the KV-only fallback —
// CUDA's SECOND asymmetry — does not exist here. Metal has exactly one exposure to this bug and
// it is gated behind this variable.
func TestMetalSpecPrefillRegression(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1 to opt in (loads a multi-GB model from ~/models)")
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

	// Force the batched prefill ON. Without this Metal declines and both arms take the same
	// per-token path — see the METAL-SPECIFIC PRECONDITION above.
	t.Setenv("GOINFER_METAL_BATCHED_PREFILL", "1")

	target, err := decoder.Load(tpath, decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatalf("load target: %v", err)
	}
	defer target.Close()
	if !target.ResidentActive() {
		t.Skip("target not Metal-resident (ineligible / no residency)")
	}
	tk, err := tokenizer.LoadGGUF(tpath)
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	// Corpus: real repository source, read at run time, concatenated and encoded once. Prompt
	// lengths are then exact token prefixes of it, so the sweep varies LENGTH and nothing else.
	corpus := readRepoCorpus(t)
	allToks, err := tk.Encode(corpus, true)
	if err != nil {
		t.Fatalf("encode corpus: %v", err)
	}
	lengths := []int{33, 96, 253, 562, 839}
	if need := lengths[len(lengths)-1]; len(allToks) < need {
		t.Fatalf("corpus encodes to %d tokens, need >= %d", len(allToks), need)
	}

	ctx := context.Background()
	greedy := decoder.SamplingParams{Temperature: 0}
	const n = 64   // generated tokens per run; the gap under test is prefill-dominated
	const K = 8    // draft depth
	const reps = 3 // median of 3, per the CUDA harness
	const slopeBar = 0.50

	collect := func(ch <-chan int) []int {
		var v []int
		for id := range ch {
			v = append(v, id)
		}
		return v
	}
	runOff := func(prompt []int) ([]int, time.Duration) {
		t0 := time.Now()
		ch, _ := target.Generate(ctx, prompt, n, greedy)
		toks := collect(ch)
		return toks, time.Since(t0)
	}
	runSpec := func(prompt []int) ([]int, time.Duration) {
		t0 := time.Now()
		ch, _, gerr := target.GenerateNgramSpeculativeAdaptive(
			ctx, prompt, n, &decoder.NgramDrafter{}, &decoder.AdaptiveDepth{MaxDraft: K}, greedy)
		if gerr != nil {
			t.Fatalf("spec generate: %v", gerr)
		}
		toks := collect(ch)
		return toks, time.Since(t0)
	}
	medianMs := func(f func([]int) ([]int, time.Duration), prompt []int) float64 {
		f(prompt) // warm pipelines/caches; discarded
		ms := make([]float64, 0, reps)
		for range reps {
			_, d := f(prompt)
			ms = append(ms, float64(d.Microseconds())/1000.0)
		}
		sort.Float64s(ms)
		return ms[len(ms)/2]
	}

	// 1. LOSSLESS GATE FIRST. Speculation must be a pure speed change; if the streams differ,
	// every timing number below is measuring two different computations and means nothing.
	{
		p := allToks[:lengths[2]]
		plain, _ := runOff(p)
		spec, _ := runSpec(p)
		if !slices.Equal(plain, spec) {
			t.Fatalf("LOSSLESS GATE FAILED at %d prompt tokens: speculative stream differs from "+
				"plain greedy (plain %d tok, spec %d tok) — timings below would be meaningless",
				len(p), len(plain), len(spec))
		}
		fmt.Fprintf(os.Stderr, "[spec-prefill] lossless gate OK at %d prompt tokens (%d tokens identical)\n",
			len(p), len(plain))
	}

	// 2. PROMPT-LENGTH SWEEP.
	pts := make([]fitPoint, 0, len(lengths))
	t.Logf("%8s %10s %10s %10s", "prompt", "off_ms", "spec_ms", "gap_ms")
	start := time.Now()
	for i, L := range lengths {
		p := allToks[:L]
		offMs := medianMs(runOff, p)
		specMs := medianMs(runSpec, p)
		gap := specMs - offMs
		pts = append(pts, fitPoint{float64(L), gap})
		t.Logf("%8d %10.1f %10.1f %10.1f", L, offMs, specMs, gap)
		fmt.Fprintf(os.Stderr, "[spec-prefill] %d/%d L=%d off=%.0fms spec=%.0fms gap=%.0fms elapsed=%s\n",
			i+1, len(lengths), L, offMs, specMs, gap, time.Since(start).Round(time.Second))
	}

	// 3. LEAST-SQUARES SLOPE of gap vs prompt tokens, and the gate.
	slope, r2 := leastSquares(pts)
	t.Logf("slope = %.3f ms/prompt-token (bar %.2f), R² = %.4f", slope, slopeBar, r2)
	if slope > slopeBar {
		t.Fatalf("SPECULATIVE PREFILL REGRESSION: gap grows %.3f ms per prompt token "+
			"(bar %.2f, R² %.4f). The speculative path is prefilling one token at a time while "+
			"plain generation uses the batched Prefiller seam — see decoder/spec_ngram.go's "+
			"genNgramInto vs decoder/model.go's generateInto.", slope, slopeBar, r2)
	}
}

// readRepoCorpus concatenates real repository source files. Real code, not a hand-written
// repetition-heavy fixture: the synthetic workloads run 4-7x the copy density of real code,
// which flatters a drafter and hides a length-dependent cost behind 36-74 token prompts.
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

// leastSquares fits gap = a + slope*x and returns the slope with the fit's R². R² is reported
// rather than gated: with the bug the fit is near-perfect (CUDA R² 0.9977) and without it the
// gap is noise around zero (R² 0.33), so a LOW R² is the healthy state and gating on it would
// invert the test.
type fitPoint struct{ x, gap float64 }

func leastSquares(pts []fitPoint) (slope, r2 float64) {
	n := float64(len(pts))
	if n < 2 {
		return 0, 0
	}
	var sx, sy, sxy, sxx float64
	for _, p := range pts {
		sx += p.x
		sy += p.gap
		sxy += p.x * p.gap
		sxx += p.x * p.x
	}
	den := n*sxx - sx*sx
	if den == 0 {
		return 0, 0
	}
	slope = (n*sxy - sx*sy) / den
	intercept := (sy - slope*sx) / n
	mean := sy / n
	var ssRes, ssTot float64
	for _, p := range pts {
		pred := intercept + slope*p.x
		ssRes += (p.gap - pred) * (p.gap - pred)
		ssTot += (p.gap - mean) * (p.gap - mean)
	}
	if ssTot == 0 {
		return slope, 0
	}
	return slope, 1 - ssRes/ssTot
}
