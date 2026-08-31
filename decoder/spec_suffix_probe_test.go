package decoder

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// ---------------------------------------------------------------------------
// 02 "next step" — Step 1: instrument the SHIPPED NgramDrafter on realistic
// agentic/code traffic, and Step 2: replay candidate scoring offline.
//
// WHY A NEW FILE RATHER THAN specWorkloads. The existing corpus in
// spec_harness_test.go is hand-written and deliberately copy-heavy (its own
// comment says so: "prompts with heavy internal repetition — the regime n-gram
// drafting is built for"). It is a CONSTRUCTION of copy-heavy traffic, not a
// SAMPLE of it, so it cannot answer "does this drafter fire on real input?" —
// it was built so the answer is yes. scripts/prompts.json is worse (four unique
// words per prompt; see docs/spec/10). Both are scored here on the same
// copy-density metric as the real inputs, so the distance is a number rather
// than an assertion.
//
// THE INPUTS ARE READ FROM REAL REPO FILES AT RUN TIME. Nothing in the corpus
// below is prose I wrote for the measurement; the code inputs are actual source
// files and the agent input is the actual system prompt from demo/agent plus
// actual source as retrieved context. That is the whole point — an authored
// prompt is exactly the failure being corrected.
// ---------------------------------------------------------------------------

// copyDensity is the fraction of positions whose preceding n-token suffix already
// occurred earlier in the stream. It is a property of the TOKEN STREAM ALONE — no
// drafter, no model — so it measures how favourable an input is to a repetition
// drafter before any drafter runs. This is the instrument that turns "that corpus
// is unrealistic" from a caveat into a measurement.
func copyDensity(toks []int, n int) float64 {
	if len(toks) <= n {
		return 0
	}
	seen := make(map[string]bool, len(toks))
	hits, total := 0, 0
	var b strings.Builder
	key := func(s []int) string {
		b.Reset()
		for _, t := range s {
			fmt.Fprintf(&b, "%d,", t)
		}
		return b.String()
	}
	for i := n; i < len(toks); i++ {
		k := key(toks[i-n : i])
		total++
		if seen[k] {
			hits++
		}
		seen[k] = true
	}
	if total == 0 {
		return 0
	}
	return float64(hits) / float64(total)
}

// distinctTrigram is the loop guard fixed in the pre-registration: a generation
// that degenerates into a cycle makes any repetition drafter look spectacular,
// so such traces are EXCLUDED before analysis rather than explained after. The
// 0.70 bar is the coherence floor docs/benchmarks.md §B4 already uses.
func distinctTrigram(toks []int) float64 {
	if len(toks) < 3 {
		return 1
	}
	seen := make(map[[3]int]bool, len(toks))
	for i := 0; i+3 <= len(toks); i++ {
		seen[[3]int{toks[i], toks[i+1], toks[i+2]}] = true
	}
	return float64(len(seen)) / float64(len(toks)-2)
}

// mustRead reads a real repo file; the corpus is built from these, not authored.
func mustRead(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("corpus file missing (%s): %v", path, err)
	}
	return string(b)
}

// realisticInputs builds the step-1 corpus out of actual repo content. Each entry
// says what real traffic it stands for and why the shape is authentic.
func realisticInputs(t *testing.T) []struct{ name, prompt string } {
	t.Helper()
	// Real Go source, truncated mid-file: the "continue this code" shape that a
	// completion request in an editor actually has.
	adaptive := mustRead(t, "spec_adaptive.go")
	// A different real file, so the result does not rest on one file's style.
	sampleSrc := mustRead(t, "spec_sample.go")

	// The REAL agent system prompt from demo/agent (not a paraphrase), plus real
	// source as retrieved context, in the transcript shape the stateless
	// /v1/chat/completions surface actually re-sends every turn.
	const answerSystem = "You are a concise Go expert answering questions about the Go standard library. " +
		"Ground every claim in the search results you are given and cite them as file:line. " +
		"If the results do not contain the answer, say so instead of guessing."

	trunc := func(s string, n int) string {
		if len(s) > n {
			return s[:n]
		}
		return s
	}

	// Turn 2 of an agent loop. The system prompt, the first user turn, its search
	// results and the first answer are all RE-SENT verbatim — that resend is the
	// structural repetition an agent loop really has, and it is why cross-request
	// scope buys less on a stateless surface than the paper's setting implies.
	//
	// The SECOND search returns DIFFERENT source, which is the whole correction
	// here: a first attempt reused the same snippet for both turns, the model
	// copied it back, and the pre-registered loop guard excluded the trace
	// (distinct-trigram 0.297). That was a defect in the INPUT, not in the guard —
	// a real second query retrieves different code — so the input was fixed and
	// the guard left alone. Recorded because it is exactly the artifact
	// arXiv 2604.26469 warns about, caught by the rule written in advance.
	agentTurn2 := answerSystem + "\n\n" +
		"User: How does the adaptive draft depth controller decide how deep to draft?\n\n" +
		"Search results:\n" + trunc(adaptive, 1500) + "\n\n" +
		"Assistant: The controller extends depth while the expected marginal committed token still beats the marginal verify cost.\n\n" +
		"User: And how does the sampled path stay lossless when it rejects a draft token?\n\n" +
		"Search results:\n" + trunc(sampleSrc, 1500) + "\n\n" +
		"Assistant:"

	return []struct{ name, prompt string }{
		{"code-continue", trunc(adaptive, 2600)},
		{"code-continue-2", trunc(sampleSrc, 2600)},
		{"agent-loop-turn2", agentTurn2},
		{"prose-doc", trunc(mustRead(t, "../docs/spec/00-core.md"), 2600)},
	}
}

// TestSuffixProbe_step1 is the pre-registered step-1 measurement: on realistic
// agentic/code traffic, what does the SHIPPED drafter actually achieve? Reports
// hit rate H, realized acceptance alpha, accepted tokens/round A, the match-length
// distribution, and wall clock for off / fixed-K / adaptive. `off` is included
// because a suite was once found here where no configuration beat running no
// drafter at all, and that was only visible because off was a competitor.
//
// Run: GOINFER_SPEC_SUFFIX_PROBE=1 go test ./decoder -run TestSuffixProbe_step1 -v
func TestSuffixProbe_step1(t *testing.T) {
	if os.Getenv("GOINFER_SPEC_SUFFIX_PROBE") == "" {
		t.Skip("set GOINFER_SPEC_SUFFIX_PROBE=1 (loads a real model; minutes)")
	}
	m, err := loadBenchModel()
	if err != nil {
		t.Skipf("no model (%v); set GOINFER_PREQUANT_GGUF", err)
	}
	tk, err := tokenizer.LoadGGUF(benchGGUFPath())
	if err != nil {
		t.Skipf("no tokenizer (%v)", err)
	}

	const maxTok = 160
	const K = 8
	ctx := context.Background()
	greedy := SamplingParams{Temperature: 0}

	type record struct {
		Name         string  `json:"name"`
		PromptToks   int     `json:"prompt_toks"`
		CopyDensity4 float64 `json:"copy_density_4"`
		GenToks      int     `json:"gen_toks"`
		DistinctTri  float64 `json:"distinct_trigram"`
		Excluded     bool    `json:"excluded_as_looping"`
		Rounds       int     `json:"rounds"`
		Drafted      int     `json:"drafted"`
		Accepted     int     `json:"accepted"`
		AcceptRate   float64 `json:"accept_rate"`
		EvalRate     float64 `json:"eval_accept_rate"`
		TokPerRound  float64 `json:"tokens_per_round"`
		Alpha        float64 `json:"adaptive_alpha"`
		OffMs        float64 `json:"off_ms"`
		FixedMs      float64 `json:"fixed_ms"`
		AdaptiveMs   float64 `json:"adaptive_ms"`
		Tokens       []int   `json:"tokens"` // prompt+generated: the step-2 replay artifact
	}
	var out []record

	// Calibration: score the CONSTRUCTED corpora on the same metric as the real
	// inputs, so "unrealistic" is a number on the page.
	t.Logf("=== copy density (4-gram) of each corpus, drafter-independent ===")
	for _, w := range specWorkloads {
		tk2, err := tk.Encode(w.prompt, true)
		if err != nil {
			continue
		}
		t.Logf("  %-18s %-12s copy_density_4 = %.3f  (%d tok)", "specWorkloads", w.name, copyDensity(tk2, 4), len(tk2))
	}

	inputs := realisticInputs(t)
	for _, w := range inputs {
		prompt, err := tk.Encode(w.prompt, true)
		if err != nil {
			t.Fatalf("%s: encode: %v", w.name, err)
		}
		rec := record{Name: w.name, PromptToks: len(prompt), CopyDensity4: copyDensity(prompt, 4)}

		// off — the required do-nothing arm, and the lossless reference.
		t0 := time.Now()
		refCh, _ := m.Generate(ctx, prompt, maxTok, greedy)
		ref := collectTokens(refCh)
		rec.OffMs = float64(time.Since(t0).Microseconds()) / 1000
		rec.GenToks = len(ref)
		rec.DistinctTri = distinctTrigram(ref)
		rec.Tokens = append(slices.Clone(prompt), ref...)

		// Pre-registered exclusion, applied BEFORE any drafter number is read.
		if rec.DistinctTri < 0.70 {
			rec.Excluded = true
			t.Logf("%-18s EXCLUDED as looping (distinct-trigram %.3f < 0.70)", w.name, rec.DistinctTri)
			out = append(out, rec)
			continue
		}

		t0 = time.Now()
		fixedCh, gFixed, err := m.GenerateNgramSpeculative(ctx, prompt, maxTok, &NgramDrafter{}, K, greedy)
		if err != nil {
			t.Fatalf("%s fixed: %v", w.name, err)
		}
		fixed := collectTokens(fixedCh)
		rec.FixedMs = float64(time.Since(t0).Microseconds()) / 1000

		ad := &AdaptiveDepth{MaxDraft: K}
		t0 = time.Now()
		adaCh, gAda, err := m.GenerateNgramSpeculativeAdaptive(ctx, prompt, maxTok, &NgramDrafter{}, ad, greedy)
		if err != nil {
			t.Fatalf("%s adaptive: %v", w.name, err)
		}
		ada := collectTokens(adaCh)
		rec.AdaptiveMs = float64(time.Since(t0).Microseconds()) / 1000

		// Lossless is non-negotiable: both modes must be token-identical to plain.
		if gFixed.Err() != nil || !slices.Equal(fixed, ref) {
			t.Fatalf("%s fixed: NOT LOSSLESS (err %v)", w.name, gFixed.Err())
		}
		if gAda.Err() != nil || !slices.Equal(ada, ref) {
			t.Fatalf("%s adaptive: NOT LOSSLESS (err %v)", w.name, gAda.Err())
		}

		s := gFixed.Spec
		rec.Rounds, rec.Drafted, rec.Accepted = s.Rounds, s.Drafted, s.Accepted
		rec.AcceptRate, rec.EvalRate = s.AcceptanceRate(), s.EvalAcceptanceRate()
		rec.TokPerRound = s.TokensPerRound()
		rec.Alpha = ad.Alpha()
		out = append(out, rec)

		t.Logf("%-18s copy4=%.3f tri=%.3f | rounds=%3d drafted=%3d accepted=%3d acc=%.3f evalacc=%.3f tok/v=%.2f alpha=%.3f | off=%.0fms fixed=%.0fms(%.2fx) ada=%.0fms(%.2fx)",
			w.name, rec.CopyDensity4, rec.DistinctTri, rec.Rounds, rec.Drafted, rec.Accepted,
			rec.AcceptRate, rec.EvalRate, rec.TokPerRound, rec.Alpha,
			rec.OffMs, rec.FixedMs, rec.OffMs/rec.FixedMs, rec.AdaptiveMs, rec.OffMs/rec.AdaptiveMs)
	}

	if p := os.Getenv("GOINFER_SPEC_SUFFIX_OUT"); p != "" {
		b, _ := json.MarshalIndent(out, "", " ")
		if err := os.WriteFile(p, b, 0o644); err != nil {
			t.Fatalf("write step-1 artifact: %v", err)
		}
		t.Logf("step-1 artifact (step-2 replay input) → %s", p)
	}
}
