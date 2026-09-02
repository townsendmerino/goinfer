package tokenizer

import (
	"strings"
	"testing"
)

// PROMPTS THAT ACTUALLY DIVERGE.
//
// Every real-GGUF gate that tokenizes through goinfer uses "The capital of France is" — no digit,
// no contraction, no CamelCase, no combining mark. So the three alternations agree on it EXACTLY,
// and the C-10 divergence is identically zero in the gate. That is the minimal-repro rule in
// CLAUDE.md, live: the shrunk input removed the only dimension the defect lives in.
//
// This pins the divergence itself. It is not a parity test against HF — no reference ids are
// available on every machine — it is the weaker and still useful claim that the three walkers
// DISAGREE where they are supposed to, so a change that quietly collapsed them into one would be
// caught. The differential oracles in the sibling files are what pin each walker to its pattern.
func TestSplitShapes_divergeOnTheInputsTheGatesNeverUse(t *testing.T) {
	cl := func(s string) []string { return splitGPT2(s, 1) }
	cl3 := func(s string) []string { return splitGPT2(s, 3) }

	for _, tc := range []struct {
		name, in string
		// each walker's expected piece COUNT; the shapes must not agree
		cl1, cl3n, o200, gpt2 int
	}{
		{"digits", "2024", 4, 2, 2, 1},
		{"spaced digits", " 2020", 5, 3, 3, 1},
		{"contraction", "don't", 2, 2, 1, 2},
		{"CamelCase", "CamelCase", 1, 1, 2, 1},
	} {
		got := [4][]string{cl(tc.in), cl3(tc.in), splitO200k(tc.in), splitGPT2Original(tc.in)}
		want := [4]int{tc.cl1, tc.cl3n, tc.o200, tc.gpt2}
		names := [4]string{"cl100k/1", "cl100k/3", "o200k", "gpt2-original"}
		for i := range got {
			if len(got[i]) != want[i] {
				t.Errorf("%s %q: %s gave %d piece(s) %q, want %d",
					tc.name, tc.in, names[i], len(got[i]), got[i], want[i])
			}
		}
	}

	// The prompt every gate uses. All four agree on it, which is exactly why none of them could
	// ever have caught this — asserted so the claim is checked rather than repeated.
	const gatePrompt = "The capital of France is"
	base := strings.Join(cl(gatePrompt), "\x00")
	for name, got := range map[string][]string{
		"cl100k/3":      cl3(gatePrompt),
		"o200k":         splitO200k(gatePrompt),
		"gpt2-original": splitGPT2Original(gatePrompt),
	} {
		if strings.Join(got, "\x00") != base {
			t.Errorf("%s disagrees with cl100k on %q — then the gates WOULD have caught C-10 and "+
				"the premise of this test is wrong: %q vs %q", name, gatePrompt, got, cl(gatePrompt))
		}
	}

	// A combining mark, written as an ESCAPE so precomposed and decomposed cannot be confused by
	// eye — they behave completely differently here and look identical in a source file.
	// Decomposed: cl100k drops the mark into the punctuation run, o200k keeps it as word content.
	const decomposed = "cafe\u0301"
	if o, c := splitO200k(decomposed), cl(decomposed); len(o) != 1 || len(c) != 2 {
		t.Errorf("decomposed cafe+U+0301: o200k %q (want 1 piece), cl100k %q (want 2) — "+
			"\\p{M} as word content is one of the four differences", o, c)
	}
	// PRECOMPOSED is the same for both, which is why a test written with a literal é proves
	// nothing: the byte sequence someone types decides whether this case exists at all.
	const precomposed = "caf\u00e9"
	if o, c := splitO200k(precomposed), cl(precomposed); len(o) != 1 || len(c) != 1 {
		t.Errorf("precomposed café: o200k %q, cl100k %q — both should be one piece", o, c)
	}
}
