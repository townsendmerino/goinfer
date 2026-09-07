package main

import (
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// R10 (docs/measurements/cold-user-2026-09-06-nobara-pc.md): a README+pkg.go.dev-only reader's
// own ≤40-line embed program compiled and ran on the first try — a genuine success — but printed
// "Hello. Hello. Hello. Hello. ..." instead of a coherent reply, because it encoded the raw
// prompt directly with no chat template. This example already applies one (chat.Detect); this
// test is the gate that would have caught a regression back to that shape: it runs the REAL
// binary against a real, tiny, chat-tuned checkpoint and asserts the output is not one token (or
// one short phrase) repeated into a loop.
func TestEmbedExample_outputIsNotARepeatedToken(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a binary and loads a real checkpoint")
	}
	fixture := "../../testdata/tinyllama-gguf/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("no go toolchain")
	}

	bin := filepath.Join(t.TempDir(), "embed")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	build := exec.Command("go", "build", "-o", bin, ".")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}

	out, err := exec.Command(bin, fixture, "Say hello in five words").CombinedOutput()
	if err != nil {
		t.Skipf("could not run against %s (fixture likely absent on this box): %v\n%s", fixture, err, out)
	}
	text := strings.TrimSpace(string(out))
	if text == "" {
		t.Fatal("embed example produced no output")
	}

	// A CONSECUTIVE-repeat check, not "this substring appears 3 times anywhere" — the latter
	// false-positives on real, varied output that shares a phrase template (this fixture likes
	// numbered lists that each start "I'm not X, but I'm Y", which is thematically repetitive
	// and not the defect). What both the original "Hello. Hello. Hello." shape and a
	// mid-generation "...Byeagain.User:Byeagain.Assistant:Byeagain..." loop have in common is a
	// substring immediately followed by itself, back to back, several times running — a real
	// loop, not similar sentences scattered through otherwise-different text. No word-split (this
	// example decodes token-by-token with no UTF-8/space holdback — an accepted limitation of
	// staying under the 40-line cap — so whitespace between decoded pieces is unreliable).
	if run, reps := longestConsecutiveRepeat(text, 8); reps >= 3 {
		t.Fatalf("%q repeats %d times back to back — degenerate output:\n%s", run, reps, text)
	}
	t.Logf("output (%d chars): %s", len(text), text)
}

// longestConsecutiveRepeat checks every window size from minLen up to a cap for a substring
// that occurs immediately followed by itself two or more times running (the repeating unit's
// PERIOD, which a real loop can land on at any width — a fixed handful of guessed widths missed
// a real one: "Byeagain.User:Byeagain.Assistant:" repeating at period 34), and returns the
// occurrence covering the most total text along with its repeat count.
func longestConsecutiveRepeat(s string, minLen int) (string, int) {
	bestUnit, bestReps := "", 1
	maxW := minLen * 8
	for w := minLen; w <= maxW; w++ {
		if w*2 > len(s) {
			break
		}
		for i := 0; i+w*2 <= len(s); i++ {
			unit := s[i : i+w]
			reps := 1
			for j := i + w; j+w <= len(s) && s[j:j+w] == unit; j += w {
				reps++
			}
			if reps >= 2 && w*reps > len(bestUnit)*bestReps {
				bestUnit, bestReps = unit, reps
			}
		}
	}
	return bestUnit, bestReps
}
