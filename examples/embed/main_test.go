package main

import (
	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/tokenizer"
	"os"
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
	// A fixture whose chat template chat.Detect RECOGNISES (ChatML), so the example's templated path
	// is the one this drives (audit-2026-09-10 G-13(g)). TinyLlama's Zephyr-style template is not one
	// Detect knows: the example fed it the raw prompt with or without chat.Detect, and its output was
	// byte-identical either way, so removing the call could not fail this test.
	fixture := "../../testdata/qwen2-gguf/qwen2.5-0.5b-instruct-q8_0.gguf"
	if _, err := os.Stat(fixture); err != nil {
		t.Skipf("no fixture at %s", fixture)
	}
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("no go toolchain")
	}
	tok, err := tokenizer.LoadGGUF(fixture)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	if _, err := chat.Detect(chat.Meta{ChatTemplate: tok.ChatTemplate(), HasToken: tok.Has}); err != nil {
		t.Fatalf("chat.Detect does not recognise %s's template (%v), so the templated path would go untested", fixture, err)
	}

	bin := filepath.Join(t.TempDir(), "embed")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	build := exec.Command("go", "build", "-o", bin, ".")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}

	// The fixture is present, so a failure to run is a failure, not a skip.
	out, err := exec.Command(bin, fixture, "Say hello in five words").CombinedOutput()
	if err != nil {
		t.Fatalf("run against %s: %v\n%s", fixture, err, out)
	}
	text := strings.TrimSpace(string(out))
	if text == "" {
		t.Fatal("embed example produced no output")
	}
	// A templated instruct model answers and ENDS its turn; fed the raw prompt it runs on to the
	// 256-token cap. Measured on this fixture: 32 chars with the template, 852 without it. The 400
	// bound was fixed before the run that tested it.
	if len(text) > 400 {
		t.Fatalf("the reply ran %d chars — the model never ended its turn, which is the shape of a prompt "+
			"sent without its chat template:\n%s", len(text), text)
	}

	if run, reps := longestConsecutiveRepeat(text, 8); reps >= 3 {
		t.Fatalf("%q repeats %d times back to back — degenerate output:\n%s", run, reps, text)
	}
	t.Logf("output (%d chars): %s", len(text), text)
}

// TestBuildPrompt_rendersOnlyOneLeadingBOS is M-39 (audit-2026-09-10): buildPrompt must not
// prepend a second BOS when the chat template it just rendered already carries the family's own
// BOS marker. chatml-tiny.gguf (tokenizer/testdata) is a small fixture whose ChatML template
// chat.Detect recognizes AND whose vocab carries a real BOS id (unlike Qwen's real ChatML
// checkpoints, whose BOS is -1 — the reason the full-binary integration test above can't exercise
// this path at all, per the audit's own note). Encoding the rendered text with addBOS=true (the
// pre-fix bug) visibly double-prepends the BOS id; this asserts exactly one.
func TestBuildPrompt_rendersOnlyOneLeadingBOS(t *testing.T) {
	fixture := filepath.Join("..", "..", "tokenizer", "testdata", "chatml-tiny.gguf")
	tok, err := tokenizer.LoadGGUF(fixture)
	if err != nil {
		t.Skipf("no committed tiny tokenizer fixture at %s: %v", fixture, err)
	}
	if _, terr := chat.Detect(chat.Meta{ChatTemplate: tok.ChatTemplate(), HasToken: tok.Has}); terr != nil {
		t.Fatalf("chat.Detect does not recognise %s's template (%v), so the templated path would go untested", fixture, terr)
	}

	ids, err := buildPrompt(tok, "hi")
	if err != nil {
		t.Fatalf("buildPrompt: %v", err)
	}
	if len(ids) < 2 {
		t.Fatalf("buildPrompt returned only %d ids: %v", len(ids), ids)
	}
	bosID := ids[0]
	count := 0
	for _, id := range ids {
		if id == bosID {
			count++
		} else {
			break // only the LEADING run of BOS matters; the id can legitimately reappear later
		}
	}
	if count != 1 {
		t.Fatalf("ids = %v start with %d copies of id %d back to back — the rendered template's own "+
			"BOS marker plus addBOS=true both fired (M-39)", ids, count, bosID)
	}
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
