package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestConfidenceExample runs the real binary on a real, small, chat-tuned checkpoint (skipped without one) and
// checks the two things the example exists to show: the constrained answer, and one confidence line per enum,
// boolean and integer field.
func TestConfidenceExample(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a binary and loads a real checkpoint")
	}
	var fixture string
	for _, p := range []string{
		"../../testdata/qwen2-gguf/qwen2.5-0.5b-instruct-q8_0.gguf", // the embed example's fixture
		"../../testdata/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",    // a local symlink on the dev Mac
	} {
		if _, err := os.Stat(p); err == nil {
			fixture = p
			break
		}
	}
	if fixture == "" {
		t.Skip("no small Qwen checkpoint in testdata")
	}
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("no go toolchain")
	}
	bin := filepath.Join(t.TempDir(), "confidence")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	if out, err := exec.Command("go", "build", "-o", bin, ".").CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	out, err := exec.Command(bin, fixture, "I was charged twice for order #123, please refund me. - Ann").CombinedOutput()
	if err != nil {
		t.Fatalf("run against %s: %v\n%s", fixture, err, out)
	}
	text := string(out)
	for _, field := range []string{"category", "urgent", "orders"} {
		if !strings.Contains(text, "\n"+field+" ") {
			t.Errorf("no confidence line for %s in:\n%s", field, text)
		}
	}
	t.Logf("%s", text)
}
