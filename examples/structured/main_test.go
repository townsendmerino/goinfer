package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

// TestStructuredExample runs the real binary on a real, small, chat-tuned checkpoint (skipped without one) and checks
// the one thing the example exists to show: the output unmarshals into Person.
func TestStructuredExample(t *testing.T) {
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
	bin := filepath.Join(t.TempDir(), "structured")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	if out, err := exec.Command("go", "build", "-o", bin, ".").CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	out, err := exec.Command(bin, fixture, "Ann Lee is 34 and writes Go and Rust.").Output()
	if err != nil {
		t.Fatalf("run against %s: %v\n%s", fixture, err, out)
	}
	if !strings.HasPrefix(string(out), "{Name:") {
		t.Errorf("not a filled-in Person: %q", out)
	}
	t.Logf("%s", out)
}

var goBlockRE = regexp.MustCompile("(?s)```go\n(.*?)\n```")

// docBlocksNotFromExamples returns each ```go block of doc that is not a verbatim run of lines from one of srcs.
func docBlocksNotFromExamples(doc string, srcs []string) []string {
	var bad []string
	for _, m := range goBlockRE.FindAllStringSubmatch(doc, -1) {
		block, found := m[1], false
		for _, s := range srcs {
			if strings.Contains("\n"+s, "\n"+block+"\n") {
				found = true
				break
			}
		}
		if !found {
			bad = append(bad, block)
		}
	}
	return bad
}

// TestUseFromGoDoc_codeIsFromExamples keeps docs/use-from-go.md compiling: its every Go block must be a run of whole
// lines copied from a program under examples/, which CI builds and vets. A block edited in the doc alone, or an example
// changed under its excerpt, fails here.
func TestUseFromGoDoc_codeIsFromExamples(t *testing.T) {
	doc, err := os.ReadFile("../../docs/use-from-go.md")
	if err != nil {
		t.Fatal(err)
	}
	mains, err := filepath.Glob("../*/main.go")
	if err != nil || len(mains) == 0 {
		t.Fatalf("no examples: %v", err)
	}
	var srcs []string
	for _, p := range mains {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		srcs = append(srcs, string(b))
	}
	if n := len(goBlockRE.FindAllString(string(doc), -1)); n < 8 {
		t.Fatalf("docs/use-from-go.md has %d Go blocks; the walkthrough has at least 8", n)
	}
	for _, b := range docBlocksNotFromExamples(string(doc), srcs) {
		t.Errorf("a Go block in docs/use-from-go.md is not copied from examples/*/main.go:\n%s", b)
	}
	// The mutation: the same check must refuse a block edited in the doc alone.
	edited := strings.Replace(string(doc), "GrammarFromStruct(Person{})", "GrammarFromStruct(&Person{})", 1)
	if edited == string(doc) {
		t.Fatal("the mutation found nothing to edit")
	}
	if len(docBlocksNotFromExamples(edited, srcs)) != 1 {
		t.Error("the check did not catch a block edited in the doc alone")
	}
}
