package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"
)

func TestParseGiwConfigs(t *testing.T) {
	got, err := parseGiwConfigs("int4:metal:e4h, int8int8:cpu ,f32:canonical")
	if err != nil {
		t.Fatal(err)
	}
	want := []giwConfig{{"int4", "metal", true}, {"int8int8", "cpu", false}, {"f32", "canonical", false}}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("config %d: %v, want %v", i, got[i], want[i])
		}
	}
	if s := got[0].String(); s != "int4:metal:e4h" {
		t.Errorf("String() = %q", s)
	}
	for _, bad := range []string{"int4", "int4:", ":metal", "int4:metal:e4", "int4:metal:e4h:x"} {
		if _, err := parseGiwConfigs(bad); err == nil {
			t.Errorf("%q parsed", bad)
		}
	}
	if c, err := parseGiwConfigs(""); err != nil || len(c) != 0 {
		t.Errorf("empty: %v, %v", c, err)
	}
}

// TestJudgeGiwBytes_verdicts: equal hashes are IDENTICAL; a source neither build transcodes is NOT RUN, never a pass;
// one build failing, the new build disagreeing with itself, and different hashes are each DIFFERENT.
func TestJudgeGiwBytes_verdicts(t *testing.T) {
	ok := func(sha string) giwResult { return giwResult{OK: true, SHA: sha + strings.Repeat("0", 64), Size: 10} }
	bad := func(why string) giwResult { return giwResult{Err: why} }
	for _, c := range []struct {
		name        string
		n1, n2, old giwResult
		want, has   string
	}{
		{"identical", ok("a"), ok("a"), ok("a"), vIdentical, ""},
		{"neither", bad("unsupported model_type"), bad("unsupported model_type"), bad("unsupported model_type"), vNotRun, "neither build"},
		{"new fails", bad("boom"), bad("boom"), ok("a"), vDifferent, "new build fails"},
		{"old fails", ok("a"), ok("a"), bad("boom"), vDifferent, "old build fails"},
		{"second new run fails", ok("a"), bad("boom"), ok("a"), vDifferent, "second transcode"},
		{"new disagrees with itself", ok("a"), ok("b"), ok("a"), vDifferent, "NONDETERMINISTIC"},
		{"different", ok("b"), ok("b"), ok("a"), vDifferent, "sha256 old"},
	} {
		got, why := judgeGiwBytes(c.n1, c.n2, c.old)
		if got != c.want || !strings.Contains(why, c.has) {
			t.Errorf("%s: %s (%q), want %s containing %q", c.name, got, why, c.want, c.has)
		}
	}
}

// TestPlanGiwSources_tiny: the walk takes the directories that hold a .safetensors file under testdata/ and
// decoder/testdata/, names the second set apart, leaves other directories out, and marks one over the cap NOT RUN.
func TestPlanGiwSources_tiny(t *testing.T) {
	root := t.TempDir()
	mk := func(rel string, size int) {
		p := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, make([]byte, size), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	mk("testdata/a-tiny/model.safetensors", 100)
	mk("testdata/big/model.safetensors", 5000)
	mk("testdata/goldens/x.json", 10)
	mk("testdata/loose.safetensors", 10)
	mk("decoder/testdata/b-tiny/model.safetensors", 100)
	defer func(v int64) { giwTinyCapBytes = v }(giwTinyCapBytes)
	giwTinyCapBytes = 1000
	srcs, err := planGiwSources(identityOpts{FixtureRoot: root, Assets: "tiny"})
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]string{}
	for _, s := range srcs {
		got[s.Name] = s.NotRun
	}
	if len(got) != 3 || got["a-tiny"] != "" || got["decoder/b-tiny"] != "" || !strings.Contains(got["big"], "cap") {
		t.Errorf("sources %v, want a-tiny and decoder/b-tiny runnable and big over the cap", got)
	}
	only, _ := planGiwSources(identityOpts{FixtureRoot: root, Assets: "tiny", Only: []string{"decoder/b-tiny"}})
	if len(only) != 1 || only[0].Name != "decoder/b-tiny" {
		t.Errorf("-only: %v", only)
	}
}

// ---- equivalence, at real revisions (opt-in, as identity's own) ----

func (fx *identityFixture) runGiw(t *testing.T, mode, oldRev, newRev string) (int, string) {
	t.Helper()
	for _, d := range []string{"llama-tiny", "qwen3moe-tiny", "siglip-tiny"} {
		if _, err := os.Stat(filepath.Join(fx.root, "testdata", d, "config.json")); err != nil {
			t.Skipf("fixture %s is not in this checkout: %v", d, err)
		}
	}
	var out bytes.Buffer
	t0 := time.Now()
	o := identityOpts{Repo: fx.clone, FixtureRoot: fx.root, OldRev: oldRev, NewRev: newRev, Backend: "cpu", Assets: "tiny",
		Giw: mode, LogDir: t.TempDir(), Progress: os.Stderr}
	if mode == "bytes" {
		o.Only = []string{"llama-tiny", "qwen3moe-tiny", "siglip-tiny"}
		o.Configs = []giwConfig{{Quant: "int4", Target: "cpu"}, {Quant: "int8int8", Target: "cpu"}}
	} else {
		o.Families, o.Quants, o.Steps = []string{"llama", "qwen3_moe"}, []string{"int4"}, 4
	}
	rc := identityMain(o, &out)
	s := regexp.MustCompile(`\x1b\[[0-9;]*m`).ReplaceAllString(out.String(), "")
	fmt.Fprintf(os.Stderr, "identity -giw %s %s..%s: rc=%d in %s\n", mode, shortSHA(oldRev), shortSHA(newRev), rc, time.Since(t0).Round(time.Second))
	return rc, s
}

// A comment-only change writes the same bytes; a vision tower neither build transcodes is NOT RUN, not a pass.
func TestIdentityGiw_bytesCommentOnlyChangeIsIdentical(t *testing.T) {
	fx := newIdentityFixture(t)
	rev := fx.commit(t, fx.head, "decoder/model.go", "\t\tparallel := placement == NormParallel\n",
		"\t\t// gate identity -giw equivalence: a comment, and nothing else.\n\t\tparallel := placement == NormParallel\n", "comment only")
	rc, out := fx.runGiw(t, "bytes", fx.head, rev)
	for _, src := range []string{"llama-tiny", "qwen3moe-tiny"} {
		if !regexp.MustCompile(`(?m)^  IDENTICAL\s+` + src + `\s+2 cell`).MatchString(out) {
			t.Errorf("%s is not IDENTICAL on both configurations", src)
		}
	}
	if !regexp.MustCompile(`(?m)^  NOT RUN\s+siglip-tiny\s`).MatchString(out) || !strings.Contains(out, "4 IDENTICAL, 0 DIFFERENT, 2 NOT RUN") {
		t.Errorf("siglip-tiny (a tower neither build transcodes) must be NOT RUN and counted apart")
	}
	if rc != 0 {
		t.Errorf("rc=%d, want 0", rc)
	}
	if t.Failed() || os.Getenv("GATE_IDENTITY_SHOW") != "" {
		t.Logf("report:\n%s", out)
	}
}

// One change to the bytes a bundle carries and to no number in it (the id the header records): the bytes mode is red on
// every cell that ran, and the logits mode, on each side's own file, is green. That is the pair of questions the two
// modes answer.
func TestIdentityGiw_headerChangeIsDifferentBytesAndIdenticalLogits(t *testing.T) {
	fx := newIdentityFixture(t)
	rev := fx.commit(t, fx.head, "decoder/serialize.go", "\twr.str(id)\n", "\twr.str(id + \"-x\") // equivalence: other bytes, the same numbers\n", "header id")
	rc, out := fx.runGiw(t, "bytes", fx.head, rev)
	if !strings.Contains(out, "0 IDENTICAL, 4 DIFFERENT, 2 NOT RUN") || !strings.Contains(out, "sha256 old") {
		t.Errorf("a changed header must make every transcoded cell DIFFERENT, with both hashes shown")
	}
	if rc != 1 {
		t.Errorf("bytes: rc=%d, want 1", rc)
	}
	if t.Failed() || os.Getenv("GATE_IDENTITY_SHOW") != "" {
		t.Logf("bytes report:\n%s", out)
	}
	rc, out = fx.runGiw(t, "logits", fx.head, rev)
	got := familyVerdicts(t, out)
	for _, f := range []string{"llama", "qwen3_moe"} {
		if got[f] != vIdentical {
			t.Errorf("logits: %s is %q, want IDENTICAL", f, got[f])
		}
	}
	if rc != 0 {
		t.Errorf("logits: rc=%d, want 0", rc)
	}
	if !strings.Contains(out, "giw logits: each revision's own cmd/prequant") {
		t.Errorf("the report must say the logits came from each side's own transcode")
	}
	if t.Failed() || os.Getenv("GATE_IDENTITY_SHOW") != "" {
		t.Logf("logits report:\n%s", out)
	}
}
