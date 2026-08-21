package main

import (
	"strings"
	"testing"
)

// The GPU gate's distinctive machinery is GROUP ACCOUNTING and a THREE-STATE verdict. Both exist
// because a tally computed from what emitted can never detect what did not, and because "the code is
// broken" and "the evidence is broken" are different answers a reader needs told apart.

func newTestGate(w *strings.Builder, backend string, expect ...string) *gpuGate {
	return &gpuGate{w: w, backend: backend, expect: expect, emitted: map[string]bool{}}
}

// A declared group that emits NO verdict is itself a failure. This is the generalisation of the
// per-block guard: a block that dies mid-way emits nothing, and "ran 3" and "ran 4" are both
// plausible-looking numbers, so only reconciling against the DECLARED set makes silence detectable.
func TestGPU_declaredGroupThatEmitsNothingIsAFailure(t *testing.T) {
	var buf strings.Builder
	g := newTestGate(&buf, "none", "alpha", "beta")
	g.grp("alpha")
	g.ok("alpha ran")
	g.ran++
	// beta never emits — the block "died".
	rc := g.verdict()
	out := buf.String()
	if rc != 1 {
		t.Fatalf("rc=%d, want 1 — a group that produced no verdict must fail the gate\n%s", rc, out)
	}
	if !strings.Contains(out, "declared but emitted NO verdict: beta") {
		t.Errorf("the silent group is not named:\n%s", out)
	}
	if !strings.Contains(out, "2 declared -> 1 reported") {
		t.Errorf("the declared/reported counts are not stated together:\n%s", out)
	}
}

// The other direction: a group that emits but was never declared means the tally has stopped
// describing the gate.
func TestGPU_undeclaredGroupThatEmitsIsAFailure(t *testing.T) {
	var buf strings.Builder
	g := newTestGate(&buf, "none", "alpha")
	g.grp("alpha")
	g.ok("alpha ran")
	g.ran++
	g.grp("surprise")
	g.ok("where did this come from")
	if rc := g.verdict(); rc != 1 {
		t.Fatalf("rc=%d, want 1\n%s", rc, buf.String())
	}
	if !strings.Contains(buf.String(), "emitted but not declared: surprise") {
		t.Errorf("the undeclared group is not named:\n%s", buf.String())
	}
}

// THREE STATES, NOT TWO. A dirty tree with every check green is INCONCLUSIVE, not PASS: the verdict
// names a commit, and an uncommitted edit means it does not describe what that commit contains.
// Collapsing that into PASS is how a verdict reading "PASS at <sha>" gets pasted into a tag message
// for a tree that is not <sha>.
func TestGPU_verdictHasThreeStates(t *testing.T) {
	mk := func(dirty bool, fails int, ran int) (int, string) {
		var buf strings.Builder
		g := newTestGate(&buf, "none", "alpha")
		g.dirty, g.commit, g.ran = dirty, "abc1234", ran
		g.grp("alpha")
		if fails > 0 {
			for i := 0; i < fails; i++ {
				g.bad("something broke")
			}
		} else {
			g.ok("fine")
		}
		return g.verdict(), buf.String()
	}

	if rc, out := mk(false, 0, 1); rc != 0 || !strings.Contains(out, "— none on") || !strings.Contains(out, "Paste this block for the tag") {
		t.Errorf("clean+green should PASS with rc 0: rc=%d\n%s", rc, out)
	}
	rc, out := mk(true, 0, 1)
	if rc != 1 || !strings.Contains(out, "INCONCLUSIVE") {
		t.Errorf("dirty+green should be INCONCLUSIVE with rc 1: rc=%d\n%s", rc, out)
	}
	if strings.Contains(out, "Paste this block for the tag") {
		t.Errorf("a dirty tree reported a PASS verdict:\n%s", out)
	}
	if rc, out := mk(false, 1, 1); rc != 1 || !strings.Contains(out, "Do not tag") {
		t.Errorf("a failing check should FAIL with rc 1: rc=%d\n%s", rc, out)
	}
	// NO GATE: nothing ran at all. Distinct from a pass, and it must not read as one.
	if rc, out := mk(false, 0, 0); rc != 1 || !strings.Contains(out, "NO GATE") {
		t.Errorf("zero checks run should be NO GATE with rc 1: rc=%d\n%s", rc, out)
	}
}

// A `-run` pattern matching nothing exits 0 and prints "ok" — so renaming a test away silently
// deletes a check while the gate stays green. ranCount() is how that is detected, and it must count
// TOP-LEVEL results only.
func TestGPU_zeroMatchedTestsIsDetectable(t *testing.T) {
	empty := newResults()
	if empty.ranCount() != 0 {
		t.Fatalf("an empty stream reports %d tests", empty.ranCount())
	}
	r := newResults()
	r.cur = "c"
	r.add(testEvent{Action: "pass", Package: "p", Test: "TestA"})
	r.add(testEvent{Action: "pass", Package: "p", Test: "TestA/sub"})
	if got := r.ranCount(); got != 1 {
		t.Fatalf("ranCount = %d, want 1 — subtests must not inflate it", got)
	}
}

// The drain group is DERIVED FROM A MARKER, never listed. A hand-kept -run list would be a constant
// restating a property, which is the drift shape this repo keeps finding — and a derivation that
// silently finds nothing is worse than a list, so zero is a failure the caller checks.
func TestGPU_drainGroupIsDerivedFromTheMarker(t *testing.T) {
	got := drainingTests()
	if len(got) == 0 {
		t.Skip("no drainsDevice() marker in cuda/ on this checkout — nothing to derive")
	}
	for _, name := range got {
		if !strings.HasPrefix(name, "Test") {
			t.Errorf("derived a non-test name: %q", name)
		}
	}
	// The derivation must be a set: a test calling the marker twice must appear once.
	seen := map[string]bool{}
	for _, n := range got {
		if seen[n] {
			t.Errorf("duplicate in the derived drain group: %s", n)
		}
		seen[n] = true
	}
}

// A group that fails and explains nothing is the silence-reads-as-health shape one level up in the
// tooling: a run killed by a signal, an OOM or a timeout emits neither "--- FAIL" nor a "file.go:N:"
// line, so the filter matches nothing and the explanation is empty for exactly the failures that are
// hardest to reproduce.
func TestGPU_detailFallsBackToTheRawTail(t *testing.T) {
	var buf strings.Builder
	g := &gpuGate{w: &buf}
	g.detail("signal: killed\nFAIL\tgithub.com/x/cuda\t612.004s\n", failLineRe)
	out := buf.String()
	if !strings.Contains(out, "NO ASSERTION LINE MATCHED") {
		t.Fatalf("no fallback for an assertion-less failure:\n%s", out)
	}
	if !strings.Contains(out, "signal: killed") {
		t.Errorf("the fallback did not carry the evidence:\n%s", out)
	}

	buf.Reset()
	g.detail("--- FAIL: TestX (1.00s)\n    x_test.go:9: cosine 0.31\n", failLineRe)
	if o := buf.String(); strings.Contains(o, "NO ASSERTION LINE MATCHED") || !strings.Contains(o, "cosine 0.31") {
		t.Errorf("a real assertion line was not used:\n%s", o)
	}
}

// A cosine of EXACTLY zero is an all-zero buffer — what a failed allocation leaves behind — and the
// note must state the READING without naming a mechanism, since the obvious one is disproven.
func TestGPU_vramNoteFiresOnlyOnExactZeroCosine(t *testing.T) {
	var buf strings.Builder
	g := &gpuGate{w: &buf}
	g.vramNote("    x_test.go:9: cosine 0.998000 vs golden")
	if buf.String() != "" {
		t.Errorf("fired on a non-zero cosine:\n%s", buf.String())
	}
	g.vramNote("    x_test.go:9: cosine 0.000000")
	out := buf.String()
	if !strings.Contains(out, "all-zero buffer") || !strings.Contains(out, "A12") {
		t.Fatalf("the note does not state the reading or point at the entry:\n%s", out)
	}
	if strings.Contains(strings.ToLower(out), "retention") {
		t.Errorf("the note names a mechanism the gate cannot see:\n%s", out)
	}
}

// ci_checks.py packs a multi-line step with \n escapes, which the shell expanded via printf '%b'.
func TestGPU_unescapeCIMatchesPrintfB(t *testing.T) {
	got := unescapeCI(`unformatted=$(gofmt -l .)\nif [ -n "$unformatted" ]; then\n  exit 1\nfi`)
	if strings.Count(got, "\n") != 3 {
		t.Fatalf("newlines not expanded: %q", got)
	}
	if strings.Contains(got, `\n`) {
		t.Errorf("literal escape survived: %q", got)
	}
}

// A skip is not a pass, and the notes block is what carries that into the verdict. Every skip must
// appear there — a gate that skips silently is the failure this whole file guards against.
func TestGPU_everySkipReachesTheNotesBlock(t *testing.T) {
	var buf strings.Builder
	g := newTestGate(&buf, "none", "alpha")
	g.dirty, g.commit, g.ran = false, "abc1234", 1
	g.grp("alpha")
	g.skip("the thing that did not run")
	rc := g.verdict()
	out := buf.String()
	if rc != 0 {
		t.Fatalf("a skip alone should not fail the gate: rc=%d\n%s", rc, out)
	}
	if !strings.Contains(out, "a skip is not a pass") || !strings.Contains(out, "the thing that did not run") {
		t.Fatalf("the skip did not reach the notes block:\n%s", out)
	}
}

// The PTX check reads each artifact's OWN recorded toolchain. A single-toolchain rebuild reports a
// false FAIL on every file from another era, which is what made this check unpassable for two minor
// versions — so the version must come from the file, never from the box's default.
func TestGPU_ptxVersionComesFromTheArtifact(t *testing.T) {
	dir := t.TempDir()
	p := dir + "/x.ptx"
	if err := writeFile(p, "//\n// Cuda compilation tools, release 12.6, V12.6.85\n//\n"); err != nil {
		t.Fatal(err)
	}
	if got := recordedNVRTC(p); got != "12.6.85" {
		t.Fatalf("recordedNVRTC = %q, want 12.6.85", got)
	}
	if err := writeFile(p, "// no provenance header\n"); err != nil {
		t.Fatal(err)
	}
	if got := recordedNVRTC(p); got != "" {
		t.Fatalf("recordedNVRTC = %q, want empty for an unversioned artifact", got)
	}
}
