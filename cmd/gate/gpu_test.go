package main

import (
	"bytes"
	"os"
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

// The Metal gate reported a FAIL and then printed a dozen PASSING parity lines as
// its "detail", because failLineRe matched every `file.go:N:` line a t.Log emits.
// The real cause — a SIGSEGV in objc_msgSend — appeared nowhere, and detail()'s
// own crash fallback never fired because the filter had already matched. This
// pins both halves against the shape of the run that exposed it.
func TestGPU_detailNamesTheCrashingTest(t *testing.T) {
	// Abbreviated from a real captured run: passing t.Log lines, then the crash.
	out := `=== RUN   TestAttention_GQA
    attention_test.go:132: GQA online-softmax attention on Metal GPU vs CPU: cosine=1.0000000 — PARITY
--- PASS: TestAttention_GQA (0.30s)
=== RUN   TestSAQVFusion_correctnessAndThroughput
SIGSEGV: segmentation violation
PC=0x1803d3c60 m=0 sigcode=2 addr=0x10
signal arrived during cgo execution

goroutine 2523 gp=0x303cfba82780 m=0 mp=0x1008af6a0 [syscall]:
github.com/ebitengine/purego/objc.ID.Send(...)
github.com/townsendmerino/aikit/gpu.(*Encoder).End(0x303d7f73a840)
github.com/townsendmerino/goinfer/metal.TestSAQVFusion_correctnessAndThroughput(0x303d8c1ba248)
	/x/metal/sa_qv_fusion_test.go:173 +0xc5c
r14     0x8000000000000000
r15     0x1ed2965b8
fault   0x10
FAIL	github.com/townsendmerino/goinfer/metal	119.290s
`
	var buf strings.Builder
	g := &gpuGate{w: &buf}
	g.detail(out, failLineRe)
	got := buf.String()

	if !strings.Contains(got, "PROCESS DIED") {
		t.Errorf("a signal death was not reported as one:\n%s", got)
	}
	if !strings.Contains(got, "TestSAQVFusion_correctnessAndThroughput") {
		t.Errorf("the detail does not NAME the test that died — the whole point:\n%s", got)
	}
	if strings.Contains(got, "PARITY") || strings.Contains(got, "cosine=1.0000000") {
		t.Errorf("passing t.Log lines were reported as failure detail:\n%s", got)
	}
	if strings.Contains(got, "r14") || strings.Contains(got, "0x8000000000000000") {
		t.Errorf("the excerpt ran into the register dump instead of stopping at it:\n%s", got)
	}
}

// Assertion lines still belong to the failure they came from — and only to it.
func TestGPU_detailKeepsFailingAssertionsAndDropsPassingLogs(t *testing.T) {
	out := `=== RUN   TestGood
    good_test.go:11: cosine=1.0000000 all fine
--- PASS: TestGood (0.10s)
=== RUN   TestBad
    bad_test.go:42: cosine 0.31 vs golden
--- FAIL: TestBad (0.20s)
FAIL	github.com/x/metal	1.000s
`
	var buf strings.Builder
	g := &gpuGate{w: &buf}
	g.detail(out, failLineRe)
	got := buf.String()

	if !strings.Contains(got, "--- FAIL: TestBad") {
		t.Errorf("the FAIL header is missing:\n%s", got)
	}
	if strings.Contains(got, "all fine") {
		t.Errorf("a passing test's log was reported as failure detail:\n%s", got)
	}
	if strings.Contains(got, "NO ASSERTION LINE MATCHED") {
		t.Errorf("a real test failure took the crash/unknown path:\n%s", got)
	}
}

// A filtered cell whose -run matches nothing is not a pass: zero tests ran, so it proves nothing,
// and the aggregate ran==0 check cannot see it once any other cell has run. This is the same shape
// as the qwen3next oracle, whose -run pattern could not match a required gate and reported
// "DID NOT RUN" for five weeks while the investigation went after a 163GB asset.
func TestGPUGate_emptyFilteredCellIsNotAPass(t *testing.T) {
	var buf bytes.Buffer
	g := &gpuGate{w: &buf, logDir: t.TempDir()}
	// A pattern that cannot match anything in ./cmd/gate/ — the rename case, exactly.
	g.run(cell{Name: "phantom", Pkgs: []string{"./"}, Run: "TestNoSuchTestNameExists_xyzzy"}, false)

	if len(g.emptyCells) != 1 {
		t.Fatalf("emptyCells = %v, want the phantom cell recorded", g.emptyCells)
	}
	if out := buf.String(); !strings.Contains(out, "MATCHED NO TESTS") {
		t.Errorf("an empty cell must say so loudly; got:\n%s", out)
	}
}

// The guard must not fire on a cell that legitimately ran something, and must not fire on an
// UNFILTERED cell either (an empty -run means "everything", so emptiness there is a different bug).
//
// Driven through the results directly rather than by running a real cell: pointing a cell at "./"
// from inside cmd/gate makes `go test` re-run this very suite, which re-runs the cell, which... The
// first draft of this test did exactly that and sat there for ten minutes.
func TestGPUGate_populatedAndUnfilteredCellsAreNotFlagged(t *testing.T) {
	var buf bytes.Buffer
	g := &gpuGate{w: &buf, logDir: t.TempDir()}

	populated := newResults()
	populated.add(testEvent{Action: "run", Package: "p", Test: "TestSomething"})
	populated.add(testEvent{Action: "pass", Package: "p", Test: "TestSomething"})
	g.noteIfEmpty(cell{Name: "real", Run: "TestSomething"}, populated)

	g.noteIfEmpty(cell{Name: "unfiltered", Run: ""}, newResults())

	if len(g.emptyCells) != 0 {
		t.Errorf("flagged a populated or unfiltered cell: %v", g.emptyCells)
	}
	if s := buf.String(); strings.Contains(s, "MATCHED NO TESTS") {
		t.Errorf("printed the empty-cell warning wrongly:\n%s", s)
	}

	// And the real case still fires.
	g.noteIfEmpty(cell{Name: "phantom", Run: "TestGone"}, newResults())
	if len(g.emptyCells) != 1 {
		t.Errorf("a filtered cell with no results must be flagged, got %v", g.emptyCells)
	}
}

// TestGPU_metalPrefillCellChecksVacuous pins V-21 (docs/review-2026-09-04.md): the sibling
// cells (metal-parity, metal-lifecycle) already gate on `cr.RC != 0 || cr.vacuous()`, but
// metal-prefill checked only cr.RC != 0 — a cell whose named tests (TestPrefillParity,
// TestPrefillNoNaN) all skipped for a reason unrelated to the os.Stat guard above it would have
// RC==0 and print PASS despite verifying nothing. Source-text guard rather than driving the real
// cell (which shells out to `go test` against a Metal checkpoint): the fix is a one-line addition
// to an existing condition, and what needs pinning is that the addition stays, not the mechanics
// of vacuous() itself (already exercised by the sibling cells' identical shape).
func TestGPU_metalPrefillCellChecksVacuous(t *testing.T) {
	src, err := os.ReadFile("gpu.go")
	if err != nil {
		t.Fatal(err)
	}
	body := string(src)
	i := strings.Index(body, `Name: "metal-prefill"`)
	if i < 0 {
		t.Fatal("metal-prefill cell not found — this guard is watching nothing")
	}
	j := strings.Index(body[i:], "\n}\n")
	if j < 0 {
		j = len(body) - i
	}
	block := body[i : i+j]
	// Comment lines are not the check — only look at actual code, or a stray comment mentioning
	// cr.vacuous() near removed code would fool this the same way a doc comment fooled the audit's
	// own G-07 finding.
	found := false
	for _, line := range strings.Split(block, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "//") {
			continue
		}
		if strings.Contains(line, "cr.vacuous()") {
			found = true
			break
		}
	}
	if !found {
		t.Error("metal-prefill's failure check lost cr.vacuous() — an all-skip run of " +
			"TestPrefillParity/TestPrefillNoNaN would print PASS having verified nothing (V-21)")
	}
}

// TestClassifyAdapterProbe_distinguishesNoAdapterFromProbeFailure pins the other half of V-21:
// a regex miss on the found-adapter line used to mean "no adapter" regardless of WHY it missed —
// a genuine "TestAdapterProbe ran and found nothing" (which prints its own explicit
// "ADAPTER_PROBE: none" line) looked identical to a build break of ./gpu/, a panic, or anything
// else that kept the subprocess from ever reaching either print.
func TestClassifyAdapterProbe_distinguishesNoAdapterFromProbeFailure(t *testing.T) {
	for name, tc := range map[string]struct {
		out          string
		wantPresent  bool
		wantBackend  string
		wantNoteText string // "" = no note expected
	}{
		"found an adapter": {
			out:         "=== RUN   TestAdapterProbe\nADAPTER_PROBE: backend=vulkan software=false\n--- PASS: TestAdapterProbe (0.01s)\n",
			wantPresent: true, wantBackend: "vulkan",
		},
		"genuinely no adapter (the test ran and said so)": {
			out:          "=== RUN   TestAdapterProbe\nADAPTER_PROBE: none (no adapter found)\n--- SKIP: TestAdapterProbe (0.00s)\n",
			wantPresent:  false,
			wantNoteText: "", // no note — this IS the hardware case
		},
		"build break (V-21): neither line ever printed": {
			out:          "# github.com/townsendmerino/goinfer/gpu\ngpu/adapter_probe_test.go:9:2: undefined: someSymbol\nFAIL\tgithub.com/townsendmerino/goinfer/gpu [build failed]\n",
			wantPresent:  false,
			wantNoteText: "not \"no hardware\"",
		},
	} {
		t.Run(name, func(t *testing.T) {
			present, backend, note := classifyAdapterProbe([]byte(tc.out), nil)
			if present != tc.wantPresent {
				t.Errorf("present = %v, want %v", present, tc.wantPresent)
			}
			if backend != tc.wantBackend {
				t.Errorf("backend = %q, want %q", backend, tc.wantBackend)
			}
			if tc.wantNoteText == "" && note != "" {
				t.Errorf("unexpected note for %q: %q", name, note)
			}
			if tc.wantNoteText != "" && !strings.Contains(note, tc.wantNoteText) {
				t.Errorf("note = %q, want it to contain %q (V-21)", note, tc.wantNoteText)
			}
		})
	}
}
