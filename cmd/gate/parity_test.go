package main

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

// The sweep's decision is a CHECKSET, and each of its four outcomes carries a different weight.
// These are the mutation checks for that arithmetic: change the outcome, the verdict must move.

func parityResults(t *testing.T, entries map[string]string) *results {
	t.Helper()
	r := newResults()
	r.cur = "cell"
	for name, act := range entries {
		r.add(testEvent{Action: "output", Package: "p", Test: name, Output: "=== RUN   " + name + "\n"})
		r.add(testEvent{Action: act, Package: "p", Test: name})
	}
	return r
}

func TestParity_fourOutcomesCarryDifferentWeights(t *testing.T) {
	checks := []gateCheck{
		{"fam-pass", "TestPasses"},
		{"fam-fail", "TestFails"},
		{"fam-first", "TestFirstRun"},
		{"fam-changed", "TestSourceChanged"},
		{"fam-skip", "TestSkips"},
		{"fam-missing", "TestNeverRan"},
	}
	res := parityResults(t, map[string]string{
		"TestPasses":        "pass",
		"TestFails":         "fail",
		"TestFirstRun":      "fail",
		"TestSourceChanged": "fail",
		"TestSkips":         "skip",
		// TestNeverRan deliberately absent
	})
	ledger := func(name string) string {
		switch name {
		case "TestFirstRun":
			return "FIRST-RUN"
		case "TestSourceChanged":
			return "SOURCE-CHANGED"
		}
		return "CONFIRMED"
	}
	rows, blockers, gaps, firstRuns := classifyChecks(res, checks, ledger, oneUnfilteredCell)

	// fail(CONFIRMED) + fail(SOURCE-CHANGED) + skip + missing = 4 blockers; the FIRST-RUN failure
	// is an ITEM, not a blocker.
	if blockers != 4 || firstRuns != 1 || gaps != 0 {
		t.Fatalf("blockers=%d firstRuns=%d gaps=%d, want 4/1/0", blockers, firstRuns, gaps)
	}
	want := map[string]string{
		"TestPasses":        "pass",
		"TestFails":         "FAIL (blocker)",
		"TestFirstRun":      "FIRST-RUN",
		"TestSourceChanged": "confirmed before this gate last changed",
		"TestSkips":         "SKIP - asset missing (blocker)",
		"TestNeverRan":      "DID NOT RUN (blocker)",
	}
	for _, r := range rows {
		if !strings.Contains(r.Mark, want[r.Test]) {
			t.Errorf("%s: mark %q does not contain %q", r.Test, r.Mark, want[r.Test])
		}
	}
}

// THE OUTCOME A TALLY CANNOT SEE. A required gate that produces no result at all leaves PASS/SKIP/
// FAIL counts entirely undisturbed — a renamed test, a -run filter that stopped matching, a package
// that failed to build. If MISSING ever stopped blocking, the sweep would go green on a gate that
// silently left the suite, which is the single worst failure this gate can have.
func TestParity_missingGateBlocksEvenThoughNothingFailed(t *testing.T) {
	res := parityResults(t, map[string]string{"TestPasses": "pass"})
	_, blockers, _, _ := classifyChecks(res,
		[]gateCheck{{"a", "TestPasses"}, {"b", "TestRenamedAway"}},
		func(string) string { return "CONFIRMED" }, oneUnfilteredCell)
	if blockers != 1 {
		t.Fatalf("blockers=%d, want 1 — a gate that never ran must block", blockers)
	}
	// And the tally is genuinely blind to it: nothing failed and nothing skipped.
	var fail, skip int
	for _, a := range res.final {
		switch a {
		case "fail":
			fail++
		case "skip":
			skip++
		}
	}
	if fail != 0 || skip != 0 {
		t.Fatalf("the premise broke: fail=%d skip=%d, want 0/0", fail, skip)
	}
}

// A gate in assetNeverBuilt is a COVERAGE GAP, not a blocker: no invocation on any machine can make
// it green, and a permanent blocker is not a gate — it is an override habit. The list is empty
// today, so this exercises the mechanism rather than a live entry.
func TestParity_assetNeverBuiltIsAGapNotABlocker(t *testing.T) {
	assetNeverBuilt["TestNoAssetAnywhere"] = true
	defer delete(assetNeverBuilt, "TestNoAssetAnywhere")

	res := parityResults(t, map[string]string{"TestNoAssetAnywhere": "skip", "TestOtherSkip": "skip"})
	_, blockers, gaps, _ := classifyChecks(res,
		[]gateCheck{{"a", "TestNoAssetAnywhere"}, {"b", "TestOtherSkip"}},
		func(string) string { return "CONFIRMED" }, oneUnfilteredCell)
	if blockers != 1 || gaps != 1 {
		t.Fatalf("blockers=%d gaps=%d, want 1/1 — the listed gate is a gap, the unlisted skip blocks", blockers, gaps)
	}
}

// A broken or absent ledger must fail SAFE: an unreachable classifier means the failure stays a
// BLOCKER. The other direction would let one broken script silently downgrade every regression in
// the sweep to an item.
func TestParity_unreachableLedgerKeepsFailuresBlocking(t *testing.T) {
	res := parityResults(t, map[string]string{"TestFails": "fail"})
	_, blockers, _, firstRuns := classifyChecks(res, []gateCheck{{"a", "TestFails"}},
		func(string) string { return "CONFIRMED" }, oneUnfilteredCell) // what ledgerClassify returns when python3 errors
	if blockers != 1 || firstRuns != 0 {
		t.Fatalf("blockers=%d firstRuns=%d, want 1/0", blockers, firstRuns)
	}
}

// lookupTop reproduces `grep '^--- X: NAME (' | tail -1`: LAST result wins (the sweep runs some
// gates in both the plain and the realckpt cell) and the match is EXACT (the trailing `(` in that
// grep is what stopped TestFoo from matching TestFooBar).
func TestParity_lookupTopIsExactAndLastWins(t *testing.T) {
	r := newResults()
	r.cur = "plain"
	r.add(testEvent{Action: "skip", Package: "p", Test: "TestQwen35Real_gate2FullModel"})
	r.add(testEvent{Action: "pass", Package: "p", Test: "TestQwen35Real_gate2FullModelExtra"})
	r.cur = "realckpt"
	r.add(testEvent{Action: "pass", Package: "p", Test: "TestQwen35Real_gate2FullModel"})

	if act, seen := r.lookupTop("TestQwen35Real_gate2FullModel"); !seen || act != "pass" {
		t.Fatalf("lookupTop = %q/%v, want pass/true — the realckpt cell's result must win", act, seen)
	}
	if act, seen := r.lookupTop("TestQwen35Real"); seen {
		t.Fatalf("prefix matched: %q — the name match must be exact", act)
	}
}

// The catch-all is the "a family we forgot" net: a parity/gate/golden-shaped test that skipped and
// is in no list. It must not re-report the checkset's own gates.
func TestParity_catchAllFindsUnlistedSkipsOnly(t *testing.T) {
	res := parityResults(t, map[string]string{
		"TestListed_parity":     "skip",
		"TestForgotten_parity":  "skip",
		"TestSomeGolden_gate":   "skip",
		"TestUnrelatedThing":    "skip", // not parity/gate/golden-shaped
		"TestForgotten_passing": "pass", // not a skip
	})
	got := catchAllSkips(res, []gateCheck{{"a", "TestListed_parity"}})
	want := []string{"TestForgotten_parity", "TestSomeGolden_gate"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("catchAllSkips = %v, want %v", got, want)
	}
}

// PARITY_ROW lines are emitted with fmt.Printf, so they arrive as ordinary output events. Collecting
// them during the single parse is what lets EMIT_MANIFEST work without re-grepping a log.
func TestParity_collectsManifestRowsInStreamOrder(t *testing.T) {
	r := newResults()
	r.cur = "c"
	r.add(testEvent{Action: "output", Package: "p", Test: "TestA",
		Output: `PARITY_ROW {"family":"phi3","method":"real-model-oracle"}` + "\n"})
	r.add(testEvent{Action: "output", Package: "p", Output: `PARITY_ROW {"family":"cohere","method":"real-model-oracle"}` + "\n"})
	if len(r.parityRows) != 2 {
		t.Fatalf("collected %d rows, want 2 (one test-level, one package-level)", len(r.parityRows))
	}
	if got := jsonField(r.parityRows[0], "family"); got != "phi3" {
		t.Errorf("family = %q, want phi3", got)
	}
	if got := jsonField(r.parityRows[1], "family"); got != "cohere" {
		t.Errorf("family = %q, want cohere", got)
	}
}

// parseExports reads what asset_registry.py writes for the shell to source.
func TestParity_parseExports(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/assets.env"
	body := "export GOINFER_PREQUANT_GGUF=\"/home/x/models/a.gguf\"\nexport GOINFER_QWEN35_REAL=\"/home/x/models/q\"\n\n"
	if err := writeFile(path, body); err != nil {
		t.Fatal(err)
	}
	got := parseExports(path)
	if got["GOINFER_PREQUANT_GGUF"] != "/home/x/models/a.gguf" || got["GOINFER_QWEN35_REAL"] != "/home/x/models/q" {
		t.Fatalf("parseExports = %v", got)
	}
}

func writeFile(path, body string) error { return os.WriteFile(path, []byte(body), 0o644) }

// ---------------------------------------------------------------------------
// Composition census (migrated from scripts/sweep_composition.py).
// ---------------------------------------------------------------------------

// A gate whose test source cannot be located is UNKNOWN, never f32. Defaulting would inflate the
// f32 count with gates nobody checked — the opposite of what the composition is for, and it would
// hide exactly the collapse the census exists to catch.
func TestComposition_unlocatableGateIsUnknownNotF32(t *testing.T) {
	old := parityGates
	parityGates = []gateCheck{{"ghost", "TestThisFunctionDoesNotExistAnywhere"}}
	defer func() { parityGates = old }()

	var buf strings.Builder
	if rc := composition(&buf, true); rc != 0 {
		t.Fatalf("rc=%d", rc)
	}
	out := buf.String()
	if !strings.Contains(out, "UNKNOWN") {
		t.Fatalf("missing gate not reported UNKNOWN:\n%s", out)
	}
	if strings.Contains(out, "f32=1") {
		t.Errorf("an unlocatable gate was counted as f32:\n%s", out)
	}
	if !strings.Contains(out, "NOT counted as f32") {
		t.Errorf("the report does not say what UNKNOWN means:\n%s", out)
	}
}

// The collapse warning is the entire reason the composition is printed: an accurate pass count
// cannot distinguish "the axis is covered" from "the axis collapsed to one value".
func TestComposition_warnsWhenTheQuantAxisCollapses(t *testing.T) {
	old := parityGates
	// Two real gates that both derive the same quant — a single-valued axis.
	parityGates = []gateCheck{{"a", "TestGGUF_Q8_0_parity"}, {"b", "TestGGUF_Q4_0_parity"}}
	defer func() { parityGates = old }()

	var buf strings.Builder
	composition(&buf, false)
	q := buf.String()
	if strings.Count(q, "quant :") != 1 {
		t.Fatalf("unexpected report shape:\n%s", q)
	}
	// Whether it warns depends on whether those two gates' sources declare one quant or several;
	// assert the INVARIANT instead: the warning appears if and only if the axis has one value.
	oneValue := !strings.Contains(strings.SplitN(strings.SplitN(q, "quant :  ", 2)[1], "\n", 2)[0], "  ")
	warned := strings.Contains(q, "collapsed to a single value")
	if oneValue != warned {
		t.Fatalf("collapse warning disagrees with the axis it describes (oneValue=%v warned=%v):\n%s",
			oneValue, warned, q)
	}
}

// Composite labels must be atomised before the cross-gate comparison, or a gate whose test file
// drives two quantizations produces a difference that is purely NOTATIONAL — a permanent false
// positive in the check built to make real differences visible.
func TestComposition_atomisesCompositeLabels(t *testing.T) {
	got := atoms(map[string]bool{"int4/int8": true, "f32": true})
	for _, want := range []string{"int4", "int8", "f32"} {
		if !got[want] {
			t.Errorf("atoms() lost %q: %v", want, sortedSet(got))
		}
	}
	if got["int4/int8"] {
		t.Errorf("atoms() kept the composite label: %v", sortedSet(got))
	}
}

func TestComposition_loaderAxisComesFromTheName(t *testing.T) {
	for name, want := range map[string]string{
		"TestGGUF_qwen2_parity":        "gguf",
		"TestQwen2_forwardParity":      "safetensors",
		"TestLoadGGUF_tinyllamaParity": "gguf",
	} {
		if got := loaderOf(name); got != want {
			t.Errorf("loaderOf(%q) = %q, want %q", name, got, want)
		}
	}
}

// A required gate that the cell's -run filter cannot match never executes, and the sweep reports
// it as DID NOT RUN — indistinguishable from a missing asset. That is exactly what happened to
// TestQwen3NextReal_oracle: named "Real_oracle" while the filter accepted only "Qwen35|Real_gate",
// so it was unreachable by construction while every investigation hunted the 163GB checkpoint.
func TestRealckptCellCanReachEveryGate(t *testing.T) {
	cells := parityCells(nil, true, "1m")
	var rc *cell
	for i := range cells {
		for _, tag := range cells[i].Tags {
			if tag == "realckpt" {
				rc = &cells[i]
			}
		}
	}
	if rc == nil {
		t.Fatal("no realckpt cell built with realckpt=true")
	}
	re, err := regexp.Compile(rc.Run)
	if err != nil {
		t.Fatalf("realckpt cell -run %q does not compile: %v", rc.Run, err)
	}
	for _, g := range parityRealckptGates {
		if !re.MatchString(g.Test) {
			t.Errorf("required gate %s (%s) is UNREACHABLE: -run %q never matches it, so the sweep "+
				"can only ever report it as DID NOT RUN", g.Test, g.Family, rc.Run)
		}
	}
}

// The base cell must stay unfiltered: every non-realckpt required gate lives in ./decoder/ or
// ./tokenizer/ and is reached only because nothing narrows it.
func TestBaseCellIsUnfiltered(t *testing.T) {
	cells := parityCells(nil, false, "1m")
	if len(cells) != 1 {
		t.Fatalf("realckpt=false built %d cells, want 1", len(cells))
	}
	if cells[0].Run != "" {
		t.Errorf("base cell -run = %q, want empty; a filter here would silently drop required "+
			"gates the same way the realckpt filter dropped the qwen3next oracle", cells[0].Run)
	}
}

// oneUnfilteredCell stands in for a normal sweep cell in the classifier tests: unfiltered, so every
// gate is reachable and "no result" means the test really did not report.
var oneUnfilteredCell = []cell{{Name: "./decoder/", Pkgs: []string{"./decoder/"}}}

// The two causes of "no result" are fixed in different places and used to read identically. A gate
// no -run pattern selects cannot be made to run by any asset or machine; one that IS selected and
// still reported nothing is a build failure, an absent asset, or a dead cell. TestQwen3NextReal_oracle
// was the first kind and the report implied the second, which sent three sessions after a 163 GB
// checkpoint that was fine.
func TestParity_missingGateSaysWhichCause(t *testing.T) {
	res := parityResults(t, map[string]string{"TestPasses": "pass"})
	checks := []gateCheck{{"a", "TestPasses"}, {"b", "TestQwen3NextReal_oracle"}}

	// Filtered cell that cannot match the oracle — the real bug.
	rows, _, _, _ := classifyChecks(res, checks, func(string) string { return "CONFIRMED" },
		[]cell{{Name: "realckpt", Run: "Qwen35|Real_gate"}})
	got := rowFor(t, rows, "TestQwen3NextReal_oracle")
	if !strings.Contains(got, "UNREACHABLE") {
		t.Errorf("an unselectable gate must say so; got %q", got)
	}

	// Same gate, a cell that DOES select it: a different cause, and it must not say UNREACHABLE.
	rows, _, _, _ = classifyChecks(res, checks, func(string) string { return "CONFIRMED" },
		[]cell{{Name: "realckpt", Run: "Real_gate|Real_oracle"}})
	got = rowFor(t, rows, "TestQwen3NextReal_oracle")
	if strings.Contains(got, "UNREACHABLE") {
		t.Errorf("a selectable gate that reported nothing is not unreachable; got %q", got)
	}
	if !strings.Contains(got, "reported nothing") {
		t.Errorf("want the other cause named; got %q", got)
	}
}

func rowFor(t *testing.T, rows []checkRow, test string) string {
	t.Helper()
	for _, r := range rows {
		if r.Test == test {
			return r.Mark
		}
	}
	t.Fatalf("no row for %s", test)
	return ""
}
