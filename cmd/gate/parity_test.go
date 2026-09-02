package main

import (
	"encoding/json"
	"os"
	"path/filepath"
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
	rc := realckptCell(t)
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

// EVERY REQUIRED GATE MUST HAVE A CONFIRMED PRIOR RESULT, OR SAY IN CODE WHY IT DOES NOT.
//
// B14's first-run outcome is only safe while the ledger is maintained: a gate with no entry fails
// as an ITEM, not a blocker, so an UNMAINTAINED ledger silently converts regressions into notes.
// That is not hypothetical — the ledger was bulk-seeded on 2026-08-14 and never touched again, and
// by 2026-09-02 five required gates were still first-run INCLUDING TestInt4_forwardParity, which
// the gate list itself calls "the broadest quant check here". Each of the five had a PASS sitting
// in the v0.15.0 sweep log the whole time; `reconcile` printed them every run and never exits
// non-zero (deliberately — see gate_ledger.py), and nothing else looked. So the assertion lives
// here, where CI already runs it.
//
// A missing entry is a red test with one of two fixes, both deliberate: promote the gate from a
// sweep log, or add it to neverConfirmed with a written reason.
func TestParity_everyRequiredGateIsConfirmed(t *testing.T) {
	root, err := repoRoot()
	if err != nil {
		t.Fatalf("cannot locate the repo root: %v", err)
	}
	b, err := os.ReadFile(filepath.Join(root, "testdata", "gate_ledger.json"))
	if err != nil {
		// NOT a skip. An unreadable ledger is the state in which every failing gate becomes a
		// non-blocking item, which is precisely what this test exists to notice.
		t.Fatalf("cannot read testdata/gate_ledger.json: %v", err)
	}
	var led struct {
		Entries []struct {
			Gate       string `json:"gate"`
			Value      string `json:"value"`
			PromotedBy string `json:"promoted_by"`
			Date       string `json:"date"`
			Commit     string `json:"commit"`
		} `json:"entries"`
	}
	if err := json.Unmarshal(b, &led); err != nil {
		t.Fatalf("gate_ledger.json does not parse: %v", err)
	}
	confirmed := map[string]bool{}
	for _, e := range led.Entries {
		// An entry missing a required field is a NOTE, NOT A CONFIRMATION (gate_ledger.py's five
		// required fields). Counting it here would let a blank row satisfy this assertion, which is
		// the same false-green one level down.
		if e.Gate == "" || e.Value == "" || e.PromotedBy == "" || e.Date == "" || e.Commit == "" {
			t.Errorf("ledger entry for %q is missing a required field (gate/value/promoted_by/date/"+
				"commit) — a note, not a confirmation: %+v", e.Gate, e)
			continue
		}
		confirmed[e.Gate] = true
	}

	required := map[string]bool{}
	for _, g := range append(append([]gateCheck{}, parityGates...), parityRealckptGates...) {
		required[g.Test] = true
		exempt, isExempt := neverConfirmed[g.Test]
		pending, isPending := awaitingFirstConfirmation[g.Test]
		states := 0
		for _, in := range []bool{confirmed[g.Test], isExempt, isPending} {
			if in {
				states++
			}
		}
		switch {
		case states > 1:
			// The three states answer the same question differently, so belonging to two of them is
			// not redundancy — it is two answers with nothing choosing between them.
			t.Errorf("%s is in more than one of {ledger, neverConfirmed, awaitingFirstConfirmation}; "+
				"they disagree about whether its failure blocks", g.Test)
		case confirmed[g.Test]:
		case isExempt && strings.TrimSpace(exempt) == "":
			t.Errorf("%s is in neverConfirmed with an EMPTY reason — an unexplained exemption is "+
				"the state this map exists to prevent", g.Test)
		case isExempt:
			// Deliberately accepted as permanently first-run.
		case isPending && !isoDatePrefix(pending):
			t.Errorf("%s is in awaitingFirstConfirmation without a leading ISO date (got %q) — the "+
				"date is what makes a pending entry that has quietly become permanent visible",
				g.Test, pending)
		case isPending:
			// Never run, so nothing to confirm yet.
		default:
			t.Errorf("required gate %s (%s) has no ledger entry, so it is FIRST-RUN: a failure is "+
				"reported as an ITEM and cannot block a tag. Promote it from a sweep log "+
				"(scripts/gate_ledger.py promote --gate %s --value PASS --by <you>), or — if it has "+
				"never run — add it to awaitingFirstConfirmation with the date, or to neverConfirmed "+
				"with a reason.", g.Test, g.Family, g.Test)
		}
	}
	for _, m := range []struct {
		name string
		set  map[string]string
	}{{"neverConfirmed", neverConfirmed}, {"awaitingFirstConfirmation", awaitingFirstConfirmation}} {
		for gate, reason := range m.set {
			if !required[gate] {
				t.Errorf("%s names %q (%q), which is not a required gate — a stale exemption reads "+
					"as a covered gate", m.name, gate, reason)
			}
		}
	}
}

// isoDatePrefix reports whether a reason opens with a YYYY-MM-DD date.
func isoDatePrefix(s string) bool {
	if len(s) < 10 {
		return false
	}
	for i, c := range s[:10] {
		if i == 4 || i == 7 {
			if c != '-' {
				return false
			}
		} else if c < '0' || c > '9' {
			return false
		}
	}
	return true
}

// A FAILURE IN A TEST NOBODY LISTED IS STILL A FAILURE. The sweep's decision is a checkset, so
// `blockers` came only from the named gates and a FAIL anywhere else changed nothing — 36 family
// parity tests (Cohere, LFM2, Laguna, InternLM, GLM4-MoE, the VL text parities, 12 *Real_gates)
// could go red and the verdict still read ALL REQUIRED GATES GREEN, exit 0.
func TestParity_unlistedFailureIsABlocker(t *testing.T) {
	res := parityResults(t, map[string]string{
		"TestListedGate":           "pass",
		"TestCohere_forwardParity": "fail",
		"TestSomethingElse":        "fail", // not even parity-shaped: still a failure
		"TestUnlistedSkip":         "skip", // a skip is the advisory half, not a blocker
	})
	got := unlistedFailures(res, []gateCheck{{"a", "TestListedGate"}})
	want := []string{"TestCohere_forwardParity", "TestSomethingElse"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("unlistedFailures = %v, want %v", got, want)
	}
}

// B14 SURVIVES THE NEW RULE, and it survives because the exclusion is EXACT. A named gate failing
// with no confirmed prior result is an ITEM; if the unlisted-failure sweep re-counted it under a
// looser match, the first-run outcome would be silently repealed.
func TestParity_firstRunGateIsNotReCountedAsAnUnlistedFailure(t *testing.T) {
	res := parityResults(t, map[string]string{"TestFirstRun": "fail"})
	checks := []gateCheck{{"a", "TestFirstRun"}}
	if got := unlistedFailures(res, checks); len(got) != 0 {
		t.Fatalf("unlistedFailures = %v, want none — a NAMED gate is classified by the checkset", got)
	}
	_, blockers, _, firstRuns := classifyChecks(res, checks,
		func(string) string { return "FIRST-RUN" }, oneUnfilteredCell)
	if blockers != 0 || firstRuns != 1 {
		t.Fatalf("blockers=%d firstRuns=%d, want 0/1", blockers, firstRuns)
	}
}

// A test whose name CONTAINS a gate name is a different test, and its failure must block. This is
// the one place the containment rule catchAllSkips uses would be actively wrong.
func TestParity_unlistedFailureMatchIsExactNotContainment(t *testing.T) {
	res := parityResults(t, map[string]string{"TestListedGateExtra": "fail"})
	got := unlistedFailures(res, []gateCheck{{"a", "TestListedGate"}})
	if len(got) != 1 || got[0] != "TestListedGateExtra" {
		t.Fatalf("unlistedFailures = %v, want [TestListedGateExtra] — containment would hide it", got)
	}
}

// Last-writer-wins, the same rule lookupTop applies to the checkset: several gates run in BOTH the
// plain and the realckpt cell, and the plain cell's skip/fail is not the sweep's answer.
func TestParity_unlistedFailureUsesTheLastCellsResult(t *testing.T) {
	r := newResults()
	r.cur = "plain"
	r.add(testEvent{Action: "fail", Package: "p", Test: "TestRunsInBothCells"})
	r.cur = "realckpt"
	r.add(testEvent{Action: "pass", Package: "p", Test: "TestRunsInBothCells"})
	if got := unlistedFailures(r, nil); len(got) != 0 {
		t.Fatalf("unlistedFailures = %v, want none — the realckpt cell's PASS is the result", got)
	}
}

// EVERY GATE-SHAPED realckpt TEST IS LISTED, ONE WAY OR THE OTHER.
//
// The five-week TestQwen3NextReal_oracle incident was a gate no -run could select. Five more were
// in that state on 2026-09-02 — TestGemma4_26B_gate, TestGlm4MoeAir_gate, TestLagunaGGUF_gate,
// TestQwen38GGUF_gate, TestGptOssReal_logitParity — and because they were also in no list, the
// sweep could not even report them as DID NOT RUN. It had no way to say a word about them.
func TestRealckptGateIsListedOrExplicitlyNotRequired(t *testing.T) {
	root, err := repoRoot()
	if err != nil {
		t.Fatalf("cannot locate the repo root: %v", err)
	}
	found := realckptGateTests(root)
	if len(found) == 0 {
		// NOT a skip: an empty scan is how this assertion would silently stop asserting, and it is
		// also what would silently narrow the cell's -run.
		t.Fatalf("no gate-shaped test found in any //go:build realckpt file under %v — the scan is "+
			"broken, and a broken scan both empties this check and narrows the sweep's -run", realckptDirs)
	}
	required := map[string]bool{}
	for _, g := range parityRealckptGates {
		required[g.Test] = true
	}
	for _, test := range found {
		reason, excluded := realckptNotRequired[test]
		switch {
		case required[test] && excluded:
			t.Errorf("%s is BOTH a required gate and in realckptNotRequired — the two disagree "+
				"about whether its SKIP blocks a tag", test)
		case required[test]:
		case excluded && strings.TrimSpace(reason) == "":
			t.Errorf("%s is in realckptNotRequired with an EMPTY reason — an unexplained exemption "+
				"is the state that map exists to prevent", test)
		case excluded:
		default:
			t.Errorf("realckpt gate %s is in neither parityRealckptGates nor realckptNotRequired. "+
				"Unlisted means the sweep has nothing to say about it: not required, and not even "+
				"reportable as DID NOT RUN. Add it to one list or the other.", test)
		}
	}
	inTree := map[string]bool{}
	for _, test := range found {
		inTree[test] = true
	}
	for test := range realckptNotRequired {
		if !inTree[test] {
			t.Errorf("realckptNotRequired names %q, which is not a gate-shaped test in any realckpt "+
				"file — a stale exemption reads as a gate somebody considered", test)
		}
	}
}

// The derived -run must reach every gate it knows about, INCLUDING the required gates that are not
// gate-shaped: TestQwen35GGUF_weightDiff is required and matches neither `_gate`, `_oracle` nor
// `Parity`, so the scan alone would drop it and the union is load-bearing.
func TestRealckptRunReachesEveryScannedAndRequiredGate(t *testing.T) {
	root, err := repoRoot()
	if err != nil {
		t.Fatalf("cannot locate the repo root: %v", err)
	}
	if _, note := realckptRun(); strings.HasPrefix(note, "!!") {
		t.Fatalf("realckptRun fell back: %s", note)
	}
	// THE CELL'S OWN -run, not realckptRun() in isolation. A derivation nothing wires up is a
	// function with a test, not a gate — and re-pinning the cell to legacyRealckptRun has to be the
	// thing that goes red, since that is the state this fix moved away from.
	pattern := realckptCell(t).Run
	if pattern == legacyRealckptRun {
		t.Fatalf("the realckpt cell is pinned to the hand-written %q again; it must use the "+
			"pattern derived from the tagged files", legacyRealckptRun)
	}
	re, err := regexp.Compile(pattern)
	if err != nil {
		t.Fatalf("cell -run %q does not compile: %v", pattern, err)
	}
	for _, test := range realckptGateTests(root) {
		if !re.MatchString(test) {
			t.Errorf("derived -run does not select %s", test)
		}
	}
	for _, g := range parityRealckptGates {
		if !re.MatchString(g.Test) {
			t.Errorf("derived -run does not select REQUIRED gate %s (%s)", g.Test, g.Family)
		}
	}
	// And it is anchored: a pattern that also matched everything containing a gate name would drag
	// the realckpt tag's perf and diagnostic tests into a release sweep.
	if re.MatchString("TestQwen35GGUF_gateExtraSlowDiagnostic") {
		t.Errorf("derived -run %q is unanchored — it selects tests merely CONTAINING a gate name", pattern)
	}
}

// THE TRAP THE SCAN MUST NOT FALL INTO. decoder/int4_golden_test.go is an ordinary untagged file
// that DISCUSSES "//go:build realckpt" in a comment, and TestInt4_forwardParity is a required gate
// of the PLAIN cell. A substring scan pulls that file into the realckpt cell and then demands the
// gate be listed among the realckpt gates, which it is not and should not be.
func TestRealckptScanReadsTheBuildLineNotTheProse(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "decoder"), 0o755); err != nil {
		t.Fatal(err)
	}
	write := func(name, body string) {
		if err := os.WriteFile(filepath.Join(dir, "decoder", name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("prose_test.go", "package decoder\n\n// sits behind `//go:build realckpt` plus a missing "+
		"checkpoint.\nfunc TestProse_forwardParity(t *testing.T) {}\n")
	write("tagged_test.go", "//go:build realckpt\n\npackage decoder\n\n"+
		"func TestTagged_gate(t *testing.T) {}\nfunc TestTagged_speed(t *testing.T) {}\n")
	write("combined_test.go", "//go:build realckpt && cgo\n\npackage decoder\n\n"+
		"func TestCombined_oracle(t *testing.T) {}\n")
	write("other_test.go", "//go:build realckpt_lookalike\n\npackage decoder\n\n"+
		"func TestLookalike_gate(t *testing.T) {}\n")

	got := realckptGateTests(dir)
	want := []string{"TestCombined_oracle", "TestTagged_gate"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("realckptGateTests = %v, want %v (prose is not a build tag; realckpt_lookalike is "+
			"not realckpt; _speed is not gate-shaped)", got, want)
	}
}

// THE FIX MUST NOT SILENTLY DROP WHAT THE OLD PATTERN RAN. legacyRealckptRun's bare "Qwen35"
// alternative selected four tests that are not gate-shaped and are on no list; they have been
// running in every sweep. A filter fix that quietly stops running them is a coverage loss dressed
// as a correctness fix, which is the shape this repo keeps catching.
func TestRealckptRunIsAdditiveOverTheLegacyPattern(t *testing.T) {
	root, err := repoRoot()
	if err != nil {
		t.Fatalf("cannot locate the repo root: %v", err)
	}
	pattern, _ := realckptRun()
	re := regexp.MustCompile(pattern)
	legacy := regexp.MustCompile(legacyRealckptRun)
	for _, test := range realckptTests(root, legacy.MatchString) {
		if !re.MatchString(test) {
			t.Errorf("%s was selected by the legacy -run and the derived one drops it — removing a "+
				"test from the sweep is its own change, not a side effect of fixing the filter", test)
		}
	}
}

// realckptCell returns the realckpt cell parityCells builds, so assertions read the -run the sweep
// will actually run rather than a function's return value nothing is wired to.
func realckptCell(t *testing.T) cell {
	t.Helper()
	for _, c := range parityCells(nil, true, "1m") {
		for _, tag := range c.Tags {
			if tag == "realckpt" {
				return c
			}
		}
	}
	t.Fatal("no realckpt cell built with realckpt=true")
	return cell{}
}

// THROUGH A REAL CELL RUN, NOT A HAND-BUILT results. The pieces below were each provable in
// isolation while the sweep still exited 0, because what was broken was the ARITHMETIC IN THE
// CALLER: `blockers` came only from the checkset. So this drives runCell over a scratch module and
// asserts on what extraBlockers — runParity's one call site for both categories — returns.
func TestParity_extraBlockersCountsUnlistedFailuresAndDeadCells(t *testing.T) {
	const src = `package scratch

import "testing"

func TestNamedGate(t *testing.T)          {}
func TestCohere_forwardParity(t *testing.T) { t.Fatalf("cosine 0.31, want >= 0.99") }
`
	cfg := &gateConfig{Name: "parity", Decision: "checkset", TopLevelOnly: true, RCIsFailure: true}
	_, _, res, cells := runScratch(t, cfg, scratchModule(t, src))

	var buf strings.Builder
	got := extraBlockers(&buf, res, []gateCheck{{"a", "TestNamedGate"}}, cells, false)
	if got != 1 {
		t.Fatalf("extraBlockers = %d, want 1 (the unlisted parity failure)\n%s", got, buf.String())
	}
	if !strings.Contains(buf.String(), "TestCohere_forwardParity") {
		t.Errorf("the blocking failure is not named in the report:\n%s", buf.String())
	}
	// The premise: this failure is invisible to the checkset, which is why it needed a second path.
	_, checksetBlockers, _, _ := classifyChecks(res, []gateCheck{{"a", "TestNamedGate"}},
		func(string) string { return "CONFIRMED" }, oneUnfilteredCell)
	if checksetBlockers != 0 {
		t.Fatalf("the premise broke: the checkset already blocks (%d) — this test proves nothing",
			checksetBlockers)
	}
}

// A CELL THAT DIED WITHOUT A SINGLE --- FAIL LINE. A panic in a goroutine, a fatal error, a timeout
// or a build failure aborts `go test` with a non-zero rc and no per-test result, and the sweep ran
// with RCIsFailure:false — so it delivered a verdict about a cell that never finished.
func TestParity_deadCellIsABlocker(t *testing.T) {
	const src = `package scratch

import (
	"os"
	"testing"
)

func TestExitsHard(t *testing.T) { os.Exit(3) }
`
	cfg := &gateConfig{Name: "parity", Decision: "checkset", TopLevelOnly: true, RCIsFailure: true}
	_, _, res, cells := runScratch(t, cfg, scratchModule(t, src))
	if len(cells) != 1 || !cells[0].Hidden {
		t.Fatalf("the premise broke: cells=%+v (want one cell marked Hidden)", cells)
	}
	var buf strings.Builder
	if got := extraBlockers(&buf, res, nil, cells, false); got != 1 {
		t.Fatalf("extraBlockers = %d, want 1 for a cell that exited rc=%d with no FAIL line\n%s",
			got, cells[0].RC, buf.String())
	}

	// RCIsFailure:false is the state this fixed, and it must be visibly different: nothing at all.
	cfgOld := &gateConfig{Name: "parity", Decision: "checkset", TopLevelOnly: true, RCIsFailure: false}
	_, _, resOld, cellsOld := runScratch(t, cfgOld, scratchModule(t, src))
	var bufOld strings.Builder
	if got := extraBlockers(&bufOld, resOld, nil, cellsOld, false); got != 0 {
		t.Fatalf("premise broke: the old config already blocked (%d)", got)
	}
}
