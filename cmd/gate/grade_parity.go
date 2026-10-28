package main

import (
	"fmt"
	"io"
	"regexp"
	"sort"
	"strings"
)

// checkRow is one named gate's outcome, ready to print.
type checkRow struct{ Family, Test, Mark string }

// classifyChecks applies the sweep's four-outcome decision to the named gates. Split out from
// runParity so the decision — which is the entire gate — is mutation-testable without a real sweep.
// ledger is injected for the same reason.
func classifyChecks(res *results, checks []gateCheck, ledger func(string) string, cells []cell) (rows []checkRow, blockers, gaps, firstRuns int) {
	for _, g := range checks {
		act, seen := res.lookupTop(g.Test)
		var mark string
		switch {
		case !seen:
			// THE OUTCOME A TALLY CANNOT SEE. A required gate that produced no result at all — a
			// renamed test, a -run filter that stopped matching, a package that failed to build —
			// leaves a pass/skip/fail count entirely undisturbed. It is a blocker.
			mark = "⛔ DID NOT RUN (blocker) — " + whyNoResult(g.Test, cells)
			blockers++
		case act == "pass":
			mark = "✅ pass"
		case act == "fail" && neverConfirmed[g.Test] != "":
			// neverConfirmed's own promise ("its failure is reported as an ITEM and never blocks a tag") is checked
			// FIRST, so a live sweep enforces it and not only the static TestParity_everyRequiredGateIsConfirmed.
			mark = "FAIL - but in neverConfirmed (NOT a blocker): " + neverConfirmed[g.Test]
			gaps++
		case act == "fail":
			// THE FOURTH OUTCOME (B14). A gate failing with no confirmed prior result is asserting a
			// DELTA IT HAS NO SECOND POINT TO COMPUTE, so it is an ITEM, not a blocker — the change
			// that made a failure visible is the one change that provably did not cause it. This is
			// about ATTRIBUTION, not harmlessness: the first observed value must not be banked as a
			// baseline without a person deciding it is correct (`gate ledger promote`).
			switch ledger(g.Test) {
			case "FIRST-RUN":
				mark = "FIRST-RUN - failed, no confirmed prior result (ITEM, not a blocker)"
				firstRuns++
			case "SOURCE-CHANGED":
				mark = "FAIL (blocker) - NOTE: confirmed before this gate last changed"
				blockers++
			default:
				mark = "FAIL (blocker)"
				blockers++
			}
		case act == "skip":
			switch {
			case assetNeverBuilt[g.Test]:
				mark = "COVERAGE GAP - asset never built (NOT a blocker; see assetNeverBuilt)"
				gaps++
			case neverConfirmed[g.Test] != "":
				// neverConfirmed's promise ("never blocks a tag") covers a SKIP as well as a FAIL: an entry whose asset is
				// absent on this box would otherwise block every run, and the deferral would be written down but never
				// take effect.
				mark = "SKIP - asset missing, but in neverConfirmed (NOT a blocker): " + neverConfirmed[g.Test]
				gaps++
			default:
				mark = "SKIP - asset missing (blocker)"
				blockers++
			}
		}
		rows = append(rows, checkRow{g.Family, g.Test, mark})
	}
	return rows, blockers, gaps, firstRuns
}

// catchAllSkips lists top-level skipped tests whose names look like a parity/gate/golden check and
// are not already in the checkset — the "a family we forgot" net.
func catchAllSkips(res *results, checks []gateCheck) []string {
	seen := map[string]bool{}
	var out []string
	for _, k := range res.order {
		if isSubtest(k) || res.final[k] != "skip" || !strings.HasPrefix(k.Test, "Test") {
			continue
		}
		low := strings.ToLower(k.Test)
		if !strings.Contains(low, "parity") && !strings.Contains(low, "gate") && !strings.Contains(low, "golden") {
			continue
		}
		// Substring containment, not equality: the shell excluded by grepping the whole result line
		// against the gate names, so a test whose name CONTAINS a checkset name was excluded too.
		skip := false
		for _, g := range checks {
			if strings.Contains(k.Test, g.Test) {
				skip = true
				break
			}
		}
		if skip || seen[k.Test] {
			continue
		}
		seen[k.Test] = true
		out = append(out, k.Test)
	}
	sort.Strings(out)
	return out
}

// whyNoResult distinguishes the two causes of "no result" that look identical in a report and are fixed in
// completely different places. A gate no pattern selects cannot be fixed by any machine, asset or
// environment; a gate that IS selected and still produced nothing is a different problem. Saying which
// halves the search.
//
// KNOWN LIMITATION: it checks -run selection only, not build-tag reachability. parityCells' base cell has
// an empty Run and no Tags, so the first branch fires for every test name at the sweep's one call site and
// UNREACHABLE cannot fire there; a realckpt-tagged test that cannot even compile is reported as "selected
// (unfiltered) but reported nothing", which reads as an asset or build problem. Open: needs per-cell
// reachability (`go test -tags <cell's Tags> -list '^<test>$' <cell's Pkgs>`). See
// docs/code-notes/cmd-gate.md, "whyNoResult".
func whyNoResult(test string, cells []cell) string {
	for _, c := range cells {
		if c.Run == "" {
			return fmt.Sprintf("selected by cell %q (unfiltered) but reported nothing — "+
				"asset absent, build failure, or the cell died before reaching it", c.Name)
		}
		re, err := regexp.Compile(c.Run)
		if err != nil {
			return fmt.Sprintf("cell %q has an invalid -run %q: %v", c.Name, c.Run, err)
		}
		if re.MatchString(test) {
			return fmt.Sprintf("selected by cell %q (-run %q) but reported nothing — "+
				"asset absent, build failure, or the cell died before reaching it", c.Name, c.Run)
		}
	}
	return "UNREACHABLE: no cell's -run pattern selects this test, so no asset, machine or " +
		"environment could make it run. Fix the pattern or the test name, not the box."
}

// unlistedFailures returns the tests that FAILED and are not one of the named gates. The checkset says a
// NAMED gate is green; "nothing else in the sweep failed" is a separate claim, and without it a FAIL in a
// family test outside the checkset leaves the verdict reading ALL REQUIRED GATES GREEN.
//
// EXACT match, unlike catchAllSkips' containment: a test whose name merely CONTAINS a gate name is a
// different test, and hiding its failure is the defect. Exactness also keeps B14 intact: a named gate's
// FIRST-RUN failure matches exactly, so it is excluded here and stays an ITEM.
func unlistedFailures(res *results, checks []gateCheck) []string {
	named := map[string]bool{}
	for _, g := range checks {
		named[g.Test] = true
	}
	seen := map[string]bool{}
	var out []string
	for _, k := range res.topLevel("fail") {
		if named[k.Test] || seen[k.Test] {
			continue
		}
		seen[k.Test] = true
		// Last-writer-wins, the same rule the checkset reads by: a test that failed in the plain
		// cell and passed in the realckpt one is not a failure.
		if act, _ := res.lookupTop(k.Test); act != "fail" {
			continue
		}
		out = append(out, k.Test)
	}
	sort.Strings(out)
	return out
}

// sortedKeys is sortedSet for a string-valued map.
func sortedKeys(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// extraBlockers is everything blocking that the CHECKSET CANNOT SEE, and it returns the count so there is
// exactly one place to get the arithmetic wrong: a FAIL in a family parity test outside the checkset, and
// a cell that died without printing a single --- FAIL line. Either would leave the verdict reading ALL
// REQUIRED GATES GREEN.
func extraBlockers(w io.Writer, res *results, checks []gateCheck, cells []cellResult, realckpt bool) int {
	extra := 0

	// THE CATCH-ALL, FOR FAILURES — and here it BLOCKS. Its skip half is advisory because a missing
	// asset on one box is ordinary; a failing test is not, whatever list it is on. No name-shape
	// filter either: the shape filter exists to keep the skip review readable, and a failure nobody
	// wants to read is still a failure.
	if unlisted := unlistedFailures(res, checks); len(unlisted) > 0 {
		fmt.Fprintf(w, "\n-- unlisted FAILURES (blockers): tests that failed and are in no gate list --\n")
		for _, name := range unlisted {
			fmt.Fprintf(w, "   ❌ failed: %s\n", name)
		}
		extra += len(unlisted)
	}

	// A cell that exited non-zero with no --- FAIL line: a panic in a goroutine, a fatal error, a
	// timeout, a build failure. runCell marks it Hidden under RCIsFailure; without counting it the
	// sweep delivers a verdict about a cell that never finished.
	for _, c := range cells {
		if c.Hidden {
			if after, running := res.timeoutPanic(); after != "" {
				fmt.Fprintf(w, "\n   ❌ cell %q TIMED OUT at -timeout %s (rc=%d), not a test failure: in flight %s; every gate after "+
					"it did not run — raise TIMEOUT, or re-run just those gates with GATE_RUN — BLOCKER: %s\n",
					c.Cell.Name, after, c.RC, strings.Join(running, ", "), c.LogPath)
			} else {
				fmt.Fprintf(w, "\n   ❌ cell %q exited rc=%d with zero --- FAIL lines (crash, timeout or "+
					"build failure) — BLOCKER: %s\n", c.Cell.Name, c.RC, c.LogPath)
			}
			extra++
		}
	}

	// The realckpt gates the sweep RUNS but does not require. Printed so "not in the table" is a
	// statement the report makes rather than an absence a reader has to notice.
	if realckpt && len(realckptNotRequired) > 0 {
		fmt.Fprintf(w, "\n-- realckpt gates that run but are NOT required (%d) --\n", len(realckptNotRequired))
		for _, name := range sortedKeys(realckptNotRequired) {
			act, seen := res.lookupTop(name)
			if !seen {
				act = "no result"
			}
			fmt.Fprintf(w, "   %-32s %-9s %s\n", name, act, realckptNotRequired[name])
		}
	}
	return extra
}

// parityInputs is what the parity verdict reads beside the cells' streams. runParity resolves each from the tree and
// the environment and records it, so a replay grades without either.
type parityInputs struct {
	Commit           string            `json:"commit"`
	Realckpt         bool              `json:"realckpt"`
	ScopeRaw         string            `json:"scope_raw,omitempty"`   // GATE_RUN, when it narrowed the checkset
	ScopedFrom       int               `json:"scoped_from,omitempty"` // the checkset's size before GATE_RUN narrowed it
	Checks           []gateCheck       `json:"checks"`
	Ledger           map[string]string `json:"ledger"`             // each check's ledger class (ledgerClassify)
	Reconcile        string            `json:"reconcile"`          // reconcileLedger's advisory text
	Emit             bool              `json:"emit,omitempty"`     // an EMIT_MANIFEST run: Manifest and its blockers follow
	Manifest         string            `json:"manifest,omitempty"` // mergeManifest's text
	ManifestBlockers int               `json:"manifest_blockers,omitempty"`
}

// gradeParity is the sweep's verdict on finished cells: the checkset table, the catch-alls, the extra blockers, the
// first-run, ledger and coverage-gap notes, and the verdict line, printed as the live sweep prints them. It returns the
// counts the sweep's record and exit code take.
func gradeParity(w io.Writer, res *results, cfgCells []cell, cells []cellResult, in parityInputs) (blockers, gaps, firstRuns int) {
	if in.ScopeRaw != "" {
		fmt.Fprintf(w, "\n== GATE_RUN=%q: checkset narrowed from %d to %d required gate(s) — "+
			"THIS IS A SCOPED RE-RUN, NOT A RELEASE-READY VERDICT ==\n", in.ScopeRaw, in.ScopedFrom, len(in.Checks))
	}
	ledger := func(test string) string { return in.Ledger[test] }
	rows, blockers, gaps, firstRuns := classifyChecks(res, in.Checks, ledger, cfgCells)
	fmt.Fprintf(w, "\n%-20s %-34s %s\n", "FAMILY", "GATE", "RESULT")
	fmt.Fprintln(w, strings.Repeat("-", 72))
	for _, r := range rows {
		fmt.Fprintf(w, "%-20s %-34s %s\n", r.Family, r.Test, r.Mark)
	}

	// Safety net: any OTHER parity/gate-shaped test that skipped — a family the lists forgot.
	fmt.Fprintf(w, "\n-- catch-all: other *Parity / *gate tests that SKIPPED (review) --\n")
	for _, name := range catchAllSkips(res, in.Checks) {
		fmt.Fprintf(w, "   skipped: %s\n", name)
	}

	blockers += extraBlockers(w, res, in.Checks, cells, in.Realckpt)

	if in.Emit {
		fmt.Fprint(w, in.Manifest)
		blockers += in.ManifestBlockers
	}

	fmt.Fprintln(w)
	if firstRuns > 0 {
		fmt.Fprintf(w, "== %d FIRST-RUN: failed with no confirmed prior result ==\n", firstRuns)
		fmt.Fprintf(w, "   Reported, NOT counted as blockers -- there is no second point to compute a delta from.\n")
		fmt.Fprintf(w, "   NOT a claim they are harmless: each is an ITEM. Confirm a value deliberately with\n")
		fmt.Fprintf(w, "   go run ./cmd/gate ledger promote --gate <G> --value <V> --by <you>.\n")
	}
	fmt.Fprint(w, in.Reconcile) // advisory, exactly as the shell had it
	if gaps > 0 {
		fmt.Fprintf(w, "== %d COVERAGE GAP(S): required gates whose asset has never been built ==\n", gaps)
		fmt.Fprintf(w, "   Reported, not counted as blockers. No invocation can clear them -- only building the\n")
		fmt.Fprintf(w, "   asset can. A permanent blocker is not a gate, it is an override habit.\n")
	}
	verdict := "ALL REQUIRED GATES GREEN"
	if blockers > 0 {
		verdict = fmt.Sprintf("%d BLOCKER(S)", blockers)
	}
	extra := ""
	if gaps > 0 {
		extra += fmt.Sprintf(" (+%d coverage gap)", gaps)
	}
	if firstRuns > 0 {
		extra += fmt.Sprintf(" (+%d first-run)", firstRuns)
	}
	fmt.Fprintf(w, "== %s: %s%s ==\n", in.Commit, verdict, extra)
	for _, c := range cells {
		fmt.Fprintf(w, "full log: %s\n", c.LogPath)
	}
	return blockers, gaps, firstRuns
}
