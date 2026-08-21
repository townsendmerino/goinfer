// Package main implements `gate` — one runner over `go test -json` for the tallying
// gates and censuses that used to be six separate shell/Python scripts (QUEUE E8).
//
// THE INSIGHT E8 IS BUILT ON. Three shell gates (parity_sweep, gpu_gate, heavy_gate) and
// three Python censuses (skip_census, sweep_composition, selector_coverage) are one program
// wearing six hats: each runs `go test -json` over a matrix of package × family × quant ×
// build-tag, tallies PASS/SKIP/FAIL with SKIPs bucketed by reason, and applies a decision.
// They differ only in WHICH matrix and WHICH decision. So the matrix and the decision become
// config; the tallying core is written once, here.
//
// WHY GO IS STRICTLY BETTER HERE, not merely same-language:
//
//   - The `-e`/tally tension disappears. The shell gates omit `set -e` deliberately — running
//     N cells and tallying is the whole point, and `-e` aborts on the first failure and loses
//     the count — which is a discipline you must remember not to "fix". Here, running every
//     cell and keeping every exit code is the natural shape of the code, not a restraint.
//   - PIPESTATUS capture vanishes: os/exec hands back each subprocess's code directly.
//   - The silent-skip anti-pattern cannot recur. `command -v tool && tool` PASSES when the
//     tool is absent; exec.LookPath returning not-found is an error you must handle. Same for
//     asset detection: "assets absent → refuse a verdict" is a decision, and it lives in code
//     that cannot fail open.
//   - The tallying layer stops SCRAPING TEXT and starts consuming events. skip_census.py had
//     already made this move ("a reader, not a parser"); the shell gates had not — heavy_gate
//     counted `grep -cE '^--- PASS: '`, which is correct only as long as nothing else in the
//     stream starts a line that way.
//
// It orchestrates `go test`; it does not reimplement it. Stdlib only — per E7's constraint,
// a consumer's module graph must not grow because a gate changed language.
package main

import (
	"bufio"
	"encoding/json"
	"io"
	"strings"
)

// testEvent is the `go test -json` record (cmd/test2json). Only the fields the gates
// actually decide on are declared; the rest are ignored by encoding/json.
type testEvent struct {
	Action  string  `json:"Action"`
	Package string  `json:"Package"`
	Test    string  `json:"Test"`
	Output  string  `json:"Output"`
	Elapsed float64 `json:"Elapsed"`
}

// testKey identifies one test result.
//
// ALL THREE FIELDS ARE LOAD-BEARING, and the third was added because a mutation test caught its
// absence. Package is in the key because the same test NAME legitimately exists in several
// packages. Cell is in the key because a matrix legitimately runs the SAME package AND test more
// than once — parity_sweep and gpu_gate run one package under several tag/quant combinations,
// which is the entire point of a matrix. Keying on (Pkg, Test) alone let a later cell overwrite an
// earlier one, so N runs of a test reported as one: an UNDERCOUNT that still printed a confident
// verdict, which is the failure shape this program exists to eliminate.
type testKey struct {
	Cell string
	Pkg  string
	Test string
}

// results is the accumulated view of one or more `go test -json` streams.
//
// Every field here exists because some gate's verdict reads it, and the union is smaller than
// the six scripts suggested: a terminal action per test, that test's output (for the skip
// reason), and the PACKAGE-level events — which are how a build error or a native crash shows
// up at all, since a package that dies before running a test emits no per-test failure.
type results struct {
	final map[testKey]string   // last terminal action per test: pass | fail | skip
	out   map[testKey][]string // that test's output lines, for reason extraction
	order []testKey            // insertion order — map iteration must not decide report order

	pkgOut  map[string][]string // package-level output, for crash detection
	pkgFail []string            // packages that failed at the package level
	pkgSeen map[string]bool     // dedupe pkgFail

	// cur is the cell currently being consumed; it stamps every key so cells cannot collide.
	cur string

	// parityRows are `PARITY_ROW {json}` lines in stream order. The real-oracle gates emit them
	// with fmt.Printf under GOINFER_MANIFEST_EMIT, so they arrive as ordinary output events; the
	// sweep folds them into testdata/parity_manifest.json. Collected during the single parse rather
	// than by re-grepping a log, which is the whole point of consuming events.
	parityRows []string

	// runLines counts top-level `=== RUN` lines. heavy_gate reported this next to its tally so
	// that "0 passed" could be told apart from "0 attempted", which are different bugs.
	runLines int
}

func newResults() *results {
	return &results{
		final:   map[testKey]string{},
		out:     map[testKey][]string{},
		pkgOut:  map[string][]string{},
		pkgSeen: map[string]bool{},
	}
}

// consume stream-parses a `go test -json` stream into r. It never returns early on a malformed
// line: test2json can interleave non-JSON on stderr-ish paths, and one bad line must not cost
// the rest of the tally — losing the count is the failure mode this whole program exists to
// avoid.
func (r *results) consume(rd io.Reader) {
	sc := bufio.NewScanner(rd)
	// Test output lines can be long (a parity dump, a stack trace). The default 64 KiB token
	// limit would truncate them into scan errors and silently drop events after the first one.
	sc.Buffer(make([]byte, 0, 256*1024), 16*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || line[0] != '{' {
			continue
		}
		var ev testEvent
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			continue
		}
		r.add(ev)
	}
}

func (r *results) add(ev testEvent) {
	if ev.Test == "" { // package-level event
		switch ev.Action {
		case "output":
			r.pkgOut[ev.Package] = append(r.pkgOut[ev.Package], ev.Output)
			if strings.HasPrefix(ev.Output, "=== RUN ") {
				r.runLines++
			}
			r.noteParityRow(ev.Output)
		case "fail":
			if !r.pkgSeen[ev.Package] {
				r.pkgSeen[ev.Package] = true
				r.pkgFail = append(r.pkgFail, ev.Package)
			}
		}
		return
	}
	key := testKey{r.cur, ev.Package, ev.Test}
	switch ev.Action {
	case "output":
		if _, ok := r.out[key]; !ok && r.final[key] == "" {
			// first sighting — remember the order for a deterministic report
			r.noteOrder(key)
		}
		r.out[key] = append(r.out[key], ev.Output)
		r.noteParityRow(ev.Output)
	case "pass", "fail", "skip":
		if _, ok := r.out[key]; !ok && r.final[key] == "" {
			r.noteOrder(key)
		}
		r.final[key] = ev.Action
	}
}

func (r *results) noteParityRow(out string) {
	if strings.HasPrefix(out, "PARITY_ROW ") {
		r.parityRows = append(r.parityRows, strings.TrimRight(out, "\n"))
	}
}

// lookupTop returns the LAST terminal action recorded for a TOP-LEVEL test of this exact name,
// across every cell and package, and whether it was seen at all.
//
// "Last" and "exact" both reproduce `grep -E "^--- (PASS|FAIL|SKIP): NAME \(" | tail -1`: the sweep
// runs some gates twice (the plain cell and the realckpt cell), and the trailing `(` in that grep is
// what stops TestFoo from matching TestFooBar. Not-seen is a FOURTH outcome, not a flavour of skip —
// a required gate that never ran is the one case where the sweep learned nothing at all.
func (r *results) lookupTop(name string) (string, bool) {
	act, found := "", false
	for _, k := range r.order {
		if k.Test != name || isSubtest(k) {
			continue
		}
		if a, ok := r.final[k]; ok {
			act, found = a, true
		}
	}
	return act, found
}

func (r *results) noteOrder(key testKey) {
	for _, k := range r.order {
		if k == key {
			return
		}
	}
	r.order = append(r.order, key)
}

// isSubtest reports whether the key names a subtest (`TestFoo/case`) rather than a top-level test.
//
// THIS DISTINCTION IS LOAD-BEARING AND THE TWO MIGRATED SCRIPTS DISAGREED ON IT. heavy_gate.sh
// counted `^--- PASS:` anchored at column 0, so it tallied TOP-LEVEL tests only (go test indents
// subtest result lines). skip_census.py keyed on (Package, Test) from the JSON, which counts
// every subtest as its own result. Both are defensible; they are not the same number, and E8's
// acceptance (a) requires each migrated gate to reproduce ITS OWN tally — so this is a per-config
// choice (topLevelOnly), not a house style.
func isSubtest(k testKey) bool { return strings.Contains(k.Test, "/") }
