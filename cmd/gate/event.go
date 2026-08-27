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
	"sync"
	"sync/atomic"
	"time"
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

	// Live progress, for the heartbeat in runCell. ATOMIC, and separate from the maps above
	// on purpose: the heartbeat runs on its own goroutine while consume() is writing those
	// maps, so reading them would be a data race. These two are the only things it reads.
	liveDone atomic.Int64
	liveLast atomic.Value // string — the most recent test to reach a terminal action
	// liveRun holds the tests that have started and not yet finished, keyed by name, valued by
	// start time. "Last finished" alone cannot answer the question a stalled run actually raises:
	// during the v0.15.0 sweep the count sat at 430 for four minutes while the line kept naming a
	// test that had already completed, which says nothing about what is holding the cell up.
	// sync.Map for the same reason the two atomics above exist — the heartbeat goroutine reads
	// this while consume() writes it.
	liveRun sync.Map // string -> time.Time

	// outAll is every output line in stream order — the reconstruction of what `go test -v` would
	// have printed. The GPU gate's failure explainer needs it: a run killed by a signal, an OOM or a
	// timeout emits neither a "--- FAIL" line nor a "file.go:N:" line, only "FAIL <pkg> <secs>", so
	// a filter over structured results alone would report an EMPTY explanation for exactly the
	// failures that are hardest to reproduce.
	outAll []string

	// stream, when set, is called for each terminal test result as it arrives. Liveness is
	// load-bearing for a 28-minute group: buffering means a running tier and a HUNG one produce
	// byte-identical output (none), so progress has to be visible as it happens.
	stream func(pkg, test, action string)

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
			r.noteOutput(ev.Output)
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
		r.noteOutput(ev.Output)
	case "run":
		r.liveRun.Store(ev.Test, time.Now())
	case "pass", "fail", "skip":
		r.liveDone.Add(1)
		r.liveLast.Store(ev.Test)
		r.liveRun.Delete(ev.Test)
		if _, ok := r.out[key]; !ok && r.final[key] == "" {
			r.noteOrder(key)
		}
		r.final[key] = ev.Action
		if r.stream != nil && !isSubtest(key) {
			r.stream(ev.Package, ev.Test, ev.Action)
		}
	}
}

// noteOutput folds one output line into the stream-wide accumulators.
//
// runLines counts EVERY `=== RUN` line, top-level and subtest alike. That is deliberate and it is
// not the same unit as the skip count beside it: go test does NOT indent a subtest's `=== RUN` line
// (only its `--- PASS` result line is indented), so the shell gate's `grep -cE '^=== RUN'` counted
// subtests while its `grep -cE '^--- SKIP'` did not. The pair "ran 238 tests, skipped 22" therefore
// mixes units — 238 tests-and-subtests started against 22 top-level skips. Reproduced exactly,
// because E8 changes the substrate and not what a gate reports; flagged in docs/task-gate-runner.md
// §10 as a number that should probably say which unit it is in.
func (r *results) noteOutput(out string) {
	if strings.HasPrefix(out, "=== RUN ") {
		r.runLines++
	}
	r.noteParityRow(out)
	r.outAll = append(r.outAll, out)
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

// text reconstructs the `go test -v` output this stream carried.
func (r *results) text() string { return strings.Join(r.outAll, "") }

// topLevel returns the top-level tests with the given terminal action, in stream order.
func (r *results) topLevel(action string) []testKey {
	var out []testKey
	for _, k := range r.order {
		if !isSubtest(k) && r.final[k] == action {
			out = append(out, k)
		}
	}
	return out
}

// ranCount is how many top-level tests produced a result. ZERO IS A FAILURE, NOT A PASS: a `-run`
// pattern that matches nothing exits 0 and prints "ok", so renaming a test away silently deletes a
// check while the gate stays green — the same shape as a skip counted as a pass.
func (r *results) ranCount() int {
	n := 0
	for _, k := range r.order {
		if !isSubtest(k) && r.final[k] != "" {
			n++
		}
	}
	return n
}

// inFlight returns the test that has been running longest without finishing, and for how long.
// The oldest is the useful one: when a parent test is slow its subtests come and go beneath it,
// and the parent is the name worth printing. Returns ok=false when nothing is in flight.
func (r *results) inFlight() (name string, since time.Duration, ok bool) {
	var oldest time.Time
	r.liveRun.Range(func(k, v any) bool {
		t, isTime := v.(time.Time)
		if !isTime {
			return true
		}
		if oldest.IsZero() || t.Before(oldest) {
			oldest, name, ok = t, k.(string), true
		}
		return true
	})
	if ok {
		since = time.Since(oldest).Round(time.Second)
	}
	return name, since, ok
}
