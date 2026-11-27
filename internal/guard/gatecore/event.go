// Package gatecore implements the go test -json event accumulator, matrix cell runner, run-record codec,
// process memory sampler, git/pre-move path lookups, and pure tally/skip/timeout graders shared by cmd/gate.
package gatecore

import (
	"bufio"
	"encoding/json"
	"io"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// TestEvent is one `go test -json` record (cmd/test2json) carrying the fields gate graders inspect.
type TestEvent struct {
	Action  string  `json:"Action"`
	Package string  `json:"Package"`
	Test    string  `json:"Test"`
	Output  string  `json:"Output"`
	Elapsed float64 `json:"Elapsed"`
}

// TestKey identifies one test result across cells and packages so repeated runs under distinct build tags
// or quant modes never overwrite earlier cells.
type TestKey struct {
	Cell string
	Pkg  string
	Test string
}

// Results accumulates one or more `go test -json` streams: terminal actions per test, per-test output lines,
// package-level output/failures, live progress atomics for heartbeats, and `PARITY_ROW` emissions.
type Results struct {
	Final map[TestKey]string   // last terminal action per test: pass | fail | skip
	Out   map[TestKey][]string // per-test output lines, for skip/failure extraction
	Order []TestKey            // insertion order for deterministic reporting

	PkgOut  map[string][]string // package-level output, for crash/build failure inspection
	PkgFail []string            // packages that failed at the package level
	pkgSeen map[string]bool     // dedupe PkgFail

	// Cur is the cell currently being consumed; it stamps every key so cells cannot collide.
	Cur string

	// LiveDone, LiveLast, and liveRun track progress concurrently for the heartbeat goroutine.
	LiveDone atomic.Int64
	LiveLast atomic.Value // string — most recent test to reach a terminal action
	liveRun  sync.Map     // string -> time.Time

	// OutAll holds every output line in stream order.
	OutAll []string

	// Stream, when set, is called for each top-level terminal test result as it arrives.
	Stream func(pkg, test, action string)

	// ParityRows holds `PARITY_ROW {json}` lines in stream order.
	ParityRows []string

	// RunLines counts `=== RUN ` lines across top-level tests and subtests.
	RunLines int
}

// NewResults allocates an empty Results accumulator.
func NewResults() *Results {
	return &Results{
		Final:   map[TestKey]string{},
		Out:     map[TestKey][]string{},
		PkgOut:  map[string][]string{},
		pkgSeen: map[string]bool{},
	}
}

// Consume parses a `go test -json` stream into r, ignoring non-JSON lines interleaved on stderr-like paths.
func (r *Results) Consume(rd io.Reader) error {
	sc := bufio.NewScanner(rd)
	sc.Buffer(make([]byte, 0, 256*1024), 16*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || line[0] != '{' {
			continue
		}
		var ev TestEvent
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			continue
		}
		r.Add(ev)
	}
	return sc.Err()
}

// Add folds one decoded TestEvent into r.
func (r *Results) Add(ev TestEvent) {
	if ev.Test == "" {
		switch ev.Action {
		case "output":
			r.PkgOut[ev.Package] = append(r.PkgOut[ev.Package], ev.Output)
			r.noteOutput(ev.Output)
		case "fail":
			if !r.pkgSeen[ev.Package] {
				r.pkgSeen[ev.Package] = true
				r.PkgFail = append(r.PkgFail, ev.Package)
			}
		}
		return
	}
	key := TestKey{r.Cur, ev.Package, ev.Test}
	switch ev.Action {
	case "output":
		if _, ok := r.Out[key]; !ok && r.Final[key] == "" {
			r.noteOrder(key)
		}
		r.Out[key] = append(r.Out[key], ev.Output)
		r.noteOutput(ev.Output)
	case "run":
		r.liveRun.Store(ev.Test, time.Now())
	case "pass", "fail", "skip":
		r.LiveDone.Add(1)
		r.LiveLast.Store(ev.Test)
		r.liveRun.Delete(ev.Test)
		if _, ok := r.Out[key]; !ok && r.Final[key] == "" {
			r.noteOrder(key)
		}
		r.Final[key] = ev.Action
		if r.Stream != nil && !IsSubtest(key) {
			r.Stream(ev.Package, ev.Test, ev.Action)
		}
	}
}

func (r *Results) noteOutput(out string) {
	if strings.HasPrefix(out, "=== RUN ") {
		r.RunLines++
	}
	if strings.HasPrefix(out, "PARITY_ROW ") {
		r.ParityRows = append(r.ParityRows, strings.TrimRight(out, "\n"))
	}
	r.OutAll = append(r.OutAll, out)
}

func (r *Results) noteOrder(key TestKey) {
	if slices.Contains(r.Order, key) {
		return
	}
	r.Order = append(r.Order, key)
}

// RanCount returns how many top-level tests produced a terminal result.
func (r *Results) RanCount() int {
	n := 0
	for _, k := range r.Order {
		if !IsSubtest(k) && r.Final[k] != "" {
			n++
		}
	}
	return n
}

// InFlight returns the longest-running unfinished test and its elapsed duration, or ok=false when none is active.
func (r *Results) InFlight() (name string, since time.Duration, ok bool) {
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
