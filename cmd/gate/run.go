package main

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// cell is one `go test` invocation in a gate's matrix: a package set under a tag set, with an
// env and a -run filter. The six migrated scripts differ mostly in what their cells ARE.
type cell struct {
	Name    string            // display name (usually the package)
	Pkgs    []string          // package patterns, e.g. ./decoder/
	Tags    []string          // build tags
	Run     string            // -run regex ("" = everything)
	Env     map[string]string // extra env for this cell
	Timeout string            // -timeout
	Serial  bool              // -p 1 — for cells whose tests each load a multi-GB model
	Extra   []string          // any further `go test` args (passthrough)
	Dir     string            // working directory ("" = the process's own) — used by the
	//                          mutation tests to point a cell at a scratch module

}

// gateConfig is a whole gate: its matrix, and the knobs its DECISION needs. Everything that
// varies between the migrated scripts is here; nothing that varies lives in the runner.
type gateConfig struct {
	Name string
	Desc string

	Cells []cell

	// Decision selects the report+verdict shape: "tally" (the shell gates) or "census".
	Decision string

	// TopLevelOnly counts top-level tests only, excluding subtests. heavy_gate.sh did this by
	// anchoring its grep at column 0; skip_census.py did not. See isSubtest.
	TopLevelOnly bool

	// ZeroPolicy says which flavour of "nothing happened" is RED:
	//   "no-pass"  — zero PASSES is red even if tests ran and skipped (heavy_gate)
	//   "no-tests" — zero test EVENTS at all is red (skip_census: an empty stream is the
	//                absence of a census, not a clean one)
	ZeroPolicy string

	// RCIsFailure counts a non-zero `go test` exit with zero --- FAIL lines as a failure. This is
	// heavy_gate's hard-won rc-awareness: a panic in a goroutine, a fatal error, a timeout or a
	// zero-match all abort the binary WITHOUT a per-test FAIL line, and counting only FAIL lines
	// reports GREEN on a crashed package.
	RCIsFailure bool

	// PkgFailIsFailure counts a package-level fail (build error / native crash) toward the verdict.
	//
	// FALSE FOR THE CENSUS ON PURPOSE, AND IT IS NOT A TYPO. skip_census.py computes `rc = 1 if
	// nfail else 0` and prints package-level fails without counting them — so a build error in one
	// package exits 0 today. E8 changes the SUBSTRATE, not what a gate decides (acceptance a), so
	// the runner reproduces that. The knob exists so flipping it later is one bool rather than an
	// archaeology exercise; see the warning the census report prints when it suppresses one.
	PkgFailIsFailure bool

	// Precondition refuses a verdict rather than reporting one. heavy_gate exits 2 when the models
	// dir is missing: with no assets, both "green" and "red" would be lies.
	Precondition func() (why string, ok bool)
}

// cellResult is one cell's outcome: the tally the gate reads, plus the exit code and log path a
// reader needs to diagnose it.
type cellResult struct {
	Cell             cell
	RC               int
	Err              error
	LogPath          string
	Pass, Skip, Fail int
	RunLines         int
	Hidden           bool // rc != 0 with zero counted failures
	Dur              time.Duration
}

// runCell executes one cell and folds its events into res. It returns a cellResult and NEVER a
// fatal error: a cell that fails to start is a red cell, not an abandoned matrix. That property —
// run every cell, keep every count — is the one `set -e` would have broken, and here it is the
// shape of the code rather than a comment asking you not to add `-e`.
// cellHeartbeatInterval is how often a running cell reports progress. A var so tests can drive
// it; GOINFER_GATE_HEARTBEAT overrides it (e.g. "0" to silence it in a noisy CI log).
var cellHeartbeatInterval = 60 * time.Second

// startCellHeartbeat prints one progress line per interval until the returned stop is called.
// stop JOINS the goroutine, so nothing prints after the cell's own summary.
func startCellHeartbeat(cell string, res *results, t0 time.Time) (stop func()) {
	every := cellHeartbeatInterval
	if v := os.Getenv("GOINFER_GATE_HEARTBEAT"); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			every = d
		}
	}
	if every <= 0 {
		return func() {}
	}
	done, finished := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(finished)
		t := time.NewTicker(every)
		defer t.Stop()
		for {
			select {
			case <-done:
				return
			case <-t.C:
				last, _ := res.liveLast.Load().(string)
				if last == "" {
					last = "(no test has finished yet)"
				}
				line := fmt.Sprintf("   ... %s: %s elapsed, %d tests finished, last: %s",
					cell, time.Since(t0).Round(time.Second), res.liveDone.Load(), last)
				if name, since, ok := res.inFlight(); ok {
					line += fmt.Sprintf(" | in flight: %s (%s)", name, since)
				}
				fmt.Fprintln(os.Stderr, line)
			}
		}
	}()
	return func() { close(done); <-finished }
}

func runCell(c cell, cfg *gateConfig, res *results, logDir string) cellResult {
	args := []string{"test", "-json", "-count=1"}
	if len(c.Tags) > 0 {
		args = append(args, "-tags", strings.Join(c.Tags, " "))
	}
	if c.Run != "" {
		args = append(args, "-run", c.Run)
	}
	if c.Timeout != "" {
		args = append(args, "-timeout", c.Timeout)
	}
	if c.Serial {
		args = append(args, "-p", "1")
	}
	args = append(args, c.Extra...)
	args = append(args, c.Pkgs...)

	// `go` itself must be present. exec.LookPath failing is an ERROR here — the shell idiom
	// `command -v go && go test` would have passed silently, which is the exact fail-open this
	// migration exists to make impossible.
	if _, err := exec.LookPath("go"); err != nil {
		return cellResult{Cell: c, RC: -1, Err: fmt.Errorf("go toolchain not on PATH: %w", err), Fail: 1, Hidden: true}
	}

	logPath := filepath.Join(logDir, "gate_"+sanitize(cfg.Name+"_"+c.Name)+".json")
	logf, err := os.Create(logPath)
	if err != nil {
		return cellResult{Cell: c, RC: -1, Err: err, Fail: 1, Hidden: true}
	}
	defer logf.Close()

	cmd := exec.Command("go", args...)
	cmd.Dir = c.Dir
	cmd.Env = os.Environ()
	for k, v := range c.Env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return cellResult{Cell: c, RC: -1, Err: err, Fail: 1, Hidden: true}
	}
	// stderr into the same log: a build error or a linker message never appears as a JSON event,
	// and a reader chasing "0 tests observed" needs it.
	cmd.Stderr = logf

	res.cur = c.Name
	t0 := time.Now()
	if err := cmd.Start(); err != nil {
		return cellResult{Cell: c, RC: -1, Err: err, Fail: 1, Hidden: true}
	}
	// HEARTBEAT while the cell runs (~/.claude/rules/long-tests.md). `go test -json` reports a
	// cell only when it finishes, and the realckpt cells run 55-90 minutes — so without this a
	// reader cannot tell a working gate from a hung one without ps'ing the box, which is exactly
	// what happened on 2026-08-26. Prints elapsed, tests finished so far, and the most recent
	// test name. There is no done-of-TOTAL because `go test` never announces a total; claiming
	// one would be inventing it.
	//
	// stderr, not the report writer: the report is a verdict document and a progress line is not
	// part of it. Interval is env-tunable per the rule's "make it configurable rather than
	// removing it" — 0 or a bad value disables.
	stopBeat := startCellHeartbeat(c.Name, res, t0)
	// Tee: the raw stream survives on disk (a panic that kills test2json mid-line still leaves
	// evidence) while being parsed in one pass.
	res.consume(io.TeeReader(stdout, logf))
	stopBeat()
	waitErr := cmd.Wait()
	dur := time.Since(t0)

	rc := 0
	if waitErr != nil {
		if ee, ok := waitErr.(*exec.ExitError); ok {
			rc = ee.ExitCode()
		} else {
			rc = -1
		}
	}

	cr := res.tally(c.Name, cfg.TopLevelOnly)
	cr.Cell, cr.RC, cr.LogPath, cr.Dur = c, rc, logPath, dur
	if rc != 0 && cr.Fail == 0 && cfg.RCIsFailure {
		cr.Fail++
		cr.Hidden = true
	}
	return cr
}

// tally counts one cell's results out of the shared set. Cell-stamped keys make this exact — an
// earlier design diffed a before/after snapshot of the map, which silently lost any test whose
// result did not CHANGE between cells (a skip that stayed a skip counted once for two runs).
func (r *results) tally(cellName string, topOnly bool) cellResult {
	var cr cellResult
	for k, act := range r.final {
		if k.Cell != cellName || (topOnly && isSubtest(k)) {
			continue
		}
		switch act {
		case "pass":
			cr.Pass++
		case "skip":
			cr.Skip++
		case "fail":
			cr.Fail++
		}
	}
	return cr
}

func sanitize(s string) string {
	rep := strings.NewReplacer("/", "_", ".", "_", " ", "_", ":", "_")
	return strings.Trim(rep.Replace(s), "_")
}

// vacuous reports a cell in which every test that ran chose to skip. `go test`
// exits 0 on an all-skip package, so RC alone cannot tell this from a real pass —
// which is how two Metal groups in `gate gpu` vouched for claims they never
// executed (audit 2026-09-02 G-01).
func (c cellResult) vacuous() bool { return c.Pass == 0 && c.Fail == 0 && c.Skip > 0 }
