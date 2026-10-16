package multimodal

import (
	"hash/fnv"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// raceSubsetStride is the share of raceSubset's tests that run under the race detector: one in this many, chosen by name.
const raceSubsetStride = 3

// raceSubset marks a test whose code under test starts no goroutine (counted with /sched/goroutines-created, the
// record in docs/measurements/test-audit-2026-10-15.md, "Race-detector subset"): the detector has nothing to find in
// it, so under -race only a spread-out sample of these tests runs, and every one runs in full in CI's race-free root
// jobs (TestRaceSubset_runsInFullSomewhere). A test that starts goroutines, or whose subtests run in parallel, must not
// call it.
func raceSubset(t *testing.T) {
	t.Helper()
	if !raceDetector {
		return
	}
	h := fnv.New32a()
	h.Write([]byte(t.Name()))
	if h.Sum32()%raceSubsetStride != 0 {
		t.Skip("no goroutines under test: outside the -race sample, run in full by the race-free root jobs")
	}
}

// TestRaceSubset_runsInFullSomewhere fails when no CI step runs the whole module without -race and without -short, the
// run in which raceSubset's tests are not sampled.
func TestRaceSubset_runsInFullSomewhere(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", ".github", "workflows", "ci.yml"))
	if err != nil {
		t.Skipf("no ci.yml to check (%v): this gate only applies in the repo", err)
	}
	for line := range strings.SplitSeq(string(b), "\n") {
		s := strings.TrimSpace(line)
		if strings.HasPrefix(s, "run:") && strings.Contains(s, "go test") && strings.Contains(s, "./...") &&
			!strings.Contains(s, "-race") && !strings.Contains(s, "-short") {
			return
		}
	}
	t.Error("no CI step runs `go test ./...` without -race and -short: raceSubset's tests would run only as a -race sample")
}
