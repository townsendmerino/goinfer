package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// runFxRecorded is runFx keeping the log directory, so the run's record can be found.
func runFxRecorded(t *testing.T, qc *quickConfig, lint bool) (int, string, string) {
	t.Helper()
	logDir := t.TempDir()
	var out bytes.Buffer
	rc := quickMain(qc, quickOpts{Base: "HEAD", Jobs: 4, Timeout: "5m", LogDir: logDir, NoLint: !lint, Progress: io.Discard}, &out)
	recs, _ := filepath.Glob(filepath.Join(logDir, "gate-quick-*", quickRunRecordFile))
	if len(recs) != 1 {
		t.Fatalf("want one run record under %s, found %v", logDir, recs)
	}
	return rc, out.String(), recs[0]
}

// TestQuickReplay_reproducesTheRun is SB12-C5 for quick, on the fixture repository: a green run, a red one with the
// lint steps, and a red one (a leaf mutation that fails tests in three cells), each
// replayed from the record it wrote, print byte for byte what the run printed and exit as it exited.
func TestQuickReplay_reproducesTheRun(t *testing.T) {
	qc := fxRepo(t)
	check := func(name string, wantRC int, lint bool) string {
		rc, live, rec := runFxRecorded(t, qc, lint)
		if wantRC >= 0 && rc != wantRC {
			t.Fatalf("%s: the live run exited %d, want %d:\n%s", name, rc, wantRC, live)
		}
		var replay bytes.Buffer
		if got := replayQuick(&replay, rec); got != rc {
			t.Errorf("%s: the replay exits %d, the run exited %d", name, got, rc)
		}
		if replay.String() != live {
			t.Errorf("%s: the replay differs from the run:\n%s", name, lineDiff(live, replay.String()))
		}
		t.Logf("%s: exit %d, %d bytes of report, replayed identically", name, rc, len(live))
		return rec
	}
	check("green", 0, false)
	mutate(t, qc, "leaf/leaf.go", "a + b", "a - b")
	lintRec := check("red, with lint", 1, true)
	if !strings.Contains(string(mustRead(t, lintRec)), `"kind": "lint"`) {
		t.Error("the run with lint recorded no lint job: the lint half of the replay is untested")
	}
	if *updateGoldens {
		captureQuickGolden(t, lintRec)
	}
	rec := check("red", 1, false)

	// C4: the red record's failing cells turned green by planting their streams' failures as passes, and the green
	// run's verdict follows.
	dir := filepath.Dir(rec)
	streams, _ := filepath.Glob(filepath.Join(dir, "*.json"))
	planted := 0
	for _, s := range streams {
		if filepath.Base(s) == quickRunRecordFile {
			continue
		}
		b, err := os.ReadFile(s)
		if err != nil {
			t.Fatal(err)
		}
		if nb := strings.ReplaceAll(string(b), `"Action":"fail"`, `"Action":"pass"`); nb != string(b) {
			planted++
			if err := os.WriteFile(s, []byte(nb), 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
	var r quickRunRecord
	b, _ := os.ReadFile(rec)
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	for i := range r.Jobs {
		r.Jobs[i].RC = 0 // go test exits 0 when nothing fails
	}
	b, _ = json.Marshal(r)
	if err := os.WriteFile(rec, b, 0o644); err != nil {
		t.Fatal(err)
	}
	if planted == 0 {
		t.Fatal("the red run's streams hold no failure to plant")
	}
	var out bytes.Buffer
	if rc := replayQuick(&out, rec); rc != 0 {
		t.Errorf("with every failure planted as a pass and every exit 0, the replay exits %d, want 0", rc)
	}
	// A test cell planted with a non-zero exit and no failing test is a hidden failure: red.
	for i := range r.Jobs {
		if r.Jobs[i].Kind == "test" && r.Jobs[i].Stream != "" {
			r.Jobs[i].RC = 2
			break
		}
	}
	b, _ = json.Marshal(r)
	if err := os.WriteFile(rec, b, 0o644); err != nil {
		t.Fatal(err)
	}
	if rc := replayQuick(io.Discard, rec); rc != 1 {
		t.Errorf("with a cell exiting 2 and no failing test, the replay exits %d, want 1", rc)
	}
}

// captureQuickGolden copies a recorded run (its record, its streams and its report) into the goldens, under
// -update-goldens: the case TestGraderGoldens then regrades with quick-replay.
func captureQuickGolden(t *testing.T, rec string) {
	t.Helper()
	dst := filepath.Join(goldenRoot, "quick-fixture-red")
	if err := os.MkdirAll(dst, 0o755); err != nil {
		t.Fatal(err)
	}
	files, _ := filepath.Glob(filepath.Join(filepath.Dir(rec), "*.json"))
	for _, f := range files {
		if err := os.WriteFile(filepath.Join(dst, filepath.Base(f)), mustRead(t, f), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	var live bytes.Buffer
	rc := replayQuick(&live, rec) // identical to the run's report: checked above
	if err := os.WriteFile(filepath.Join(dst, "live.txt"), live.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	c := goldenCase{Grader: "quick-replay", Note: "a gate quick run on cmd/gate's fixture repository (fxRepo) with a leaf mutation and the lint steps, recorded by TestQuickReplay_reproducesTheRun under -update-goldens; live.txt is its report",
		Record: quickRunRecordFile, Live: "live.txt", Exit: rc}
	b, _ := json.MarshalIndent(c, "", "  ")
	if err := os.WriteFile(filepath.Join(dst, "case.json"), append(b, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
}

func mustRead(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
