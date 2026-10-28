package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestRunRecord_plantsTurnTheReplayRed is SB12-C4 for the tally graders: from a copy of a recorded green heavy run,
// a passing test's terminal event planted as a failure, and the cell's exit code planted non-zero with no failing
// event, each make the replay red, and the copy left alone stays green.
func TestRunRecord_plantsTurnTheReplayRed(t *testing.T) {
	src := filepath.Join(goldenRoot, "heavy-narrowed")
	plants := map[string]func(t *testing.T, dir string){
		"none": func(*testing.T, string) {},
		"a pass turned to fail": func(t *testing.T, dir string) {
			p := filepath.Join(dir, "stream.json")
			b, err := os.ReadFile(p)
			if err != nil {
				t.Fatal(err)
			}
			const pass = `"Action":"pass","Package":"github.com/townsendmerino/goinfer/decoder","Test":"TestCacheStateGrid_rewindCellsOnRealCaches"`
			if !strings.Contains(string(b), pass) {
				t.Fatal("the recorded stream has no passing top-level test to plant")
			}
			if err := os.WriteFile(p, []byte(strings.Replace(string(b), pass, strings.Replace(pass, "pass", "fail", 1), 1)), 0o644); err != nil {
				t.Fatal(err)
			}
		},
		"a non-zero exit with no failing test": func(t *testing.T, dir string) {
			p := filepath.Join(dir, "run.json")
			var rec runRecord
			b, err := os.ReadFile(p)
			if err == nil {
				err = json.Unmarshal(b, &rec)
			}
			if err != nil {
				t.Fatal(err)
			}
			rec.Cells[0].RC = 2
			b, _ = json.Marshal(rec)
			if err := os.WriteFile(p, b, 0o644); err != nil {
				t.Fatal(err)
			}
		},
	}
	for name, plant := range plants {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			for _, f := range []string{"run.json", "stream.json"} {
				b, err := os.ReadFile(filepath.Join(src, f))
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(dir, f), b, 0o644); err != nil {
					t.Fatal(err)
				}
			}
			plant(t, dir)
			cfg, res, cells, prov, req, err := loadRunRecord(filepath.Join(dir, "run.json"))
			if err != nil {
				t.Fatal(err)
			}
			var out strings.Builder
			rc := gradeTallyRun(&out, cfg, res, cells, prov, req)
			if want := map[bool]int{true: 0, false: 1}[name == "none"]; rc != want {
				t.Errorf("exit %d, want %d:\n%s", rc, want, out.String())
			}
		})
	}
}

// TestRunRecord_replayRefusesAnotherGatesRecord: a record names its gate, and loadRunRecord gives it back, so `gate
// census -replay` of a heavy record is refused rather than graded under the wrong policy.
func TestRunRecord_replayRefusesAnotherGatesRecord(t *testing.T) {
	cfg, _, _, _, _, err := loadRunRecord(filepath.Join(goldenRoot, "heavy-narrowed", "run.json"))
	if err != nil {
		t.Fatal(err)
	}
	var b strings.Builder
	if rc := run([]string{"census", "-replay", filepath.Join(goldenRoot, "heavy-narrowed", "run.json")}, &b); rc != 2 || cfg.Name != "heavy" {
		t.Errorf("census -replay of a %s record exited %d, want 2", cfg.Name, rc)
	}
}

// TestParityReplay_plants is SB12-C4 for the parity grader: from the recorded sweep narrowed to its two passing gates
// (green), a passing gate's end event planted as a failure is a blocker, the same failure with the ledger saying
// FIRST-RUN is not, a cell planted with a non-zero exit and no failing test is a blocker, and an EMIT_MANIFEST run
// whose merge reported a blocker is red.
func TestParityReplay_plants(t *testing.T) {
	src := filepath.Join(goldenRoot, "parity-replay")
	const passing = "TestVoxtral_tinyTextParity"
	failIt := func(t *testing.T, dir string) {
		matches, _ := filepath.Glob(filepath.Join(dir, "gate_parity_*.json"))
		if len(matches) != 1 {
			t.Fatalf("want one stream, found %v", matches)
		}
		b, err := os.ReadFile(matches[0])
		if err != nil {
			t.Fatal(err)
		}
		pass := `"Action":"pass","Package":"github.com/townsendmerino/goinfer/decoder","Test":"` + passing + `"`
		if !strings.Contains(string(b), pass) {
			t.Fatalf("the recorded stream has no passing %s", passing)
		}
		if err := os.WriteFile(matches[0], []byte(strings.Replace(string(b), pass, strings.Replace(pass, `"pass"`, `"fail"`, 1), 1)), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	cases := []struct {
		name  string
		plant func(t *testing.T, dir string, rec *runRecord)
		exit  int
	}{
		{"none", func(*testing.T, string, *runRecord) {}, 0},
		{"a passing gate fails", func(t *testing.T, dir string, _ *runRecord) { failIt(t, dir) }, 1},
		{"it fails with no confirmed prior result", func(t *testing.T, dir string, rec *runRecord) {
			failIt(t, dir)
			rec.Parity.Ledger[passing] = "FIRST-RUN"
		}, 0},
		{"a cell exits non-zero with no failing test", func(_ *testing.T, _ string, rec *runRecord) { rec.Cells[0].RC = 2 }, 1},
		{"the manifest merge reports a blocker", func(_ *testing.T, _ string, rec *runRecord) {
			rec.Parity.Emit, rec.Parity.ManifestBlockers = true, 1
		}, 1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dir := t.TempDir()
			ents, err := os.ReadDir(src)
			if err != nil {
				t.Fatal(err)
			}
			for _, e := range ents {
				b, err := os.ReadFile(filepath.Join(src, e.Name()))
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(dir, e.Name()), b, 0o644); err != nil {
					t.Fatal(err)
				}
			}
			var rec runRecord
			b, _ := os.ReadFile(filepath.Join(dir, "run.json"))
			if err := json.Unmarshal(b, &rec); err != nil {
				t.Fatal(err)
			}
			var keep []gateCheck // the base: only the gates the recorded sweep passed
			for _, g := range rec.Parity.Checks {
				if g.Test == passing || g.Test == "TestQwen3ASR_tinyTextParity" {
					keep = append(keep, g)
				}
			}
			if len(keep) != 2 {
				t.Fatalf("the recorded checks %v lack the two passing gates", rec.Parity.Checks)
			}
			rec.Parity.Checks = keep
			c.plant(t, dir, &rec)
			b, _ = json.Marshal(rec)
			if err := os.WriteFile(filepath.Join(dir, "run.json"), b, 0o644); err != nil {
				t.Fatal(err)
			}
			var out strings.Builder
			if rc := replayParity(&out, filepath.Join(dir, "run.json")); rc != c.exit {
				t.Errorf("exit %d, want %d:\n%s", rc, c.exit, out.String()[strings.LastIndex(out.String(), "FAMILY"):])
			}
		})
	}
}

// TestHeavyStream_refusesAndNamesReplay: `gate heavy -stream` exits 2 and names -replay, the decision of 2026-10-27
// (docs/measurements/sb12-2026-10.md §8), even on a stream whose run was green.
func TestHeavyStream_refusesAndNamesReplay(t *testing.T) {
	var b strings.Builder
	rc := run([]string{"heavy", "-stream", filepath.Join(goldenRoot, "heavy-narrowed", "stream.json")}, &b)
	if rc != 2 || !strings.Contains(b.String(), "gate heavy -replay FILE") {
		t.Errorf("heavy -stream exited %d with %q; want 2 and the -replay instruction", rc, b.String())
	}
}
