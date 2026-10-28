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
