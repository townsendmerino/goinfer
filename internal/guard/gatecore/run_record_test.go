package gatecore

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/testpath"
)

func TestRunRecord_plantsTurnTheReplayRed(t *testing.T) {
	src := testpath.Testdata(t, "gate", "goldens", "heavy-narrowed")
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
			var rec RunRecord
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
			cfg, res, cells, prov, req, err := LoadRunRecord(filepath.Join(dir, "run.json"))
			if err != nil {
				t.Fatal(err)
			}
			var out strings.Builder
			rc := GradeTallyRun(&out, cfg, res, cells, prov, req)
			if want := map[bool]int{true: 0, false: 1}[name == "none"]; rc != want {
				t.Errorf("exit %d, want %d:\n%s", rc, want, out.String())
			}
		})
	}
}

func TestRunRecord_refusesAnUnreadableStream(t *testing.T) {
	write := func(t *testing.T, dir string, rec RunRecord) string {
		b, _ := json.Marshal(rec)
		p := filepath.Join(dir, "run.json")
		if err := os.WriteFile(p, b, 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	base := RunRecord{Schema: RunRecordSchema, Gate: "heavy", Decision: "tally"}

	dir := t.TempDir()
	rec := base
	rec.Cells = []RecordCell{{Name: "./decoder/"}}
	if _, _, _, _, _, err := LoadRunRecord(write(t, dir, rec)); err == nil || !strings.Contains(err.Error(), "neither a stream nor an error") {
		t.Errorf("a cell with no stream and no error: err %v", err)
	}

	dir = t.TempDir()
	long := strings.Repeat("x", 17*1024*1024)
	if err := os.WriteFile(filepath.Join(dir, "s.json"), []byte(long+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	rec = base
	rec.Cells = []RecordCell{{Name: "./decoder/", Stream: "s.json"}}
	if _, _, _, _, _, err := LoadRunRecord(write(t, dir, rec)); err == nil {
		t.Error("a stream the scanner cannot read was graded")
	}

	dir = t.TempDir()
	rec = base
	rec.Cells = []RecordCell{{Name: "./decoder/", RC: -1, Err: "go toolchain not on PATH", LogPath: "/x.json", DurNS: 3e9}}
	_, _, cells, _, _, err := LoadRunRecord(write(t, dir, rec))
	if err != nil || len(cells) != 1 || cells[0].Fail != 1 || !cells[0].Hidden || cells[0].LogPath != "/x.json" || cells[0].Dur != 3e9 {
		t.Errorf("a cell that never started: %+v, %v; want one hidden failure keeping its log path and duration", cells, err)
	}
}
