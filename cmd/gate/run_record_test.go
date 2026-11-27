package main

import (
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/guard/gatecore"
)

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
// FIRST-RUN is not, nor is it when the record's own neverConfirmed list holds the gate, a cell planted with a non-zero exit and no failing test is a blocker, and an EMIT_MANIFEST run
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
		plant func(t *testing.T, dir string, rec *runRecord, in *parityInputs)
		exit  int
	}{
		{"none", func(*testing.T, string, *runRecord, *parityInputs) {}, 0},
		{"a passing gate fails", func(t *testing.T, dir string, _ *runRecord, _ *parityInputs) { failIt(t, dir) }, 1},
		{"it fails with no confirmed prior result", func(t *testing.T, dir string, _ *runRecord, in *parityInputs) {
			failIt(t, dir)
			in.Ledger[passing] = "FIRST-RUN"
		}, 0},
		{"it fails, and the record's neverConfirmed list holds it", func(t *testing.T, dir string, _ *runRecord, in *parityInputs) {
			failIt(t, dir)
			in.Lists.NeverConfirmed = map[string]string{passing: "deferred"}
		}, 0},
		{"a cell exits non-zero with no failing test", func(_ *testing.T, _ string, rec *runRecord, _ *parityInputs) { rec.Cells[0].RC = 2 }, 1},
		{"the manifest merge reports a blocker", func(_ *testing.T, _ string, _ *runRecord, in *parityInputs) {
			in.Emit, in.ManifestBlockers = true, 1
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
			parIn, err := gatecore.UnmarshalPayload[parityInputs](rec.Parity)
			if err != nil || parIn == nil {
				t.Fatalf("parity payload: %v", err)
			}
			var keep []gateCheck // the base: only the gates the recorded sweep passed
			for _, g := range parIn.Checks {
				if g.Test == passing || g.Test == "TestQwen3ASR_tinyTextParity" {
					keep = append(keep, g)
				}
			}
			if len(keep) != 2 {
				t.Fatalf("the recorded checks %v lack the two passing gates", parIn.Checks)
			}
			parIn.Checks = keep
			c.plant(t, dir, &rec, parIn)
			rec.Parity, _ = json.Marshal(parIn)
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

// TestGradeGPU_plants is SB12-C4 for the gpu verdict: from a green tally, each plant gives its label, in gradeGPU's
// order, and a declared group that reported nothing is a failure.
func TestGradeGPU_plants(t *testing.T) {
	green := func() gpuTally {
		return gpuTally{Expect: []string{"seam", "suite"}, Emitted: map[string]bool{"seam": true, "suite": true}, Pass: 5, Ran: 2}
	}
	cases := []struct {
		name  string
		plant func(*gpuTally, *gpuVerdictInputs)
		label string
	}{
		{"none", func(*gpuTally, *gpuVerdictInputs) {}, "PASS"},
		{"nothing ran", func(t *gpuTally, _ *gpuVerdictInputs) { t.Ran = 0 }, "NO_GATE"},
		{"a cell's tests all skipped", func(t *gpuTally, _ *gpuVerdictInputs) { t.Vacuous = []string{"suite"} }, "VACUOUS"},
		{"a -run matched nothing", func(t *gpuTally, _ *gpuVerdictInputs) { t.Empty = []string{"suite"} }, "EMPTY"},
		{"a check failed", func(t *gpuTally, _ *gpuVerdictInputs) { t.Fail = 1 }, "FAIL"},
		{"a declared group reported nothing", func(t *gpuTally, _ *gpuVerdictInputs) { delete(t.Emitted, "suite") }, "FAIL"},
		{"the tree is dirty", func(_ *gpuTally, in *gpuVerdictInputs) { in.Dirty = true }, "INCONCLUSIVE"},
	}
	for _, c := range cases {
		tl, in := green(), gpuVerdictInputs{Backend: "cuda", Commit: "abc", Host: "Linux", Now: "2026-10-27T00:00:00Z"}
		c.plant(&tl, &in)
		code, label := gradeGPU(io.Discard, &tl, in)
		if label != c.label || (code == 0) != (c.label == "PASS") {
			t.Errorf("%s: label %s exit %d, want %s", c.name, label, code, c.label)
		}
	}
}

// TestGradeGPU_noticesDoNotChangeTheVerdict: a notice is printed in the verdict block whether or not anything skipped, and leaves the
// exit code and the label as they were.
func TestGradeGPU_noticesDoNotChangeTheVerdict(t *testing.T) {
	for _, fail := range []int{0, 1} {
		tl := gpuTally{Expect: []string{"seam"}, Emitted: map[string]bool{"seam": true}, Pass: 1, Ran: 1, Fail: fail, Notices: []string{"KV10 depth 65536 drifted"}}
		var w strings.Builder
		code, label := gradeGPU(&w, &tl, gpuVerdictInputs{Backend: "cuda", Commit: "abc", Host: "Linux", Now: "2026-11-22T00:00:00Z"})
		if want := map[int]string{0: "PASS", 1: "FAIL"}[fail]; label != want || (code == 0) != (fail == 0) {
			t.Errorf("fail %d: label %s exit %d, want %s", fail, label, code, want)
		}
		if !strings.Contains(w.String(), "- KV10 depth 65536 drifted") || !strings.Contains(w.String(), "Notices") {
			t.Errorf("fail %d: the verdict block lacks the notice:\n%s", fail, w.String())
		}
	}
}
