package main

import (
	"encoding/json"
	"io"
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
		plant func(t *testing.T, dir string, rec *runRecord)
		exit  int
	}{
		{"none", func(*testing.T, string, *runRecord) {}, 0},
		{"a passing gate fails", func(t *testing.T, dir string, _ *runRecord) { failIt(t, dir) }, 1},
		{"it fails with no confirmed prior result", func(t *testing.T, dir string, rec *runRecord) {
			failIt(t, dir)
			rec.Parity.Ledger[passing] = "FIRST-RUN"
		}, 0},
		{"it fails, and the record's neverConfirmed list holds it", func(t *testing.T, dir string, rec *runRecord) {
			failIt(t, dir)
			rec.Parity.Lists.NeverConfirmed = map[string]string{passing: "deferred"}
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

// TestRunRecord_refusesAnUnreadableStream: a cell with neither a stream nor an error, and a stream the scanner cannot
// read (a line over its limit), are errors, never a silently empty cell.
func TestRunRecord_refusesAnUnreadableStream(t *testing.T) {
	write := func(t *testing.T, dir string, rec runRecord) string {
		b, _ := json.Marshal(rec)
		p := filepath.Join(dir, "run.json")
		if err := os.WriteFile(p, b, 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	base := runRecord{Schema: runRecordSchema, Gate: "heavy", Decision: "tally"}

	dir := t.TempDir()
	rec := base
	rec.Cells = []recordCell{{Name: "./decoder/"}}
	if _, _, _, _, _, err := loadRunRecord(write(t, dir, rec)); err == nil || !strings.Contains(err.Error(), "neither a stream nor an error") {
		t.Errorf("a cell with no stream and no error: err %v", err)
	}

	dir = t.TempDir()
	long := strings.Repeat("x", 17*1024*1024) // over consume's 16 MiB line limit
	if err := os.WriteFile(filepath.Join(dir, "s.json"), []byte(long+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	rec = base
	rec.Cells = []recordCell{{Name: "./decoder/", Stream: "s.json"}}
	if _, _, _, _, _, err := loadRunRecord(write(t, dir, rec)); err == nil {
		t.Error("a stream the scanner cannot read was graded")
	}

	dir = t.TempDir()
	rec = base
	rec.Cells = []recordCell{{Name: "./decoder/", RC: -1, Err: "go toolchain not on PATH", LogPath: "/x.json", DurNS: 3e9}}
	_, _, cells, _, _, err := loadRunRecord(write(t, dir, rec))
	if err != nil || len(cells) != 1 || cells[0].Fail != 1 || !cells[0].Hidden || cells[0].LogPath != "/x.json" || cells[0].Dur != 3e9 {
		t.Errorf("a cell that never started: %+v, %v; want one hidden failure keeping its log path and duration", cells, err)
	}
}

// TestIdentityReplay_plants is SB12-C4 for identity, on a copy of the recorded tiny llama run (all IDENTICAL): one
// byte changed in the old side's f32 dump makes llama DIFFERENT (exit 1); one byte changed in the new build's second
// run makes the CPU nondeterministic (exit 1); a dump the run had, lost or cut short since, is refused (exit 2), never
// graded NOT RUN or IDENTICAL; and a record whose run did not keep its dumps is refused (exit 2).
func TestIdentityReplay_plants(t *testing.T) {
	src := filepath.Join(goldenRoot, "identity-llama-tiny")
	copyTree := func(t *testing.T) string {
		dst := t.TempDir()
		err := filepath.WalkDir(src, func(p string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			rel, _ := filepath.Rel(src, p)
			if d.IsDir() {
				return os.MkdirAll(filepath.Join(dst, rel), 0o755)
			}
			b, err := os.ReadFile(p)
			if err != nil {
				return err
			}
			return os.WriteFile(filepath.Join(dst, rel), b, 0o644)
		})
		if err != nil {
			t.Fatal(err)
		}
		return dst
	}
	flip := func(t *testing.T, p string) {
		b, err := os.ReadFile(p)
		if err != nil || len(b) < 16 {
			t.Fatalf("%s: %v (%d bytes)", p, err, len(b))
		}
		b[len(b)/2] ^= 0x01
		if err := os.WriteFile(p, b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	cases := []struct {
		name  string
		plant func(t *testing.T, dir string)
		exit  int
		says  string
	}{
		{"none", func(*testing.T, string) {}, 0, "GREEN"},
		{"the old side's logits differ", func(t *testing.T, dir string) { flip(t, filepath.Join(dir, "dumps/old/llama/llama__tiny__f32.bin")) }, 1, "DIFFERENT"},
		{"the new build's second run differs", func(t *testing.T, dir string) { flip(t, filepath.Join(dir, "dumps/new2/llama/llama__tiny__f32.bin")) }, 1, "NONDETERMINISTIC"},
		{"an old dump's logits were lost", func(t *testing.T, dir string) {
			if err := os.Remove(filepath.Join(dir, "dumps/old/llama/llama__tiny__f32.bin")); err != nil {
				t.Fatal(err)
			}
		}, 2, ""},
		{"a new dump's logits were cut short", func(t *testing.T, dir string) {
			p := filepath.Join(dir, "dumps/new1/llama/llama__tiny__int4.bin")
			b := mustRead(t, p)
			if err := os.WriteFile(p, b[:len(b)-4], 0o644); err != nil {
				t.Fatal(err)
			}
		}, 2, ""},
		{"a dump's result was lost", func(t *testing.T, dir string) {
			if err := os.Remove(filepath.Join(dir, "dumps/new2/llama/llama__tiny__int8int8.json")); err != nil {
				t.Fatal(err)
			}
		}, 2, ""},
		{"the dumps were not kept", func(t *testing.T, dir string) {
			p := filepath.Join(dir, identityRunRecordFile)
			var rec identityRunRecord
			if err := json.Unmarshal(mustRead(t, p), &rec); err != nil {
				t.Fatal(err)
			}
			rec.Opts.Keep = false
			b, _ := json.Marshal(rec)
			if err := os.WriteFile(p, b, 0o644); err != nil {
				t.Fatal(err)
			}
		}, 2, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dir := copyTree(t)
			c.plant(t, dir)
			var out strings.Builder
			rc := replayIdentity(&out, filepath.Join(dir, identityRunRecordFile))
			if rc != c.exit || !strings.Contains(out.String(), c.says) {
				t.Errorf("exit %d, want %d and %q in the report:\n%s", rc, c.exit, c.says, out.String())
			}
		})
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
