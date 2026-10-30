package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// replayEditedRelease replays the release-replay golden's record with edit applied to its reads, and returns the report
// and the exit code.
func replayEditedRelease(t *testing.T, edit func(in *releaseInputs)) (string, int) {
	t.Helper()
	rec, err := readRunRecord(filepath.Join(goldenRoot, "release-replay", "gate_release_run.json.gz"))
	if err != nil {
		t.Fatal(err)
	}
	edit(rec.Release)
	b, err := json.Marshal(rec)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "gate_release_run.json")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	rc := replayRelease(&out, path)
	return out.String(), rc
}

// setRead changes the answer to one recorded read; the test fails if the record holds no such read.
func setRead(t *testing.T, in *releaseInputs, kind string, args []string, out string, failed bool) {
	t.Helper()
	for i, l := range in.Looks {
		if l.Kind == kind && strings.Join(l.Args, "\x00") == strings.Join(args, "\x00") {
			in.Looks[i].Out, in.Looks[i].Failed = out, failed
			if failed {
				in.Looks[i].Err = "exit status 1"
			}
			return
		}
	}
	t.Fatalf("the record holds no %s %v read", kind, args)
}

// rowLines is the report's lines for row n, from its state line up to the next row's.
func rowLines(report string, n int) string {
	lines := strings.Split(report, "\n")
	for i, ln := range lines {
		if strings.HasPrefix(ln, strings.Repeat(" ", 2-len(itoa(n)))+itoa(n)+". ") {
			j := i + 1
			for j < len(lines) && strings.HasPrefix(lines[j], "             ") {
				j++
			}
			return strings.Join(lines[i:j], "\n")
		}
	}
	return ""
}

func itoa(n int) string { b, _ := json.Marshal(n); return string(b) }

// TestReleaseReplay_plants is SB12-C4 for step 12: from the recorded pre-flight, one read changed per row family turns that
// row OWED (or UNKNOWN) with its detail, and a read the record lacks is exit 2 with no report.
func TestReleaseReplay_plants(t *testing.T) {
	head := "96623e9641118c81a8461257e131d44954a741b1"
	for name, tc := range map[string]struct {
		edit  func(t *testing.T, in *releaseInputs)
		row   int
		state string
		want  string
	}{
		"a pending gate older than the previous release": {func(t *testing.T, in *releaseInputs) {
			setRead(t, in, "pending", nil, `{"TestOld_gate":"2026-09-01 waiting for its asset"}`, false)
		}, 1, "OWED", "TestOld_gate (pending since 2026-09-01)"},
		"pins that differ": {func(t *testing.T, in *releaseInputs) {
			for i, l := range in.Looks {
				if l.Kind == "read" && l.Args[0] == "cuda/go.mod" {
					in.Looks[i].Out = strings.Replace(l.Out, "github.com/townsendmerino/aikit v1.76.0", "github.com/townsendmerino/aikit v1.75.0", 1)
				}
			}
		}, 2, "OWED", "aikit differs: v1.75.0: cuda | v1.76.0: ., demo/agent, gpu, metal"},
		"a dirty gofmt": {func(t *testing.T, in *releaseInputs) {
			setRead(t, in, "run", []string{"cuda", "gofmt", "-l", "."}, "x.go", false)
		}, 8, "OWED", "cuda/x.go"},
		"a red lint": {func(t *testing.T, in *releaseInputs) {
			setRead(t, in, "run", []string{".", "python3", "scripts/queue_citation_lint.py"}, "RED: docs/x.md cites a7f3", true)
		}, 9, "OWED", "RED: docs/x.md cites a7f3"},
		"a red CI run": {func(t *testing.T, in *releaseInputs) {
			in.Offline = false
			in.Looks = append(in.Looks, relLookup{Kind: "git", Args: []string{"rev-parse", "origin/main"}, Out: head},
				relLookup{Kind: "run", Args: []string{".", "gh", "run", "list", "--limit", "30", "--json", "headSha,status,conclusion,workflowName"},
					Out: `[{"headSha":"` + head + `","status":"completed","conclusion":"failure","workflowName":"ci"}]`})
		}, 10, "OWED", "ci"},
	} {
		report, rc := replayEditedRelease(t, func(in *releaseInputs) { tc.edit(t, in) })
		got := rowLines(report, tc.row)
		if rc != 1 || !strings.Contains(got, tc.state) || !strings.Contains(got, tc.want) {
			t.Errorf("%s: exit %d, row %d:\n%s\nwant %s naming %q", name, rc, tc.row, got, tc.state, tc.want)
		}
	}
	report, rc := replayEditedRelease(t, func(in *releaseInputs) {
		for i, l := range in.Looks {
			if l.Kind == "read" && l.Args[0] == "metal/go.mod" {
				in.Looks = append(in.Looks[:i], in.Looks[i+1:]...)
				break
			}
		}
	})
	if rc != 2 || !strings.Contains(report, "holds no answer for: read metal/go.mod") || strings.Contains(report, "gate release v") {
		t.Errorf("a read the record lacks: exit %d\n%s", rc, report)
	}
}
