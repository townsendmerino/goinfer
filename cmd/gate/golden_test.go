package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/paritykey"
)

// SB12's goldens (docs/tasks/task-subsystems-2026-10.md §8 SB12, gate SB12-C2): each case under goldenRoot is a recorded
// input and the verdict today's grader gives on it. TestGraderGoldens regrades every case from its recorded inputs alone,
// so a grader that moves out of its runner must reproduce each want.txt byte for byte. -update-goldens captures a case:
// it resolves the git lookups the grader asks for with real git, stores the answers in case.json, and writes want.txt.
// Without it, a lookup the case did not record fails the case, so a grader cannot read git, the clock or the machine.
var updateGoldens = flag.Bool("update-goldens", false, "capture SB12's grader goldens (git lookups and want.txt) instead of checking them")

const goldenRoot = "../../testdata/gate/goldens"

// goldenCase is one case.json. Grader names the function under test; the other fields are its inputs, with Lookups
// holding every git answer the grader asked for at capture ("kind\x00a\x00b" -> answer, or "error: ..." for a failed one).
type goldenCase struct {
	Grader      string            `json:"grader"`
	Note        string            `json:"note,omitempty"`
	Heads       []string          `json:"heads,omitempty"`
	Paths       []string          `json:"paths,omitempty"`
	Records     string            `json:"records,omitempty"` // a directory, relative to the case or (for dated records) to the repo root
	Pattern     string            `json:"pattern,omitempty"` // the record files' glob within Records
	InCase      bool              `json:"in_case,omitempty"` // Records is relative to the case, not the repo root
	Details     bool              `json:"details,omitempty"` // giw-bytes: print each row's detail, not only the counts
	Stream      string            `json:"stream,omitempty"`
	Record      string            `json:"record,omitempty"`      // a run record (run_record.go)
	Live        string            `json:"live,omitempty"`        // the run's own output: the verdict text, instead of want.txt
	Exit        int               `json:"exit,omitempty"`        // with Live, the run's exit code
	GoRunExit   bool              `json:"go_run_exit,omitempty"` // Live was captured from `go run`, which adds "exit status N" after a non-zero exit
	Passthrough []string          `json:"passthrough,omitempty"`
	Lookups     map[string]string `json:"lookups,omitempty"`
}

// goldenGraders grades one case: the verdict text it returns is compared with want.txt. look answers a git lookup from
// the case's recording (or, under -update-goldens, from git, recording the answer).
var goldenGraders = map[string]func(t *testing.T, dir string, c *goldenCase, look lookupFn) string{
	"census":          gradeGoldenCensus,
	"tally-replay":    gradeGoldenTallyReplay,
	"parity-replay":   gradeGoldenParityReplay,
	"quick-replay":    gradeGoldenQuickReplay,
	"identity-replay": gradeGoldenIdentityReplay,
	"gpu-verdict":     gradeGoldenGPUVerdict,
	"gpu-coverage":    gradeGoldenGPUCoverage,
	"parity-coverage": gradeGoldenParityCoverage,
	"quick-coverage":  gradeGoldenQuickCoverage,
	"giw-bytes":       gradeGoldenGiwBytes,
}

// goldenOwed is the cases whose grader cannot regrade a recorded input yet: each holds its input and the live run's
// output (live.txt), the golden its family's step must reproduce. The list is exact, so a case cannot fall into it
// silently.
var goldenOwed = map[string]string{}

type lookupFn func(kind string, args ...string) (string, error)

func TestGraderGoldens(t *testing.T) {
	ents, err := os.ReadDir(goldenRoot)
	if err != nil {
		t.Fatal(err)
	}
	graded, owed := 0, map[string]bool{}
	for _, e := range ents {
		if !e.IsDir() {
			continue
		}
		name, dir := e.Name(), filepath.Join(goldenRoot, e.Name())
		if why, ok := goldenOwed[name]; ok {
			checkOwedCase(t, name, why, dir)
			owed[name] = true
			t.Logf("%s: NOT regraded yet: %s", name, why)
			continue
		}
		t.Run(name, func(t *testing.T) { runGoldenCase(t, dir) })
		graded++
	}
	for name := range goldenOwed {
		if !owed[name] {
			t.Errorf("goldenOwed names %s, which is not a case", name)
		}
	}
	if graded == 0 {
		t.Fatal("no golden case graded: nothing compared")
	}
}

// checkOwedCase holds an owed case to what its step will need: a case.json that parses and names its grader, the live
// output, and the stream it names, parsing as go test -json with at least one test event.
func checkOwedCase(t *testing.T, name, why, dir string) {
	t.Helper()
	var c goldenCase
	b, err := os.ReadFile(filepath.Join(dir, "case.json"))
	if err == nil {
		err = json.Unmarshal(b, &c)
	}
	if err != nil || c.Grader == "" {
		t.Errorf("%s: owed (%s) but its case.json does not parse or names no grader: %v", name, why, err)
		return
	}
	if _, err := os.Stat(filepath.Join(dir, "live.txt")); err != nil {
		t.Errorf("%s: owed (%s) but holds no live.txt", name, why)
	}
	f, err := os.Open(filepath.Join(dir, c.Stream))
	if err != nil {
		t.Errorf("%s: owed (%s) but its stream: %v", name, why, err)
		return
	}
	defer f.Close()
	res := newResults()
	res.consume(f)
	if len(res.final) == 0 {
		t.Errorf("%s: owed (%s) but its stream holds no test result", name, why)
	}
}

func runGoldenCase(t *testing.T, dir string) {
	b, err := os.ReadFile(filepath.Join(dir, "case.json"))
	if err != nil {
		t.Fatal(err)
	}
	var c goldenCase
	if err := json.Unmarshal(b, &c); err != nil {
		t.Fatal(err)
	}
	grade, ok := goldenGraders[c.Grader]
	if !ok {
		t.Fatalf("unknown grader %q", c.Grader)
	}
	recorded, used := map[string]string{}, map[string]bool{}
	look := func(kind string, args ...string) (string, error) {
		key := kind + "\x00" + strings.Join(args, "\x00")
		used[key] = true
		if *updateGoldens {
			v, err := liveLookup(kind, args...)
			if err != nil {
				v = "error: " + err.Error()
			}
			recorded[key] = v
		}
		v, ok := c.Lookups[key]
		if *updateGoldens {
			v, ok = recorded[key], true
		}
		if !ok {
			t.Fatalf("the grader asked for %s %q, which the case did not record (run -update-goldens)", kind, args)
		}
		if msg, isErr := strings.CutPrefix(v, "error: "); isErr {
			return "", fmt.Errorf("%s", msg)
		}
		return v, nil
	}
	got := grade(t, dir, &c, look)
	if !*updateGoldens {
		for key := range c.Lookups {
			if !used[key] {
				t.Errorf("the case recorded the lookup %q, which the grader never asked for: a check it made at capture is gone", strings.ReplaceAll(key, "\x00", " "))
			}
		}
	}
	want := filepath.Join(dir, "want.txt")
	if c.Live != "" {
		want = filepath.Join(dir, c.Live) // the run's output is the golden; capturing never rewrites it
	}
	if *updateGoldens && c.Live == "" {
		c.Lookups = recorded
		out, err := json.MarshalIndent(c, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "case.json"), append(out, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(want, []byte(got), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	w, err := os.ReadFile(want)
	if err != nil {
		t.Fatal(err)
	}
	if got != string(w) {
		t.Errorf("the grader's verdict differs from %s:\n%s", want, lineDiff(string(w), got))
	}
}

// liveLookup answers a lookup from the repository's git, only while capturing.
func liveLookup(kind string, args ...string) (string, error) {
	root, err := repoRoot()
	if err != nil {
		return "", err
	}
	switch kind {
	case "tree":
		return gitTree(root, args[0], args[1])
	case "object":
		return gitObject(root, args[0], args[1])
	case "diff":
		return gitOut(root, "diff", "--name-only", args[0], args[1])
	}
	return "", fmt.Errorf("unknown lookup %q", kind)
}

func lineDiff(want, got string) string {
	wl, gl := strings.Split(want, "\n"), strings.Split(got, "\n")
	var b strings.Builder
	for i := 0; i < len(wl) || i < len(gl); i++ {
		var a, c string
		if i < len(wl) {
			a = wl[i]
		}
		if i < len(gl) {
			c = gl[i]
		}
		if a != c {
			fmt.Fprintf(&b, "  line %d\n    want %q\n    got  %q\n", i+1, a, c)
		}
	}
	return b.String()
}

// gradeGoldenCensus is `gate census -stream`'s grading, without the provenance block: the stream's results through
// reportCensus.
func gradeGoldenCensus(t *testing.T, dir string, c *goldenCase, _ lookupFn) string {
	f, err := os.Open(filepath.Join(dir, c.Stream))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	res := newResults()
	res.consume(f)
	var b strings.Builder
	rc := reportCensus(&b, censusConfig(c.Passthrough), res, false)
	return fmt.Sprintf("exit %d\n%s", rc, b.String())
}

// gradeGoldenTallyReplay is `gate census|heavy -replay`: the run record and its streams through gradeTallyRun, which
// must print what the run printed and exit as it exited.
func gradeGoldenTallyReplay(t *testing.T, dir string, c *goldenCase, _ lookupFn) string {
	cfg, res, cells, prov, requireFixtures, err := loadRunRecord(filepath.Join(dir, c.Record))
	if err != nil {
		t.Fatal(err)
	}
	var b strings.Builder
	if rc := gradeTallyRun(&b, cfg, res, cells, prov, requireFixtures); rc != c.Exit {
		t.Errorf("the replay exits %d; the run exited %d", rc, c.Exit)
	}
	return b.String()
}

// gradeGoldenParityReplay is `gate parity -replay`: the runner's recorded prelude, then gradeParity on the recorded
// cells and inputs.
func gradeGoldenParityReplay(t *testing.T, dir string, c *goldenCase, _ lookupFn) string {
	var b strings.Builder
	rc := replayParity(&b, filepath.Join(dir, c.Record))
	if rc != c.Exit {
		t.Errorf("the replay exits %d; the run exited %d", rc, c.Exit)
	}
	if c.GoRunExit && rc != 0 {
		fmt.Fprintf(&b, "exit status %d\n", rc)
	}
	return b.String()
}

// gradeGoldenQuickReplay is `gate quick -replay`: the recorded prelude, then gradeQuick on the recorded jobs.
func gradeGoldenQuickReplay(t *testing.T, dir string, c *goldenCase, _ lookupFn) string {
	var b strings.Builder
	if rc := replayQuick(&b, filepath.Join(dir, c.Record)); rc != c.Exit {
		t.Errorf("the replay exits %d; the run exited %d", rc, c.Exit)
	}
	return b.String()
}

// gradeGoldenIdentityReplay is `gate identity -replay`: the recorded prelude, each cell judged again from its kept
// dumps, and gradeIdentity.
func gradeGoldenIdentityReplay(t *testing.T, dir string, c *goldenCase, _ lookupFn) string {
	var b strings.Builder
	if rc := replayIdentity(&b, filepath.Join(dir, c.Record)); rc != c.Exit {
		t.Errorf("the replay exits %d; the run exited %d", rc, c.Exit)
	}
	return b.String()
}

// gradeGoldenGPUVerdict is gradeGPU on each gpu-gate record's tally. A record from SB12 step 7 on holds the groups by
// name and the vacuous and empty cells; an older one holds counts, so its groups are stood in for by placeholders and a
// VACUOUS or EMPTY label, whose cell list was never recorded, is reported as not regradable rather than assumed.
func gradeGoldenGPUVerdict(t *testing.T, dir string, c *goldenCase, _ lookupFn) string {
	recs, err := loadGPURecords(filepath.Join(dir, c.Records))
	if err != nil {
		t.Fatal(err)
	}
	var b strings.Builder
	for _, r := range recs {
		tl := gpuTally{Pass: r.Groups.Pass, Skip: r.Groups.Skip, Fail: r.Groups.Fail, Ran: r.Groups.Ran,
			Vacuous: r.VacuousCells, Empty: r.EmptyCells, Notes: r.Notes, Emitted: map[string]bool{}}
		full := len(r.Expect) > 0
		if full {
			tl.Expect = r.Expect
			for _, e := range r.Emitted {
				tl.Emitted[e] = true
			}
		} else {
			for i := range r.Groups.Declared {
				tl.Expect = append(tl.Expect, fmt.Sprintf("group%d", i))
			}
			for i := range r.Groups.Reported {
				tl.Emitted[fmt.Sprintf("group%d", i)] = true
			}
		}
		id := r.Backend + " " + short8(r.Commit) + " " + r.Date
		if !full && (r.Verdict == "VACUOUS" || r.Verdict == "EMPTY") {
			fmt.Fprintf(&b, "%s: recorded %s; not regradable (a counts-only record holds no cell list)\n", id, r.Verdict)
			continue
		}
		// The record's fail count already holds the reconciliation's failures, which gradeGPU adds again.
		recon := 0
		var missing, unexpected bool
		for _, e := range tl.Expect {
			missing = missing || !tl.Emitted[e]
		}
		decl := map[string]bool{}
		for _, e := range tl.Expect {
			decl[e] = true
		}
		for e := range tl.Emitted {
			unexpected = unexpected || !decl[e]
		}
		if missing {
			recon++
		}
		if unexpected {
			recon++
		}
		tl.Fail -= recon
		code, label := gradeGPU(io.Discard, &tl, gpuVerdictInputs{Backend: r.Backend, Dirty: r.Dirty})
		mark := "agrees"
		if label != r.Verdict {
			mark = "DIFFERS"
		}
		fmt.Fprintf(&b, "%s: recorded %s, regraded %s (exit %d) — %s\n", id, r.Verdict, label, code, mark)
	}
	return b.String()
}

// gradeGoldenGPUCoverage is rowGPU's decision for each recorded head and backend.
func gradeGoldenGPUCoverage(t *testing.T, dir string, c *goldenCase, look lookupFn) string {
	recs, err := loadGPURecords(filepath.Join(dir, c.Records))
	if err != nil {
		t.Fatal(err)
	}
	treeOf := func(rev, d string) (string, error) { return look("tree", rev, d) }
	var b strings.Builder
	for _, head := range c.Heads {
		for _, be := range []string{"cuda", "metal"} {
			r, why := gpuCoverage(recs, be, head, treeOf)
			fmt.Fprintf(&b, "%s %s: %s\n", short8(head), be, coverageLine(r != nil, why, func() string { return r.Date + " @ " + short8(r.Commit) }))
		}
	}
	return b.String()
}

// gradeGoldenParityCoverage is rowParity's decision for each recorded head, over the recorded qualifying paths.
func gradeGoldenParityCoverage(t *testing.T, dir string, c *goldenCase, look lookupFn) string {
	recs, err := loadParityRecords(filepath.Join(dir, c.Records))
	if err != nil {
		t.Fatal(err)
	}
	if *updateGoldens && len(c.Paths) == 0 {
		root, err := repoRoot()
		if err != nil {
			t.Fatal(err)
		}
		mb, err := paritykey.ManifestJSON(paritykey.Dir(root))
		if err != nil {
			t.Fatal(err)
		}
		if c.Paths, err = parityQualifyingPaths(mb); err != nil {
			t.Fatal(err)
		}
	}
	objOf := func(rev, p string) (string, error) { return look("object", rev, p) }
	var b strings.Builder
	for _, head := range c.Heads {
		r, why := parityCoverage(recs, head, c.Paths, objOf)
		fmt.Fprintf(&b, "%s: %s\n", short8(head), coverageLine(r != nil, why, func() string { return r.Date + " @ " + short8(r.Commit) }))
	}
	return b.String()
}

// gradeGoldenQuickCoverage is rowQuickFull's decision for each recorded head and system.
func gradeGoldenQuickCoverage(t *testing.T, dir string, c *goldenCase, look lookupFn) string {
	recs, err := loadQuickRecords(filepath.Join(dir, c.Records))
	if err != nil {
		t.Fatal(err)
	}
	changed := func(from, to string) ([]string, error) {
		out, err := look("diff", from, to)
		if err != nil || out == "" {
			return nil, err
		}
		return strings.Split(out, "\n"), nil
	}
	var b strings.Builder
	for _, head := range c.Heads {
		for _, goos := range quickRecordSystems {
			r, why := quickCoverage(recs, goos, head, changed)
			fmt.Fprintf(&b, "%s %s: %s\n", short8(head), goos, coverageLine(r != nil, why, func() string { return r.Date + " @ " + short8(r.Commit) }))
		}
	}
	return b.String()
}

func coverageLine(ok bool, why string, which func() string) string {
	if ok {
		return "covered by " + which()
	}
	return "OWED: " + why
}

// gradeGoldenGiwBytes is judgeGiwBytes on every row of the dated .giw byte-identity records: a row holds the old side
// and one new transcode, so the new side stands for both of the live run's new transcodes. Each verdict is also
// compared with the one the live run wrote, and a disagreement is part of the golden, not hidden.
func gradeGoldenGiwBytes(t *testing.T, dir string, c *goldenCase, _ lookupFn) string {
	base := filepath.Join("../..", c.Records)
	if c.InCase {
		base = filepath.Join(dir, c.Records)
	}
	files, err := filepath.Glob(filepath.Join(base, c.Pattern))
	if err != nil || len(files) == 0 {
		t.Fatalf("no records in %s: %v", c.Records, err)
	}
	sort.Strings(files)
	counts := map[string]int{}
	var b strings.Builder
	rows := 0
	for _, f := range files {
		fh, err := os.Open(f)
		if err != nil {
			t.Fatal(err)
		}
		sc := bufio.NewScanner(fh)
		sc.Buffer(make([]byte, 1<<20), 1<<20)
		for sc.Scan() {
			var r struct {
				Source, Config, Verdict, Why string
				OldSHA                       string `json:"old_sha256"`
				NewSHA                       string `json:"new_sha256"`
				OldBytes                     int64  `json:"old_bytes"`
				NewBytes                     int64  `json:"new_bytes"`
				OldLine                      string `json:"old_line"`
				NewLine                      string `json:"new_line"`
				New2SHA                      string `json:"new2_sha256"` // plants only: a second new transcode
				NewErr                       string `json:"new_err"`     // plants only: why the new side failed
			}
			if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
				t.Fatalf("%s: %v", f, err)
			}
			old := giwResult{OK: r.OldSHA != "", SHA: r.OldSHA, Size: r.OldBytes, Line: r.OldLine, Err: errIfNoSHA(r.OldSHA, r.Why)}
			nw := giwResult{OK: r.NewSHA != "", SHA: r.NewSHA, Size: r.NewBytes, Line: r.NewLine, Err: errIfNoSHA(r.NewSHA, r.Why)}
			if r.NewErr != "" {
				nw.Err = r.NewErr
			}
			n2 := nw
			if r.New2SHA != "" {
				n2.SHA = r.New2SHA
			}
			v, detail := judgeGiwBytes(nw, n2, old)
			counts[v]++
			rows++
			if v != r.Verdict {
				fmt.Fprintf(&b, "DIFFERS %s %s %s: recorded %s, judged %s\n", filepath.Base(f), r.Source, r.Config, r.Verdict, v)
			}
			if c.Details && detail != "" {
				fmt.Fprintf(&b, "%s %s: %s: %s\n", r.Source, r.Config, v, detail)
			}
		}
		if err := sc.Err(); err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		fh.Close()
	}
	keys := make([]string, 0, len(counts))
	for k := range counts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	fmt.Fprintf(&b, "%d rows in %d records\n", rows, len(files))
	for _, k := range keys {
		fmt.Fprintf(&b, "  %s %d\n", k, counts[k])
	}
	return b.String()
}

func errIfNoSHA(sha, why string) string {
	if sha == "" {
		return why
	}
	return ""
}
