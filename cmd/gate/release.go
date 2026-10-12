package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// `gate release vX.Y.Z`: RELEASING.md's pre-flight as one command. Each read is a row that is `done`, `OWED` or `UNKNOWN` (a read that could
// not be made: offline, no gh), with the record that proves it or that is missing. It runs nothing long: it reads records and calls the
// functions the existing checks are built on, so the command and the checks cannot disagree. Exit 0 only when every row is done, 1 when
// any is owed or unknown, 2 when refused (a precondition for tagging is unmet, so neither answer would be true).
//
// Why it exists: v0.21.0 and v0.22.0 shipped with the Metal GPU-gate run and no recorded CUDA one, because that read was a sentence nobody counted.

// relRow is one pre-flight read.
type relRow struct {
	Name   string
	State  string // "done", "OWED" or "UNKNOWN"
	Detail []string
	Record string // the path that proves it, or that is missing
}

// relEnv is what a row reads from: the repository, the tag being cut and the one before it, and the means to run a command.
type relEnv struct {
	root, tag, prev string
	offline         bool
	run             func(dir, name string, args ...string) (string, error) // combined output
}

func osRun(dir, name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	return strings.TrimSpace(string(out)), err
}

func done(name, record string, detail ...string) relRow {
	return relRow{Name: name, State: "done", Record: record, Detail: detail}
}
func owed(name, record string, detail ...string) relRow {
	return relRow{Name: name, State: "OWED", Record: record, Detail: detail}
}
func unknown(name, record string, detail ...string) relRow {
	return relRow{Name: name, State: "UNKNOWN", Record: record, Detail: detail}
}

// changelogRelease matches a released CHANGELOG header, "## [v0.17.2] — 2026-09-08". The dates come
// from the CHANGELOG rather than git tags because the release workflow's checkout carries no tags.
var changelogRelease = regexp.MustCompile(`(?m)^## \[(v[0-9][^\]]*)\]\s*[—–-]+\s*(\d{4}-\d{2}-\d{2})`)

// pendingOutlivingRelease finds the release before `releasing` (the newest dated CHANGELOG release
// that is not `releasing` itself) and returns every pending entry dated before it, sorted.
func pendingOutlivingRelease(pending map[string]string, changelog, releasing string) (prevVer, prevDate string, blockers []string, err error) {
	for _, m := range changelogRelease.FindAllStringSubmatch(changelog, -1) {
		if m[1] != releasing {
			prevVer, prevDate = m[1], m[2]
			break
		}
	}
	if prevVer == "" {
		return "", "", nil, fmt.Errorf("no dated release before %s in CHANGELOG.md (want a header like \"## [v0.17.2] — 2026-09-08\")", releasing)
	}
	for gate, reason := range pending {
		if !isoDatePrefix(reason) {
			return "", "", nil, fmt.Errorf("%s has no leading ISO date (%q), so it cannot be placed against a release", gate, reason)
		}
		if reason[:10] < prevDate {
			blockers = append(blockers, fmt.Sprintf("%s (pending since %s)", gate, reason[:10]))
		}
	}
	sort.Strings(blockers)
	return prevVer, prevDate, blockers, nil
}

// isoDatePrefix reports whether a reason opens with a YYYY-MM-DD date.
func isoDatePrefix(s string) bool {
	if len(s) < 10 {
		return false
	}
	for i, c := range s[:10] {
		if i == 4 || i == 7 {
			if c != '-' {
				return false
			}
		} else if c < '0' || c > '9' {
			return false
		}
	}
	return true
}

// ---- row 1: no required gate pending since before the previous release ----

func rowPending(e *relEnv) relRow {
	const name = "no required gate pending since before the previous release"
	cl, err := os.ReadFile(filepath.Join(e.root, "CHANGELOG.md"))
	if err != nil {
		return owed(name, "CHANGELOG.md", err.Error())
	}
	prevVer, prevDate, blockers, err := pendingOutlivingRelease(awaitingFirstConfirmation, string(cl), e.tag)
	if err != nil {
		return owed(name, "cmd/gate/parity.go awaitingFirstConfirmation", err.Error())
	}
	if len(blockers) > 0 {
		return owed(name, "cmd/gate/parity.go awaitingFirstConfirmation", append([]string{
			fmt.Sprintf("pending since before %s (%s); run each and `gate ledger promote`, or move it to neverConfirmed with its reason:", prevVer, prevDate)}, blockers...)...)
	}
	return done(name, "cmd/gate/parity.go awaitingFirstConfirmation", fmt.Sprintf("nothing pending since before %s (%s)", prevVer, prevDate))
}

// ---- row 2: the five modules agree on aikit, cuda and metal on aikit/gpu ----

var relModules = []string{".", "gpu", "cuda", "metal", "demo/agent"}

// modRequire is the version a go.mod requires for module, or "".
func modRequire(gomod, module string) string {
	m := regexp.MustCompile(`(?m)^\s*(?:require\s+)?` + regexp.QuoteMeta(module) + `\s+(v[^\s]+)`).FindStringSubmatch(gomod)
	if m == nil {
		return ""
	}
	return m[1]
}

func rowPins(e *relEnv) relRow {
	const name = "the modules agree on the aikit pins"
	aikit, gpuPins := map[string]string{}, map[string]string{}
	for _, m := range relModules {
		b, err := os.ReadFile(filepath.Join(e.root, m, "go.mod"))
		if err != nil {
			return owed(name, m+"/go.mod", err.Error())
		}
		if v := modRequire(string(b), "github.com/townsendmerino/aikit"); v != "" {
			aikit[m] = v
		}
		if m == "cuda" || m == "metal" {
			if v := modRequire(string(b), "github.com/townsendmerino/aikit/gpu"); v != "" {
				gpuPins[m] = v
			}
		}
	}
	var bad []string
	check := func(label string, pins map[string]string) {
		vs := map[string][]string{}
		for m, v := range pins {
			vs[v] = append(vs[v], m)
		}
		if len(vs) > 1 {
			var parts []string
			for v, ms := range vs {
				sort.Strings(ms)
				parts = append(parts, fmt.Sprintf("%s: %s", v, strings.Join(ms, ", ")))
			}
			sort.Strings(parts)
			bad = append(bad, label+" differs: "+strings.Join(parts, " | "))
		}
	}
	check("aikit", aikit)
	check("aikit/gpu (cuda, metal)", gpuPins)
	if len(aikit) == 0 {
		bad = append(bad, "no module requires aikit: the reader found nothing")
	}
	if len(bad) > 0 {
		return owed(name, "go.mod of "+strings.Join(relModules, ", "), bad...)
	}
	vs := ""
	for _, v := range aikit {
		vs = v
	}
	return done(name, "go.mod of "+strings.Join(relModules, ", "), fmt.Sprintf("aikit %s in %d modules; aikit/gpu %s", vs, len(aikit), gpuPins["cuda"]))
}

// ---- row 3: a cold-user record for the previous tag ----

var coldUserTitle = regexp.MustCompile(`(?m)^# .*\bgoinfer (v[0-9]+\.[0-9]+\.[0-9]+)\b`)

func rowColdUser(e *relEnv) relRow {
	name := "a cold-user record for the previous tag (" + e.prev + ")"
	files, _ := filepath.Glob(filepath.Join(e.root, "docs/measurements/cold-user-*.md"))
	sort.Strings(files)
	var seen []string
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		head := string(b)
		if len(head) > 600 {
			head = head[:600]
		}
		if m := coldUserTitle.FindStringSubmatch(head); m != nil {
			rel, _ := filepath.Rel(e.root, f)
			if m[1] == e.prev {
				return done(name, rel)
			}
			seen = append(seen, m[1])
		}
	}
	return owed(name, "docs/measurements/cold-user-<date>.md", "none of the records names "+e.prev+" (they name: "+strings.Join(seen, ", ")+"); run task-first-hour.md section 1 on the published tag")
}

// ---- row 4: the GPU gate, both backends ----

func rowGPU(e *relEnv) relRow {
	const name = "the GPU gate on both backends covers HEAD"
	dir := filepath.Join(e.root, gpuRecordDirDefault)
	recs, err := loadGPURecords(dir)
	if err != nil {
		return owed(name, gpuRecordDirDefault, err.Error())
	}
	head, err := gitOut(e.root, "rev-parse", "HEAD")
	if err != nil {
		return owed(name, gpuRecordDirDefault, err.Error())
	}
	treeOf := func(rev, d string) (string, error) { return gitTree(e.root, rev, d) }
	var bad, ok []string
	for _, be := range []string{"cuda", "metal"} {
		if r, why := gpuCoverage(recs, be, head, treeOf); r != nil {
			ok = append(ok, fmt.Sprintf("%s: %s @ %s", be, r.Date, r.Commit[:8]))
		} else {
			bad = append(bad, be+": "+why)
		}
	}
	if len(bad) > 0 {
		return owed(name, gpuRecordDirDefault+"/<backend>-<time>-<commit>.json", append(bad, "run `go run ./cmd/gate gpu` on that box (cuda: about 83 minutes, queue it at night) and commit the record")...)
	}
	return done(name, gpuRecordDirDefault, ok...)
}

// ---- row 5: the hardware matrix is current, and what is newly without a record ----

type hwCoverage struct {
	Entries []struct {
		ID           string            `json:"id"`
		LastExecuted []json.RawMessage `json:"last_executed"`
	} `json:"entries"`
}

// neverExecuted returns the ids of the entries with no record of ever running.
func neverExecuted(b []byte) ([]string, error) {
	var h hwCoverage
	if err := json.Unmarshal(b, &h); err != nil {
		return nil, err
	}
	var ids []string
	for _, e := range h.Entries {
		if len(e.LastExecuted) == 0 {
			ids = append(ids, e.ID)
		}
	}
	sort.Strings(ids)
	return ids, nil
}

func rowHardware(e *relEnv) relRow {
	const name = "the hardware matrix is current; the never-executed list is read"
	const rec = "docs/hardware-matrix.md (from docs/hardware-coverage.json)"
	if out, err := e.run(e.root, "go", "test", "./decoder", "-run", "HardwareMatrix", "-count=1"); err != nil {
		return owed(name, rec, "the generated section is stale or the test failed (regenerate: go test ./decoder -run HardwareMatrix -update):", lastLines(out, 3))
	}
	head, err := os.ReadFile(filepath.Join(e.root, "docs/hardware-coverage.json"))
	if err != nil {
		return owed(name, rec, err.Error())
	}
	now, err := neverExecuted(head)
	if err != nil {
		return owed(name, rec, err.Error())
	}
	detail := []string{fmt.Sprintf("%d hardware-selected paths have no record of ever running: %s", len(now), strings.Join(now, ", ")),
		"make sure nothing in the release notes, the README or the site claims support for them"}
	if out, err := e.run(e.root, "git", "show", e.prev+":docs/hardware-coverage.json"); err == nil {
		if before, err := neverExecuted([]byte(out)); err == nil {
			had := map[string]bool{}
			for _, id := range before {
				had[id] = true
			}
			var fresh []string
			for _, id := range now {
				if !had[id] {
					fresh = append(fresh, id)
				}
			}
			if len(fresh) > 0 {
				detail = append(detail, "NEW since "+e.prev+" with no record: "+strings.Join(fresh, ", "))
			}
		}
	}
	return done(name, rec, detail...)
}

func lastLines(s string, n int) string {
	ls := strings.Split(strings.TrimSpace(s), "\n")
	if len(ls) > n {
		ls = ls[len(ls)-n:]
	}
	return strings.Join(ls, " / ")
}

// ---- row 6: no unmerged branch carries a fix ----

var fixSubject = regexp.MustCompile(`(?i)(^|[^a-z])(fix|fixes|fixed|bugfix|hotfix)([^a-z]|$)`)

// branchesWithFixes lists the branches (local and remote, but not main) holding a commit that is not on main, not patch-equal to
// one that is, and whose subject says it is a fix; each with its unmerged commits. Other unmerged branches are listed apart.
func branchesWithFixes(e *relEnv) (fixes, others []string, err error) {
	out, err := e.run(e.root, "git", "for-each-ref", "--format=%(refname:short)", "refs/heads", "refs/remotes")
	if err != nil {
		return nil, nil, err
	}
	for _, br := range strings.Fields(out) {
		if br == "main" || br == "origin/main" || strings.HasSuffix(br, "/HEAD") || br == "origin" {
			continue
		}
		cherry, err := e.run(e.root, "git", "cherry", "origin/main", br)
		if err != nil {
			continue
		}
		var subjects []string
		for _, ln := range strings.Split(cherry, "\n") {
			if strings.HasPrefix(ln, "+ ") {
				sha := strings.TrimPrefix(ln, "+ ")
				subj, _ := e.run(e.root, "git", "log", "-1", "--format=%s", sha)
				subjects = append(subjects, subj)
			}
		}
		if len(subjects) == 0 {
			continue
		}
		hasFix := false
		for _, s := range subjects {
			if fixSubject.MatchString(s) {
				hasFix = true
			}
		}
		line := fmt.Sprintf("%s (%d unmerged commit(s); newest: %s)", br, len(subjects), subjects[len(subjects)-1])
		if hasFix {
			fixes = append(fixes, line)
		} else {
			others = append(others, line)
		}
	}
	sort.Strings(fixes)
	sort.Strings(others)
	return fixes, others, nil
}

func rowBranches(e *relEnv) relRow {
	const name = "no unmerged branch carries a fix"
	const rec = "git branch -a, against origin/main"
	fixes, others, err := branchesWithFixes(e)
	if err != nil {
		return unknown(name, rec, err.Error())
	}
	if len(fixes) > 0 {
		d := append([]string{"merge, abandon or explain each (a branch may be another session's in-flight work):"}, fixes...)
		if len(others) > 0 {
			d = append(d, fmt.Sprintf("(and %d other unmerged branch(es) with no fix-like subject)", len(others)))
		}
		return owed(name, rec, d...)
	}
	d := []string{}
	if len(others) > 0 {
		d = append(d, fmt.Sprintf("%d unmerged branch(es) with no fix-like subject, listed for reading:", len(others)))
		d = append(d, others...)
	}
	return done(name, rec, d...)
}

// ---- row 7: the CHANGELOG ----

func rowChangelog(e *relEnv) relRow {
	const name = "the CHANGELOG names what changed"
	section := "Unreleased"
	if b, err := os.ReadFile(filepath.Join(e.root, "CHANGELOG.md")); err == nil {
		if _, ok := clSection(string(b), e.tag); ok {
			section = e.tag
		}
	}
	var buf bytes.Buffer
	rc := runChangelog([]string{"-base", e.prev, "-section", section}, &buf)
	var lines []string
	for _, ln := range strings.Split(strings.TrimSpace(buf.String()), "\n") {
		if strings.Contains(ln, "OWED") || strings.HasPrefix(ln, "gate changelog:") {
			lines = append(lines, strings.TrimSpace(ln))
		}
	}
	rec := "CHANGELOG.md [" + section + "], docs/changelog-waivers.txt"
	if rc == 0 {
		return done(name, rec)
	}
	return owed(name, rec, lines...)
}

// ---- rows 8-10: gofmt, the citation lint, CI at HEAD ----

func rowGofmt(e *relEnv) relRow {
	const name = "gofmt -l is empty in every module"
	var dirty []string
	for _, m := range relModules {
		out, err := e.run(filepath.Join(e.root, m), "gofmt", "-l", ".")
		if err != nil {
			return owed(name, m, err.Error())
		}
		for _, f := range strings.Fields(out) {
			dirty = append(dirty, filepath.Join(m, f))
		}
	}
	if len(dirty) > 0 {
		return owed(name, "gofmt -l .", dirty...)
	}
	return done(name, "gofmt -l .")
}

func rowLint(e *relEnv) relRow {
	const name = "the citation lint is green"
	out, err := e.run(e.root, "python3", "scripts/queue_citation_lint.py")
	if err != nil {
		return owed(name, "scripts/queue_citation_lint.py", lastLines(out, 4))
	}
	return done(name, "scripts/queue_citation_lint.py")
}

// ciRuns is what `gh run list` reports for one run.
type ciRun struct {
	HeadSha    string `json:"headSha"`
	Status     string `json:"status"`
	Conclusion string `json:"conclusion"`
	Workflow   string `json:"workflowName"`
}

// ciVerdict reads the runs for the full sha: the ci workflow must be completed (at the RUN level, not job by job) and successful.
func ciVerdict(runs []ciRun, sha string) (state, detail string) {
	found := false
	for _, r := range runs {
		if r.HeadSha != sha || r.Workflow != "ci" {
			continue
		}
		found = true
		if r.Status != "completed" {
			return "OWED", "the ci run at HEAD is " + r.Status + ", not completed"
		}
		if r.Conclusion != "success" {
			return "OWED", "the ci run at HEAD concluded " + r.Conclusion
		}
	}
	if !found {
		return "OWED", "no ci run for " + sha[:8] + " (is HEAD pushed?)"
	}
	return "done", ""
}

func rowCI(e *relEnv) relRow {
	const name = "CI is green at HEAD (run level)"
	const rec = "gh run list, matched on the full sha"
	if e.offline {
		return unknown(name, rec, "offline: not read")
	}
	sha, err := gitOut(e.root, "rev-parse", "HEAD")
	if err != nil {
		return owed(name, rec, err.Error())
	}
	if tip, err := gitOut(e.root, "rev-parse", "origin/main"); err == nil && tip != sha {
		return owed(name, rec, "HEAD is not origin/main's tip: push (and wait for the run) first")
	}
	out, err := e.run(e.root, "gh", "run", "list", "--limit", "30", "--json", "headSha,status,conclusion,workflowName")
	if err != nil {
		return unknown(name, rec, "gh run list failed: "+lastLines(out, 2))
	}
	var runs []ciRun
	if err := json.Unmarshal([]byte(out), &runs); err != nil {
		return unknown(name, rec, "gh output not read: "+err.Error())
	}
	if st, d := ciVerdict(runs, sha); st != "done" {
		return owed(name, rec, d)
	}
	return done(name, rec)
}

// ---- the command ----

var tagRE = regexp.MustCompile(`^v[0-9]+\.[0-9]+\.[0-9]+$`)

// relRefusal returns why tagging cannot start, or "".
func relRefusal(e *relEnv) string {
	if !tagRE.MatchString(e.tag) {
		return fmt.Sprintf("%q is not a version tag like v0.23.0", e.tag)
	}
	if br, err := gitOut(e.root, "rev-parse", "--abbrev-ref", "HEAD"); err != nil || br != "main" {
		return "not on main (on " + br + ")"
	}
	if st, err := gitOut(e.root, "status", "--porcelain"); err != nil || st != "" {
		return "the working tree is not clean"
	}
	if _, err := gitOut(e.root, "rev-parse", "-q", "--verify", "refs/tags/"+e.tag); err == nil {
		return e.tag + " already exists"
	}
	return ""
}

func runRelease(args []string, w io.Writer) int {
	fs := flag.NewFlagSet("gate release", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	offline := fs.Bool("offline", false, "do not use the network: the CI row reads UNKNOWN")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(w, "usage: gate release [-offline] vX.Y.Z")
		return 2
	}
	root, err := repoRoot()
	if err != nil {
		fmt.Fprintln(w, err)
		return 2
	}
	e := &relEnv{root: root, tag: fs.Arg(0), offline: *offline, run: osRun}
	if why := relRefusal(e); why != "" {
		fmt.Fprintf(w, "gate release: refused: %s\n", why)
		return 2
	}
	prev, err := newestVersionTag(root, e.tag)
	if err != nil {
		fmt.Fprintf(w, "gate release: refused: %v\n", err)
		return 2
	}
	e.prev = prev
	return reportRelease(w, e, releaseRows)
}

// releaseRows are the pre-flight's reads, in the order they are printed.
var releaseRows = []func(*relEnv) relRow{rowPending, rowPins, rowColdUser, rowGPU, rowHardware, rowBranches, rowChangelog, rowGofmt, rowLint, rowCI}

// reportRelease runs the rows in order, prints them and returns the exit code.
func reportRelease(w io.Writer, e *relEnv, rows []func(*relEnv) relRow) int {
	fmt.Fprintf(w, "gate release %s (previous tag %s)\n", e.tag, e.prev)
	code, nDone := 0, 0
	for i, f := range rows {
		r := f(e)
		fmt.Fprintf(w, "%2d. %-7s %s\n", i+1, r.State, r.Name)
		if r.Record != "" {
			fmt.Fprintf(w, "             record: %s\n", r.Record)
		}
		for _, d := range r.Detail {
			fmt.Fprintf(w, "             %s\n", d)
		}
		if r.State == "done" {
			nDone++
		} else {
			code = 1
		}
	}
	fmt.Fprintf(w, "%d of %d done", nDone, len(rows))
	if code != 0 {
		fmt.Fprint(w, "; not ready to tag")
	}
	fmt.Fprintln(w)
	return code
}
