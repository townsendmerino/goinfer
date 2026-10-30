package main

import (
	"encoding/json"
	"fmt"
	"io"
	"regexp"
	"sort"
	"strings"
)

// The rows of `gate release`, graders of what they read: every read goes through relEnv.look, which the runner answers
// live and records (release.go), and a replay answers from that record. A row's state, detail and record are a function of
// the answers alone.

// relRow is one pre-flight read.
type relRow struct {
	Name   string
	State  string // "done", "OWED" or "UNKNOWN"
	Detail []string
	Record string // the path that proves it, or that is missing
}

// relEnv is what a row reads from: the tag being cut and the one before it, whether the network may be used, and look,
// the one reader. root and run are the live reader's (liveRelEnv); a row never reads them.
type relEnv struct {
	root, tag, prev string
	offline         bool
	run             func(dir, name string, args ...string) (string, error) // combined output
	look            func(kind string, args ...string) (string, error)
	looks           []relLookup // what the live reader answered, each read once, in the order first asked
}

// relLookup is one recorded read: its kind and arguments, the answer, and the error's text when it failed.
type relLookup struct {
	Kind   string   `json:"kind"`
	Args   []string `json:"args,omitempty"`
	Out    string   `json:"out,omitempty"`
	Err    string   `json:"err,omitempty"`
	Failed bool     `json:"failed,omitempty"`
}

// releaseInputs is what a release run records beside its provenance: what the report header names and every read.
type releaseInputs struct {
	Tag     string      `json:"tag"`
	Prev    string      `json:"prev"`
	Offline bool        `json:"offline,omitempty"`
	Looks   []relLookup `json:"looks"`
}

func relLookupKey(kind string, args []string) string {
	return kind + "\x00" + strings.Join(args, "\x00")
}

// relReplayLook answers each read from a record. A read the record does not hold fails, and is added to *missing.
func relReplayLook(looks []relLookup, missing *[]string) func(kind string, args ...string) (string, error) {
	ans := map[string]relLookup{}
	for _, l := range looks {
		ans[relLookupKey(l.Kind, l.Args)] = l
	}
	return func(kind string, args ...string) (string, error) {
		l, ok := ans[relLookupKey(kind, args)]
		if !ok {
			*missing = append(*missing, strings.TrimSpace(kind+" "+strings.Join(args, " ")))
			return "", fmt.Errorf("not in the record: %s %s", kind, strings.Join(args, " "))
		}
		if l.Failed {
			return l.Out, fmt.Errorf("%s", l.Err)
		}
		return l.Out, nil
	}
}

// The reads. Paths are relative to the repository root.
func (e *relEnv) read(path string) (string, error)   { return e.look("read", path) }
func (e *relEnv) git(args ...string) (string, error) { return e.look("git", args...) }
func (e *relEnv) cmd(dir, name string, args ...string) (string, error) {
	return e.look("run", append([]string{dir, name}, args...)...)
}

// glob is the sorted repository-relative paths matching pattern.
func (e *relEnv) glob(pattern string) []string {
	out, err := e.look("glob", pattern)
	if err != nil || out == "" {
		return nil
	}
	return strings.Split(out, "\n")
}

// gitObject is the git object id of path (a file or a directory) at rev, and "" when rev has no such path. An unknown rev is an error.
func (e *relEnv) gitObject(rev, path string) (string, error) {
	out, err := e.git("ls-tree", rev, "--", path)
	if err != nil {
		return "", err
	}
	f := strings.Fields(out) // <mode> <type> <object>\t<path>
	if len(f) < 3 {
		return "", nil
	}
	return f[2], nil
}

// relPath joins a module directory and a path inside it as filepath.Join does for these clean, relative names.
func relPath(dir, name string) string {
	if dir == "." || dir == "" {
		return name
	}
	return dir + "/" + name
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
	cl, err := e.read("CHANGELOG.md")
	if err != nil {
		return owed(name, "CHANGELOG.md", err.Error())
	}
	var pending map[string]string
	out, err := e.look("pending")
	if err == nil {
		err = json.Unmarshal([]byte(out), &pending)
	}
	if err != nil {
		return owed(name, "cmd/gate/parity.go awaitingFirstConfirmation", err.Error())
	}
	prevVer, prevDate, blockers, err := pendingOutlivingRelease(pending, cl, e.tag)
	if err != nil {
		return owed(name, "cmd/gate/parity.go awaitingFirstConfirmation", err.Error())
	}
	if len(blockers) > 0 {
		return owed(name, "cmd/gate/parity.go awaitingFirstConfirmation", append([]string{
			fmt.Sprintf("pending since before %s (%s); run each and `gate ledger promote`, or move it to neverConfirmed with its reason:", prevVer, prevDate)}, blockers...)...)
	}
	return done(name, "cmd/gate/parity.go awaitingFirstConfirmation", fmt.Sprintf("nothing pending since before %s (%s)", prevVer, prevDate))
}

// ---- row 2: the five modules agree on a tagged aikit, and none requires aikit/gpu ----

var relModules = []string{".", "gpu", "cuda", "metal", "demo/agent"}

// rePseudo matches the three pseudo-version forms Go writes (vX.0.0-yyyymmddhhmmss-hash, vX.Y.Z-pre.0.yyyymmddhhmmss-hash,
// vX.Y.Z-0.yyyymmddhhmmss-hash), with or without build metadata such as +incompatible. goinfer's main may require one
// between weekly aikit tags; a release may not (the owner's decision, docs/tasks/task-subsystems-2026-10.md §14).
var rePseudo = regexp.MustCompile(`[-.][0-9]{14}-[0-9a-f]{12}(?:\+[0-9A-Za-z.-]+)?$`)

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
	aikit := map[string]string{}
	var gpuReq []string // aikit's gpu module moved into goinfer (Track AX); a require is a module that did not switch
	for _, m := range relModules {
		b, err := e.read(relPath(m, "go.mod"))
		if err != nil {
			return owed(name, m+"/go.mod", err.Error())
		}
		if v := modRequire(b, "github.com/townsendmerino/aikit"); v != "" {
			aikit[m] = v
		}
		if v := modRequire(b, "github.com/townsendmerino/aikit/gpu"); v != "" {
			gpuReq = append(gpuReq, m+" "+v)
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
	var pseudo []string
	for m, v := range aikit {
		if rePseudo.MatchString(v) {
			pseudo = append(pseudo, m+" "+v)
		}
	}
	if len(pseudo) > 0 {
		sort.Strings(pseudo)
		bad = append(bad, "requires an untagged aikit (a pseudo-version); a release needs a tagged aikit (SD3): tag aikit and bump to it: "+strings.Join(pseudo, ", "))
	}
	if len(gpuReq) > 0 {
		bad = append(bad, "requires aikit/gpu, which goinfer no longer uses (cuda/internal/gpudev, metal/internal/gpudev): "+strings.Join(gpuReq, ", "))
	}
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
	return done(name, "go.mod of "+strings.Join(relModules, ", "), fmt.Sprintf("aikit %s in %d modules; no aikit/gpu", vs, len(aikit)))
}

// ---- row 3: a cold-user record for the previous tag ----

var coldUserTitle = regexp.MustCompile(`(?m)^# .*\bgoinfer (v[0-9]+\.[0-9]+\.[0-9]+)\b`)

func rowColdUser(e *relEnv) relRow {
	name := "a cold-user record for the previous tag (" + e.prev + ")"
	var seen []string
	for _, rel := range e.glob("docs/measurements/cold-user-*.md") {
		head, err := e.read(rel)
		if err != nil {
			continue
		}
		if len(head) > 600 {
			head = head[:600]
		}
		if m := coldUserTitle.FindStringSubmatch(head); m != nil {
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
	var recs []gpuRecord
	out, err := e.look("records", "gpu")
	if err == nil {
		err = json.Unmarshal([]byte(out), &recs)
	}
	if err != nil {
		return owed(name, gpuRecordDirDefault, err.Error())
	}
	head, err := e.git("rev-parse", "HEAD")
	if err != nil {
		return owed(name, gpuRecordDirDefault, err.Error())
	}
	treeOf := func(rev, d string) (string, error) { return e.git("rev-parse", rev+":"+d) }
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
	if out, err := e.cmd(".", "go", "test", "./decoder", "-run", "HardwareMatrix", "-count=1"); err != nil {
		return owed(name, rec, "the generated section is stale or the test failed (regenerate: go test ./decoder -run HardwareMatrix -update):", lastLines(out, 3))
	}
	head, err := e.read("docs/hardware-coverage.json")
	if err != nil {
		return owed(name, rec, err.Error())
	}
	now, err := neverExecuted([]byte(head))
	if err != nil {
		return owed(name, rec, err.Error())
	}
	detail := []string{fmt.Sprintf("%d hardware-selected paths have no record of ever running: %s", len(now), strings.Join(now, ", ")),
		"make sure nothing in the release notes, the README or the site claims support for them"}
	if out, err := e.cmd(".", "git", "show", e.prev+":docs/hardware-coverage.json"); err == nil {
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
	out, err := e.cmd(".", "git", "for-each-ref", "--format=%(refname:short)", "refs/heads", "refs/remotes")
	if err != nil {
		return nil, nil, err
	}
	for _, br := range strings.Fields(out) {
		if br == "main" || br == "origin/main" || strings.HasSuffix(br, "/HEAD") || br == "origin" {
			continue
		}
		cherry, err := e.cmd(".", "git", "cherry", "origin/main", br)
		if err != nil {
			continue
		}
		var subjects []string
		for _, ln := range strings.Split(cherry, "\n") {
			if strings.HasPrefix(ln, "+ ") {
				sha := strings.TrimPrefix(ln, "+ ")
				subj, _ := e.cmd(".", "git", "log", "-1", "--format=%s", sha)
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

// clSection returns the body of the CHANGELOG section named `Unreleased` or a version (`v0.23.0`), up to the next `## [`.
func clSection(doc, name string) (string, bool) {
	head := "## [" + name + "]"
	_, after, ok := strings.Cut(doc, head)
	if !ok {
		return "", false
	}
	if j := strings.Index(after, "\n## ["); j >= 0 {
		after = after[:j]
	}
	return after, true
}

// rowChangelog reads `gate changelog -base <prev> -section <s>` as the "changelog" read: its report, and an error when it
// exits non-zero.
func rowChangelog(e *relEnv) relRow {
	const name = "the CHANGELOG names what changed"
	section := "Unreleased"
	if b, err := e.read("CHANGELOG.md"); err == nil {
		if _, ok := clSection(b, e.tag); ok {
			section = e.tag
		}
	}
	out, err := e.look("changelog", "-base", e.prev, "-section", section)
	var lines []string
	for _, ln := range strings.Split(strings.TrimSpace(out), "\n") {
		if strings.Contains(ln, "OWED") || strings.HasPrefix(ln, "gate changelog:") {
			lines = append(lines, strings.TrimSpace(ln))
		}
	}
	rec := "CHANGELOG.md [" + section + "], docs/changelog-waivers.txt"
	if err == nil {
		return done(name, rec)
	}
	return owed(name, rec, lines...)
}

// ---- rows 8-10: gofmt, the citation lint, CI at HEAD ----

func rowGofmt(e *relEnv) relRow {
	const name = "gofmt -l is empty in every module"
	var dirty []string
	for _, m := range relModules {
		out, err := e.cmd(m, "gofmt", "-l", ".")
		if err != nil {
			return owed(name, m, err.Error())
		}
		for _, f := range strings.Fields(out) {
			dirty = append(dirty, relPath(m, f))
		}
	}
	if len(dirty) > 0 {
		return owed(name, "gofmt -l .", dirty...)
	}
	return done(name, "gofmt -l .")
}

func rowLint(e *relEnv) relRow {
	const name = "the citation lint is green"
	out, err := e.cmd(".", "python3", "scripts/queue_citation_lint.py")
	if err != nil {
		return owed(name, "scripts/queue_citation_lint.py", lastLines(out, 4))
	}
	return done(name, "scripts/queue_citation_lint.py")
}

// ciRun is what `gh run list` reports for one run.
type ciRun struct {
	HeadSha    string `json:"headSha"`
	Status     string `json:"status"`
	Conclusion string `json:"conclusion"`
	Workflow   string `json:"workflowName"`
}

func rowCI(e *relEnv) relRow {
	const name = "CI is green at HEAD (run level)"
	const rec = "gh run list, matched on the full sha"
	if e.offline {
		return unknown(name, rec, "offline: not read")
	}
	sha, err := e.git("rev-parse", "HEAD")
	if err != nil {
		return owed(name, rec, err.Error())
	}
	if tip, err := e.git("rev-parse", "origin/main"); err == nil && tip != sha {
		return owed(name, rec, "HEAD is not origin/main's tip: push (and wait for the run) first")
	}
	out, err := e.cmd(".", "gh", "run", "list", "--limit", "30", "--json", "headSha,status,conclusion,workflowName")
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

// ---- row 11: the §C1 parity sweep ----

// rowParity is the release command's §C1 row: a complete parity sweep covers HEAD. The paths a sweep vouches for
// (parityQualifyingPaths of the manifest) are the "parity-paths" read, one per line.
func rowParity(e *relEnv) relRow {
	const name = "the §C1 parity sweep covers HEAD"
	const want = parityRecordDirDefault + "/parity-<time>-<commit>.json"
	const how = "run `go run ./cmd/gate parity -logdir <durable dir>` on nobara, at night (it takes hours), and commit the record it writes"
	var recs []parityRecord
	out, err := e.look("records", "parity")
	if err == nil {
		err = json.Unmarshal([]byte(out), &recs)
	}
	if err != nil {
		return owed(name, want, err.Error())
	}
	pout, err := e.look("parity-paths")
	if err != nil {
		return owed(name, want, err.Error())
	}
	var paths []string
	if pout != "" {
		paths = strings.Split(pout, "\n")
	}
	head, err := e.git("rev-parse", "HEAD")
	if err != nil {
		return owed(name, want, err.Error())
	}
	r, why := parityCoverage(recs, head, paths, e.gitObject)
	if r == nil {
		return owed(name, want, why, how)
	}
	detail := fmt.Sprintf("%s @ %s on %s: %d required gates, no blocker", r.Date, short8(r.Commit), r.Host, r.Required)
	if r.Gaps > 0 || r.FirstRuns > 0 {
		detail += fmt.Sprintf(" (%d coverage gap(s) and %d first-run failure(s) reported, as the sweep prints them)", r.Gaps, r.FirstRuns)
	}
	return done(name, parityRecordDirDefault, detail)
}

// ---- row 12: the deferred tests on both systems ----

// quickRecordSystems are the operating systems a release needs a record from: darwin builds and runs the metal
// module, linux the cuda one, and neither box can run the other's.
var quickRecordSystems = []string{"darwin", "linux"}

// rowQuickFull is the release row for the tests the day loop defers: a covering `gate quick -full` record from each
// of quickRecordSystems.
func rowQuickFull(e *relEnv) relRow {
	const name = "the deferred tests (gate quick -full) on both systems cover HEAD"
	var recs []quickRecord
	out, err := e.look("records", "quick")
	if err == nil {
		err = json.Unmarshal([]byte(out), &recs)
	}
	if err != nil {
		return owed(name, quickRecordDirDefault, err.Error())
	}
	head, err := e.git("rev-parse", "HEAD")
	if err != nil {
		return owed(name, quickRecordDirDefault, err.Error())
	}
	changed := func(from, to string) ([]string, error) {
		out, err := e.git("diff", "--name-only", from, to)
		if err != nil {
			return nil, err
		}
		if out == "" {
			return nil, nil
		}
		return strings.Split(out, "\n"), nil
	}
	var bad, ok []string
	for _, goos := range quickRecordSystems {
		if r, why := quickCoverage(recs, goos, head, changed); r != nil {
			ok = append(ok, fmt.Sprintf("%s: %s @ %s (%d pass, %d skip)", goos, r.Date, r.Commit[:8], r.Pass, r.Skip))
		} else {
			bad = append(bad, goos+": "+why)
		}
	}
	if len(bad) > 0 {
		return owed(name, quickRecordDirDefault+"/<os>-<arch>-<time>-<commit>.json",
			append(bad, "run `go run ./cmd/gate quick -full -base <the previous tag>` on that box (about 10 minutes on the Mac; queue it at night) and commit the record")...)
	}
	return done(name, quickRecordDirDefault, ok...)
}

// ---- the report ----

// releaseRows are the pre-flight's reads, in the order they are printed.
var releaseRows = []func(*relEnv) relRow{rowPending, rowPins, rowColdUser, rowGPU, rowHardware, rowBranches, rowChangelog, rowGofmt, rowLint, rowCI, rowParity, rowQuickFull}

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
