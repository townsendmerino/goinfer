package main

import (
	"fmt"
	"strings"
)

// The graders of the gates that read records: each decides from records and recorded lookups alone (SB12,
// docs/tasks/task-subsystems-2026-10.md §8). TestGraders_arePure holds every file named grade_*.go to that: no exec,
// environment, clock or file I/O, and no call to a package function outside such a file.

// valid says whether the run itself is one a release can lean on: a PASS, on a clean tree, with the whole heavy tier run.
func (r gpuRecord) valid() (bool, string) {
	switch {
	case r.Verdict != "PASS":
		return false, "verdict " + r.Verdict
	case r.Dirty:
		return false, "the run was made on a dirty tree"
	case r.HeavySkipped:
		return false, "the heavy tier (real models) was skipped"
	case len(r.HeavyGroups) > 0:
		return false, "the heavy tier was narrowed to " + strings.Join(r.HeavyGroups, ", ") + " (-heavy-group)"
	}
	return true, ""
}

// gpuCoverageR is, for one backend, whether the records cover head. The newest record made at the same code (every
// gpuQualifyingTrees directory identical) decides, whatever its verdict: a later FAIL at unchanged code is not hidden by an
// earlier PASS. With no record at the same code, the reason names the newest record and what changed since it.
func gpuCoverageR(recs []gpuRecord, backend, head string, treeOf func(rev, dir string) (string, error)) (cover, failed *gpuRecord, why string) {
	var newest *gpuRecord
	var newestWhy string
	for i := len(recs) - 1; i >= 0; i-- {
		r := &recs[i]
		if r.Backend != backend {
			continue
		}
		changed, err := r.changedSince(head, treeOf)
		if err != nil {
			return nil, nil, err.Error()
		}
		if changed == "" {
			if ok, why := r.valid(); !ok {
				return nil, r, fmt.Sprintf("the newest record at this code (%s @ %s) does not count: %s", r.Date, r.Commit[:8], why)
			}
			return r, nil, ""
		}
		if newest == nil {
			newest, newestWhy = r, fmt.Sprintf("newest record %s @ %s: %s/ changed since", r.Date, r.Commit[:8], changed)
		}
	}
	if newest == nil {
		return nil, nil, "no record"
	}
	return nil, nil, newestWhy
}

// gpuCoverage is gpuCoverageR without the record that failed to count.
func gpuCoverage(recs []gpuRecord, backend, head string, treeOf func(rev, dir string) (string, error)) (*gpuRecord, string) {
	cover, _, why := gpuCoverageR(recs, backend, head, treeOf)
	return cover, why
}

// valid says whether the run itself is one a release can lean on: no blocker, on a clean tree, with the real-checkpoint tier
// run and the checkset not narrowed.
func (r parityRecord) valid() (bool, string) {
	switch {
	case r.Verdict != "PASS":
		return false, "verdict " + r.Verdict
	case r.Dirty:
		return false, "the sweep was made on a dirty tree"
	case !r.Realckpt:
		return false, "the real-checkpoint tier was not run (REALCKPT=0)"
	case r.Scoped != "":
		return false, "the checkset was narrowed (GATE_RUN=" + r.Scoped + ")"
	}
	return true, ""
}

// parityCoverageR is whether the records cover head. The newest record made at the same code decides, whatever its verdict: a
// later sweep with blockers at unchanged code is not hidden by an earlier clean one. With no record at the same code, the reason
// names the newest record and what changed since it.
func parityCoverageR(recs []parityRecord, head string, paths []string, objOf func(rev, path string) (string, error)) (cover, failed *parityRecord, why string) {
	newestWhy := ""
	for i := len(recs) - 1; i >= 0; i-- {
		r := &recs[i]
		changed, err := r.changedSince(head, paths, objOf)
		if err != nil {
			return nil, nil, err.Error()
		}
		if changed == "" {
			if ok, why := r.valid(); !ok {
				return nil, r, fmt.Sprintf("the newest record at this code (%s @ %s) does not count: %s", r.Date, short8(r.Commit), why)
			}
			return r, nil, ""
		}
		if newestWhy == "" {
			newestWhy = fmt.Sprintf("newest record %s @ %s: %s changed since", r.Date, short8(r.Commit), changed)
		}
	}
	if newestWhy == "" {
		return nil, nil, "no record"
	}
	return nil, nil, newestWhy
}

// parityCoverage is parityCoverageR without the record that failed to count.
func parityCoverage(recs []parityRecord, head string, paths []string, objOf func(rev, path string) (string, error)) (*parityRecord, string) {
	cover, _, why := parityCoverageR(recs, head, paths, objOf)
	return cover, why
}

// valid says whether the run is one a release can lean on: every cell run, green, on a clean tree.
func (r quickRecord) valid() (bool, string) {
	switch {
	case !r.Full:
		return false, "the run deferred cells (not -full)"
	case r.Verdict != "PASS":
		return false, "verdict " + r.Verdict
	case r.Dirty:
		return false, "the run was made on a dirty tree"
	case r.Pass == 0:
		return false, "no test passed"
	}
	return true, ""
}

// quickCoverageR is, for one operating system, whether the records cover head. The newest record made there at the
// same code decides, whatever its verdict, so a later FAIL at unchanged code is not hidden by an earlier PASS.
// changed lists the paths that differ between two revisions.
func quickCoverageR(recs []quickRecord, goos, head string, changed func(from, to string) ([]string, error)) (cover, failed *quickRecord, why string) {
	var newestWhy string
	for i := len(recs) - 1; i >= 0; i-- {
		r := &recs[i]
		if r.OS != goos {
			continue
		}
		paths, err := changed(r.Commit, head)
		if err != nil {
			return nil, nil, fmt.Sprintf("could not compare %s with HEAD: %v", r.Commit[:8], err)
		}
		first := ""
		for _, p := range paths {
			if quickRecordVoidedBy(p) {
				first = p
				break
			}
		}
		if first == "" {
			if ok, why := r.valid(); !ok {
				return nil, r, fmt.Sprintf("the newest record at this code (%s @ %s) does not count: %s", r.Date, r.Commit[:8], why)
			}
			return r, nil, ""
		}
		if newestWhy == "" {
			newestWhy = fmt.Sprintf("newest record %s @ %s: %s changed since", r.Date, r.Commit[:8], first)
		}
	}
	if newestWhy == "" {
		return nil, nil, "no record"
	}
	return nil, nil, newestWhy
}

// quickCoverage is quickCoverageR without the record that failed to count.
func quickCoverage(recs []quickRecord, goos, head string, changed func(from, to string) ([]string, error)) (*quickRecord, string) {
	cover, _, why := quickCoverageR(recs, goos, head, changed)
	return cover, why
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

// judgeGiwBytes decides one cell from its three transcodes (new#1, new#2, old).
func judgeGiwBytes(n1, n2, old giwResult) (verdict, detail string) {
	switch {
	case !n1.OK && !old.OK:
		detail = "neither build transcodes it — new: " + n1.Err
		if old.Err != n1.Err {
			detail += " | old: " + old.Err
		}
		return vNotRun, detail
	case !n1.OK:
		return vDifferent, "the new build fails where the old transcodes: " + n1.Err
	case !old.OK:
		return vDifferent, "the old build fails where the new transcodes: " + old.Err
	case !n2.OK:
		return vDifferent, "the new build's second transcode failed: " + n2.Err
	case n1.SHA != n2.SHA:
		return vDifferent, fmt.Sprintf("NONDETERMINISTIC: the new build wrote %s… (%d B) then %s… (%d B) from the same source",
			n1.SHA[:16], n1.Size, n2.SHA[:16], n2.Size)
	case n1.SHA != old.SHA:
		return vDifferent, fmt.Sprintf("sha256 old %s… (%d B), new %s… (%d B)", old.SHA[:16], old.Size, n1.SHA[:16], n1.Size)
	}
	return vIdentical, ""
}

// changedSince names the first gpuQualifyingTrees directory whose tree at head differs from its tree at the run's commit, or "".
// An error reading either side is returned, not read as "unchanged".
func (r gpuRecord) changedSince(head string, treeOf func(rev, dir string) (string, error)) (string, error) {
	for _, d := range gpuQualifyingTrees {
		then, err1 := treeOf(r.Commit, d)
		now, err2 := treeOf(head, d)
		if err1 != nil || err2 != nil {
			return "", fmt.Errorf("could not compare %s/ (%v %v)", d, err1, err2)
		}
		if then != now {
			return d, nil
		}
	}
	return "", nil
}

func short8(sha string) string {
	if len(sha) > 8 {
		return sha[:8]
	}
	return sha
}

// quickRecordVoidedBy reports whether a change to path, since a run, voids the run. Everything a test can compile or
// read does; the documentation tree, the website, the CI definitions, the CHANGELOG's entries (changelog.d/) and the
// top-level prose files do not, so a release commit that writes the CHANGELOG and commits the records keeps its run.
func quickRecordVoidedBy(path string) bool {
	for _, p := range []string{"docs/", "apps/site/", ".github/", changelogFragmentDir + "/"} {
		if strings.HasPrefix(path, p) {
			return false
		}
	}
	return strings.Contains(path, "/") || !strings.HasSuffix(path, ".md")
}

// changedSince names the first qualifying path whose object at head differs from its object at the sweep's commit (a directory
// with a trailing slash), or "". A path one side lacks has changed. An error reading either side is returned, not read as
// "unchanged".
func (r parityRecord) changedSince(head string, paths []string, objOf func(rev, path string) (string, error)) (string, error) {
	for _, p := range paths {
		then, err1 := objOf(r.Commit, p)
		now, err2 := objOf(head, p)
		if err1 != nil || err2 != nil {
			return "", fmt.Errorf("could not compare %s (%v %v)", p, err1, err2)
		}
		if then != now {
			c := canonicalPath(p)
			for _, d := range parityQualifyingTrees {
				if c == d {
					return c + "/", nil
				}
			}
			return c, nil
		}
	}
	return "", nil
}

// gpuQualifyingTrees are the directories whose content a GPU-gate run vouches for. A change in any of them since the run
// voids the run for a later commit; a change anywhere else (docs, the site, cmd) does not.
var gpuQualifyingTrees = []string{"backend/metal", "backend/cuda", "backend/webgpu", "decoder"}

// parityQualifyingTrees are the directories whose every file a sweep vouches for, hashed by the manifest or not.
var parityQualifyingTrees = []string{"decoder", "backend/cuda", "backend/metal", "backend/webgpu"}

// backendMoveMap is the layout Tier 2, PR 1 move: each GPU backend directory's path after the move (key) and before it
// (value). A commit from before the move names the old path, so git reports the new one as absent there; treeAt and
// objectAt then read the old name once. A record written before the move names the old path, so canonicalPath maps it
// to the current name before the grader prints or compares it.
// internalMoveMap is the layout Tier 2, PR 2 regroup: each internal/ package's path after the move (key) and before it
// (value). GENERATED from scripts/layout/tier2-2.sh (its MAPPING dict); regenerate from that dict, do not hand-edit.
var internalMoveMap = map[string]string{
	"internal/runtime/adapter":    "internal/adapter",
	"internal/app/chatapp":        "internal/chatapp",
	"internal/app/cliutil":        "internal/cliutil",
	"internal/app/connect":        "internal/connect",
	"internal/app/decidecmd":      "internal/decidecmd",
	"internal/runtime/diskfree":   "internal/diskfree",
	"internal/runtime/f16":        "internal/f16",
	"internal/app/fitcmd":         "internal/fitcmd",
	"internal/runtime/giw":        "internal/giw",
	"internal/diag/hwcensus":      "internal/hwcensus",
	"internal/diag/hwreport":      "internal/hwreport",
	"internal/runtime/hybridattn": "internal/hybridattn",
	"internal/app/inproc":         "internal/inproc",
	"internal/diag/kernelprint":   "internal/kernelprint",
	"internal/runtime/kvmgr":      "internal/kvmgr",
	"internal/app/loadflags":      "internal/loadflags",
	"internal/runtime/mamba2":     "internal/mamba2",
	"internal/runtime/mem":        "internal/mem",
	"internal/guard/memgate":      "internal/memgate",
	"internal/runtime/modelload":  "internal/modelload",
	"internal/guard/movedpaths":   "internal/movedpaths",
	"internal/guard/paritykey":    "internal/paritykey",
	"internal/runtime/prequant":   "internal/prequant",
	"internal/guard/prezerolint":  "internal/prezerolint",
	"internal/runtime/program":    "internal/program",
	"internal/app/pullcmd":        "internal/pullcmd",
	"internal/runtime/sampling":   "internal/sampling",
	"internal/diag/sb2census":     "internal/sb2census",
	"internal/runtime/scheduler":  "internal/scheduler",
	"internal/app/serveapp":       "internal/serveapp",
	"internal/guard/servecheck":   "internal/servecheck",
	"internal/guard/swapguard":    "internal/swapguard",
	"internal/diag/towerid":       "internal/towerid",
	"internal/runtime/whisper":    "internal/whisper",
}

// allMoveMap is backendMoveMap and internalMoveMap together: the moves preMovePath reads through.
var allMoveMap = mergeMoveMaps(backendMoveMap, internalMoveMap)

func mergeMoveMaps(maps ...map[string]string) map[string]string {
	out := map[string]string{}
	for _, m := range maps {
		for k, v := range m {
			out[k] = v
		}
	}
	return out
}

var backendMoveMap = map[string]string{
	"backend/cuda":   "cuda",
	"backend/metal":  "metal",
	"backend/webgpu": "gpu",
}

// preMovePath returns p with its moved backend prefix replaced by the pre-move one, and whether p had such a prefix.
func preMovePath(p string) (string, bool) {
	for nw, old := range allMoveMap {
		if p == nw {
			return old, true
		}
		if strings.HasPrefix(p, nw+"/") {
			return old + p[len(nw):], true
		}
	}
	return p, false
}

// canonicalPath is the current name of p: a pre-move backend path (a record's "cuda/", "metal/", "gpu/") maps to its
// backend/ name; anything else is p itself.
func canonicalPath(p string) string {
	for nw, old := range backendMoveMap {
		if p == old {
			return nw
		}
		if strings.HasPrefix(p, old+"/") {
			return nw + p[len(old):]
		}
	}
	return p
}

// absentFromRev reports whether err is git saying a path is not in a revision: `git rev-parse rev:path` says "does not
// exist" or "exists on disk, but not in" when the working tree still has the path.
func absentFromRev(err error) bool {
	return err != nil && (strings.Contains(err.Error(), "does not exist") || strings.Contains(err.Error(), "but not in"))
}

// treeAt is the git tree id of dir at rev, read through lookup. When git says a moved backend path is absent at rev, it
// reads the pre-move name once.
func treeAt(lookup func(rev, dir string) (string, error), rev, dir string) (string, error) {
	id, err := lookup(rev, dir)
	if err == nil {
		return id, nil
	}
	old, moved := preMovePath(dir)
	if !moved || !absentFromRev(err) {
		return id, err
	}
	return lookup(rev, old)
}

// objectAt is the git object id of path at rev, read through lookup. `git ls-tree` prints nothing for a missing path rather
// than failing, so an empty id counts as absent too; a moved backend path then reads the pre-move name once.
func objectAt(lookup func(rev, path string) (string, error), rev, path string) (string, error) {
	id, err := lookup(rev, path)
	if err == nil && id != "" {
		return id, nil
	}
	old, moved := preMovePath(path)
	if !moved || (err != nil && !absentFromRev(err)) {
		return id, err
	}
	return lookup(rev, old)
}

// absentBeforeMove answers a lookup that has no entry in a record, when the record holds the pre-move twin of it: the record
// was captured before the backend move, so git would say the moved path is absent at that commit, and the grader then reads
// the pre-move name (treeAt, objectAt). has reports whether a lookup has an entry. handled is false when there is no twin;
// the caller then fails as before.
func absentBeforeMove(kind string, args []string, has func(kind string, args ...string) bool) (string, error, bool) {
	old, moved := mapMovedArgs(args)
	if !moved || !has(kind, old...) {
		return "", nil, false
	}
	if kind == "object" || (kind == "git" && len(args) > 0 && args[0] == "ls-tree") {
		return "", nil, true // ls-tree prints nothing for a missing path
	}
	return "", fmt.Errorf("fatal: path does not exist"), true
}

// mapMovedArgs replaces each moved backend path in args (a path, or the path after "rev:") with its pre-move name.
func mapMovedArgs(args []string) ([]string, bool) {
	out := make([]string, len(args))
	moved := false
	for i, a := range args {
		if rev, p, ok := strings.Cut(a, ":"); ok {
			if q, m := preMovePath(p); m {
				out[i], moved = rev+":"+q, true
				continue
			}
		}
		if q, m := preMovePath(a); m {
			out[i], moved = q, true
			continue
		}
		out[i] = a
	}
	return out, moved
}
