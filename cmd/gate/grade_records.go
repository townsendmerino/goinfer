package main

import (
	"fmt"
	"strings"
)

// The graders of the gates that read records: each decides from records and recorded lookups alone (SB12,
// docs/tasks/task-subsystems-2026-10.md §8). TestGraders_arePure holds every file named grade_*.go to that: no exec,
// environment, clock or file I/O, and no call to a package function outside such a file.

// valid says whether the run itself is one a release can lean on: a PASS, on a clean tree, with the heavy tier run.
func (r gpuRecord) valid() (bool, string) {
	switch {
	case r.Verdict != "PASS":
		return false, "verdict " + r.Verdict
	case r.Dirty:
		return false, "the run was made on a dirty tree"
	case r.HeavySkipped:
		return false, "the heavy tier (real models) was skipped"
	}
	return true, ""
}

// gpuCoverage is, for one backend, whether the records cover head. The newest record made at the same code (every
// gpuQualifyingTrees directory identical) decides, whatever its verdict: a later FAIL at unchanged code is not hidden by an
// earlier PASS. With no record at the same code, the reason names the newest record and what changed since it.
func gpuCoverage(recs []gpuRecord, backend, head string, treeOf func(rev, dir string) (string, error)) (*gpuRecord, string) {
	var newest *gpuRecord
	var newestWhy string
	for i := len(recs) - 1; i >= 0; i-- {
		r := &recs[i]
		if r.Backend != backend {
			continue
		}
		changed, err := r.changedSince(head, treeOf)
		if err != nil {
			return nil, err.Error()
		}
		if changed == "" {
			if ok, why := r.valid(); !ok {
				return nil, fmt.Sprintf("the newest record at this code (%s @ %s) does not count: %s", r.Date, r.Commit[:8], why)
			}
			return r, ""
		}
		if newest == nil {
			newest, newestWhy = r, fmt.Sprintf("newest record %s @ %s: %s/ changed since", r.Date, r.Commit[:8], changed)
		}
	}
	if newest == nil {
		return nil, "no record"
	}
	return nil, newestWhy
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

// parityCoverage is whether the records cover head. The newest record made at the same code decides, whatever its verdict: a
// later sweep with blockers at unchanged code is not hidden by an earlier clean one. With no record at the same code, the reason
// names the newest record and what changed since it.
func parityCoverage(recs []parityRecord, head string, paths []string, objOf func(rev, path string) (string, error)) (*parityRecord, string) {
	newestWhy := ""
	for i := len(recs) - 1; i >= 0; i-- {
		r := &recs[i]
		changed, err := r.changedSince(head, paths, objOf)
		if err != nil {
			return nil, err.Error()
		}
		if changed == "" {
			if ok, why := r.valid(); !ok {
				return nil, fmt.Sprintf("the newest record at this code (%s @ %s) does not count: %s", r.Date, short8(r.Commit), why)
			}
			return r, ""
		}
		if newestWhy == "" {
			newestWhy = fmt.Sprintf("newest record %s @ %s: %s changed since", r.Date, short8(r.Commit), changed)
		}
	}
	if newestWhy == "" {
		return nil, "no record"
	}
	return nil, newestWhy
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

// quickCoverage is, for one operating system, whether the records cover head. The newest record made there at the
// same code decides, whatever its verdict, so a later FAIL at unchanged code is not hidden by an earlier PASS.
// changed lists the paths that differ between two revisions.
func quickCoverage(recs []quickRecord, goos, head string, changed func(from, to string) ([]string, error)) (*quickRecord, string) {
	var newestWhy string
	for i := len(recs) - 1; i >= 0; i-- {
		r := &recs[i]
		if r.OS != goos {
			continue
		}
		paths, err := changed(r.Commit, head)
		if err != nil {
			return nil, fmt.Sprintf("could not compare %s with HEAD: %v", r.Commit[:8], err)
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
				return nil, fmt.Sprintf("the newest record at this code (%s @ %s) does not count: %s", r.Date, r.Commit[:8], why)
			}
			return r, ""
		}
		if newestWhy == "" {
			newestWhy = fmt.Sprintf("newest record %s @ %s: %s changed since", r.Date, r.Commit[:8], first)
		}
	}
	if newestWhy == "" {
		return nil, "no record"
	}
	return nil, newestWhy
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
	for _, p := range []string{"docs/", "site/", ".github/", changelogFragmentDir + "/"} {
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
			for _, d := range parityQualifyingTrees {
				if p == d {
					return p + "/", nil
				}
			}
			return p, nil
		}
	}
	return "", nil
}
