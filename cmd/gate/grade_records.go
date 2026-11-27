package main

import (
	"fmt"
	"slices"
	"strings"

	"github.com/townsendmerino/goinfer/internal/guard/gatecore"
)

// The graders of the gates that read records: each decides from records and recorded lookups alone. TestGraders_arePure holds every file named grade_*.go to that: no exec,
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
				return nil, r, fmt.Sprintf("the newest record at this code (%s @ %s) does not count: %s", r.Date, gatecore.Short8(r.Commit), why)
			}
			return r, nil, ""
		}
		if newestWhy == "" {
			newestWhy = fmt.Sprintf("newest record %s @ %s: %s changed since", r.Date, gatecore.Short8(r.Commit), changed)
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
			c := gatecore.CanonicalPath(p)
			if slices.Contains(parityQualifyingTrees, c) {
				return c + "/", nil
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
