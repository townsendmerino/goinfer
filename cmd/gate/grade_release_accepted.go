package main

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// The accepted-reds file of a release, docs/releases/<tag>-accepted.json. A gate record that is not a PASS leaves its release row OWED. Francis can
// accept a failing record whose failures were read and classified; the file says which, and `gate release` then prints the row as "accepted" with the
// entries, and does not count it against the tag. It never turns a row done: the record stays red, and the report says so.
//
//	{"schema": 1, "release": "v0.25.0", "accepted": [
//	  {"record": "cuda-20261105T022250Z-ccdfb3ad.json", "item": "cuda-heavy-kernels: TestAttnBatchedBandwidth",
//	   "class": "test-bug", "evidence": "docs/measurements/post-tag-test-fixes-2026-11-05/README.md",
//	   "accepted_by": "Francis", "accepted_on": "2026-11-05"}]}
//
// An entry names the record by its file name, which holds the backend or system, the time and the commit; the row builds that name from the failing
// record it judged, so an entry for another record, or for a rerun, does not match. A record is accepted only when its entries name as many distinct items
// as it has reds (coversReds), so a second, new red is never hidden behind an accepted one. Records carry counts, not the names of what failed: the gate
// compares counts, and the report shows the entries for the reader to match.
const acceptedDirDefault = "docs/releases"

// acceptedClasses are the classes an entry may name. A product defect is not among them: it is fixed, or the tag waits.
var acceptedClasses = []string{"test-bug", "gate-definition", "fixture", "toolchain-artifact", "environment", "stale-golden"}

// acceptedApprover is the only name that makes an entry count. The file records a decision of the owner's; an entry another name wrote is a
// draft, and the row stays OWED naming it.
const acceptedApprover = "Francis"

type acceptedFile struct {
	Schema   int             `json:"schema"`
	Release  string          `json:"release"`
	Accepted []acceptedEntry `json:"accepted"`
}

type acceptedEntry struct {
	Record     string `json:"record"`      // the record's file name
	Item       string `json:"item"`        // the test, group or cell
	Class      string `json:"class"`       // one of acceptedClasses
	Evidence   string `json:"evidence"`    // where the classification is written down
	AcceptedBy string `json:"accepted_by"` // acceptedApprover
	AcceptedOn string `json:"accepted_on"` // YYYY-MM-DD
}

// recordFileName is the name a gate record is committed under: prefix, the run's UTC time as 20060102T150405Z, and the first 8 of its commit.
func recordFileName(prefix, date, commit string) string {
	t, err := time.Parse(time.RFC3339, date)
	if err != nil || len(commit) < 8 {
		return prefix + "-unreadable.json"
	}
	return fmt.Sprintf("%s-%s-%s.json", prefix, t.UTC().Format("20060102T150405Z"), commit[:8])
}

// acceptedFor returns the valid entries the release's accepted file holds for one record file, and a problem to print when the file is unreadable
// or an entry for that record is not valid. Entries for other records are not looked at.
func acceptedFor(e *relEnv, record string) (entries []acceptedEntry, problem string) {
	text, err := e.look("accepted", e.tag)
	if err != nil {
		return nil, fmt.Sprintf("%s/%s-accepted.json could not be read: %v", acceptedDirDefault, e.tag, err)
	}
	if strings.TrimSpace(text) == "" {
		return nil, ""
	}
	var f acceptedFile
	if err := json.Unmarshal([]byte(text), &f); err != nil {
		return nil, fmt.Sprintf("%s/%s-accepted.json is not valid JSON: %v", acceptedDirDefault, e.tag, err)
	}
	if f.Schema != 1 || f.Release != e.tag {
		return nil, fmt.Sprintf("%s/%s-accepted.json is schema %d for release %q, want schema 1 for %q", acceptedDirDefault, e.tag, f.Schema, f.Release, e.tag)
	}
	var bad []string
	for i, a := range f.Accepted {
		if a.Record != record {
			continue
		}
		switch why := a.invalid(); {
		case why != "":
			bad = append(bad, fmt.Sprintf("entry %d (%q): %s", i+1, a.Item, why))
		default:
			entries = append(entries, a)
		}
	}
	if len(bad) > 0 {
		return nil, "not counted: " + strings.Join(bad, "; ")
	}
	return entries, ""
}

func (a acceptedEntry) invalid() string {
	switch {
	case strings.TrimSpace(a.Item) == "":
		return "no item"
	case !isAcceptedClass(a.Class):
		return fmt.Sprintf("class %q is none of %s", a.Class, strings.Join(acceptedClasses, ", "))
	case strings.TrimSpace(a.Evidence) == "":
		return "no evidence"
	case a.AcceptedBy != acceptedApprover:
		return fmt.Sprintf("accepted_by %q, want %q", a.AcceptedBy, acceptedApprover)
	}
	if _, err := time.Parse("2006-01-02", a.AcceptedOn); err != nil {
		return fmt.Sprintf("accepted_on %q is not a date (YYYY-MM-DD)", a.AcceptedOn)
	}
	return ""
}

// acceptedLines is how the report shows accepted entries.
func acceptedLines(label string, entries []acceptedEntry) []string {
	var out []string
	for _, a := range entries {
		out = append(out, fmt.Sprintf("accepted %s: %s [%s] — %s (accepted by %s, %s)", label, a.Item, a.Class, a.Evidence, a.AcceptedBy, a.AcceptedOn))
	}
	return out
}

// accepted is a row that is red but whose every failing record has accepted entries: it does not block the tag, and is not "done".
func accepted(name, record string, detail ...string) relRow {
	return relRow{Name: name, State: "accepted", Record: record, Detail: detail}
}

func isAcceptedClass(c string) bool {
	for _, k := range acceptedClasses {
		if k == c {
			return true
		}
	}
	return false
}

// A record's reds are counted from what it holds: a GPU record's failed groups and its vacuous and empty cells, a quick -full record's failed tests, a parity
// record's blockers; a record that is not a PASS with none of those (a dirty tree, a narrowed run) has one red item. An accepted file covers a record only when it
// names that many distinct items: one entry for one red does not hide a second, new one.
func gpuReds(r *gpuRecord) int     { return max(r.Groups.Fail+len(r.VacuousCells)+len(r.EmptyCells), 1) }
func quickReds(r *quickRecord) int { return max(r.Fail, 1) }
func parityReds(r *parityRecord) int {
	return max(r.Blockers, 1)
}

// coversReds says whether entries name at least reds distinct items, and when they do not, how many are unlisted. The records carry counts, not names, so the count
// is what the gate can compare.
func coversReds(entries []acceptedEntry, reds int) (bool, string) {
	items := map[string]bool{}
	for _, a := range entries {
		items[strings.TrimSpace(a.Item)] = true
	}
	if len(items) >= reds {
		return true, ""
	}
	return false, fmt.Sprintf("the record has %d red item(s), %d accepted: %d unlisted (it carries counts, not names)", reds, len(items), reds-len(items))
}
