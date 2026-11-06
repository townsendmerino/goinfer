package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

const acceptedTestHead = "ccdfb3ad68b8ed9120f5712cc0d91e49cfb59d9f"

// acceptedEnv is a release env that answers from canned records at one commit (no code changed since any of them), and from acceptedText for the
// accepted file ("" is no file).
func acceptedEnv(t *testing.T, gpu []gpuRecord, quick []quickRecord, parity []parityRecord, acceptedText string) *relEnv {
	t.Helper()
	marshal := func(v any) string {
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	e := &relEnv{tag: "v0.25.0", prev: "v0.24.0", offline: true}
	e.look = func(kind string, args ...string) (string, error) {
		switch kind {
		case "records":
			switch args[0] {
			case "gpu":
				return marshal(gpu), nil
			case "quick":
				return marshal(quick), nil
			case "parity":
				return marshal(parity), nil
			}
		case "parity-paths":
			return "decoder", nil
		case "accepted":
			return acceptedText, nil
		case "git":
			switch args[0] {
			case "rev-parse":
				if args[1] == "HEAD" {
					return acceptedTestHead, nil
				}
				return "tree-the-same-everywhere", nil
			case "ls-tree":
				return "100644 blob object-the-same-everywhere\t" + args[len(args)-1], nil
			case "diff":
				return "", nil
			}
		}
		return "", fmt.Errorf("unexpected read %s %v", kind, args)
	}
	return e
}

func failingRecords() (gpu []gpuRecord, quick []quickRecord, parity []parityRecord) {
	gpu = []gpuRecord{
		{Schema: 1, Backend: "cuda", Commit: acceptedTestHead, Date: "2026-11-05T02:22:50Z", Verdict: "FAIL", Groups: gpuRecordGroups{Fail: 2}},
		{Schema: 1, Backend: "metal", Commit: acceptedTestHead, Date: "2026-11-04T20:00:00Z", Verdict: "PASS", Trees: map[string]string{}},
	}
	quick = []quickRecord{
		{Schema: 1, OS: "linux", Arch: "amd64", Commit: acceptedTestHead, Date: "2026-11-05T03:20:44Z", Verdict: "FAIL", Full: true, Pass: 3391, Fail: 4},
		{Schema: 1, OS: "darwin", Arch: "arm64", Commit: acceptedTestHead, Date: "2026-11-04T20:00:00Z", Verdict: "PASS", Full: true, Pass: 100},
	}
	parity = []parityRecord{{Schema: 1, Commit: acceptedTestHead, Date: "2026-11-05T07:10:07Z", Verdict: "BLOCKERS", Realckpt: true, Blockers: 7}}
	return
}

func acceptedJSON(release string, entries ...acceptedEntry) string {
	b, _ := json.Marshal(acceptedFile{Schema: 1, Release: release, Accepted: entries})
	return string(b)
}

func goodEntry(record, item string) acceptedEntry {
	return acceptedEntry{Record: record, Item: item, Class: "test-bug", Evidence: "docs/measurements/x/README.md", AcceptedBy: "Francis", AcceptedOn: "2026-11-05"}
}

// entriesFor is n distinct good entries for a record, as a record with n red items needs.
func entriesFor(record string, n int) []acceptedEntry {
	var out []acceptedEntry
	for i := 1; i <= n; i++ {
		out = append(out, goodEntry(record, fmt.Sprintf("red item %d", i)))
	}
	return out
}

func TestRecordFileName(t *testing.T) {
	for _, tc := range []struct{ prefix, date, commit, want string }{
		{"cuda", "2026-11-05T02:22:50Z", acceptedTestHead, "cuda-20261105T022250Z-ccdfb3ad.json"},
		{"linux-amd64", "2026-11-05T03:20:44Z", acceptedTestHead, "linux-amd64-20261105T032044Z-ccdfb3ad.json"},
		{"parity", "2026-11-05T07:10:07Z", acceptedTestHead, "parity-20261105T071007Z-ccdfb3ad.json"},
	} {
		if got := recordFileName(tc.prefix, tc.date, tc.commit); got != tc.want {
			t.Errorf("recordFileName(%q, %q) = %q, want %q", tc.prefix, tc.date, got, tc.want)
		}
	}
}

// A failing record with accepted entries reads "accepted", prints the entries, and does not count against the tag. Nothing else does: another
// record's entry, an entry another name wrote, a class that is not one, a file for another release, a missing field.
func TestRowsAccepted(t *testing.T) {
	gpu, quick, parity := failingRecords()
	cuda := recordFileName("cuda", gpu[0].Date, gpu[0].Commit)
	linux := recordFileName("linux-amd64", quick[0].Date, quick[0].Commit)
	par := recordFileName("parity", parity[0].Date, parity[0].Commit)
	rows := map[string]func(*relEnv) relRow{"gpu": rowGPU, "quick": rowQuickFull, "parity": rowParity}
	record := map[string]string{"gpu": cuda, "quick": linux, "parity": par}
	reds := map[string]int{"gpu": 2, "quick": 4, "parity": 7} // the failing records' own counts

	for name, row := range rows {
		// no accepted file: OWED, as before
		e := acceptedEnv(t, gpu, quick, parity, "")
		if r := row(e); r.State != "OWED" {
			t.Errorf("%s with no accepted file: %+v", name, r)
		}
		// the entry for this record: accepted, with the entry printed
		e = acceptedEnv(t, gpu, quick, parity, acceptedJSON("v0.25.0", entriesFor(record[name], reds[name])...))
		r := row(e)
		d := strings.Join(r.Detail, "\n")
		if r.State != "accepted" || !strings.Contains(d, "red item 1") || !strings.Contains(d, "test-bug") || !strings.Contains(d, "Francis") {
			t.Errorf("%s with its entry: state %q\n%s", name, r.State, d)
		}
		for why, text := range map[string]string{
			"an entry for another record": acceptedJSON("v0.25.0", goodEntry("cuda-20260101T000000Z-deadbeef.json", "x")),
			"accepted_by someone else":    acceptedJSON("v0.25.0", func() acceptedEntry { a := goodEntry(record[name], "x"); a.AcceptedBy = "Claude"; return a }()),
			"a class that is not one":     acceptedJSON("v0.25.0", func() acceptedEntry { a := goodEntry(record[name], "x"); a.Class = "product"; return a }()),
			"no evidence":                 acceptedJSON("v0.25.0", func() acceptedEntry { a := goodEntry(record[name], "x"); a.Evidence = ""; return a }()),
			"no date":                     acceptedJSON("v0.25.0", func() acceptedEntry { a := goodEntry(record[name], "x"); a.AcceptedOn = "yesterday"; return a }()),
			"a file for another release":  acceptedJSON("v0.24.0", goodEntry(record[name], "x")),
			"not JSON":                    "{",
		} {
			if r := row(acceptedEnv(t, gpu, quick, parity, text)); r.State != "OWED" {
				t.Errorf("%s with %s: state %q, want OWED\n%s", name, why, r.State, strings.Join(r.Detail, "\n"))
			}
		}
	}
	// a rejected entry says why, so the reader can fix the file
	e := acceptedEnv(t, gpu, quick, parity, acceptedJSON("v0.25.0", func() acceptedEntry { a := goodEntry(cuda, "x"); a.AcceptedBy = "Claude"; return a }()))
	if d := strings.Join(rowGPU(e).Detail, "\n"); !strings.Contains(d, `accepted_by "Claude"`) {
		t.Errorf("a rejected entry does not say why:\n%s", d)
	}
}

// The report counts an accepted row apart, prints it, and exits 0 when nothing else is owed.
func TestReportRelease_acceptedDoesNotBlock(t *testing.T) {
	gpu, quick, parity := failingRecords()
	e := acceptedEnv(t, gpu, quick, parity, acceptedJSON("v0.25.0", entriesFor(recordFileName("cuda", gpu[0].Date, gpu[0].Commit), 2)...))
	var out bytes.Buffer
	if code := reportRelease(&out, e, []func(*relEnv) relRow{rowGPU}); code != 0 {
		t.Errorf("exit %d with only an accepted row\n%s", code, out.String())
	}
	if s := out.String(); !strings.Contains(s, "accepted") || !strings.Contains(s, "0 of 1 done, 1 accepted") || strings.Contains(s, "not ready to tag") {
		t.Errorf("the report:\n%s", s)
	}
	// with a second row still owed the tag is not ready, and the accepted row is still shown as accepted
	out.Reset()
	e = acceptedEnv(t, gpu, quick, parity, acceptedJSON("v0.25.0", entriesFor(recordFileName("cuda", gpu[0].Date, gpu[0].Commit), 2)...))
	if code := reportRelease(&out, e, []func(*relEnv) relRow{rowGPU, rowParity}); code != 1 || !strings.Contains(out.String(), "not ready to tag") {
		t.Errorf("exit %d with a row still owed\n%s", code, out.String())
	}
}

// One entry does not accept a record with two reds: the new red would hide behind the old one. Each red item needs its own entry (the record's count of failed groups,
// failed tests or blockers), and the row names how many are unlisted. Two entries for the same item are one.
func TestRowsAccepted_oneEntryDoesNotCoverTwoReds(t *testing.T) {
	gpu, quick, parity := failingRecords()
	file := map[string]string{
		"gpu": recordFileName("cuda", gpu[0].Date, gpu[0].Commit), "quick": recordFileName("linux-amd64", quick[0].Date, quick[0].Commit),
		"parity": recordFileName("parity", parity[0].Date, parity[0].Commit)}
	rows := map[string]func(*relEnv) relRow{"gpu": rowGPU, "quick": rowQuickFull, "parity": rowParity}
	reds := map[string]int{"gpu": 2, "quick": 4, "parity": 7}
	for name, row := range rows {
		one := acceptedEnv(t, gpu, quick, parity, acceptedJSON("v0.25.0", goodEntry(file[name], "the red already read")))
		r := row(one)
		d := strings.Join(r.Detail, "\n")
		want := fmt.Sprintf("%d red item(s), %d accepted: %d unlisted", reds[name], 1, reds[name]-1)
		if r.State != "OWED" || !strings.Contains(d, want) {
			t.Errorf("%s: one entry for a record with %d reds: state %q, want OWED naming %q\n%s", name, reds[name], r.State, want, d)
		}
		// one fewer than the record's reds is still short, and the same item twice counts once
		short := acceptedJSON("v0.25.0", entriesFor(file[name], reds[name]-1)...)
		if r := row(acceptedEnv(t, gpu, quick, parity, short)); r.State != "OWED" {
			t.Errorf("%s: %d entries for %d reds read %q", name, reds[name]-1, reds[name], r.State)
		}
		dup := acceptedJSON("v0.25.0", append(entriesFor(file[name], reds[name]-1), goodEntry(file[name], "red item 1"))...)
		if r := row(acceptedEnv(t, gpu, quick, parity, dup)); r.State != "OWED" {
			t.Errorf("%s: a repeated item counted twice, state %q", name, r.State)
		}
		// as many distinct entries as reds, or more
		more := acceptedJSON("v0.25.0", entriesFor(file[name], reds[name]+1)...)
		if r := row(acceptedEnv(t, gpu, quick, parity, more)); r.State != "accepted" {
			t.Errorf("%s: %d entries for %d reds read %q, want accepted", name, reds[name]+1, reds[name], r.State)
		}
	}
	// a non-pass record with no count of reds (a dirty tree, a narrowed run) is one red item
	g := []gpuRecord{{Schema: 1, Backend: "cuda", Commit: acceptedTestHead, Date: "2026-11-05T02:22:50Z", Verdict: "INCONCLUSIVE", Dirty: true},
		{Schema: 1, Backend: "metal", Commit: acceptedTestHead, Date: "2026-11-04T20:00:00Z", Verdict: "PASS", Trees: map[string]string{}}}
	one := acceptedJSON("v0.25.0", goodEntry(recordFileName("cuda", g[0].Date, g[0].Commit), "the dirty-tree run"))
	if r := rowGPU(acceptedEnv(t, g, quick, parity, one)); r.State != "accepted" {
		t.Errorf("a record that is red for no counted failure needs one entry; state %q\n%s", r.State, strings.Join(r.Detail, "\n"))
	}
}
