package site

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A book claim must sit in its chapter's "Try it" section and in its record, with the record's date; each case
// corrupts one thing on a synthetic chapter and record and must be refused, and a chapter with a "Try it" section
// but no claim must be refused too.
func TestBookClaims_refuse(t *testing.T) {
	root := t.TempDir()
	write := func(rel, body string) {
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("docs/book/05-x.md", "# Five\n\nText 9.9 tok/s.\n\n## Try it\n\nRun it; on an M1 Pro you should see about 88.0 tok/s.\n\n## Next\n\n77.7\n")
	write("docs/measurements/r.md", "# R\n\n## Cells (2026-10-02)\n\nMeasured 2026-10-02.\n\n| depth | tok/s |\n|---|---|\n| 128 | 88.0 |\n\n## Other\n\n77.7 2026-10-03\n")
	good := BookClaim{ID: "ch5", Chapter: "docs/book/05-x.md", Figure: "88.0", What: "decode", Date: "2026-10-02",
		Source: Source{Path: "docs/measurements/r.md", Heading: "Cells (2026-10-02)"}}
	check := func(b []BookClaim) error {
		return CheckClaims(root, &Inputs{Claims: Claims{Book: b}})
	}
	if err := check([]BookClaim{good}); err != nil {
		t.Fatalf("the good claim was refused: %v", err)
	}
	for _, tc := range []struct {
		name, want string
		edit       func(b *BookClaim)
	}{
		{"a figure the record does not carry", `figure "88.1" is not under`, func(b *BookClaim) { b.Figure = "88.1" }},
		{"a figure outside the chapter's Try it section", `"77.7" is not under`, func(b *BookClaim) { b.Figure = "77.7" }},
		{"a figure only outside Try it in the chapter", `is not in docs/book/05-x.md's "Try it" section`, func(b *BookClaim) {
			b.Figure, b.Source.Heading, b.Date = "77.7", "Other", "2026-10-03"
		}},
		{"a date the record does not carry", `date "2026-10-04"`, func(b *BookClaim) { b.Date = "2026-10-04" }},
		{"a missing heading", `heading "Nope"`, func(b *BookClaim) { b.Source.Heading = "Nope" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := good
			tc.edit(&b)
			err := check([]BookClaim{b})
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error containing %q", err, tc.want)
			}
		})
	}
	if err := check(nil); err == nil || !strings.Contains(err.Error(), "has a \"Try it\" section and no book claim") {
		t.Fatalf("a Try it section with no claim: got %v", err)
	}
}
