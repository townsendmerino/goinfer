package qualityscan

import (
	"go/ast"
	"go/parser"
	"go/token"
	"slices"
	"strings"
	"testing"
)

func TestParseAllowlist_needsAReasonAndAPositiveCeiling(t *testing.T) {
	got, err := ParseAllowlist([]byte("# header\n\na/b.go:F -- a reason\na/c.go:G 12 -- ratchet\n"))
	if err != nil {
		t.Fatal(err)
	}
	if got["a/b.go:F"].Reason != "a reason" || got["a/b.go:F"].Ceiling != 0 || got["a/c.go:G"].Ceiling != 12 {
		t.Fatalf("parsed %+v", got)
	}
	for _, bad := range []string{"a/b.go:F\n", "a/b.go:F -- \n", "a/b.go:F 0 -- r\n", "a/b.go:F x -- r\n", "a b c -- r\n", "k -- r\nk -- r\n"} {
		if _, err := ParseAllowlist([]byte(bad)); err == nil {
			t.Errorf("%q parsed without error", bad)
		}
	}
}

func TestCompare_failsOnGrowthAndReportsShrinkage(t *testing.T) {
	allow := map[string]Entry{
		"set":     {Key: "set", Reason: "r"},
		"at":      {Key: "at", Ceiling: 3, Reason: "r"},
		"under":   {Key: "under", Ceiling: 3, Reason: "r"},
		"over":    {Key: "over", Ceiling: 3, Reason: "r"},
		"removed": {Key: "removed", Reason: "r"},
	}
	found := map[string]int{"set": 9, "at": 3, "under": 2, "over": 4, "new": 1}
	grown, shrunk := Compare(found, allow)
	if want := []string{"new (1)", "over (4, ceiling 3)"}; !slices.Equal(grown, want) {
		t.Errorf("grown = %q, want %q", grown, want)
	}
	if want := []string{"removed (gone)", "under (2, ceiling 3)"}; !slices.Equal(shrunk, want) {
		t.Errorf("shrunk = %q, want %q", shrunk, want)
	}
}

func TestFuncKey_namesTheReceiverType(t *testing.T) {
	f, err := parser.ParseFile(token.NewFileSet(), "x.go", "package x\nfunc F(){}\nfunc (r *R) M(){}\nfunc (g G[T]) N(){}\n", 0)
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, d := range f.Decls {
		got = append(got, FuncKey("p/x.go", d.(*ast.FuncDecl)))
	}
	if want := []string{"p/x.go:F", "p/x.go:R.M", "p/x.go:G.N"}; !slices.Equal(got, want) {
		t.Fatalf("got %q, want %q", got, want)
	}
}

// The file-length ratchet's plant: a new long file and a grown one fail, and so does every allowlist line the tree no
// longer matches (shrunk, gone, or without a ceiling); a long test file and a file at its line pass.
func TestFileLengthRatchet_plantsGoRed(t *testing.T) {
	lines := func(n int) []byte { return []byte(strings.Repeat("x\n", n)) }
	files := []*File{
		{Path: "a/new.go", Src: lines(1501)},
		{Path: "a/grown.go", Src: lines(1602)},
		{Path: "a/held.go", Src: lines(1700)},
		{Path: "a/shrunk.go", Src: lines(1550)},
		{Path: "a/under.go", Src: lines(1400)},
		{Path: "a/long_test.go", Src: lines(3000), Test: true},
		{Path: "a/open.go", Src: append(lines(1600), 'x')},
	}
	allow, err := ParseAllowlist([]byte(strings.Join([]string{
		"a/grown.go 1601 -- r", "a/held.go 1700 -- r", "a/shrunk.go 1600 -- r", "a/under.go 1600 -- r",
		"a/gone.go 1600 -- r", "a/open.go -- r",
	}, "\n")))
	if err != nil {
		t.Fatal(err)
	}
	found := FileLengths(files, 1500)
	if found["a/open.go"] != 1601 {
		t.Errorf("a/open.go counted %d lines, want 1601 (a last line without a newline counts)", found["a/open.go"])
	}
	grown, stale := CompareExact(found, allow)
	if want := []string{"a/grown.go (1602, ceiling 1601)", "a/new.go (1501)"}; !slices.Equal(grown, want) {
		t.Errorf("grown = %q, want %q", grown, want)
	}
	if want := []string{"a/gone.go (gone)", "a/open.go (no ceiling)", "a/shrunk.go (1550, ceiling 1600)", "a/under.go (gone)"}; !slices.Equal(stale, want) {
		t.Errorf("stale = %q, want %q", stale, want)
	}
}
