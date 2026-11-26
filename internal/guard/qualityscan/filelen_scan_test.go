//go:build quality_scan

package qualityscan

import (
	"strings"
	"testing"
)

// fileLenCeiling is the longest a non-test Go file may be without an allowlist line.
const fileLenCeiling = 1500

// A file over fileLenCeiling lines holds several concerns, and every change to one of them diffs and conflicts in the
// others. The ones that exist are held at their exact length (testdata/filelen.allow): a new one or a longer one fails, and
// so does one that has shrunk until its line is lowered or removed, so the list always reads as the tree does.
func TestQualityScan_fileLengthRatchet(t *testing.T) {
	tr := loadTree(t)
	allow, err := ReadAllowlist("testdata/filelen.allow")
	if err != nil {
		t.Fatal(err)
	}
	grown, stale := CompareExact(FileLengths(tr.Files, fileLenCeiling), allow)
	if len(grown) > 0 {
		t.Errorf("%d file(s) over %d lines not in testdata/filelen.allow or over their ceiling (split the file, or allow it with its reason):\n\t%s",
			len(grown), fileLenCeiling, strings.Join(grown, "\n\t"))
	}
	if len(stale) > 0 {
		t.Errorf("%d line(s) of testdata/filelen.allow no longer match the tree (lower the ceiling to the file's length, or remove the line):\n\t%s",
			len(stale), strings.Join(stale, "\n\t"))
	}
}
