package pull

import (
	"encoding/json"
	"os"
	"regexp"
	"strconv"
	"testing"
)

// TestReadme_familyCountMatchesTheMatrix: the README's "N model families" is a claim the generated
// capability matrix can check. The matrix (docs/capability-matrix.json, and the site built from it) is the
// source of truth; every README occurrence must equal its length.
//
// Mutation: change either "37 model families" in README.md and this goes red naming the number.
func TestReadme_familyCountMatchesTheMatrix(t *testing.T) {
	raw, err := os.ReadFile("../docs/capability-matrix.json")
	if err != nil {
		t.Fatalf("read matrix: %v", err)
	}
	var rows []struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal(raw, &rows); err != nil || len(rows) == 0 {
		t.Fatalf("parse matrix: %v (%d rows)", err, len(rows))
	}
	readme, err := os.ReadFile("../README.md")
	if err != nil {
		t.Fatalf("read README.md: %v", err)
	}
	ms := regexp.MustCompile(`(\d+) model families`).FindAllSubmatch(readme, -1)
	if len(ms) == 0 {
		t.Fatal(`README.md no longer says "N model families" — if that was deliberate, delete this test; ` +
			"if the count moved somewhere else, point this regexp at it")
	}
	for _, m := range ms {
		if n, _ := strconv.Atoi(string(m[1])); n != len(rows) {
			t.Errorf("README says %d model families; docs/capability-matrix.json has %d", n, len(rows))
		}
	}
}

// TestBenchmarks_architectureCountMatchesTheMatrix: docs/benchmarks.md's "N architectures ʲ" is the matrix's row
// count, as its footnote ʲ says, so every occurrence must equal the matrix's length.
func TestBenchmarks_architectureCountMatchesTheMatrix(t *testing.T) {
	raw, err := os.ReadFile("../docs/capability-matrix.json")
	if err != nil {
		t.Fatalf("read matrix: %v", err)
	}
	var rows []struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal(raw, &rows); err != nil || len(rows) == 0 {
		t.Fatalf("parse matrix: %v (%d rows)", err, len(rows))
	}
	doc, err := os.ReadFile("../docs/benchmarks.md")
	if err != nil {
		t.Fatalf("read docs/benchmarks.md: %v", err)
	}
	ms := regexp.MustCompile(`(\d+) architectures\*{0,2} ʲ|ʲ \*\*(\d+) architectures`).FindAllSubmatch(doc, -1)
	if len(ms) < 3 {
		t.Fatalf(`docs/benchmarks.md says "N architectures ʲ" %d time(s), want its summary line, its coverage row and its footnote`, len(ms))
	}
	for _, m := range ms {
		got := string(m[1]) + string(m[2])
		if n, _ := strconv.Atoi(got); n != len(rows) {
			t.Errorf("docs/benchmarks.md says %s architectures (%q); docs/capability-matrix.json has %d", got, m[0], len(rows))
		}
	}
}
