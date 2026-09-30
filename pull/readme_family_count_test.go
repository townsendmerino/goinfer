package pull

import (
	"encoding/json"
	"os"
	"regexp"
	"strconv"
	"testing"
)

// TestReadme_familyCountMatchesTheMatrix: the README's "N model families" is a claim the generated
// capability matrix can check, and it did drift — the README said 35 while docs/capability-matrix.json
// (and the site built from it) said 37. The matrix is the source of truth; every README occurrence must
// equal its length.
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
