package decidecmd

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func TestEachRow(t *testing.T) {
	in := `{"id":1,"kind":"noul","state":"s","question":"q","options":["false","true"],"target":[0,1],"extra":"ignored"}

{"id":"x","kind":"choice","state":"s","question":"q","options":["a","b"]}
`
	var rows []Row
	if err := eachRow(strings.NewReader(in), func(r Row) error { rows = append(rows, r); return nil }); err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 || rows[0].Kind != "noul" || len(rows[0].Target) != 2 || string(rows[1].ID) != `"x"` {
		t.Errorf("rows %+v", rows)
	}
	if err := eachRow(strings.NewReader("{\"kind\":\"noul\"}\nnot json\n"), func(Row) error { return nil }); err == nil ||
		!strings.Contains(err.Error(), "line 2") {
		t.Errorf("a bad line: %v, want an error naming line 2", err)
	}
}

// An output line always states whether it is calibrated and which route produced it, even when false/empty — the
// task's §2: every number is labelled with what it is.
func TestOut_alwaysSaysCalibratedAndRoute(t *testing.T) {
	b, _ := json.Marshal(Out{Kind: "noul", Route: "label"})
	if s := string(b); !strings.Contains(s, `"calibrated":false`) || !strings.Contains(s, `"route":"label"`) {
		t.Errorf("out line %s", s)
	}
}

// TestHeadFlag_refusesAConflictingLoRA: an unmerged head brings its own adapter, so a different --lora is refused before
// anything loads, rather than one of the two silently winning.
func TestHeadFlag_refusesAConflictingLoRA(t *testing.T) {
	c := newCommon("decide", decideUsage)
	if err := c.fs.Parse([]string{"--model", "unused", "--head", "../../testdata/decisions/judge-tiny", "--lora", "/some/other/adapter"}); err != nil {
		t.Fatal(err)
	}
	_, _, err := c.open(context.Background(), nil)
	if err == nil || !strings.Contains(err.Error(), "brings its adapter") {
		t.Fatalf("got %v", err)
	}
}
