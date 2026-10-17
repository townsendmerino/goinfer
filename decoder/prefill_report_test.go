package decoder

import (
	"context"
	"strings"
	"testing"
)

// TestPrefillReport_keepsTheFirstNote: a ctx with no report takes a note as a no-op, and a report keeps the first fallback.
func TestPrefillReport_keepsTheFirstNote(t *testing.T) {
	NotePrefillFallback(context.Background(), "sequential", "no report here")
	ctx, rep := WithPrefillReport(context.Background())
	if p, r := rep.Fallback(); p != "" || r != "" {
		t.Fatalf("a fresh report holds %q, %q", p, r)
	}
	NotePrefillFallback(ctx, "layer-major", "first")
	NotePrefillFallback(ctx, "sequential", "second")
	if p, r := rep.Fallback(); p != "layer-major" || r != "first" {
		t.Errorf("the report holds %q, %q; want the first note", p, r)
	}
	if p, _ := (*PrefillReport)(nil).Fallback(); p != "" {
		t.Error("a nil report reported a fallback")
	}
}

// TestPrefillReport_declinedPassIsNoted: when the resident's batched pass declines, residentPrefillSeed runs the
// sequential loop and the request's report names that path and the resident's reason; a pass that does not decline
// leaves the report empty.
func TestPrefillReport_declinedPassIsNoted(t *testing.T) {
	for _, declineAt := range []int{0, 1} {
		m, _ := loadWithBlockFake(t, func(f *blockFake) { f.declineAt = declineAt })
		ctx, rep := WithPrefillReport(context.Background())
		if _, err := m.residentPrefillSeed(ctx, blockPrompt(m, 20), 0, false); err != nil {
			t.Fatal(err)
		}
		path, reason := rep.Fallback()
		switch {
		case declineAt == 0 && path != "":
			t.Errorf("no decline, but the report holds %q (%q)", path, reason)
		case declineAt == 1 && (path != "sequential" || !strings.Contains(reason, "this pass declines")):
			t.Errorf("a declined pass: the report holds %q, %q; want sequential and the resident's reason", path, reason)
		}
	}
}
