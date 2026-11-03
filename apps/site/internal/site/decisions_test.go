package site

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The decisions row's figures go through the claims check like a speed: each must be under its record's heading, with
// its date, and must reach its checkpoint's page. Each case corrupts one thing and must be refused.
func TestDecisionClaims_refuse(t *testing.T) {
	in := realInputs(t)
	if len(in.Claims.Decisions) == 0 {
		t.Fatal("no decision claims: the check would be vacuous")
	}
	if err := CheckClaims(repoRoot, in); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, want string
		edit       func(d *DecisionClaim)
	}{
		{"a mistyped top-1", `top1 "0.2775"`, func(d *DecisionClaim) { d.Top1 = "0.2775" }},
		{"a mistyped ECE", `ece "0.2103"`, func(d *DecisionClaim) { d.ECE = "0.2103" }},
		{"a date the record does not carry", `date "2026-09-27"`, func(d *DecisionClaim) { d.Date = "2026-09-27" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := realInputs(t)
			tc.edit(&in.Claims.Decisions[0])
			err := CheckClaims(repoRoot, in)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want an error containing %q", err, tc.want)
			}
		})
	}
	in = realInputs(t)
	in.Claims.Decisions[0].Checkpoint = "ghost"
	if _, err := Derive(in); err == nil || !strings.Contains(err.Error(), "ghost") {
		t.Fatalf("an unknown checkpoint: %v", err)
	}
}

// Verify refuses a checkpoint page that drops its decision figures.
func TestVerify_decisionsRow(t *testing.T) {
	needRepo(t)
	out := t.TempDir()
	m, err := Derive(realInputs(t))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.Render(out, DefaultConfig(), nil); err != nil {
		t.Fatal(err)
	}
	var fam string
	for _, f := range m.Families {
		for _, c := range f.Checkpoints {
			if c.Decision != nil {
				fam = f.Name
			}
		}
	}
	if fam == "" {
		t.Fatal("no checkpoint carries a decision claim")
	}
	p := filepath.Join(out, "models", fam, "index.html")
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(strings.ReplaceAll(string(b), "0.2102", "0.21")), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := Verify(out, m, nil, false); err == nil || !strings.Contains(err.Error(), "decision figures") {
		t.Fatalf("Verify must refuse a page without its decision figures, got %v", err)
	}
}
