package main

import (
	"encoding/json"
	"strings"
	"testing"
)

func recFamily(verdict string, quants ...string) identityFamily {
	f := identityFamily{Name: "gemma4", Verdict: verdict}
	for _, q := range quants {
		f.Cells = append(f.Cells, cellOutcome{Cell: &identityCell{Asset: "e2b-q4_0", Quant: q}, Verdict: vIdentical})
	}
	f.Ran = len(f.Cells)
	return f
}

func recFixture() (identityOpts, *identitySide, *identitySide, manifestFamily) {
	o := identityOpts{Assets: "real", Backend: "cpu"}
	old := &identitySide{SHA: "310a2e44aaaa", Short: "310a2e44"}
	nw := &identitySide{SHA: "b327153fbbbb", Short: "b327153f"}
	mf := manifestFamily{Status: "validated", Method: "full-forward-oracle", ValidatedAt: "310a2e44", Machine: "linux-amd64",
		Reference: "HF bf16 (google/gemma-4 E2B + 12B)", Metrics: json.RawMessage(`{"argmax_pct":100,"cosine_min":0.98972}`)}
	return o, old, nw, mf
}

func TestIdentityInheritance_eligible(t *testing.T) {
	o, old, nw, mf := recFixture()
	in := identityInheritance(o, old, nw, "b327153fbbbb", "amd64", "nobara", mf, recFamily(vIdentical, "f32", "int8int8", "int4"))
	if in.Row == "" {
		t.Fatalf("should be eligible, got reasons %v", in.Reasons)
	}
	var row struct {
		Family, Method, Reference string
		Metrics                   json.RawMessage
	}
	if err := json.Unmarshal([]byte(strings.TrimPrefix(in.Row, "PARITY_ROW ")), &row); err != nil {
		t.Fatal(err)
	}
	if row.Method != "identity-inherited (full-forward-oracle @ 310a2e44)" {
		t.Fatalf("method %q", row.Method)
	}
	if string(row.Metrics) != `{"argmax_pct":100,"cosine_min":0.98972}` {
		t.Fatalf("the original metrics must be carried, got %s", row.Metrics)
	}
	for _, want := range []string{"e2b-q4_0", "310a2e44", "b327153f", "Original validation: HF bf16"} {
		if !strings.Contains(row.Reference, want) {
			t.Fatalf("reference %q lacks %q", row.Reference, want)
		}
	}
}

// A chain keeps the ORIGINAL method and the rev its oracle ran at; old-rev must be the row's current validated_at.
func TestIdentityInheritance_chainKeepsTheOriginalOracle(t *testing.T) {
	o, _, nw, mf := recFixture()
	mf.Method, mf.ValidatedAt = "identity-inherited (full-forward-oracle @ 310a2e44)", "8a472437"
	old := &identitySide{SHA: "8a472437cccc", Short: "8a472437"}
	in := identityInheritance(o, old, nw, "b327153fbbbb", "amd64", "nobara", mf, recFamily(vIdentical, "f32", "int8int8", "int4"))
	if !strings.Contains(in.Row, `identity-inherited (full-forward-oracle @ 310a2e44)`) {
		t.Fatalf("a chain must keep the original oracle rev: %s %v", in.Row, in.Reasons)
	}
}

func TestIdentityInheritance_refusals(t *testing.T) {
	cases := map[string]func(o *identityOpts, mf *manifestFamily, f *identityFamily, arch *string, head *string){
		"not IDENTICAL": func(o *identityOpts, mf *manifestFamily, f *identityFamily, arch, head *string) {
			f.Verdict = vDifferent
		},
		"not validated": func(o *identityOpts, mf *manifestFamily, f *identityFamily, arch, head *string) {
			mf.Status = "experimental"
		},
		"real-model-oracle": func(o *identityOpts, mf *manifestFamily, f *identityFamily, arch, head *string) {
			mf.Method = "real-model-oracle"
		},
		"tiny assets":   func(o *identityOpts, mf *manifestFamily, f *identityFamily, arch, head *string) { o.Assets = "tiny" },
		"metal backend": func(o *identityOpts, mf *manifestFamily, f *identityFamily, arch, head *string) { o.Backend = "metal" },
		"other arch":    func(o *identityOpts, mf *manifestFamily, f *identityFamily, arch, head *string) { *arch = "arm64" },
		"unknown machine": func(o *identityOpts, mf *manifestFamily, f *identityFamily, arch, head *string) {
			mf.Machine = "somebox"
		},
		"old is not validated": func(o *identityOpts, mf *manifestFamily, f *identityFamily, arch, head *string) {
			mf.ValidatedAt = "0b74e671"
		},
		"new is not HEAD": func(o *identityOpts, mf *manifestFamily, f *identityFamily, arch, head *string) {
			*head = "deadbeef0000"
		},
		"a quant missing": func(o *identityOpts, mf *manifestFamily, f *identityFamily, arch, head *string) {
			f.Cells = f.Cells[:2]
		},
		"a cell not identical": func(o *identityOpts, mf *manifestFamily, f *identityFamily, arch, head *string) {
			f.Cells[1].Verdict = vDifferent
		},
	}
	for name, mutate := range cases {
		o, old, nw, mf := recFixture()
		f := recFamily(vIdentical, "f32", "int8int8", "int4")
		arch, head := "amd64", "b327153fbbbb"
		mutate(&o, &mf, &f, &arch, &head)
		in := identityInheritance(o, old, nw, head, arch, "nobara", mf, f)
		if in.Row != "" || len(in.Reasons) == 0 {
			t.Errorf("%s: must be refused with a reason, got row %q", name, in.Row)
		}
	}
}

func TestRowArch(t *testing.T) {
	for m, want := range map[string]string{"linux-amd64": "amd64", "linux-62gb": "amd64", "mac": "arm64",
		"darwin-arm64": "arm64", "macbook-arm64": "arm64", "somebox": ""} {
		if got := rowArch(m); got != want {
			t.Errorf("rowArch(%q) = %q, want %q", m, got, want)
		}
	}
}
