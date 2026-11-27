package gateidentity

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/testpath"
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
	in := identityInheritance(o, old, nw, "b327153fbbbb", "amd64", "nobara", "2026-10-27", mf, recFamily(vIdentical, "f32", "int8int8", "int4"))
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
	in := identityInheritance(o, old, nw, "b327153fbbbb", "amd64", "nobara", "2026-10-27", mf, recFamily(vIdentical, "f32", "int8int8", "int4"))
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
		in := identityInheritance(o, old, nw, head, arch, "nobara", "2026-10-27", mf, f)
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

func mustRead(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// TestIdentityReplay_plants is SB12-C4 for identity, on a copy of the recorded tiny llama run (all IDENTICAL): one
// byte changed in the old side's f32 dump makes llama DIFFERENT (exit 1); one byte changed in the new build's second
// run makes the CPU nondeterministic (exit 1); a dump the run had, lost or cut short since, is refused (exit 2), never
// graded NOT RUN or IDENTICAL; and a record whose run did not keep its dumps is refused (exit 2).
func TestIdentityReplay_plants(t *testing.T) {
	src := filepath.Join(testpath.Testdata(t, "gate", "goldens"), "identity-llama-tiny")
	copyTree := func(t *testing.T) string {
		dst := t.TempDir()
		err := filepath.WalkDir(src, func(p string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			rel, _ := filepath.Rel(src, p)
			if d.IsDir() {
				return os.MkdirAll(filepath.Join(dst, rel), 0o755)
			}
			b, err := os.ReadFile(p)
			if err != nil {
				return err
			}
			return os.WriteFile(filepath.Join(dst, rel), b, 0o644)
		})
		if err != nil {
			t.Fatal(err)
		}
		return dst
	}
	flip := func(t *testing.T, p string) {
		b, err := os.ReadFile(p)
		if err != nil || len(b) < 16 {
			t.Fatalf("%s: %v (%d bytes)", p, err, len(b))
		}
		b[len(b)/2] ^= 0x01
		if err := os.WriteFile(p, b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	cases := []struct {
		name  string
		plant func(t *testing.T, dir string)
		exit  int
		says  string
	}{
		{"none", func(*testing.T, string) {}, 0, "GREEN"},
		{"the old side's logits differ", func(t *testing.T, dir string) { flip(t, filepath.Join(dir, "dumps/old/llama/llama__tiny__f32.bin")) }, 1, "DIFFERENT"},
		{"the new build's second run differs", func(t *testing.T, dir string) { flip(t, filepath.Join(dir, "dumps/new2/llama/llama__tiny__f32.bin")) }, 1, "NONDETERMINISTIC"},
		{"an old dump's logits were lost", func(t *testing.T, dir string) {
			if err := os.Remove(filepath.Join(dir, "dumps/old/llama/llama__tiny__f32.bin")); err != nil {
				t.Fatal(err)
			}
		}, 2, ""},
		{"a new dump's logits were cut short", func(t *testing.T, dir string) {
			p := filepath.Join(dir, "dumps/new1/llama/llama__tiny__int4.bin")
			b := mustRead(t, p)
			if err := os.WriteFile(p, b[:len(b)-4], 0o644); err != nil {
				t.Fatal(err)
			}
		}, 2, ""},
		{"a dump's result was lost", func(t *testing.T, dir string) {
			if err := os.Remove(filepath.Join(dir, "dumps/new2/llama/llama__tiny__int8int8.json")); err != nil {
				t.Fatal(err)
			}
		}, 2, ""},
		{"the dumps were not kept", func(t *testing.T, dir string) {
			p := filepath.Join(dir, identityRunRecordFile)
			var rec identityRunRecord
			if err := json.Unmarshal(mustRead(t, p), &rec); err != nil {
				t.Fatal(err)
			}
			rec.Opts.Keep = false
			b, _ := json.Marshal(rec)
			if err := os.WriteFile(p, b, 0o644); err != nil {
				t.Fatal(err)
			}
		}, 2, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dir := copyTree(t)
			c.plant(t, dir)
			var out strings.Builder
			rc := ReplayIdentity(&out, filepath.Join(dir, identityRunRecordFile))
			if rc != c.exit || !strings.Contains(out.String(), c.says) {
				t.Errorf("exit %d, want %d and %q in the report:\n%s", rc, c.exit, c.says, out.String())
			}
		})
	}
}
