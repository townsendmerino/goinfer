package pullcmd

import (
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/pull"
)

// R15 (docs/measurements/cold-user-2026-09-07-macbook-arm64.md): `pull` and `--model` are meant
// to take the same reference forms — a ref copied from one flag to the other should just work.
// Before this fix, resolveRunRef (then inlined in Run) called pull.ParseRef directly, which has
// no idea what "hf:" means; --model's path (pull.Resolve, pull/resolve.go:59) strips it first.
// So every hf:-prefixed form --model accepts (pull.IsRef reports true for it) was refused by
// `pull` with a validRepo error naming "hf" as the owner.
//
// Mutation: delete the `refArg = strings.TrimPrefix(refArg, "hf:")` line in resolveRunRef — the
// hf:-prefixed cases below go red with exactly that "hf" owner error, while the bare and demo:
// cases (which never carried the prefix) stay green, proving the assertion isolates the fix.
func TestResolveRunRef_everyFormModelAcceptsPullAcceptsToo(t *testing.T) {
	cases := []struct {
		name    string
		ref     string
		wantErr bool
		want    pull.Ref
	}{
		{name: "bare owner/repo", ref: "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
			want: pull.Ref{Repo: "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"}},
		{name: "bare owner/repo:quant", ref: "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:q4_k_m",
			want: pull.Ref{Repo: "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", Quant: "q4_k_m"}},
		{name: "hf: owner/repo", ref: "hf:Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
			want: pull.Ref{Repo: "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"}},
		{name: "hf: owner/repo:quant", ref: "hf:Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:q4_k_m",
			want: pull.Ref{Repo: "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", Quant: "q4_k_m"}},
		{name: "hf: owner/repo:file.gguf", ref: "hf:Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:model.q4_k_m.gguf",
			want: pull.Ref{Repo: "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", File: "model.q4_k_m.gguf"}},
		{name: "demo: tier", ref: "demo:0.5b"}, // Ref is repo-specific (from curated.json); only checking it parses
		{name: "hf: with malformed repo still refused", ref: "hf:notarepo", wantErr: true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, _, _, err := resolveRunRef(c.ref)
			if c.wantErr {
				if err == nil {
					t.Fatalf("resolveRunRef(%q): want error, got %+v", c.ref, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("resolveRunRef(%q): %v", c.ref, err)
			}
			if c.want != (pull.Ref{}) && got != c.want {
				t.Errorf("resolveRunRef(%q) = %+v, want %+v", c.ref, got, c.want)
			}
		})
	}
}

// Every form pull.IsRef recognizes as an hf: reference must survive resolveRunRef with the
// prefix gone — this is the specific assertion the mutation above breaks.
func TestResolveRunRef_stripsTheHfPrefixBeforeParsing(t *testing.T) {
	ref := "hf:owner/repo:q4_k_m"
	if !pull.IsRef(ref) {
		t.Fatalf("test fixture %q is not recognized by pull.IsRef — table is stale", ref)
	}
	got, resolved, _, err := resolveRunRef(ref)
	if err != nil {
		t.Fatalf("resolveRunRef(%q): %v", ref, err)
	}
	if strings.Contains(resolved, "hf:") {
		t.Errorf("resolved ref %q still carries the hf: prefix", resolved)
	}
	if got.Repo != "owner/repo" || got.Quant != "q4_k_m" {
		t.Errorf("resolveRunRef(%q) = %+v, want Repo=owner/repo Quant=q4_k_m", ref, got)
	}
}

// A registry short name still resolves through the same function, and the returned ref is the
// rewritten repo:file form, not the name itself.
func TestResolveRunRef_registryShortNameRewrites(t *testing.T) {
	names := pull.RecommendedNames()
	if len(names) == 0 {
		t.Skip("no recommended names registered")
	}
	got, resolved, note, err := resolveRunRef(names[0])
	if err != nil {
		t.Fatalf("resolveRunRef(%q): %v", names[0], err)
	}
	if note == "" {
		t.Errorf("resolveRunRef(%q): expected a rewrite note, got none", names[0])
	}
	if resolved == names[0] {
		t.Errorf("resolveRunRef(%q): resolvedArg was not rewritten", names[0])
	}
	if got.Repo == "" {
		t.Errorf("resolveRunRef(%q): empty repo in resolved ref", names[0])
	}
}
