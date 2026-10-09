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
	// M-32 (audit-2026-09-10.md): a recommended checkpoint's SHA256/Bytes used to be dropped
	// on the round trip through ParseRef(c.Ref()), which only ever sets Pin for a "demo:" ref
	// — so every OTHER registry entry verified against whatever HF's API reports today instead
	// of the digest this build vouches for.
	c, ok := pull.Recommended(names[0])
	if !ok {
		t.Fatalf("pull.Recommended(%q): not found (was in RecommendedNames)", names[0])
	}
	if c.SHA256 != "" && got.Pin != c.SHA256 {
		t.Errorf("resolveRunRef(%q).Pin = %q, want the registry's SHA256 %q", names[0], got.Pin, c.SHA256)
	}
	if c.Bytes != 0 && got.Bytes != c.Bytes {
		t.Errorf("resolveRunRef(%q).Bytes = %d, want the registry's Bytes %d", names[0], got.Bytes, c.Bytes)
	}
}

// R22: the listing marks a projector among the quants (listingLine is what `pull <repo>` prints per file).
func TestListingLine_marksVisionProjector(t *testing.T) {
	q := listingLine(pull.Listed{Path: "gemma-3-4b-it-Q4_K_M.gguf", Size: 2 << 30})
	m := listingLine(pull.Listed{Path: "mmproj-google_gemma-3-4b-it-f16.gguf", Size: 800 << 20})
	if strings.Contains(q, "vision projector") || !strings.HasSuffix(q, "\n") {
		t.Errorf("a quant row was marked or lost its newline: %q", q)
	}
	if !strings.Contains(m, "vision projector") || !strings.Contains(m, "cannot load it yet") {
		t.Errorf("the mmproj row is not marked: %q", m)
	}
}

// An embedding encoder's checkpoint opens as serve's embedding model, not as a --model (task-checkpoint-fetch P7).
func TestRunHint_encoderPointsAtEmbedModel(t *testing.T) {
	if h := runHint(pull.Plan{Family: pull.EncoderFamily}, "/c/enc"); !strings.Contains(h, "serve --embed-model /c/enc") || strings.Contains(h, " --model ") {
		t.Errorf("encoder hint: %q", h)
	}
	if h := runHint(pull.Plan{Family: "llama"}, "/c/m"); !strings.Contains(h, "--model /c/m") || strings.Contains(h, "embed-model") {
		t.Errorf("generative hint: %q", h)
	}
}

// A directory entry (P9(d)) resolves to the checkpoint form with its tree digest pinned: the same ref a user could type as
// owner/repo:safetensors, plus the Pin that makes pullCheckpoint refuse a repo that has changed since this build was cut.
func TestResolveRunRef_directoryEntryPinsTheTreeDigest(t *testing.T) {
	for _, name := range []string{"qwen3.5-0.8b", "qwen3-vl-2b"} {
		c, ok := pull.Recommended(name)
		if !ok || !c.IsDirectory() {
			t.Fatalf("%s: not a directory entry in the registry (%+v, %v)", name, c, ok)
		}
		ref, arg, note, err := resolveRunRef(name)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if !ref.Checkpoint || ref.Repo != c.Repo || ref.File != "" || ref.Pin != c.SHA256 || ref.Bytes != c.Bytes {
			t.Errorf("%s: ref %+v does not carry the checkpoint selector and the registry's pin", name, ref)
		}
		if arg != c.Repo+":safetensors" {
			t.Errorf("%s: resolved arg %q, want %s:safetensors", name, arg, c.Repo)
		}
		if !strings.Contains(note, "GB download") {
			t.Errorf("%s: the note %q does not state the download size", name, note)
		}
	}
}
