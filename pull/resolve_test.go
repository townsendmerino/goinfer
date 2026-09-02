package pull

import (
	"context"
	"strings"
	"testing"
)

// TestResolve_pathsAreUntouched is the compatibility property that lets a reference form be
// added to --model at all: anything that is not an hf:/demo: reference comes back byte-identical
// and never touches the network. If this breaks, every existing --model silently changes meaning.
func TestResolve_pathsAreUntouched(t *testing.T) {
	for _, spec := range []string{
		"/models/qwen.gguf",
		"./relative/model.giw",
		"~/models/dir",
		"C:\\models\\m.gguf",
		"model-with-hf-in-name.gguf",
		"demonstration.gguf", // starts with "demo" but is NOT "demo:"
		"",
	} {
		if IsRef(spec) {
			t.Errorf("IsRef(%q) = true, want false", spec)
		}
		got, err := Resolve(context.Background(), spec, nil)
		if err != nil {
			t.Errorf("Resolve(%q) errored: %v", spec, err)
		}
		if got != spec {
			t.Errorf("Resolve(%q) = %q, want it returned unchanged", spec, got)
		}
	}
}

func TestIsRef_recognisesReferences(t *testing.T) {
	for _, spec := range []string{
		"hf:Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:q4_k_m",
		"hf:owner/repo",
		"demo:0.5b",
		"demo:1.5b",
	} {
		if !IsRef(spec) {
			t.Errorf("IsRef(%q) = false, want true", spec)
		}
	}
}

// TestResolve_badRefsFailBeforeTheNetwork: a malformed reference must be rejected by parsing,
// not by a request. Keeps the traversal allow-list on the path a --model flag now reaches.
func TestResolve_badRefsFailBeforeTheNetwork(t *testing.T) {
	for _, spec := range []string{
		"hf:../..",
		"hf:a/../../etc",
		"hf:notarepo",
		"demo:nope",
	} {
		if _, err := Resolve(context.Background(), spec, nil); err == nil {
			t.Errorf("Resolve(%q) should have failed", spec)
		}
	}
	// A well-formed repo with no quant or file names the fix rather than guessing one.
	_, err := Resolve(context.Background(), "hf:owner/repo", nil)
	if err == nil || !strings.Contains(err.Error(), "q4_k_m") {
		t.Errorf("a ref with no selector should suggest one, got: %v", err)
	}
}
