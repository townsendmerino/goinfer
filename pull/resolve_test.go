package pull

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
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

// TestResolveOffline_verifiedCacheHitNeedsNoNetwork pins V-16 (docs/review-2026-09-04.md):
// when a ref's exact file, size AND digest are all known WITHOUT asking HuggingFace anything
// (a demo: ref's shape — File+Pin+Bytes all set from curated.json), an already-cached, matching
// copy resolves offline. resolveOffline only ever touches CacheDir + the filesystem, so this
// needs no network stub to prove the property — the function has nothing in it that COULD reach
// the network.
func TestResolveOffline_verifiedCacheHitNeedsNoNetwork(t *testing.T) {
	t.Setenv("HOME", t.TempDir())  // redirect os.UserCacheDir() away from the real machine cache
	t.Setenv("XDG_CACHE_HOME", "") // don't let a real one from the environment win on Linux

	const repo, file = "o/r", "m.gguf"
	body := []byte("pretend gguf bytes, byte-identical to what curated.json would pin")
	sum := sha256.Sum256(body)
	sha := hex.EncodeToString(sum[:])

	dir, err := CacheDir(repo)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, file), body, 0o644); err != nil {
		t.Fatal(err)
	}

	ref := Ref{Repo: repo, File: file, Pin: sha, Bytes: int64(len(body))}
	path, ok := resolveOffline(ref)
	if !ok {
		t.Fatal("resolveOffline reported no cache hit for a file that is on disk, right size, right digest")
	}
	if want := filepath.Join(dir, file); path != want {
		t.Errorf("resolveOffline path = %q, want %q", path, want)
	}
}

// A wrong digest or a wrong size must NOT resolve offline — this is what stops a stale or
// corrupted cache entry from being served as if it were verified, the same discipline
// cachedIntact already applies inside Download.
func TestResolveOffline_mismatchIsNotAHit(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", "")

	const repo, file = "o/r", "m.gguf"
	body := []byte("actual cached bytes")
	dir, err := CacheDir(repo)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, file), body, 0o644); err != nil {
		t.Fatal(err)
	}

	wrongSHA := Ref{Repo: repo, File: file, Pin: strings.Repeat("a", 64), Bytes: int64(len(body))}
	if _, ok := resolveOffline(wrongSHA); ok {
		t.Error("resolveOffline reported a hit despite a wrong pinned digest")
	}
	wrongSize := Ref{Repo: repo, File: file, Pin: "irrelevant-but-nonempty", Bytes: int64(len(body)) + 1}
	if _, ok := resolveOffline(wrongSize); ok {
		t.Error("resolveOffline reported a hit despite a wrong pinned size")
	}
	missing := Ref{Repo: repo, File: "not-cached.gguf", Pin: "irrelevant-but-nonempty", Bytes: 3}
	if _, ok := resolveOffline(missing); ok {
		t.Error("resolveOffline reported a hit for a file that was never written")
	}
}

// A quant selector (hf:repo:quant) never qualifies for the offline path — Select needs the
// repo's real file listing to pick which file matches a quant string, so File is empty and
// resolveOffline must decline regardless of what Pin/Bytes happen to hold.
func TestResolveOffline_quantOnlyRefNeverQualifies(t *testing.T) {
	ref := Ref{Repo: "o/r", Quant: "q4_k_m", Pin: "irrelevant", Bytes: 12345}
	if _, ok := resolveOffline(ref); ok {
		t.Error("a quant-only ref (no exact File) resolved offline; Select needs List's file " +
			"listing to know which file that even is (V-16)")
	}
}

// TestResolve_callsResolveOfflineBeforeCheckAccess is the wiring guard: resolveOffline works in
// isolation (the tests above), but that proves nothing about whether Resolve's actual request
// path reaches it BEFORE the network call it exists to skip — the exact shape of gap this
// session's audit keeps finding (a helper with a test, and a call site nobody checked).
func TestResolve_callsResolveOfflineBeforeCheckAccess(t *testing.T) {
	src, err := os.ReadFile("resolve.go")
	if err != nil {
		t.Fatal(err)
	}
	body := string(src)
	i := strings.Index(body, "func Resolve(")
	if i < 0 {
		t.Fatal("Resolve not found — this guard is watching nothing")
	}
	fn := body[i:]
	oi := strings.Index(fn, "resolveOffline(")
	ci := strings.Index(fn, "CheckAccess(")
	if oi < 0 {
		t.Fatal("Resolve no longer calls resolveOffline — the offline cache-hit path is gone (V-16)")
	}
	if ci < 0 {
		t.Fatal("Resolve no longer calls CheckAccess — this guard is watching the wrong shape now")
	}
	if oi > ci {
		t.Error("Resolve calls CheckAccess before resolveOffline — a cached demo: ref would hit " +
			"the network anyway, defeating V-16's fix")
	}
}
