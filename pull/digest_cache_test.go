package pull

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestCachedFileSHA256_matchesRawHash is P-12's correctness floor: the cached path must return
// the exact same digest the uncached fileSHA256 would, on a fresh file with no sidecar yet.
func TestCachedFileSHA256_matchesRawHash(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "m.gguf")
	if err := os.WriteFile(path, []byte("pretend checkpoint bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	want, err := fileSHA256(path)
	if err != nil {
		t.Fatal(err)
	}
	got, err := cachedFileSHA256(path)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Errorf("cachedFileSHA256 = %q, want %q (fileSHA256's own answer)", got, want)
	}
	if _, err := os.Stat(sidecarPath(path)); err != nil {
		t.Errorf("no sidecar written at %s: %v", sidecarPath(path), err)
	}
}

// TestCachedFileSHA256_actuallyTrustsTheSidecar is P-12's TAUTOLOGY GUARD: proving
// cachedFileSHA256 returns the RIGHT answer is not the same as proving it skipped re-hashing to
// get there — a version that silently always recomputed would pass the test above too. This
// plants a WRONG digest in the sidecar (size/mtime left matching the real file) and asserts
// cachedFileSHA256 returns that wrong value — the only way to observe, from outside, that the
// sidecar was consulted rather than the file being read again.
func TestCachedFileSHA256_actuallyTrustsTheSidecar(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "m.gguf")
	if err := os.WriteFile(path, []byte("pretend checkpoint bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	st, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	const planted = "0000000000000000000000000000000000000000000000000000000000000000"
	sc := digestSidecar{Size: st.Size(), ModTime: st.ModTime().UnixNano(), SHA256: planted}
	b, err := json.Marshal(sc)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(sidecarPath(path), b, 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := cachedFileSHA256(path)
	if err != nil {
		t.Fatal(err)
	}
	if got != planted {
		t.Errorf("cachedFileSHA256 = %q, want the planted %q — a matching-size/mtime sidecar was not trusted, so this call re-hashed the file instead of using the cache", got, planted)
	}
}

// TestCachedFileSHA256_invalidatesOnRealChange proves the (size, mtime) key actually gates the
// cache: a file whose CONTENT and SIZE genuinely change must be re-hashed, not served the old
// digest — the correctness half that TestCachedFileSHA256_actuallyTrustsTheSidecar's tautology
// guard must not come at the cost of.
func TestCachedFileSHA256_invalidatesOnRealChange(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "m.gguf")
	if err := os.WriteFile(path, []byte("version one"), 0o644); err != nil {
		t.Fatal(err)
	}
	first, err := cachedFileSHA256(path)
	if err != nil {
		t.Fatal(err)
	}

	// Advance mtime explicitly: some filesystems have coarse mtime resolution, and a same-second
	// rewrite could otherwise leave the sidecar's stale (size, mtime) key looking unchanged even
	// though the content did change — a real trap this test exists to not paper over.
	newContent := []byte("version two, a different length")
	if err := os.WriteFile(path, newContent, 0o644); err != nil {
		t.Fatal(err)
	}
	future := time.Now().Add(time.Hour)
	if err := os.Chtimes(path, future, future); err != nil {
		t.Fatal(err)
	}

	second, err := cachedFileSHA256(path)
	if err != nil {
		t.Fatal(err)
	}
	want, err := fileSHA256(path)
	if err != nil {
		t.Fatal(err)
	}
	if second != want {
		t.Errorf("cachedFileSHA256 after a real content+size+mtime change = %q, want %q (fresh hash)", second, want)
	}
	if second == first {
		t.Error("digest did not change after the file's content changed — stale cache served across a real edit")
	}
}
