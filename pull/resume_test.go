package pull

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestDownload_resumesAfterInterruption is the test the resume feature exists for: a dropped
// connection midway through a multi-gigabyte transfer must not cost the bytes already on
// disk, AND the end-to-end sha256 must still be correct afterwards.
//
// The digest is the part that is easy to get wrong. On resume the running hash has to cover
// the bytes ALREADY on disk, not just the tail — otherwise the check passes over a fragment
// and the whole verification is theatre. So this asserts the final digest, not merely the
// file size.
func TestDownload_resumesAfterInterruption(t *testing.T) {
	body := make([]byte, 512*1024)
	rand.New(rand.NewSource(11)).Read(body)
	sum := sha256.Sum256(body)
	want := hex.EncodeToString(sum[:])

	var ranges []string
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		ranges = append(ranges, r.Header.Get("Range"))
		// First attempt: drop halfway, as a flaky link would.
		if calls == 1 {
			w.Header().Set("Content-Length", "999999")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(body[:len(body)/2])
			return
		}
		http.ServeContent(w, r, "m.gguf", time.Unix(0, 0), bytes.NewReader(body))
	}))
	defer srv.Close()
	hfCDN = srv.URL
	defer func() { hfCDN = "https://huggingface.co" }()

	dir := t.TempDir()
	f := File{Path: "m.gguf", Size: int64(len(body)), SHA256: want}

	// Attempt 1 — must fail, and must LEAVE the partial behind.
	if _, err := Download(context.Background(), "o/r", f, dir, nil); err == nil {
		t.Fatal("truncated first attempt should have failed")
	}
	part := filepath.Join(dir, "m.gguf.part")
	st, err := os.Stat(part)
	if err != nil {
		t.Fatalf(".part must survive a failed attempt so the next run can resume: %v", err)
	}
	if st.Size() != int64(len(body)/2) {
		t.Fatalf(".part is %d bytes, want %d", st.Size(), len(body)/2)
	}

	// Attempt 2 — resumes, completes, verifies.
	path, err := Download(context.Background(), "o/r", f, dir, nil)
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	if len(ranges) < 2 || !strings.HasPrefix(ranges[1], "bytes=") {
		t.Errorf("second attempt should have sent a Range header, got %q", ranges)
	}
	got, err := fileSHA256(path)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Errorf("resumed file digest = %s, want %s — the running hash did not cover the resumed prefix", got, want)
	}
	if _, err := os.Stat(part); !os.IsNotExist(err) {
		t.Error(".part should be gone after a successful download")
	}
}

// TestDownload_serverIgnoresRange covers the server that answers 200 to a Range request.
// The body is then the WHOLE file, so appending it to an existing partial would silently
// produce a corrupt file — caught only by the checksum, with no clue why.
func TestDownload_serverIgnoresRange(t *testing.T) {
	body := []byte(strings.Repeat("goinfer", 5000))
	sum := sha256.Sum256(body)
	want := hex.EncodeToString(sum[:])

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK) // 200 even though a Range was asked for
		_, _ = w.Write(body)
	}))
	defer srv.Close()
	hfCDN = srv.URL
	defer func() { hfCDN = "https://huggingface.co" }()

	dir := t.TempDir()
	// Seed a stale partial that must be DISCARDED, not appended to.
	if err := os.WriteFile(filepath.Join(dir, "m.gguf.part"), []byte("stale junk"), 0o644); err != nil {
		t.Fatal(err)
	}
	path, err := Download(context.Background(), "o/r", File{Path: "m.gguf", Size: int64(len(body)), SHA256: want}, dir, nil)
	if err != nil {
		t.Fatalf("download: %v", err)
	}
	got, err := fileSHA256(path)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Errorf("digest = %s, want %s — a 200 response must reset the partial and the hash", got, want)
	}
}

// TestDownload_rangeNotSatisfiableOnAnAlreadyCompletePart is N-70 (docs/audit-2026-09-10.md):
// f.Size <= 0 (a non-LFS file with no declared size) admits ANY existing .part into the resume
// path regardless of whether it is actually complete — there is no size to compare against. If
// a prior run fetched every byte but was interrupted before the digest-check-and-rename, the
// next run's Range request starts exactly at EOF and HuggingFace answers 416, not 206/200.
// Before this fix that fell into the catch-all default case: a confusing "HuggingFace returned
// 416" error, AND the .part was never cleared — so every subsequent retry hit the identical 416
// forever. This drives that exact shape and asserts it now succeeds instead.
func TestDownload_rangeNotSatisfiableOnAnAlreadyCompletePart(t *testing.T) {
	body := []byte(strings.Repeat("goinfer-n70-", 500))
	sum := sha256.Sum256(body)
	want := hex.EncodeToString(sum[:])

	var ranges []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ranges = append(ranges, r.Header.Get("Range"))
		w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
	}))
	defer srv.Close()
	hfCDN = srv.URL
	defer func() { hfCDN = "https://huggingface.co" }()

	dir := t.TempDir()
	// Seed a .part that is ALREADY the complete, correct file — the shape a prior run leaves
	// when it finished fetching but never reached the rename.
	if err := os.WriteFile(filepath.Join(dir, "m.gguf.part"), body, 0o644); err != nil {
		t.Fatal(err)
	}
	// f.Size <= 0: the non-LFS case this finding is about — no declared size to compare the
	// .part against, so the resume guard admits it unconditionally.
	f := File{Path: "m.gguf", Size: 0, SHA256: want}

	path, err := Download(context.Background(), "o/r", f, dir, nil)
	if err != nil {
		t.Fatalf("download: %v (ranges requested: %v)", err, ranges)
	}
	got, err := fileSHA256(path)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Errorf("digest = %s, want %s", got, want)
	}
	if _, err := os.Stat(filepath.Join(dir, "m.gguf.part")); !os.IsNotExist(err) {
		t.Error(".part should be gone once the already-complete file is verified and renamed")
	}
	if len(ranges) != 1 || !strings.HasPrefix(ranges[0], "bytes=") {
		t.Errorf("expected exactly one Range request, got %v", ranges)
	}
}

// TestDownload_rangeNotSatisfiableWithBadDigestIsCleanedUp: the 416 fast path must still
// verify the digest, not just trust "the server said no more bytes" — a LOCALLY corrupted
// .part that happens to be the same length as the real file must still be caught and removed,
// not renamed straight to the final path.
func TestDownload_rangeNotSatisfiableWithBadDigestIsCleanedUp(t *testing.T) {
	body := []byte(strings.Repeat("goinfer-n70-corrupt-", 300))

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
	}))
	defer srv.Close()
	hfCDN = srv.URL
	defer func() { hfCDN = "https://huggingface.co" }()

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "m.gguf.part"), body, 0o644); err != nil {
		t.Fatal(err)
	}
	// A digest that does NOT match body — the local .part is corrupt, not really complete.
	f := File{Path: "m.gguf", Size: 0, SHA256: strings.Repeat("c", 64)}

	if _, err := Download(context.Background(), "o/r", f, dir, nil); err == nil {
		t.Fatal("a digest mismatch on the 416 fast path must fail")
	}
	if _, err := os.Stat(filepath.Join(dir, "m.gguf.part")); !os.IsNotExist(err) {
		t.Error("a corrupt .part on the 416 fast path must be removed, or every retry hits the same 416 forever")
	}
	if _, err := os.Stat(filepath.Join(dir, "m.gguf")); !os.IsNotExist(err) {
		t.Error("nothing may be written to the final path on a digest mismatch")
	}
}

// TestDownload_badDigestIsNotKept: a completed transfer whose digest is wrong must leave
// nothing behind. Otherwise the next run resumes from known-bad bytes forever.
func TestDownload_badDigestIsNotKept(t *testing.T) {
	body := []byte("some bytes")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.ServeContent(w, r, "m.gguf", time.Unix(0, 0), bytes.NewReader(body))
	}))
	defer srv.Close()
	hfCDN = srv.URL
	defer func() { hfCDN = "https://huggingface.co" }()

	dir := t.TempDir()
	f := File{Path: "m.gguf", Size: int64(len(body)), SHA256: strings.Repeat("b", 64)}
	if _, err := Download(context.Background(), "o/r", f, dir, nil); err == nil {
		t.Fatal("a digest mismatch must fail")
	}
	if _, err := os.Stat(filepath.Join(dir, "m.gguf.part")); !os.IsNotExist(err) {
		t.Error("a known-bad partial must be deleted, or every later run resumes from it")
	}
	if _, err := os.Stat(filepath.Join(dir, "m.gguf")); !os.IsNotExist(err) {
		t.Error("nothing may be written to the final path on a digest mismatch")
	}
}
