package serveapp

import (
	"bytes"
	"encoding/json"
	"flag"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestFileStore_putAndGet(t *testing.T) {
	fs := newFileStore(0, 0)
	f := fs.put("in.jsonl", "batch", []byte("hello"))
	if f.ID == "" {
		t.Fatal("put returned an empty id")
	}
	got := fs.get(f.ID)
	if got == nil {
		t.Fatal("get returned nil for a just-put file")
	}
	if got.Filename != "in.jsonl" || got.Purpose != "batch" || string(got.Bytes) != "hello" {
		t.Fatalf("get returned %+v, want filename=in.jsonl purpose=batch bytes=hello", got)
	}
}

func TestFileStore_getUnknownReturnsNil(t *testing.T) {
	fs := newFileStore(0, 0)
	if fs.get("does-not-exist") != nil {
		t.Fatal("get for an unknown id returned non-nil")
	}
}

// TestFileStore_evictionDropsOldestFirst: plain FIFO at cap, no "never evict a live one"
// protection (files.go's own doc comment — nothing polls a file the way a client polls a job).
func TestFileStore_evictionDropsOldestFirst(t *testing.T) {
	fs := newFileStore(1, 0)
	a := fs.put("a", "batch", []byte("a"))
	b := fs.put("b", "batch", []byte("b"))
	if fs.get(a.ID) != nil {
		t.Fatal("oldest file was not evicted at cap=1")
	}
	if fs.get(b.ID) == nil {
		t.Fatal("newest file was evicted instead of the oldest")
	}
}

func TestFileStore_unboundedWhenCapIsZero(t *testing.T) {
	fs := newFileStore(0, 0)
	var ids []string
	for i := range 10 {
		f := fs.put(string(rune('a'+i)), "batch", []byte{byte(i)})
		ids = append(ids, f.ID)
	}
	for _, id := range ids {
		if fs.get(id) == nil {
			t.Fatalf("file %s evicted despite cap=0 (unbounded)", id)
		}
	}
}

// put stores n bytes.
func putBytes(fs *fileStore, n int) *storedFile { return fs.put("f", "batch", make([]byte, n)) }

func TestFileStore_byteLimitDropsOldestFirst(t *testing.T) {
	fs := newFileStore(0, 100)
	a, b := putBytes(fs, 60), putBytes(fs, 60)
	if fs.get(a.ID) != nil || fs.get(b.ID) == nil {
		t.Fatal("over the byte limit, the oldest file must go and the newest stay")
	}
	c := putBytes(fs, 30) // 90: nothing goes
	if fs.get(b.ID) == nil || fs.get(c.ID) == nil {
		t.Fatal("a file was dropped under the byte limit")
	}
	d := putBytes(fs, 20) // 110: b goes, and 50 is enough
	if fs.get(b.ID) != nil || fs.get(c.ID) == nil || fs.get(d.ID) == nil {
		t.Fatal("the store dropped more or less than it had to")
	}
}

// A result larger than the limit is kept until the next file arrives: dropping it as it is stored would lose a finished batch outright.
func TestFileStore_newestIsKeptEvenAloneOverTheLimit(t *testing.T) {
	fs := newFileStore(0, 10)
	a := putBytes(fs, 50)
	if fs.get(a.ID) == nil {
		t.Fatal("the only file was dropped")
	}
	b := putBytes(fs, 50)
	if fs.get(a.ID) != nil || fs.get(b.ID) == nil {
		t.Fatal("the older oversize file must go when the next arrives")
	}
}

// Files dropped by the count limit must leave the byte account too, or the store would drop files it has room for.
func TestFileStore_countLimitKeepsTheByteAccountRight(t *testing.T) {
	fs := newFileStore(2, 100)
	putBytes(fs, 60)
	b, c := putBytes(fs, 30), putBytes(fs, 30) // the count limit drops the 60
	d := putBytes(fs, 30)                      // the count limit drops b; c and d are 60
	if fs.get(b.ID) != nil || fs.get(c.ID) == nil || fs.get(d.ID) == nil {
		t.Fatal("the count limit dropped the wrong file")
	}
	e := putBytes(fs, 30) // the count limit drops c; d and e are 60, well under 100
	if fs.get(d.ID) == nil || fs.get(e.ID) == nil {
		t.Fatal("a stale byte account dropped a file the store has room for")
	}
}

// A finished batch with both successes and failures stores an output and an error file. Over the byte limit, together or each
// alone, storing the error file must not evict the output: the older file goes and both of the batch's stay.
func TestFinalizeBatch_keepsBothResultFilesOverTheByteLimit(t *testing.T) {
	s := &server{files: newFileStore(0, 50), batches: newBatchStore(4)}
	older := putBytes(s.files, 10)
	b := s.batches.create("openai_chat", "/v1/chat/completions", "24h", "f", []string{"ok", "bad"})
	setResult(b, 0, batchLineResult{CustomID: "ok", StatusCode: 200, Body: map[string]any{"content": strings.Repeat("x", 200)}})
	setResult(b, 1, lineError("bad", 400, "invalid_request_error", strings.Repeat("y", 200)))
	b.wg.Done()
	b.wg.Done()
	s.finalizeBatch(b.ID)
	b.mu.Lock()
	outID, errID := b.OutputFileID, b.ErrorFileID
	b.mu.Unlock()
	out, errs := s.files.get(outID), s.files.get(errID)
	if out == nil || errs == nil {
		t.Fatalf("output file kept %v, error file kept %v: a batch's result files must survive their own storing", out != nil, errs != nil)
	}
	if len(out.Bytes) <= 50 || len(errs.Bytes) <= 50 {
		t.Fatalf("output %d and error %d bytes: each must be over the 50-byte limit for this to test it", len(out.Bytes), len(errs.Bytes))
	}
	if s.files.get(older.ID) != nil {
		t.Error("the older file was kept over the limit")
	}
	if c := putBytes(s.files, 10); s.files.get(outID) != nil || s.files.get(errID) != nil || s.files.get(c.ID) == nil {
		t.Error("the next put did not evict the batch's files, oldest first")
	}
}

func TestFileStore_zeroBytesIsNoByteLimit(t *testing.T) {
	fs := newFileStore(0, 0)
	a, b := putBytes(fs, 1<<20), putBytes(fs, 1<<20)
	if fs.get(a.ID) == nil || fs.get(b.ID) == nil {
		t.Fatal("-max-file-store-bytes 0 must mean no limit on bytes")
	}
}

// The limit through the flag and both routes that read a file, its object and its content: the oldest upload is gone from each, and
// the 404 says why it might be, since the client holds an id it was given.
func TestFiles_byteLimitThroughFlagAndRoute(t *testing.T) {
	fs := flag.NewFlagSet("t", flag.ContinueOnError)
	sf := registerFlags(fs)
	if err := fs.Parse(nil); err != nil {
		t.Fatal(err)
	}
	if sf.cfg.maxFileStoreBytes != 1<<30 {
		t.Errorf("the default is %d, want 1 GiB", sf.cfg.maxFileStoreBytes)
	}
	if err := fs.Parse([]string{"-max-file-store-bytes", "150"}); err != nil {
		t.Fatal(err)
	}
	cfg := sf.cfg
	srv, err := newServer(cfg)
	if err != nil {
		t.Fatal(err)
	}
	mux, _ := srv.routes(cfg, "", nil)
	upload := func() string {
		t.Helper()
		body, ctype := uploadBody(t, make([]byte, 100))
		req := httptest.NewRequest(http.MethodPost, "/v1/files", bytes.NewReader(body))
		req.Header.Set("Content-Type", ctype)
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("upload: %d %s", rec.Code, rec.Body)
		}
		var f struct {
			ID string `json:"id"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &f); err != nil {
			t.Fatal(err)
		}
		return f.ID
	}
	get := func(path string) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, path, nil))
		return rec
	}
	first, second := upload(), upload()
	for _, path := range []string{"/v1/files/" + first, "/v1/files/" + first + "/content"} {
		if rec := get(path); rec.Code != http.StatusNotFound || !strings.Contains(rec.Body.String(), "-max-file-store-bytes") {
			t.Errorf("GET %s, a dropped file: %d %q, want a 404 naming the limit", path, rec.Code, rec.Body)
		}
	}
	if rec := get("/v1/files/" + second + "/content"); rec.Code != http.StatusOK || rec.Body.Len() != 100 {
		t.Errorf("the newest file's content: %d, %d bytes", rec.Code, rec.Body.Len())
	}
	var obj struct {
		ID    string `json:"id"`
		Bytes int    `json:"bytes"`
	}
	if rec := get("/v1/files/" + second); rec.Code != http.StatusOK || json.Unmarshal(rec.Body.Bytes(), &obj) != nil || obj.ID != second || obj.Bytes != 100 {
		t.Errorf("the newest file's object: %d %q", rec.Code, rec.Body)
	}
}

func TestConfig_refusesANegativeFileStoreLimit(t *testing.T) {
	cfg := config{maxFileStoreBytes: -1}
	if err := cfg.validate(); err == nil || !strings.Contains(err.Error(), "-max-file-store-bytes") {
		t.Errorf("validate: %v, want a refusal naming the flag", err)
	}
}
