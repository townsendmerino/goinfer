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
	e := putBytes(fs, 30) // c and e are 60, well under 100: only the count limit acts
	if fs.get(d.ID) == nil || fs.get(e.ID) == nil {
		t.Fatal("a stale byte account dropped a file the store has room for")
	}
}

func TestFileStore_zeroBytesIsNoByteLimit(t *testing.T) {
	fs := newFileStore(0, 0)
	a, b := putBytes(fs, 1<<20), putBytes(fs, 1<<20)
	if fs.get(a.ID) == nil || fs.get(b.ID) == nil {
		t.Fatal("-max-file-store-bytes 0 must mean no limit on bytes")
	}
}

// The limit through the flag and the route: the oldest upload is gone, and the 404 says why it might be, since the client holds an id it was given.
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
	get := func(id string) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/files/"+id+"/content", nil))
		return rec
	}
	first, second := upload(), upload()
	if rec := get(first); rec.Code != http.StatusNotFound || !strings.Contains(rec.Body.String(), "-max-file-store-bytes") {
		t.Errorf("a dropped file: %d %q, want a 404 naming the limit", rec.Code, rec.Body)
	}
	if rec := get(second); rec.Code != http.StatusOK || rec.Body.Len() != 100 {
		t.Errorf("the newest file: %d, %d bytes", rec.Code, rec.Body.Len())
	}
}

func TestConfig_refusesANegativeFileStoreLimit(t *testing.T) {
	cfg := config{maxFileStoreBytes: -1}
	if err := cfg.validate(); err == nil || !strings.Contains(err.Error(), "-max-file-store-bytes") {
		t.Errorf("validate: %v, want a refusal naming the flag", err)
	}
}
