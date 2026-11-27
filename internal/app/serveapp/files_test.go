package serveapp

import (
	"bytes"
	"encoding/json"
	"flag"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/stores"
)

func putBytes(fs *stores.FileStore, n int) *stores.StoredFile {
	return fs.Put("f", "batch", make([]byte, n))
}

// A finished batch with both successes and failures stores an output and an error file. Over the byte limit, together or each
// alone, storing the error file must not evict the output: the older file goes and both of the batch's stay.
func TestFinalizeBatch_keepsBothResultFilesOverTheByteLimit(t *testing.T) {
	s := &server{serverStores: serverStores{files: stores.NewFileStore(0, 50), batches: stores.NewBatchStore(4)}}
	older := putBytes(s.files, 10)
	b := s.batches.Create("openai_chat", "/v1/chat/completions", "24h", "f", []string{"ok", "bad"})
	b.SetResult(0, stores.BatchLineResult{CustomID: "ok", StatusCode: 200, Body: map[string]any{"content": strings.Repeat("x", 200)}})
	b.SetResult(1, stores.LineError("bad", 400, "invalid_request_error", strings.Repeat("y", 200)))
	b.DoneLine()
	b.DoneLine()
	s.finalizeBatch(b.ID)
	outID, errID := b.OutputFileIDs()
	out, errs := s.files.Get(outID), s.files.Get(errID)
	if out == nil || errs == nil {
		t.Fatalf("output file kept %v, error file kept %v: a batch's result files must survive their own storing", out != nil, errs != nil)
	}
	if len(out.Bytes) <= 50 || len(errs.Bytes) <= 50 {
		t.Fatalf("output %d and error %d bytes: each must be over the 50-byte limit for this to test it", len(out.Bytes), len(errs.Bytes))
	}
	if s.files.Get(older.ID) != nil {
		t.Error("the older file was kept over the limit")
	}
	if c := putBytes(s.files, 10); s.files.Get(outID) != nil || s.files.Get(errID) != nil || s.files.Get(c.ID) == nil {
		t.Error("the next put did not evict the batch's files, oldest first")
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
