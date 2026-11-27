package stores

import (
	"strings"
	"testing"
)

func putBytes(fs *FileStore, n int) *StoredFile { return fs.Put("f", "batch", make([]byte, n)) }

func TestFileStore_putAndGet(t *testing.T) {
	fs := NewFileStore(0, 0)
	f := fs.Put("in.jsonl", "batch", []byte("hello"))
	if f.ID == "" {
		t.Fatal("Put returned an empty id")
	}
	got := fs.Get(f.ID)
	if got == nil {
		t.Fatal("Get returned nil for a just-put file")
	}
	if got.Filename != "in.jsonl" || got.Purpose != "batch" || string(got.Bytes) != "hello" {
		t.Fatalf("Get returned %+v, want filename=in.jsonl purpose=batch bytes=hello", got)
	}
}

func TestFileStore_getUnknownReturnsNil(t *testing.T) {
	fs := NewFileStore(0, 0)
	if fs.Get("does-not-exist") != nil {
		t.Fatal("Get for an unknown id returned non-nil")
	}
}

func TestFileStore_evictionDropsOldestFirst(t *testing.T) {
	fs := NewFileStore(1, 0)
	a := fs.Put("a", "batch", []byte("a"))
	b := fs.Put("b", "batch", []byte("b"))
	if fs.Get(a.ID) != nil {
		t.Fatal("oldest file was not evicted at cap=1")
	}
	if fs.Get(b.ID) == nil {
		t.Fatal("newest file was evicted instead of the oldest")
	}
}

func TestFileStore_unboundedWhenCapIsZero(t *testing.T) {
	fs := NewFileStore(0, 0)
	var ids []string
	for i := range 10 {
		f := fs.Put(string(rune('a'+i)), "batch", []byte{byte(i)})
		ids = append(ids, f.ID)
	}
	for _, id := range ids {
		if fs.Get(id) == nil {
			t.Fatalf("file %s evicted despite cap=0 (unbounded)", id)
		}
	}
}

func TestFileStore_byteLimitDropsOldestFirst(t *testing.T) {
	fs := NewFileStore(0, 100)
	a, b := putBytes(fs, 60), putBytes(fs, 60)
	if fs.Get(a.ID) != nil || fs.Get(b.ID) == nil {
		t.Fatal("over the byte limit, the oldest file must go and the newest stay")
	}
	c := putBytes(fs, 30)
	if fs.Get(b.ID) == nil || fs.Get(c.ID) == nil {
		t.Fatal("a file was dropped under the byte limit")
	}
	d := putBytes(fs, 20)
	if fs.Get(b.ID) != nil || fs.Get(c.ID) == nil || fs.Get(d.ID) == nil {
		t.Fatal("the store dropped more or less than it had to")
	}
}

func TestFileStore_newestIsKeptEvenAloneOverTheLimit(t *testing.T) {
	fs := NewFileStore(0, 10)
	a := putBytes(fs, 50)
	if fs.Get(a.ID) == nil {
		t.Fatal("the only file was dropped")
	}
	b := putBytes(fs, 50)
	if fs.Get(a.ID) != nil || fs.Get(b.ID) == nil {
		t.Fatal("the older oversize file must go when the next arrives")
	}
}

func TestFileStore_countLimitKeepsTheByteAccountRight(t *testing.T) {
	fs := NewFileStore(2, 100)
	putBytes(fs, 60)
	b, c := putBytes(fs, 30), putBytes(fs, 30)
	d := putBytes(fs, 30)
	if fs.Get(b.ID) != nil || fs.Get(c.ID) == nil || fs.Get(d.ID) == nil {
		t.Fatal("the count limit dropped the wrong file")
	}
	e := putBytes(fs, 30)
	if fs.Get(d.ID) == nil || fs.Get(e.ID) == nil {
		t.Fatal("a stale byte account dropped a file the store has room for")
	}
}

func TestFileStore_zeroBytesIsNoByteLimit(t *testing.T) {
	fs := NewFileStore(0, 0)
	a, b := putBytes(fs, 1<<20), putBytes(fs, 1<<20)
	if fs.Get(a.ID) == nil || fs.Get(b.ID) == nil {
		t.Fatal("-max-file-store-bytes 0 must mean no limit on bytes")
	}
}

func TestFinalizeBatch_keepsBothResultFilesOverTheByteLimit(t *testing.T) {
	fs := NewFileStore(0, 50)
	bs := NewBatchStore(4)
	older := putBytes(fs, 10)
	b := bs.Create("openai_chat", "/v1/chat/completions", "24h", "f", []string{"ok", "bad"})
	b.SetResult(0, BatchLineResult{CustomID: "ok", StatusCode: 200, Body: map[string]any{"content": strings.Repeat("x", 200)}})
	b.SetResult(1, LineError("bad", 400, "invalid_request_error", strings.Repeat("y", 200)))
	b.DoneLine()
	b.DoneLine()
	b.Finalize(fs.PutAll)
	outID, errID := b.OutputFileIDs()
	out, errs := fs.Get(outID), fs.Get(errID)
	if out == nil || errs == nil {
		t.Fatalf("output file kept %v, error file kept %v: a batch's result files must survive their own storing", out != nil, errs != nil)
	}
	if len(out.Bytes) <= 50 || len(errs.Bytes) <= 50 {
		t.Fatalf("output %d and error %d bytes: each must be over the 50-byte limit for this to test it", len(out.Bytes), len(errs.Bytes))
	}
	if fs.Get(older.ID) != nil {
		t.Error("the older file was kept over the limit")
	}
	if c := putBytes(fs, 10); fs.Get(outID) != nil || fs.Get(errID) != nil || fs.Get(c.ID) == nil {
		t.Error("the next put did not evict the batch's files, oldest first")
	}
}
