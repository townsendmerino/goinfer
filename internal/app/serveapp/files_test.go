package serveapp

import "testing"

func TestFileStore_putAndGet(t *testing.T) {
	fs := newFileStore(0)
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
	fs := newFileStore(0)
	if fs.get("does-not-exist") != nil {
		t.Fatal("get for an unknown id returned non-nil")
	}
}

// TestFileStore_evictionDropsOldestFirst: plain FIFO at cap, no "never evict a live one"
// protection (files.go's own doc comment — nothing polls a file the way a client polls a job).
func TestFileStore_evictionDropsOldestFirst(t *testing.T) {
	fs := newFileStore(1)
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
	fs := newFileStore(0)
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
