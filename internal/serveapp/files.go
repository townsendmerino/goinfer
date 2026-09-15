package serveapp

import (
	"sync"
	"time"
)

// storedFile is OpenAI's Files API object (task-work-queue-2026-09.md J4) — a batch's input
// JSONL, or an assembled output/error JSONL, held in memory keyed by id. Write-once: nothing
// mutates a storedFile's fields after put() returns it, so unlike job (job.go), get() can hand
// back the live pointer directly — no snapshot-style copy is needed, and no lock is needed to
// read one after it's been looked up.
type storedFile struct {
	ID        string
	Filename  string
	Purpose   string // "batch" (input) | "batch_output" | "batch_error"
	Bytes     []byte
	CreatedAt time.Time
}

// fileStore is a bounded, in-memory, process-wide registry, mirroring responseStore's own FIFO
// cap pattern (responses.go:49) — but deliberately WITHOUT jobStore's "never evict a live one"
// protection (job.go:198's evictLocked): nothing polls a file the way a client polls a job's
// state, so there is no in-flight condition eviction could corrupt. A generous cap is the whole
// answer here; this is a scope choice, not an oversight (see this package's J4 closure note).
type fileStore struct {
	mu    sync.Mutex
	files map[string]*storedFile
	order []string
	cap   int
}

func newFileStore(cap int) *fileStore {
	return &fileStore{files: map[string]*storedFile{}, cap: cap}
}

// put stores data under a freshly minted id and returns the resulting storedFile. filename/purpose
// are caller-supplied metadata (echoed back on GET, never interpreted).
func (fs *fileStore) put(filename, purpose string, data []byte) *storedFile {
	f := &storedFile{
		ID: "file-" + reqID(), Filename: filename, Purpose: purpose,
		Bytes: data, CreatedAt: time.Now(),
	}
	fs.mu.Lock()
	defer fs.mu.Unlock()
	fs.files[f.ID] = f
	fs.order = append(fs.order, f.ID)
	if fs.cap > 0 {
		for len(fs.order) > fs.cap {
			delete(fs.files, fs.order[0])
			fs.order = fs.order[1:]
		}
	}
	return f
}

// get returns id's storedFile, or nil if unknown or evicted.
func (fs *fileStore) get(id string) *storedFile {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	return fs.files[id]
}
