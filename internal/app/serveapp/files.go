package serveapp

import (
	"sync"
	"time"
)

// storedFile is OpenAI's Files API object: a batch's input JSONL, or an assembled output or error JSONL, held in
// memory keyed by id. Write-once: nothing mutates its fields after put() returns it, so unlike job (job.go), get()
// hands back the live pointer directly, with no snapshot copy and no lock needed to read it.
type storedFile struct {
	ID        string
	Filename  string
	Purpose   string // "batch" (input) | "batch_output" | "batch_error"
	Bytes     []byte
	CreatedAt time.Time
}

// fileStore is a bounded, in-memory, process-wide registry. Two bounds hold, whichever comes first: at most cap files, and at most maxBytes of
// their contents (either 0 for no such bound). The oldest file goes first and the newest is always kept, even alone over maxBytes, so a finished
// batch's results are never dropped as they are stored. Deliberately without jobStore's "never evict a live one" protection
// (jobStore.evictLocked): nothing polls a file the way a client polls a job's state, so there is no in-flight condition eviction could corrupt, and
// a reader holding a storedFile keeps its bytes until it is done.
type fileStore struct {
	mu       sync.Mutex
	files    map[string]*storedFile
	order    []string
	cap      int
	maxBytes int64
	bytes    int64 // the contents of files
}

func newFileStore(cap int, maxBytes int64) *fileStore {
	return &fileStore{files: map[string]*storedFile{}, cap: cap, maxBytes: maxBytes}
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
	fs.bytes += int64(len(data))
	for len(fs.order) > 1 && fs.overLimit() {
		oldest := fs.order[0]
		fs.bytes -= int64(len(fs.files[oldest].Bytes))
		delete(fs.files, oldest)
		fs.order = fs.order[1:]
	}
	return f
}

// overLimit reports whether the store is past either bound. The caller holds mu.
func (fs *fileStore) overLimit() bool {
	return (fs.cap > 0 && len(fs.order) > fs.cap) || (fs.maxBytes > 0 && fs.bytes > fs.maxBytes)
}

// get returns id's storedFile, or nil if unknown or evicted.
func (fs *fileStore) get(id string) *storedFile {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	return fs.files[id]
}
