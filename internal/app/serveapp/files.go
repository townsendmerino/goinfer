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
// their contents (either 0 for no such bound). The oldest file goes first, and the files of the newest put are always kept, even over either bound,
// so a finished batch's results, its output and error files stored by one putAll, are never dropped as they are stored. Deliberately without jobStore's "never evict a live one" protection
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

// newFile is one file for putAll: caller-supplied metadata (echoed back on GET, never interpreted) and contents.
type newFile struct {
	filename, purpose string
	data              []byte
}

// put stores data under a freshly minted id and returns the resulting storedFile.
func (fs *fileStore) put(filename, purpose string, data []byte) *storedFile {
	return fs.putAll(newFile{filename, purpose, data})[0]
}

// putAll stores files together, each under a freshly minted id, and returns them in order. Eviction takes only older files, so
// none of these is dropped by the others' storing.
func (fs *fileStore) putAll(files ...newFile) []*storedFile {
	out := make([]*storedFile, len(files))
	now := time.Now()
	fs.mu.Lock()
	defer fs.mu.Unlock()
	for i, nf := range files {
		f := &storedFile{ID: "file-" + reqID(), Filename: nf.filename, Purpose: nf.purpose, Bytes: nf.data, CreatedAt: now}
		fs.files[f.ID] = f
		fs.order = append(fs.order, f.ID)
		fs.bytes += int64(len(nf.data))
		out[i] = f
	}
	for len(fs.order) > len(files) && fs.overLimit() {
		oldest := fs.order[0]
		fs.bytes -= int64(len(fs.files[oldest].Bytes))
		delete(fs.files, oldest)
		fs.order = fs.order[1:]
	}
	return out
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
