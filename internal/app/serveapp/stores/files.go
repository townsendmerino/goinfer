package stores

import (
	"sync"
	"time"
)

// StoredFile is OpenAI's Files API object: a batch's input JSONL, or an assembled output or error JSONL, held in
// memory keyed by ID. Write-once: nothing mutates its fields after Put/PutAll returns it, so Get hands back the live
// pointer directly with no snapshot copy.
type StoredFile struct {
	ID        string
	Filename  string
	Purpose   string // "batch" (input) | "batch_output" | "batch_error"
	Bytes     []byte
	CreatedAt time.Time
}

// NewFile is one file for PutAll: caller-supplied metadata (echoed back on GET, never interpreted) and contents.
type NewFile struct {
	Filename string
	Purpose  string
	Data     []byte
}

// FileStore is a bounded, in-memory, process-wide registry. Two bounds hold, whichever comes first: at most cap files,
// and at most maxBytes of their contents (either 0 for no such bound). The oldest file goes first, and the files of the
// newest PutAll are always kept, even over either bound, so a finished batch's output and error files stored together
// are never dropped as they are stored.
type FileStore struct {
	mu       sync.Mutex
	files    map[string]*StoredFile
	order    []string
	cap      int
	maxBytes int64
	bytes    int64
}

// NewFileStore constructs a FileStore bounded by cap files and maxBytes total bytes (0 means unbounded on that axis).
func NewFileStore(cap int, maxBytes int64) *FileStore {
	return &FileStore{files: map[string]*StoredFile{}, cap: cap, maxBytes: maxBytes}
}

// Put stores data under a freshly minted ID and returns the resulting StoredFile.
func (fs *FileStore) Put(filename, purpose string, data []byte) *StoredFile {
	return fs.PutAll(NewFile{Filename: filename, Purpose: purpose, Data: data})[0]
}

// PutAll stores files together, each under a freshly minted ID, and returns them in order. Eviction takes only older
// files, so none of these is dropped by the others' storing.
func (fs *FileStore) PutAll(files ...NewFile) []*StoredFile {
	out := make([]*StoredFile, len(files))
	now := time.Now()
	fs.mu.Lock()
	defer fs.mu.Unlock()
	for i, nf := range files {
		f := &StoredFile{ID: "file-" + MintID(), Filename: nf.Filename, Purpose: nf.Purpose, Bytes: nf.Data, CreatedAt: now}
		fs.files[f.ID] = f
		fs.order = append(fs.order, f.ID)
		fs.bytes += int64(len(nf.Data))
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

func (fs *FileStore) overLimit() bool {
	return (fs.cap > 0 && len(fs.order) > fs.cap) || (fs.maxBytes > 0 && fs.bytes > fs.maxBytes)
}

// Get returns id's StoredFile, or nil if unknown or evicted.
func (fs *FileStore) Get(id string) *StoredFile {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	return fs.files[id]
}
