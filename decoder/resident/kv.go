package resident

// KVSnapshot is one prefix's KV held in host memory by a resident. Release frees it; it is idempotent.
type KVSnapshot interface {
	// Rows is how many leading positions it holds, a whole number of blocks.
	Rows() int
	// Bytes is the host memory it occupies.
	Bytes() int64
	// Serves reports whether RestoreKV can restore its first end rows into a slot: false when a windowed layer's rows behind the
	// snapshot's end, which the window of position end needs, are not among those it holds, or when the resident's geometry has
	// changed since it was taken.
	Serves(end int) bool
	Release()
}

// KVPoolDetail is a resident's shared KV block pool as the load banner names it (docs/tasks/task-kv-architecture-2026-10.md
// §5.16): the context positions it serves across every slot, the positions each slot is guaranteed (one conversation's cap,
// Caps.ContextCap, may exceed it), the rows in a block, the layer groups that hold pools, the precision the rows are stored at, and the
// device bytes the pools and block tables take. WindowRows is how many rows a sliding-window layer keeps for one conversation, its
// newest (0 when no layer has a window group): such a layer serves every position of Positions while holding only these.
type KVPoolDetail struct {
	Positions, Guaranteed, BlockRows, WindowRows int
	Groups                                       []string
	Precision                                    string
	Bytes                                        int64
}

// KVHostBudget is the host byte budget a resident's demoted copies share with the prefix snapshots.
type KVHostBudget interface {
	// Fits reports whether n bytes would fit with held snapshots and copies bytes of demoted copies dropped: bytes reserved by anyone else stay.
	Fits(n, copies int64) bool
	// Reserve charges n bytes when they fit, dropping held snapshots to make them fit when dropSnapshots is set.
	Reserve(n int64, dropSnapshots bool) bool
	Release(n int64)
	Kept()    // a slot's rows were kept
	Dropped() // a kept copy was dropped for room
	Refused() // a slot's rows did not fit
}
