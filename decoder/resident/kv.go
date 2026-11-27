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

// KVPoolDetail is a resident's shared KV block pool as the load banner names it: the context positions it serves across every slot, the positions each slot is guaranteed (one conversation's cap,
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

// KVSnapshotStats summarizes a model's host prefix snapshots and demoted copies, for /admin/status and request logs.
type KVSnapshotStats struct {
	Budget         int64            `json:"budget_bytes"`
	Used           int64            `json:"used_bytes"`
	Held           int              `json:"held"`
	Taken          int              `json:"taken"`
	Restored       int              `json:"restored"`
	RestoredRows   int64            `json:"restored_rows"`
	Demoted        int              `json:"demoted"`
	DemoteDropped  int              `json:"demote_dropped"`
	Promoted       int              `json:"promoted"`
	PromotedBlocks int64            `json:"promoted_blocks"`
	Evicted        int              `json:"evicted"`
	EvictedUnused  int              `json:"evicted_unused"`
	Unqualified    int              `json:"unqualified"`
	TakeNanos      int64            `json:"take_ns"`
	RestoreNanos   int64            `json:"restore_ns"`
	Declined       map[string]int   `json:"declined"`
	LastDecline    string           `json:"last_decline,omitempty"`
	Snapshots      []KVSnapshotInfo `json:"snapshots,omitempty"`
}

// KVSnapshotInfo describes one held host prefix snapshot.
type KVSnapshotInfo struct {
	Rows     int   `json:"rows"`
	Bytes    int64 `json:"bytes"`
	Restores int   `json:"restores"`
	AgeSecs  int64 `json:"age_s"`
}
