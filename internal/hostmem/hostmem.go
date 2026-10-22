// Package hostmem is goinfer's one reading of this machine's available memory: what a new load can take without pushing other
// processes into swap, with reclaimable memory (cached files) counted as available. The fit guard reads it through
// decoder.HostRAMAvailableBytes, and the scripts' start checks (scripts/night.py, scripts/bench_peer.py, the job runners) read the
// same number through cmd/hostmem, so no reader keeps a second formula (docs/measurements/hostmem-available-2026-10/README.md).
package hostmem

// Reading is one sample, in bytes. On macOS Available is what the kernel can hand back without compressing or swapping anything: free,
// file-backed and purgeable pages, so Free + FileBacked + Purgeable. Speculative pages are part of FileBacked, reported apart as context,
// as are App, Wired and Compressed, Activity Monitor's terms. On Linux the macOS fields are 0 and Available is the kernel's MemAvailable.
type Reading struct {
	Physical, Available                      int64
	Free, Speculative, FileBacked, Purgeable int64
	App, Wired, Compressed                   int64
	Source                                   string
}

// TaskVM is another process's memory as task_info(TASK_VM_INFO) reports it, in bytes. Compressed is its pages in the compressor, held
// in memory or swapped to disk: macOS keeps no separate per-process disk-swap figure.
type TaskVM struct {
	Resident, Internal, Compressed, CompressedPeak, PhysFootprint int64
}

// ReadTask reads pid's TaskVM; ok is false where the platform has none or the query was refused.
func ReadTask(pid int) (TaskVM, bool) { return readTask(pid) }

// Read samples the reading now. ok is false where this platform has none or the query failed; callers treat that as "cannot
// judge", never as no memory. On macOS the sample can be up to a second old: XNU rate-limits host_statistics64 for binaries
// that are not Apple's, and once callers system-wide pass 2 to 10 requests in a second it serves the last fresh reply.
func Read() (r Reading, ok bool) { return read() }

// Available is Read's Available, or 0 when there is no reading.
func Available() int64 {
	r, ok := Read()
	if !ok {
		return 0
	}
	return r.Available
}

// OracleSample is a reading bracketed by the kernel's own count of available pages under CONFIG_JETSAM (pageable external + free
// pages, read through sysctl), read immediately before and after it in one process: B41 gate 1a's instrument
// (docs/measurements/hostmem-available-2026-10/README.md, amendment A3). JetsamBefore and JetsamAfter are bytes. Fresh is true
// when Ours was computed inside the bracket: the sysctls are not rate-limited and host_statistics64 is (see Read), so Ours is
// read twice, with a page fault between that a fresh reply counts, and Fresh is whether the two replies differ. A sample that is not
// Fresh may predate the bracket and cannot be graded.
type OracleSample struct {
	Ours                      Reading
	JetsamBefore, JetsamAfter int64
	Fresh                     bool
}

// ReadWithOracle takes one OracleSample; ok is false where the platform has no oracle.
func ReadWithOracle() (OracleSample, bool) { return readWithOracle() }
