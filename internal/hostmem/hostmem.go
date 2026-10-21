// Package hostmem is goinfer's one reading of this machine's available memory: what a new load can take without pushing other
// processes into swap, with reclaimable memory (cached files) counted as available. The fit guard reads it through
// decoder.HostRAMAvailableBytes, and the scripts' start checks (scripts/night.py, scripts/bench_peer.py, the job runners) read the
// same number through cmd/hostmem, so no reader keeps a second formula (docs/measurements/hostmem-available-2026-10/README.md).
package hostmem

// Reading is one sample, in bytes. On macOS App, Wired and Compressed are Activity Monitor's Memory terms and Available is Physical
// less their sum; on Linux they are 0 and Available is the kernel's MemAvailable.
type Reading struct {
	Physical, App, Wired, Compressed, Available int64
	Source                                      string
}

// Read samples the reading now. ok is false where this platform has none or the query failed; callers treat that as "cannot
// judge", never as no memory.
func Read() (r Reading, ok bool) { return read() }

// Available is Read's Available, or 0 when there is no reading.
func Available() int64 {
	r, ok := Read()
	if !ok {
		return 0
	}
	return r.Available
}
