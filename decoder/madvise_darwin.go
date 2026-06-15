//go:build darwin

package decoder

import "golang.org/x/sys/unix"

// madviseBytes on darwin honors the willNeed (prefetch) hint but CANNOT enforce an
// on-demand RAM cap — unlike the Linux/BSD path in madvise_unix.go.
//
// The willNeed=true case issues MADV_WILLNEED, a real, working fault-ahead hint
// (syscall.Madvise doesn't exist on darwin, so this goes through x/sys/unix).
//
// The eviction case (willNeed=false) is a deliberate no-op. macOS keeps clean,
// file-backed pages of a read-only MAP_PRIVATE mapping in the Unified Buffer Cache
// and reclaims them only under memory pressure; there is no syscall that forces an
// immediate resident drop for this mapping type. Empirically (verified on
// darwin/arm64): MADV_DONTNEED and MADV_FREE leave RSS unchanged, MADV_FREE_REUSABLE
// returns EPERM (it is a malloc-zone flag, illegal on a file-backed mapping), and the
// msync MS_INVALIDATE/MS_KILLPAGES/MS_DEACTIVATE variants are no-ops on RSS too.
//
// So on darwin the expert pager's bookkeeping still runs and --stream-weights loads
// normally; the resident pages are clean and therefore freely evictable by the OS
// when RAM is tight (no writeback, no OOM risk), but goinfer does NOT claim a firm,
// self-enforced cap here. That guarantee is Linux/BSD-only. Correctness is unaffected
// regardless: the mapping is read-only and file-backed, so any page the OS reclaims
// simply re-faults identical bytes.
func madviseBytes(b []byte, willNeed bool) error {
	if len(b) == 0 || !willNeed {
		return nil // no-op eviction: see above — no darwin syscall forces a resident drop here
	}
	return unix.Madvise(b, unix.MADV_WILLNEED)
}
