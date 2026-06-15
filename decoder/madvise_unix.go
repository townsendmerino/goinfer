//go:build unix && !darwin

package decoder

import "syscall"

// madviseBytes advises the kernel on the residency of a page-aligned span of the
// read-only .giw mapping. willNeed (MADV_WILLNEED) hints to fault the span in;
// otherwise (MADV_DONTNEED) it releases the span's resident pages.
//
// Releasing is *safe regardless of timing*: the mapping is read-only and
// file-backed (MAP_PRIVATE, clean pages), so a released span simply re-faults
// from the file on the next access — the bytes are identical. That's what lets the
// expert pager cap resident RAM without any correctness risk (an evicted-then-
// reused expert costs a re-fault, never wrong data).
//
// On Linux and the BSDs, MADV_DONTNEED frees the resident pages immediately, so the
// RAM cap here is firm. darwin needs a different flag (MADV_DONTNEED is only a weak
// hint there) and lives in madvise_darwin.go; platforms without madvise fall to the
// no-op in madvise_other.go. b must be page-aligned (the caller aligns).
func madviseBytes(b []byte, willNeed bool) error {
	if len(b) == 0 {
		return nil
	}
	adv := syscall.MADV_DONTNEED
	if willNeed {
		adv = syscall.MADV_WILLNEED
	}
	return syscall.Madvise(b, adv)
}
