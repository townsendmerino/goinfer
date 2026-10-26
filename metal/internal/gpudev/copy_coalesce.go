//go:build darwin || linux

package gpudev

// coalesce merges runs of pairs that are contiguous in BOTH src and dst into
// single larger copies, preserving order. It never reorders and never merges
// across different underlying buffers.
//
// WHY: THE COST HERE IS DISPATCH COUNT, NOT BYTES. gocudrv submits every copy to
// a thread-locked executor goroutine and waits on a channel round trip
// (internal/executor's DoJobCtx), so each op carries a fixed ~9 µs that does not
// shrink with the copy. It shows up directly in the motivating consumer's numbers:
// one contiguous 20 MiB copy reaches ~347 GB/s (78% of a 2070 SUPER's 448 GB/s
// peak), while the same bytes as 36 separate copies reach ~174 GB/s (39%) —
// 36 × ~9 µs is ~324 µs of the ~446 µs total. Bytes moved are identical; only the
// op count differs.
//
// Merging is therefore worth more than any per-copy tuning, and it is the one
// lever available inside this package: the caller's buffer LAYOUT is theirs to
// choose (goinfer records making the windows contiguous as its own design note),
// but if the pairs a caller already hands us happen to be adjacent, collapsing
// them costs a linear scan and saves a channel round trip each.
//
// SAFETY. Two pairs merge only when both destinations and both sources are the
// same underlying gocudrv buffer AND each range begins exactly where the previous
// one ended. That makes the merged copy byte-for-byte the two originals, in the
// same order. Anything else — a gap, a different buffer, a reordering — is left
// alone, so a batch that coalesces to nothing behaves exactly as before at the
// cost of one pass. Overlap between src and dst is not this function's concern:
// it is neither created nor removed by merging, since the merged extents are the
// unions of ranges that were already going to be copied.
func coalesce(copies []DeviceCopy) []DeviceCopy {
	if len(copies) < 2 {
		return copies
	}
	out := make([]DeviceCopy, 0, len(copies))
	for _, c := range copies {
		if c.Bytes == 0 {
			continue
		}
		if n := len(out); n > 0 && adjacent(out[n-1], c) {
			out[n-1].Bytes += c.Bytes
			continue
		}
		out = append(out, c)
	}
	return out
}
