//go:build !unix

package decoder

// madviseBytes is a no-op on platforms without madvise (notably Windows): the
// expert pager's bookkeeping still runs, but it can't release resident pages, so
// --stream-weights loads normally without an enforced RAM cap.
func madviseBytes([]byte, bool) error { return nil }
