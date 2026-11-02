//go:build !darwin && !linux

package diskfree

// Free has no portable probe on this platform, so ok is always false and a caller proceeds, never refusing on a guess.
func Free(dir string) (free int64, ok bool) {
	return 0, false
}
