//go:build !darwin && !linux

package prequant

// statfsFreeBytes has no portable probe on this platform — every unknown proceeds
// (fitguard.go's own rule), so the disk-space guard in EnsureCachedGIW never refuses on a guess.
func statfsFreeBytes(dir string) (free int64, ok bool) {
	return 0, false
}
