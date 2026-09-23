//go:build !darwin && !linux

package decoder

// processFaultCounts has no portable probe on this platform — every unknown proceeds, same as
// this file's siblings (hostram_other.go, memwatch_other.go).
func processFaultCounts() (minflt, majflt int64, ok bool) {
	return 0, 0, false
}
