//go:build !darwin && !linux

package probe

// FaultCounts has no portable probe on this platform — every unknown proceeds, as with every
// reading here.
func FaultCounts() (minflt, majflt int64, ok bool) {
	return 0, 0, false
}
