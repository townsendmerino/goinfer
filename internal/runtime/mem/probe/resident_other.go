//go:build !darwin && !linux

package probe

// ResidentFraction reports 1, warm, where there is no mincore: a cold-load plan then does nothing.
func ResidentFraction(mapping []byte) float64 { return 1 }
