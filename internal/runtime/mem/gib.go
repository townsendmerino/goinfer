package mem

import "strconv"

// GiB formats a byte count in binary gigabytes, as "1.23 GiB": the one unit every fit, decline and banner message prints
// memory in, so two layers reporting the same free VRAM print the same figure. Disk and download sizes stay SI (GB,
// 10^9 bytes), the unit the files' own sizes are quoted in.
func GiB[N ~int | ~int64 | ~uint64](b N) string {
	return strconv.FormatFloat(float64(b)/(1<<30), 'f', 2, 64) + " GiB"
}
