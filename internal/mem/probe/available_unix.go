//go:build darwin || linux

package probe

// readAvailable is Read's Available, or 0 when there is no reading.
func readAvailable() int64 {
	r, ok := Read()
	if !ok {
		return 0
	}
	return r.Available
}
