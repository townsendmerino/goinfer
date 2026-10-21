//go:build !darwin && !linux

package hostmem

func read() (Reading, bool) { return Reading{}, false }
