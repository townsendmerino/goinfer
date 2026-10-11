//go:build !linux

package decoder

// coldLoadSupported: off Linux the load does nothing about the page cache. No other system's read-around is measured.
// A variable so a test can run the plan, with its probes and actions replaced, on any system.
var coldLoadSupported = false

func giwResidentFraction(mapping []byte) float64 { return 1 }

func giwReadThrough(mapping []byte) {}

func giwAdviseReadAround(mapping []byte, on bool) {}
