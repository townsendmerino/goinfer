//go:build !linux && !darwin

package decoder

// coldPolicyHere: on a system that is not measured, the load does nothing about the page cache.
var coldPolicyHere = coldLoadPolicy{}

func giwReadThrough(mapping []byte) {}

func giwAdviseReadAround(mapping []byte, on bool) {}
