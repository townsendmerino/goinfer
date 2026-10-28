//go:build gpu

package kernels

// Test helpers copied across the gpu / gpu/internal/kernels line: each is the same code as its original, kept
// on both sides because tests on both sides call it.

import ()

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}
