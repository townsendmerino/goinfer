//go:build goinfer_testhooks

package program

import "testing"

// SetShapePlantForTest plants p in ShapeOf and MixerOf for the rest of tb, and restores the unplanted derivation when tb ends.
func SetShapePlantForTest(tb testing.TB, p ShapePlant) {
	tb.Helper()
	prev := shapePlant
	shapePlant = p
	tb.Cleanup(func() { shapePlant = prev })
}
