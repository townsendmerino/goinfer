//go:build !race

package decoder

// sweepSeedStride selects which seeds TestTopFilterLogits_MatchesReference exercises: stride 1 =
// every seed in the range, the FULL 24,018-case exactness sweep. This is the normal build, and it
// is where the exhaustive gate actually runs.
//
// See the //go:build race sibling for why the race build strides instead.
const (
	sweepSeedStride = 1
	sweepMode       = "full (no -race)"
)
