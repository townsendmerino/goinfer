//go:build darwin

package metal

// costPlantForTest plants SB11's Metal defects: 1 commits and waits on one more, empty command buffer in the token; 2 keeps
// the executor from re-encoding a buffer it encoded ahead with no counter attached (execLoop), so the token commits it
// unattributed.
var costPlantForTest int
