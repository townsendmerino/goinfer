//go:build cuda

package kernels

import (
	"github.com/townsendmerino/goinfer/cuda/internal/launchgeom"
	"github.com/townsendmerino/goinfer/cuda/internal/wpack"
)

// The names package cuda gives the launch geometry and the KV f16 converters, so the tests moved from there read as
// they did.
const (
	faWarps       = launchgeom.FaWarps
	faMaxRows     = launchgeom.FaMaxRows
	attnFusedBM   = launchgeom.AttnFusedBM
	attnFusedKPAD = launchgeom.AttnFusedKPAD
)

type faRun = launchgeom.FaRun

func faRowsPerCTA(hd, g int) int                   { return launchgeom.FaRowsPerCTA(hd, g) }
func faRowRuns(n0, m, window, s int) []faRun       { return launchgeom.FaRowRuns(n0, m, window, s) }
func f32ToF16RNE(f float32) uint16                 { return wpack.F32ToF16RNE(f) }
func f32SliceToF16RNE(dst []uint16, src []float32) { wpack.F32SliceToF16RNE(dst, src) }
func imgBlocksMaxNWin(c, window int, b [][2]int) int {
	return launchgeom.ImgBlocksMaxNWin(c, window, b)
}
