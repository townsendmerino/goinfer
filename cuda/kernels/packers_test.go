//go:build cuda

package kernels

import (
	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/cuda/internal/wpack"
)

// The names package cuda gives the packers, over wpack, so the tests moved from there read as they did. hostW holds only a
// packed weight's fields: the moved tests do not plan.
type hostW struct {
	kind string
	wpk  []uint32
	ws   []float32
	ws16 []uint16
	N, K int
}

func hostWOf(p wpack.Packed) hostW {
	return hostW{kind: p.Kind, wpk: p.Wpk, ws: p.Ws, ws16: p.Ws16, N: p.N, K: p.K}
}

func packWeight(w *linalg.WeightMat) (hostW, error) {
	p, err := wpack.Pack(w)
	return hostWOf(p), err
}

func packWeightStack(ws ...*linalg.WeightMat) (hostW, error) {
	p, err := wpack.PackStack(nil, ws...)
	return hostWOf(p), err
}

func packI8(q8 []int8, N, K int) []uint32 { return wpack.PackI8(q8, N, K) }
func nibblePosFast(i int) int             { return wpack.NibblePos(i) }
func f32tof16(f float32) uint16           { return wpack.F32ToF16(f) }
