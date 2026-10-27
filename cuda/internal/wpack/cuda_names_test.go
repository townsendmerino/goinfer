package wpack

import "github.com/townsendmerino/aikit/linalg"

// The names package cuda gives the packers, over wpack, so the tests moved from there read as they did. hostW holds only a
// packed weight's fields: the moved tests do not plan.
type hostW struct {
	kind string
	wpk  []uint32
	ws   []float32
	ws16 []uint16
	N, K int
}

func hostWOf(p Packed) hostW {
	return hostW{kind: p.Kind, wpk: p.Wpk, ws: p.Ws, ws16: p.Ws16, N: p.N, K: p.K}
}

func packWeight(w *linalg.WeightMat) (hostW, error) {
	p, err := Pack(w)
	return hostWOf(p), err
}

func packWeightStack(ws ...*linalg.WeightMat) (hostW, error) {
	p, err := PackStack(nil, ws...)
	return hostWOf(p), err
}

func f32tof16(f float32) uint16 { return F32ToF16(f) }
