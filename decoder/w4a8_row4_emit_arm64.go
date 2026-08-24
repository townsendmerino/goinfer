//go:build arm64

package decoder

import "github.com/townsendmerino/aikit/linalg"

// repackRow4ForEmit computes the on-disk split-half + 4-row-interleaved layout
// (docs/task-w4a8-neon-bandwidth.md's "Format follow-on") from canonical int4
// bytes, for the .giw writer's opt-in kind-4 path (decoder/serialize.go).
//
// Deliberately does NOT gate on the WRITING machine's DotProd support (unlike
// aikit's own RepackInt4Row4, whose in-RAM repack only benefits THIS process'
// own decode and so correctly declines on a core that can't use it) — a
// prequant tool may run on any arm64 box, and the bytes it emits are read back
// on whatever machine loads the .giw later, which may have DotProd when the
// writer doesn't or vice versa. The shape checks are the only real eligibility
// question here; the READ side (linalg.WrapInt4Row4) is where the DotProd gate
// belongs, since that runs on the machine that will actually dispatch to the
// kernel — see aikit's row4Usable().
//
// ok=false for a shape RepackW4A8Row4/RepackW4A8Row4Scales would reject (same
// three checks their own WeightMat method uses, minus the CPU-feature one):
// group != 32, rows not a multiple of 4, or cols not a multiple of group. The
// caller falls back to kind 3 for such a tensor (the router, or any int4
// tensor whose shape doesn't qualify).
func repackRow4ForEmit(q4 []byte, q4s []float32, rows, cols, group int) (row4 []byte, row4Scales []float32, ok bool) {
	if group != 32 || rows%4 != 0 || cols%group != 0 {
		return nil, nil, false
	}
	return linalg.RepackW4A8Row4(q4, rows, cols, group), linalg.RepackW4A8Row4Scales(q4s, rows, cols, group), true
}
