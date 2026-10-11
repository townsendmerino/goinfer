//go:build arm64

package decoder

import "github.com/townsendmerino/aikit/linalg"

// repackRow4ForEmit computes the on-disk split-half + 4-row-interleaved layout from canonical int4 nibbles and their
// binary16 group scales, for the .giw writer's kind 5 (decoder/serialize.go).
//
// It does not gate on the WRITING machine's DotProd support, unlike aikit's RepackInt4Row4, whose in-RAM repack serves
// this process's own decode: a prequant tool may run on any arm64 box, and the bytes are read back on whatever machine
// loads the .giw. The READ side (linalg.WrapInt4Row4OnlyF16) is where the DotProd gate belongs.
//
// ok=false for a shape the repack rejects: group != 32, rows not a multiple of 4, or cols not a multiple of group. The
// caller then writes kind 3.
func repackRow4ForEmit(q4 []byte, q4s []uint16, rows, cols, group int) (row4 []byte, row4Scales []uint16, ok bool) {
	if group != 32 || rows%4 != 0 || cols%group != 0 {
		return nil, nil, false
	}
	return linalg.RepackW4A8Row4(q4, rows, cols, group), linalg.RepackW4A8Row4ScalesF16(q4s, rows, cols, group), true
}
