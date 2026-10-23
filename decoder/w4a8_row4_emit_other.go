//go:build !arm64

package decoder

// repackRow4ForEmit reports ok=false off arm64: the split-half + 4-row layout is NEON-specific and aikit's repack
// functions exist only there, so the .giw writer writes kind 3 for every int4 tensor. Kind 5 needs cmd/prequant run on
// an arm64 box.
func repackRow4ForEmit(q4 []byte, q4s []uint16, rows, cols, group int) (row4 []byte, row4Scales []uint16, ok bool) {
	return nil, nil, false
}

// row4EmitEligible is false off arm64: the writer writes kind 3.
func row4EmitEligible(rows, cols, group int) bool { return false }

func repackRow4ScalesForEmit(q4s []uint16, rows, cols, group int) []uint16 { return q4s }

func repackRow4CodesForEmit(q4 []byte, rows, cols, group int) []byte { return q4 }
