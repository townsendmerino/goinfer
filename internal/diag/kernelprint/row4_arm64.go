package kernelprint

import "github.com/townsendmerino/aikit/linalg"

// row4Emit is the .giw writer's row4 repack (decoder's w4a8_row4_emit_arm64.go), which exists on arm64 only.
func row4Emit(q4 []byte, q4s []uint16, rows, cols, group int) ([]byte, []uint16) {
	return linalg.RepackW4A8Row4(q4, rows, cols, group), linalg.RepackW4A8Row4ScalesF16(q4s, rows, cols, group)
}

func init() {
	suite = append(suite, kernel{"linalg.RepackW4A8Row4", func(s *sink, g *gen) {
		for _, rc := range [][2]int{{4, 32}, {8, 64}, {12, 256}} {
			q4, q4s := int4F16(g.mat(rc[0], rc[1], 1), rc[0], rc[1])
			r4, r4s := row4Emit(q4, q4s, rc[0], rc[1], 32)
			s.tag("r", rc[0], rc[1])
			s.bytes(r4)
			s.u16(r4s)
		}
	}})
}
