//go:build cuda

package cuda

import "testing"

// TestKSO1C_stageRowsFitTheScratch: a staging block of kquantStageRows rows asks packStage.bytes for no more than one stage's
// scratch, for Q5_K and Q6_K rows at every block count up to 64 (an odd count makes a Q6_K row 2 bytes short of a word) and at
// both the production and a small stage size, so the scratch never grows past stagePackWords.
func TestKSO1C_stageRowsFitTheScratch(t *testing.T) {
	saved := stagePackWords
	defer func() { stagePackWords = saved }()
	for _, words := range []int{saved, 997, 64} {
		stagePackWords = words
		for _, bb := range []int{176, 210} {
			for nSB := 1; nSB <= 64; nSB++ {
				rb := nSB * bb
				rows := kquantStageRows(rb)
				if (rb+3)/4 > words {
					continue // one row is wider than the stage: a block is one row, and the scratch grows to it by design
				}
				st := &packStage{}
				st.words(1)
				st.bytes(rows * rb)
				if cap(st.u32) != words {
					t.Errorf("stage %d words, %d-byte rows (NB=%d): a %d-row block grew the scratch to %d words", words, rb, nSB, rows, cap(st.u32))
				}
			}
		}
	}
}
