package batchplan

import (
	"fmt"
	"math"
)

// AttnTempScale returns the post-RoPE query temperature scale at absolute position pos (1 + beta*log1p(floor(pos/origMaxPos))).
// When beta == 0, it returns 1 without dividing by origMaxPos.
func AttnTempScale(pos int, beta, origMaxPos float64) float32 {
	if beta == 0 {
		return 1
	}
	return float32(1 + beta*math.Log1p(math.Floor(float64(pos)/origMaxPos)))
}

// FillAttnTempRows writes AttnTempScale(startPos+m, beta, origMaxPos) into dst[m] for each m in dst.
func FillAttnTempRows(dst []float32, startPos int, beta, origMaxPos float64) {
	for m := range dst {
		dst[m] = AttnTempScale(startPos+m, beta, origMaxPos)
	}
}

// StepRow describes one row of a multi-sequence step for host-side validation.
type StepRow struct {
	Slot    int
	Pos     int
	EmbLen  int
	HasDraw bool
}

// CheckStepRows validates n step rows: each slot must be in [0, numSlots), repeated slots within one step must advance
// by consecutive positions (p + 1), each position must be in [0, ctxCap), each embedding length must equal
// hidden + pleInputs, and device draws require sampleOK.
func CheckStepRows(prefix string, numSlots, ctxCap, hidden, pleInputs int, sampleOK bool, n int, rowAt func(i int) StepRow) error {
	wantEmb := hidden + pleInputs
	last := make(map[int]int, n)
	for i := range n {
		q := rowAt(i)
		if q.Slot < 0 || q.Slot >= numSlots {
			return fmt.Errorf("%s: slot %d out of range", prefix, q.Slot)
		}
		if p, ok := last[q.Slot]; ok && q.Pos != p+1 {
			return fmt.Errorf("%s: slot %d repeated at position %d after %d (rows on one slot must be consecutive positions of one sequence)",
				prefix, q.Slot, q.Pos, p)
		}
		last[q.Slot] = q.Pos
		if q.Pos < 0 || q.Pos >= ctxCap {
			return fmt.Errorf("%s: position %d outside the resident context %d", prefix, q.Pos, ctxCap)
		}
		if q.EmbLen != wantEmb {
			return fmt.Errorf("%s: embedding length %d, want %d (hidden %d, and %d per-layer inputs on a Gemma 4 E-model)",
				prefix, q.EmbLen, wantEmb, hidden, pleInputs)
		}
		if q.HasDraw && !sampleOK {
			return fmt.Errorf("%s: a device draw on a resident that cannot sample on-device", prefix)
		}
	}
	return nil
}

// StepPieces partitions [0, totalRows) into contiguous pieces of width 1..maxWidth and calls fn(from, n, last) for each
// piece in order, stopping on the first non-nil error.
func StepPieces(totalRows, maxWidth int, fn func(from, n int, last bool) error) error {
	for from := 0; from < totalRows; {
		n := min(maxWidth, totalRows-from)
		last := from+n == totalRows
		if err := fn(from, n, last); err != nil {
			return err
		}
		from += n
	}
	return nil
}
