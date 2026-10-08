package decoder

import "testing"

// TestAddDeepstack pins the injection's arithmetic (S10): set l is added only to the rows of the batch that fall in the
// image run, at the right offsets, whatever the batch's start position (a chunked or resumed prefill), and nowhere else.
func TestAddDeepstack(t *testing.T) {
	const hidden = 2
	ds := &deepstackRows{start: 3, n: 3, rows: [][]float32{{10, 11, 20, 21, 30, 31}}}
	for _, tc := range []struct {
		startPos, K int
		want        []float32
	}{
		{0, 8, []float32{0, 0, 0, 0, 0, 0, 10, 11, 20, 21, 30, 31, 0, 0, 0, 0}}, // the whole prompt in one batch
		{4, 3, []float32{20, 21, 30, 31, 0, 0}},                                 // a batch starting inside the run
		{0, 3, []float32{0, 0, 0, 0, 0, 0}},                                     // a batch ending before it
	} {
		h := make([]float32, tc.K*hidden)
		addDeepstack(h, ds, 0, tc.startPos, tc.K, hidden)
		for i := range h {
			if h[i] != tc.want[i] {
				t.Fatalf("startPos %d, K %d: h = %v, want %v", tc.startPos, tc.K, h, tc.want)
			}
		}
	}
}
