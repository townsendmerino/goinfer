package decoder

import "github.com/townsendmerino/aikit/linalg"

// wmBytes is one weight matrix's in-memory footprint, read from the backing slices, not computed from Rows()*Cols(): a
// quantized matrix carries per-group scales beside its packed payload and int4 packs two values per byte, so a
// dimension-derived figure is wrong both ways. Asking the matrix what it holds cannot drift when a new kind is added.
func wmBytes(w *linalg.WeightMat) int64 {
	var n int64
	if q8, sc, _, ok := w.Int8(); ok {
		n += int64(len(q8)) + 4*int64(len(sc))
	}
	if q4, sc, _, ok := w.Int4F16(); ok {
		n += int64(len(q4)) + 2*int64(len(sc)) // int4 scales are binary16
	}
	if p4, sc, ok := w.Int4Row4F16(); ok {
		n += int64(len(p4)) + 2*int64(len(sc))
	}
	if f, ok := w.F32(); ok {
		n += 4 * int64(len(f))
	}
	if raw, ok := w.Q4K(); ok {
		n += int64(len(raw))
	}
	if u, _, ok := w.Half16(); ok {
		n += 2 * int64(len(u))
	}
	n += int64(w.SplitHalfBytes())
	return n
}
