//go:build darwin

package metal

import (
	"context"

	"github.com/townsendmerino/goinfer/decoder"
)

// rowSource is a suffix's embeddings read a range of rows at a time (decoder.ResidentPrefillRows, MEM6): n rows, rows(lo, hi)
// returning rows [lo, hi), valid until the next call, and width the length every row has (-1 when the rows disagree).
type rowSource struct {
	n, width int
	rows     func(lo, hi int) [][]float32
}

// rowChunk is the rows a route reads from a source at once where it has no chunk of its own (the batched pass's input fill).
const rowChunk = 256

// sliceRows is the source of embeddings already held whole.
func sliceRows(embs [][]float32) rowSource {
	w := -1
	if len(embs) > 0 {
		w = len(embs[0])
		for _, e := range embs {
			if len(e) != w {
				w = -1
				break
			}
		}
	}
	return rowSource{n: len(embs), width: w, rows: func(lo, hi int) [][]float32 { return embs[lo:hi] }}
}

// PrefillLastRows (decoder.ResidentPrefillRows) is PrefillLast reading the suffix a chunk at a time from rows: every route takes
// its rows per chunk (the layer-major routes per 256 rows, the step per piece, the batched pass per rowChunk while it fills its
// input buffers), so the host holds a chunk of embeddings, not the suffix.
func (a *metalResident) PrefillLastRows(ctx context.Context, n, startPos int, rows func(lo, hi int) [][]float32) ([]float32, error) {
	src := rowSource{n: n, width: -1, rows: rows}
	if n > 0 {
		src.width = len(rows(0, 1)[0]) // the decoder's rows all have ResidentEmbedLen values
	}
	return a.prefillLastSrc(ctx, src, startPos)
}

var _ decoder.ResidentPrefillRows = (*metalResident)(nil)
