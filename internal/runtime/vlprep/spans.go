package vlprep

import (
	"fmt"

	"github.com/townsendmerino/goinfer/internal/runtime/cpuengine"
	"github.com/townsendmerino/goinfer/internal/runtime/residency"
)

// ImageSpan is one image or audio placeholder run in a prompt: ids[Pos, Pos+Len)
// are replaced by that media item's projected features.
type ImageSpan = cpuengine.ImageSpan

// TotalTokens returns the total placeholder token count across spans.
func TotalTokens(spans []ImageSpan) int {
	n := 0
	for _, s := range spans {
		n += s.Len
	}
	return n
}

// CheckSpans validates spans against a prompt of n tokens and their concatenated features (hidden values per row):
// at least one span, each non-empty and in range, in prompt order, and disjoint.
func CheckSpans(spans []ImageSpan, n int, feats []float32, hidden int) error {
	if len(spans) == 0 {
		return fmt.Errorf("decoder: no image span")
	}
	end := 0
	for i, s := range spans {
		if s.Pos < end || s.Len <= 0 || s.Pos+s.Len > n {
			return fmt.Errorf("decoder: image span %d [%d,%d) is out of range, overlaps or is out of order (%d tokens; the previous span ends at %d)", i, s.Pos, s.Pos+s.Len, n, end)
		}
		end = s.Pos + s.Len
	}
	tot := TotalTokens(spans)
	if want := tot * hidden; len(feats) != want {
		return fmt.Errorf("decoder: image features have %d values, want %d (%d image tokens × %d)", len(feats), want, tot, hidden)
	}
	return nil
}

// Splice overwrites each span's placeholder rows in h with its raw projected features (no embed scale).
func Splice(h []float32, spans []ImageSpan, feats []float32, hidden int) {
	off := 0
	for _, s := range spans {
		copy(h[s.Pos*hidden:(s.Pos+s.Len)*hidden], feats[off:off+s.Len*hidden])
		off += s.Len * hidden
	}
}

// SpliceRange overwrites the rows of h (holding prompt positions [lo, hi)) that fall inside spans with their raw features.
func SpliceRange(h []float32, spans []ImageSpan, feats []float32, hidden, lo, hi int) {
	off := 0
	for _, s := range spans {
		a, b := max(s.Pos, lo), min(s.Pos+s.Len, hi)
		if a < b {
			copy(h[(a-lo)*hidden:(b-lo)*hidden], feats[off+(a-s.Pos)*hidden:off+(b-s.Pos)*hidden])
		}
		off += s.Len * hidden
	}
}

// SplicedRows validates spans, embeds ids via embedN, splices feats over the placeholder runs, and returns one row slice per token.
func SplicedRows(embedN func([]int) []float32, ids []int, spans []ImageSpan, feats []float32, hidden int) ([][]float32, error) {
	if err := CheckSpans(spans, len(ids), feats, hidden); err != nil {
		return nil, err
	}
	h := embedN(ids)
	Splice(h, spans, feats, hidden)
	rows := make([][]float32, len(ids))
	for i := range rows {
		rows[i] = h[i*hidden : (i+1)*hidden : (i+1)*hidden]
	}
	return rows, nil
}

// Blocks converts spans to [start, end) position intervals for KVCache.SetImageBlocks and resident span prefills.
func Blocks(spans []ImageSpan) [][2]int {
	b := make([][2]int, len(spans))
	for i, s := range spans {
		b[i] = [2]int{s.Pos, s.Pos + s.Len}
	}
	return b
}

// ResidentBlocks converts spans to residency.ImageBlock entries for resident KV commit tracking.
func ResidentBlocks(spans []ImageSpan) []residency.ImageBlock {
	b := make([]residency.ImageBlock, len(spans))
	for i, s := range spans {
		b[i] = residency.ImageBlock{Start: s.Pos, End: s.Pos + s.Len, Hash: s.Hash}
	}
	return b
}

// ResidentClaims converts spans to residency.ImageClaim entries for resident prefix reuse matching.
func ResidentClaims(spans []ImageSpan) []residency.ImageClaim {
	c := make([]residency.ImageClaim, len(spans))
	for i, s := range spans {
		c[i] = residency.ImageClaim{Start: s.Pos, Len: s.Len, Hash: s.Hash}
	}
	return c
}

// SpanRow returns the concatenated-feature row index for prompt position pos and whether pos lies inside any span.
func SpanRow(spans []ImageSpan, pos int) (int, bool) {
	off := 0
	for _, s := range spans {
		if pos >= s.Pos && pos < s.Pos+s.Len {
			return off + pos - s.Pos, true
		}
		off += s.Len
	}
	return 0, false
}
