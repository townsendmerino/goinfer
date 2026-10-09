package decoder

import "fmt"

// ImageSpan is one image's placeholder run in a prompt (S11, docs/tasks/task-multimodal-support-2026-10.md): ids[Pos,
// Pos+Len) are replaced by that image's projected features. Hash identifies the image for resident prefix reuse (P9a);
// 0 makes no claim and never matches (M-06).
//
// A prompt's spans are in prompt order and do not overlap. Each is its own block: two images are never one block, even
// when they are adjacent and the same size (docs/multimodal.md, "Do not pair images").
type ImageSpan struct {
	Pos, Len int
	Hash     uint64
}

// imageSpansTotal is the number of placeholder positions across spans: the rows the concatenated features hold.
func imageSpansTotal(spans []ImageSpan) int {
	n := 0
	for _, s := range spans {
		n += s.Len
	}
	return n
}

// checkImageSpans validates spans against a prompt of n tokens and their concatenated features (each span's rows in
// span order, hidden values a row): at least one span, each non-empty and in range, in order and disjoint.
func checkImageSpans(spans []ImageSpan, n int, feats []float32, hidden int) error {
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
	if want := imageSpansTotal(spans) * hidden; len(feats) != want {
		return fmt.Errorf("decoder: image features have %d values, want %d (%d image tokens × %d)", len(feats), want, imageSpansTotal(spans), hidden)
	}
	return nil
}

// spliceImageSpans writes each span's rows of feats (concatenated in span order) over h's placeholder rows: the raw
// projected features with no embed scale, matching HF's masked_scatter.
func spliceImageSpans(h []float32, spans []ImageSpan, feats []float32, hidden int) {
	off := 0
	for _, s := range spans {
		copy(h[s.Pos*hidden:(s.Pos+s.Len)*hidden], feats[off:off+s.Len*hidden])
		off += s.Len * hidden
	}
}

// imageSpanBlocks is spans as [start, end) blocks, the form KVCache.SetImageBlocks and the resident prefills take.
func imageSpanBlocks(spans []ImageSpan) [][2]int {
	b := make([][2]int, len(spans))
	for i, s := range spans {
		b[i] = [2]int{s.Pos, s.Pos + s.Len}
	}
	return b
}

// residentImageBlocksOf is spans as the resident reuse bookkeeping's blocks (residentCommitIDs).
func residentImageBlocksOf(spans []ImageSpan) []residentImageBlock {
	b := make([]residentImageBlock, len(spans))
	for i, s := range spans {
		b[i] = residentImageBlock{start: s.Pos, end: s.Pos + s.Len, hash: s.Hash}
	}
	return b
}

// residentImageClaimsOf is spans as residentReuseLen's claims.
func residentImageClaimsOf(spans []ImageSpan) []residentImageClaim {
	c := make([]residentImageClaim, len(spans))
	for i, s := range spans {
		c[i] = residentImageClaim{Start: s.Pos, Len: s.Len, Hash: s.Hash}
	}
	return c
}

// imageSpanRow is the row of the concatenated features that position pos takes: its index across spans, and whether pos
// is in any span at all.
func imageSpanRow(spans []ImageSpan, pos int) (int, bool) {
	off := 0
	for _, s := range spans {
		if pos >= s.Pos && pos < s.Pos+s.Len {
			return off + pos - s.Pos, true
		}
		off += s.Len
	}
	return 0, false
}
