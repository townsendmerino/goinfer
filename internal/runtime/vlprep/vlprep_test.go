package vlprep

import (
	"slices"
	"testing"

	darch "github.com/townsendmerino/goinfer/decoder/arch"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
)

func TestSpansSpliceAndClaims(t *testing.T) {
	spans := []ImageSpan{{Pos: 1, Len: 2, Hash: 11}, {Pos: 4, Len: 1, Hash: 22}}
	feats := []float32{10, 11, 20, 21, 30, 31}
	if err := CheckSpans(spans, 6, feats, 2); err != nil {
		t.Fatalf("CheckSpans: %v", err)
	}
	if TotalTokens(spans) != 3 {
		t.Fatalf("TotalTokens = %d, want 3", TotalTokens(spans))
	}
	if r, ok := SpanRow(spans, 4); !ok || r != 2 {
		t.Fatalf("SpanRow(4) = %d, %v, want 2, true", r, ok)
	}
	rows, err := SplicedRows(func(ids []int) []float32 {
		out := make([]float32, len(ids)*2)
		for i, id := range ids {
			out[i*2], out[i*2+1] = float32(id), float32(id)
		}
		return out
	}, []int{0, 9, 9, 3, 9, 5}, spans, feats, 2)
	if err != nil || !slices.Equal(rows[1], []float32{10, 11}) || !slices.Equal(rows[3], []float32{3, 3}) || !slices.Equal(rows[4], []float32{30, 31}) {
		t.Fatalf("SplicedRows = %v, %v", rows, err)
	}
	sub := []float32{-1, -1, -1, -1}
	SpliceRange(sub, spans, feats, 2, 2, 4)
	if !slices.Equal(sub, []float32{20, 21, -1, -1}) {
		t.Fatalf("SpliceRange = %v", sub)
	}
	if blks := ResidentBlocks(spans); len(blks) != 2 || blks[0].End != 3 || blks[1].Hash != 22 {
		t.Fatalf("ResidentBlocks = %+v", blks)
	}
	if cl := ResidentClaims(spans); len(cl) != 2 || cl[0].Len != 2 || cl[1].Start != 4 {
		t.Fatalf("ResidentClaims = %+v", cl)
	}
}

func TestGuardsAndGemma4Rows(t *testing.T) {
	if err := CheckImageBlockFitsWindow(4, 5); err == nil {
		t.Fatal("expected sliding-window refusal")
	}
	if err := CheckDeepstack(2, 4, [][]float32{make([]float32, 8)}, 2); err != nil {
		t.Fatalf("CheckDeepstack: %v", err)
	}
	arch := &darch.Architecture{
		Name:       "gemma4",
		HiddenDim:  2,
		EmbedScale: 4,
		Gemma4:     &archparts.Gemma4Params{PadTokenID: 7},
	}
	ple := func(row []float32, tokenID int, dst []float32) {
		dst[0] = row[0] + float32(tokenID)
	}
	rows, err := Gemma4EModelImageRows(arch, 3, ple, func(id int) []float32 {
		return []float32{float32(id), 0, -1}
	}, []int{5, 99}, []float32{2, 3}, []ImageSpan{{Pos: 1, Len: 1}}, 0)
	if err != nil || !slices.Equal(rows[0], []float32{5, 0, -1}) || !slices.Equal(rows[1], []float32{2, 3, 9}) {
		t.Fatalf("Gemma4EModelImageRows = %v, %v", rows, err)
	}
}
