package serveapp

import (
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/multimodal"
)

// S10 (docs/tasks/task-multimodal-support-2026-10.md, "S10, Ministral 3 (Pixtral)"): a Pixtral image fills one [IMG]
// run per merged row, the [IMG_BREAK]s between them text, and becomes one span per row with the image's hash.
func TestImageSpans_pixtralRows(t *testing.T) {
	const img, brk, end = 10, 12, 13
	// Text, a 2x3 image (two rows of three), text, a 1x4 image (one row, no break), text.
	ids := []int{1, 2, img, img, img, brk, img, img, img, end, 5, img, img, img, img, end, 6}
	preps := []imagePrep{{n: 6, runs: 2, runLen: 3, hash: 7}, {n: 4, runs: 1, runLen: 4, hash: 8}}
	spans, _, err := imageSpans(multimodal.FindImageRuns(ids, img), preps, "pixtral")
	if err != nil {
		t.Fatal(err)
	}
	want := []decoder.ImageSpan{{Pos: 2, Len: 3, Hash: 7}, {Pos: 6, Len: 3, Hash: 7}, {Pos: 11, Len: 4, Hash: 8}}
	if !slices.Equal(spans, want) {
		t.Errorf("spans %v, want %v", spans, want)
	}
	total := 0
	for _, s := range spans {
		total += s.Len
	}
	if total != 10 {
		t.Errorf("spans cover %d positions, want the 10 feature rows", total)
	}

	// Planted: a row one token short, and the rows counted as one image-wide run, must each refuse.
	short := slices.Clone(ids)
	short[8] = brk
	if _, _, err := imageSpans(multimodal.FindImageRuns(short, img), preps, "pixtral"); err == nil {
		t.Error("a short row was accepted")
	}
	oneRun := []imagePrep{{n: 6, hash: 7}, preps[1]}
	if _, _, err := imageSpans(multimodal.FindImageRuns(ids, img), oneRun, "pixtral"); err == nil {
		t.Error("an image whose rows are broken by [IMG_BREAK] was accepted as one run")
	}
}
