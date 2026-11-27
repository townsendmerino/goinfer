package towers

import (
	"testing"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/media"
)

func TestPlaceImageBlocks_inlinesAtOffsetsAndFallsBackToPrefix(t *testing.T) {
	imgs := []media.ImageRef{{At: 8}, {At: 14}}
	blocks := []string{"<IMGA>", "<IMGB>"}
	got := PlaceImageBlocks("Compare  with .", "Compare  with .", true, imgs, blocks)
	if got != "Compare <IMGA> with <IMGB>." {
		t.Fatalf("PlaceImageBlocks = %q", got)
	}
	fallback := PlaceImageBlocks("and?", "earlier", false, []media.ImageRef{{At: 3}}, []string{"<IMGA>"})
	if fallback != "<IMGA>and?" {
		t.Fatalf("fallback PlaceImageBlocks = %q", fallback)
	}
}

func TestImageSpans_pairsRunsAndChecksLengths(t *testing.T) {
	runs := [][2]int{{4, 3}, {10, 2}}
	preps := []ImagePrep{
		{N: 3, Hash: 11, Grid: [3]int{1, 2, 3}},
		{N: 2, Hash: 22, Grid: [3]int{1, 1, 2}},
	}
	spans, grids, err := ImageSpans(runs, preps, "qwen")
	if err != nil || len(spans) != 2 || len(grids) != 2 || spans[1].Pos != 10 || spans[1].Hash != 22 {
		t.Fatalf("ImageSpans = %+v, %+v, %v", spans, grids, err)
	}
	if _, _, err := ImageSpans([][2]int{{4, 5}}, preps[:1], "qwen"); err == nil {
		t.Fatal("mismatched run length must return an error")
	}
}

func TestPlanGridTower_respectsBackendAndQuant(t *testing.T) {
	p, err := PlanGridTower("Qwen3.5", []string{"metal"}, false, "metal", true)
	if err != nil || p.Device != "metal" || p.Where != "Metal" {
		t.Fatalf("PlanGridTower(metal) = %+v, %v", p, err)
	}
	if _, err := PlanGridTower("Qwen3.5", []string{"metal"}, true, "metal", true); err == nil {
		t.Fatal("int8 under -require-backend must fail")
	}
}
