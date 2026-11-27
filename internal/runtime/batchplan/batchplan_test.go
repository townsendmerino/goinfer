package batchplan

import (
	"math"
	"slices"
	"testing"

	"github.com/townsendmerino/goinfer/decoder/resident"
)

func TestRaggedOrder_AndDistinctSlots(t *testing.T) {
	rows := func(n int) [][]float32 { return make([][]float32, n) }
	chunk := func(slot, pos, n, prompt int, last bool) resident.RaggedSeq {
		return resident.RaggedSeq{
			Slot: slot, Pos: pos, Embs: rows(n),
			Prefill: &resident.RaggedPrefill{PromptLen: prompt, Last: last},
		}
	}
	dec := func(slot int) resident.RaggedSeq {
		return resident.RaggedSeq{Slot: slot, Pos: 5, Embs: rows(1)}
	}

	got, err := RaggedOrder("ragged", []resident.RaggedSeq{dec(0), dec(1), chunk(2, 32, 32, 64, true), dec(3)})
	if err != nil || !slices.Equal(got, []int{2, 0, 1, 3}) {
		t.Fatalf("RaggedOrder = (%v, %v), want [2 0 1 3]", got, err)
	}
	if err := CheckDistinctSlots("ragged", 4, []resident.RaggedSeq{dec(0), dec(1), chunk(2, 32, 32, 64, true), dec(3)}, got); err != nil {
		t.Fatalf("CheckDistinctSlots unexpected error: %v", err)
	}
	if err := CheckDistinctSlots("ragged", 4, []resident.RaggedSeq{dec(1), dec(1)}, []int{0, 1}); err == nil {
		t.Fatal("CheckDistinctSlots accepted duplicate slot")
	}

	drawn := chunk(0, 0, 32, 64, false)
	drawn.Draw = &resident.BatchDraw{Temperature: 1}
	for _, bad := range [][]resident.RaggedSeq{
		{chunk(0, 0, 32, 64, false), chunk(1, 32, 32, 64, true)},
		{drawn, dec(1)},
		{chunk(0, 48, 32, 64, true)},
		{chunk(0, 0, 32, 64, true)},
		{chunk(0, 32, 32, 64, false)},
	} {
		if _, err := RaggedOrder("ragged", bad); err == nil {
			t.Fatalf("RaggedOrder accepted invalid input: %+v", bad)
		}
	}
}

func TestAttnTempScale_AndStepHelpers(t *testing.T) {
	if s := AttnTempScale(4096, 0, 0); s != 1 || math.IsNaN(float64(s)) {
		t.Fatalf("AttnTempScale(beta=0, orig=0) = %v, want 1", s)
	}
	want := float32(1 + 0.1*math.Log1p(2))
	if got := AttnTempScale(2048, 0.1, 1024); got != want {
		t.Fatalf("AttnTempScale = %v, want %v", got, want)
	}
	buf := make([]float32, 2)
	FillAttnTempRows(buf, 1023, 0.1, 1024)
	if buf[0] != 1 || buf[1] != float32(1+0.1*math.Log1p(1)) {
		t.Fatalf("FillAttnTempRows = %v", buf)
	}

	valid := []StepRow{
		{Slot: 0, Pos: 10, EmbLen: 128},
		{Slot: 0, Pos: 11, EmbLen: 128},
		{Slot: 1, Pos: 3, EmbLen: 128, HasDraw: true},
	}
	if err := CheckStepRows("step", 2, 64, 96, 32, true, len(valid), func(i int) StepRow { return valid[i] }); err != nil {
		t.Fatalf("CheckStepRows(valid) = %v", err)
	}
	nonConsec := []StepRow{{Slot: 0, Pos: 10, EmbLen: 128}, {Slot: 0, Pos: 12, EmbLen: 128}}
	if err := CheckStepRows("step", 2, 64, 128, 0, true, len(nonConsec), func(i int) StepRow { return nonConsec[i] }); err == nil {
		t.Fatal("CheckStepRows accepted non-consecutive positions on same slot")
	}

	var spans [][3]int
	err := StepPieces(19, 8, func(from, n int, last bool) error {
		l := 0
		if last {
			l = 1
		}
		spans = append(spans, [3]int{from, n, l})
		return nil
	})
	if err != nil || !slices.Equal(spans, [][3]int{{0, 8, 0}, {8, 8, 0}, {16, 3, 1}}) {
		t.Fatalf("StepPieces = (%v, %v)", spans, err)
	}
}
