package cpuengine

import "testing"

func TestCache_subStructsZeroValueAndPromotion(t *testing.T) {
	var c Cache
	if c.Pos() != 0 {
		t.Fatalf("zero-value Pos() = %d, want 0", c.Pos())
	}
	if c.HasRecurrentState() {
		t.Fatal("zero-value HasRecurrentState() = true, want false")
	}
	if !c.TruncateTo(0) {
		t.Fatal("zero-value TruncateTo(0) = false, want exact")
	}
	c.ResetRecurrent()
	c.ResetMultimodal()
	if c.RecurrentResets != 1 {
		t.Fatalf("RecurrentResets = %d, want 1", c.RecurrentResets)
	}

	c.SetImageBlocks([][2]int{{2, 6}})
	if got := c.CacheVision.attendHi(3); got != 5 {
		t.Fatalf("attendHi(3) = %d, want 5", got)
	}
	c.TruncateTo(0)
	if c.CacheVision.ImgBlocks != nil {
		t.Fatalf("TruncateTo(0) left ImgBlocks = %v, want nil", c.CacheVision.ImgBlocks)
	}

	if !c.prefillChunkable() {
		t.Fatal("clean cache prefillChunkable() = false, want true")
	}
	c.CacheDiag.SubCapture = true
	if c.prefillChunkable() {
		t.Fatal("SubCapture cache prefillChunkable() = true, want false")
	}
}
