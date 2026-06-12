package multimodal

import (
	"strings"
	"testing"
)

func TestGemma3ImageBlock(t *testing.T) {
	b := Gemma3ImageBlock(4)
	if !strings.HasPrefix(b, ImageBlockStart) || !strings.HasSuffix(b, ImageBlockEnd) {
		t.Errorf("block missing sentinels: %q", b)
	}
	if got := strings.Count(b, ImageSoftToken); got != 4 {
		t.Errorf("soft-token count = %d, want 4", got)
	}
}

func TestFindImageRun(t *testing.T) {
	pos, n := FindImageRun([]int{5, 5, 262144, 262144, 262144, 9}, 262144)
	if pos != 2 || n != 3 {
		t.Errorf("got (%d,%d), want (2,3)", pos, n)
	}
	if _, n := FindImageRun([]int{1, 2, 3}, 262144); n != 0 {
		t.Errorf("no run should give n=0, got %d", n)
	}
}
