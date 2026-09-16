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

// TestGemma3PromptBlock is M-38's gate (docs/audit-2026-09-10.md): Gemma 3's own processor
// (processing_gemma3.py, verified against the real transformers source 2026-09-16) wraps the
// image sequence in "\n\n" on BOTH sides — f"\n\n{boi_token}{image_tokens}{eoi_token}\n\n" — a
// shape internal/serveapp and demo/agent used to get wrong (a bare trailing "\n", or nothing).
func TestGemma3PromptBlock(t *testing.T) {
	inner := Gemma3ImageBlock(4)
	got := Gemma3PromptBlock(4)
	want := "\n\n" + inner + "\n\n"
	if got != want {
		t.Errorf("Gemma3PromptBlock(4) = %q, want %q", got, want)
	}
	if !strings.HasPrefix(got, "\n\n") {
		t.Errorf("Gemma3PromptBlock(4) does not lead with \\n\\n: %q", got)
	}
	if !strings.HasSuffix(got, "\n\n") {
		t.Errorf("Gemma3PromptBlock(4) does not trail with \\n\\n: %q", got)
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
