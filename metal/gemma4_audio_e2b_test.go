//go:build darwin

package metal

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/audio"
	"github.com/townsendmerino/goinfer/multimodal"
)

// TestG4AMetal_E2BTower is G-S5d's tower half (docs/tasks/task-multimodal-support-2026-10.md, S5, registered before
// this ran): Gemma 4 E2B's audio tower (~/models/gemma-4-E2B-unq, never the archive) with its 12 conformer blocks on
// Metal (the g4aAccel EmbeddingGemma 2 uses), aikit's Subsample before and FinishBlocks after, against aikit's CPU Forward:
// every soft token at cosine >= 0.9999 (0.999-0.9999 parked), on the three EmbeddingGemma 2 clips and the LibriSpeech
// clip. Times are exploratory. Heavy: GOINFER_HEAVY_TESTS=1.
func TestG4AMetal_E2BTower(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, "models", "gemma-4-E2B-unq")
	if strings.HasPrefix(dir, "/Volumes/") || strings.HasPrefix(dir, "/srv/models") {
		t.Fatalf("%s is the archive", dir)
	}
	enc, err := audio.LoadGemma4AudioEncoder(dir)
	if err != nil {
		t.Skipf("no E2B tower: %v", err)
	}
	acc, err := newG4AAccel(enc)
	if err != nil {
		t.Fatal(err)
	}
	for _, clip := range []string{"embeddinggemma2-audio/short.wav", "embeddinggemma2-audio/mid.wav", "embeddinggemma2-audio/long.wav", "speech/librispeech-1272-128104-0000.wav"} {
		raw, err := os.ReadFile(filepath.Join("../testdata", clip))
		if err != nil {
			t.Fatal(err)
		}
		samples, err := multimodal.DecodeWAV(raw)
		if err != nil {
			t.Fatal(err)
		}
		mel, T, err := audio.Gemma4Features(samples)
		if err != nil {
			t.Fatal(err)
		}
		t0 := time.Now()
		want, err := enc.Forward(mel, T)
		tc := time.Since(t0)
		if err != nil {
			t.Fatal(err)
		}
		t0 = time.Now()
		h, n, err := enc.Subsample(mel, T)
		if err != nil {
			t.Fatal(err)
		}
		hb, err := acc.Blocks(h, n)
		if err != nil {
			t.Fatal(err)
		}
		got, err := enc.FinishBlocks(hb, n)
		tm := time.Since(t0)
		if err != nil {
			t.Fatal(err)
		}
		w := gvWorst(t, got, want, enc.TextHiddenSize)
		fmt.Fprintf(os.Stderr, "[G-S5d] %-42s %d soft tokens: worst cosine %.9f; tower Metal %s, CPU %s (exploratory)\n", clip, n, w, tm.Round(time.Millisecond), tc.Round(time.Millisecond))
		switch {
		case w < 0.999:
			t.Errorf("%s: worst cosine %.9f under 0.999 (G-S5d FAIL)", clip, w)
		case w < 0.9999:
			t.Errorf("%s: worst cosine %.9f in 0.999-0.9999 (G-S5d ambiguous, parked)", clip, w)
		}
	}
}
