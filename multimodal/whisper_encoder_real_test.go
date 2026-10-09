package multimodal

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/audio"
)

// G-S14d2 (small) and G-S14d3 (large-v3) of docs/tasks/task-multimodal-support-2026-10.md (registered before this code): a real Whisper checkpoint's encoder, goinfer float32 on the CPU, against
// transformers float32 (scripts/pin_whisper_real.py: the reference is proved real by transcribing the LibriSpeech clip) on the LibriSpeech clip and 25 s of synthetic signal. Bar: every one of
// the 1500 frames' cosine >= 0.9999; the largest absolute difference is recorded. The Go front end's features are run through the encoder as well and recorded beside.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_WHISPER_REF=<pin output dir> [GOINFER_WHISPER_DIR=<checkpoint>] go test ./multimodal/ -run TestWhisperEncoder_real -v -timeout 30m
func TestWhisperEncoder_real(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	ref := os.Getenv("GOINFER_WHISPER_REF")
	if ref == "" {
		t.Skip("set GOINFER_WHISPER_REF to scripts/pin_whisper_real.py's output directory")
	}
	dir := os.Getenv("GOINFER_WHISPER_DIR")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, "models", "whisper-small")
	}
	raw, err := os.ReadFile(filepath.Join(ref, "meta.json"))
	if err != nil {
		t.Fatal(err)
	}
	var meta struct {
		Mels  int
		Cases []struct{ Name string }
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
		t.Fatal(err)
	}
	t0 := time.Now()
	enc, err := audio.LoadWhisperEncoder(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	t.Logf("loaded %s in %.1fs: %+v", filepath.Base(dir), time.Since(t0).Seconds(), enc.Cfg)
	rd := func(name string) []float32 {
		b, err := os.ReadFile(filepath.Join(ref, name))
		if err != nil {
			t.Fatal(err)
		}
		return f32s(b)
	}
	for _, c := range meta.Cases {
		t1 := time.Now()
		got, err := enc.Forward(rd(c.Name + ".feats.f32"))
		if err != nil {
			t.Fatal(err)
		}
		el := time.Since(t1).Seconds()
		cos, ab := cosineRows(got, rd(c.Name+".enc.f32"), audio.WhisperFrames/2)
		t.Logf("(a) %-9s reference features: worst per-frame cosine %.9f, max |diff| %.2e; %.1fs", c.Name, cos, ab, el)
		if cos < 0.9999 {
			t.Errorf("%s: worst per-frame cosine %.9f, bar 0.9999", c.Name, cos)
		}
		f, _, err := audio.WhisperFeatures(rd(c.Name+".in.f32"), meta.Mels)
		if err != nil {
			t.Fatal(err)
		}
		got2, err := enc.Forward(f)
		if err != nil {
			t.Fatal(err)
		}
		c2, a2 := cosineRows(got2, rd(c.Name+".enc.f32"), audio.WhisperFrames/2)
		t.Logf("(b) %-9s Go front end:        worst per-frame cosine %.9f, max |diff| %.2e", c.Name, c2, a2)
	}
}
