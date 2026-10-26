//go:build realckpt

package whisper

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/internal/audio"
	"github.com/townsendmerino/goinfer/multimodal"
)

// TestDecoderSpeed_record prints one timed greedy decode on a held encoder output, for G-S14t3's interleaved old/new record (docs/tasks/task-multimodal-support-2026-10.md, S14.6). It is a record, not a gate:
// it asserts nothing, and skips unless asked. Each process times once after GOINFER_SPEED_WARM warm-up runs, so two builds' test binaries can be run alternately.
//
//	GOINFER_SPEED_WAV=<16-bit wav> [GOINFER_WHISPER_DIR=<checkpoint>] [GOINFER_SPEED_WARM=1] go test -tags realckpt ./internal/whisper/ -run TestDecoderSpeed_record -v
func TestDecoderSpeed_record(t *testing.T) {
	wav := os.Getenv("GOINFER_SPEED_WAV")
	if wav == "" {
		t.Skip("set GOINFER_SPEED_WAV")
	}
	dir := os.Getenv("GOINFER_WHISPER_DIR")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, "models", "whisper-small")
	}
	tr, err := LoadTranscriber(dir)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(wav)
	if err != nil {
		t.Fatal(err)
	}
	samples, err := multimodal.DecodeWAVAnyRate(raw)
	if err != nil {
		t.Fatal(err)
	}
	feats, _, err := audio.WhisperFeatures(samples[:min(len(samples), MaxSamples)], tr.Mels)
	if err != nil {
		t.Fatal(err)
	}
	enc, err := tr.Enc.Forward(feats)
	if err != nil {
		t.Fatal(err)
	}
	warm := 0
	if v := os.Getenv("GOINFER_SPEED_WARM"); v != "" {
		for _, c := range v {
			warm = warm*10 + int(c-'0')
		}
	}
	for range warm {
		if _, err := tr.Dec.Generate(enc, tr.Gen, "en", "transcribe"); err != nil {
			t.Fatal(err)
		}
	}
	t0 := time.Now()
	res, err := tr.Dec.Generate(enc, tr.Gen, "en", "transcribe")
	if err != nil {
		t.Fatal(err)
	}
	d := time.Since(t0)
	t.Logf("SPEED whisper_decoder_s=%.3f tokens=%d ids=%v", d.Seconds(), len(res.IDs), res.IDs)
}
