package audio

import (
	"math"
	"os"
	"strconv"
	"testing"
	"time"
)

// TestEncoderSpeed_record prints one timed run of an audio encoder on a fixed synthetic window, for the interleaved old/new records of
// docs/tasks/task-multimodal-support-2026-10.md S14.5 (G-S14s3). It runs two builds' test binaries alternately, so each process times once.
// It is a record, not a gate: it asserts nothing and skips unless asked.
//
//	AIKIT_SPEED_WHISPER=<whisper checkpoint dir> and/or AIKIT_SPEED_VOXTRAL=<voxtral checkpoint dir> [AIKIT_SPEED_WARM=1] go test -run TestEncoderSpeed_record -v ./audio/
func TestEncoderSpeed_record(t *testing.T) {
	wdir, vdir := os.Getenv("AIKIT_SPEED_WHISPER"), os.Getenv("AIKIT_SPEED_VOXTRAL")
	if wdir == "" && vdir == "" {
		t.Skip("set AIKIT_SPEED_WHISPER and/or AIKIT_SPEED_VOXTRAL")
	}
	warm := 0 // warm-up runs before the timed one; unset or unparsable is none
	if v, err := strconv.Atoi(os.Getenv("AIKIT_SPEED_WARM")); err == nil {
		warm = v
	}
	samples := make([]float32, 10*WhisperSampleRate)
	for i := range samples {
		x := float64(i) / WhisperSampleRate
		samples[i] = float32(0.3*math.Sin(2*math.Pi*220*x) + 0.2*math.Sin(2*math.Pi*1370*x+math.Sin(3*x)))
	}
	if wdir != "" {
		enc, err := LoadWhisperEncoder(wdir)
		if err != nil {
			t.Fatal(err)
		}
		feats, _, err := WhisperFeatures(samples, enc.Cfg.MelBins)
		if err != nil {
			t.Fatal(err)
		}
		for range warm {
			if _, err := enc.Forward(feats); err != nil {
				t.Fatal(err)
			}
		}
		t0 := time.Now()
		out, err := enc.Forward(feats)
		if err != nil {
			t.Fatal(err)
		}
		var sum float64
		for _, v := range out {
			sum += float64(v)
		}
		t.Logf("SPEED whisper_encoder_s=%.3f checksum=%.6f", time.Since(t0).Seconds(), sum)
	}
	if vdir != "" {
		va, err := LoadVoxtralAudio(vdir)
		if err != nil {
			t.Fatal(err)
		}
		wins, err := WhisperFeaturesWindows(samples, 128)
		if err != nil {
			t.Fatal(err)
		}
		for range warm {
			if _, err := va.Embed(wins[:1]); err != nil {
				t.Fatal(err)
			}
		}
		t0 := time.Now()
		out, err := va.Embed(wins[:1])
		if err != nil {
			t.Fatal(err)
		}
		var sum float64
		for _, v := range out {
			sum += float64(v)
		}
		t.Logf("SPEED voxtral_tower_s=%.3f checksum=%.6f", time.Since(t0).Seconds(), sum)
	}
}
