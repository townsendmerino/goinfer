//go:build realckpt

package whisper

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"
)

// G-S14h2 of docs/tasks/task-multimodal-support-2026-10.md: openai/whisper-small with OpenAI's decode policy (conditioning on earlier text, logprob_threshold -1.0, compression_ratio_threshold 2.4, no_speech_threshold 0.6,
// one greedy attempt) against transformers 5.15.0 float32, on a 70 s composite of LibriSpeech clips and on a "silent" composite (35 s of zeros, a clip, 35 s of zeros, two clips). Bars: the same segments (count,
// tokens, start and end to 1e-9), the same sequence and decoded text. The windows' statistics are logged.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_WHISPER_POLREF=<pin output dir> [GOINFER_WHISPER_DIR=<checkpoint>] go test -tags realckpt ./internal/whisper/ -run TestWhisperSmall_policy -v -timeout 30m
func TestWhisperSmall_policy(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	ref := os.Getenv("GOINFER_WHISPER_POLREF")
	if ref == "" {
		t.Skip("set GOINFER_WHISPER_POLREF to scripts/pin_whisper_policy_real.py's output directory")
	}
	dir := os.Getenv("GOINFER_WHISPER_DIR")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, "models", "whisper-small")
	}
	raw, err := os.ReadFile(filepath.Join(ref, "golden.json"))
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Cases map[string]struct {
			Samples          int
			LogprobThreshold float64 `json:"logprob_threshold"`
			Segments         []struct {
				Start, End float64
				Tokens     []int
			}
			Sequence []int
			Text     string
		}
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	tr, err := LoadTranscriber(dir)
	if err != nil {
		t.Fatal(err)
	}
	for name, c := range g.Cases {
		opts := TranscribeOptions{Task: "transcribe", ConditionOnPrev: true, LogprobThreshold: f64(c.LogprobThreshold), CompressionRatioThreshold: f64(2.4), NoSpeechThreshold: f64(0.6)}
		b, err := os.ReadFile(filepath.Join(ref, name+".in.f32"))
		if err != nil {
			t.Fatal(err)
		}
		x := make([]float32, len(b)/4)
		for i := range x {
			x[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		}
		t0 := time.Now()
		r, err := tr.TranscribeWith(x, opts)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		text := tr.TimestampText(r)
		t.Logf("%s: %d segments (reference %d), %.0f s, text %q", name, len(r.Segments), len(c.Segments), time.Since(t0).Seconds(), text[:min(len(text), 80)])
		for _, w := range r.Windows {
			t.Logf("  window at frame %5d: prompt %3d tokens, attempts %v, compression ratio %.2f, avg logprob %.3f, no-speech %.3f, skipped %v", w.Seek, w.PromptLen, w.Attempts, w.CompressionRatio, w.AvgLogprob, w.NoSpeechProb, w.Skipped)
		}
		if len(r.Segments) != len(c.Segments) {
			t.Errorf("%s: %d segments, reference %d", name, len(r.Segments), len(c.Segments))
		} else {
			for i, s := range r.Segments {
				w := c.Segments[i]
				if math.Abs(s.Start-w.Start) > 1e-9 || math.Abs(s.End-w.End) > 1e-9 || !slices.Equal(s.Tokens, w.Tokens) {
					t.Errorf("%s: segment %d differs: got [%.2f, %.2f] %v, reference [%.2f, %.2f] %v", name, i, s.Start, s.End, s.Tokens, w.Start, w.End, w.Tokens)
					break
				}
			}
		}
		if !slices.Equal(r.Tokens, c.Sequence) || text != c.Text {
			t.Errorf("%s: the sequence or text differs from transformers' (text %q against %q)", name, text, c.Text)
		}
	}
}
