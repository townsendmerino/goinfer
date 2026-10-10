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

// G-S14g3 of docs/tasks/task-multimodal-support-2026-10.md: openai/whisper-small, goinfer float32 on the CPU, against transformers 5.15.0 float32 generate(return_timestamps=True, return_segments=True) on the
// registered 5.9 s clip (short form) and a 70 s composite of LibriSpeech clips (long form). Bars: the same segments (count, tokens, start and end to 1e-9), the same sequence, the same decoded text, the same language.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_WHISPER_TSREF=<pin output dir> [GOINFER_WHISPER_DIR=<checkpoint>] go test -tags realckpt ./internal/whisper/ -run TestWhisperSmall_timestamps -v -timeout 30m
func TestWhisperSmall_timestamps(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	ref := os.Getenv("GOINFER_WHISPER_TSREF")
	if ref == "" {
		t.Skip("set GOINFER_WHISPER_TSREF to scripts/pin_whisper_ts_real.py's output directory")
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
			Samples    int
			LanguageID int `json:"language_id"`
			Segments   []struct {
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
		b, err := os.ReadFile(filepath.Join(ref, name+".in.f32"))
		if err != nil {
			t.Fatal(err)
		}
		x := make([]float32, len(b)/4)
		for i := range x {
			x[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		}
		t0 := time.Now()
		r, err := tr.TranscribeTimestamps(x, "", "transcribe")
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		text := tr.TimestampText(r)
		t.Logf("%s: %d samples, %d segments (reference %d), language %s, %.0f s, text %q", name, len(x), len(r.Segments), len(c.Segments), r.Language, time.Since(t0).Seconds(), text[:min(len(text), 90)])
		if tr.Gen.LangToID[r.Language] != c.LanguageID {
			t.Errorf("%s: language %s (id %d), reference id %d", name, r.Language, tr.Gen.LangToID[r.Language], c.LanguageID)
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
