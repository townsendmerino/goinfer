//go:build realckpt

package whisper

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/multimodal"
)

// G-S14f3's goinfer arm (docs/tasks/task-multimodal-support-2026-10.md, "G-S14f"): every WAV in $GOINFER_S14F_DIR (scripts/s14c4_data.py's) through the Transcriber, short-form greedy, language detected, and
// {id: {text, lang}} written to $GOINFER_S14F_OUT, the shape scripts/s14f_hf.py writes; scripts/s14f_grade.py compares them.
//
//	GOINFER_HEAVY_TESTS=1 GOINFER_S14F_DIR=<wav dir> GOINFER_S14F_OUT=<json> [GOINFER_S14F_LIMIT=<clips>] [GOINFER_WHISPER_DIR=<checkpoint>] go test -tags realckpt ./internal/whisper/ -run TestWhisperSmall_clips -v -timeout 60m
func TestWhisperSmall_clips(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	data, out := os.Getenv("GOINFER_S14F_DIR"), os.Getenv("GOINFER_S14F_OUT")
	if data == "" || out == "" {
		t.Skip("set GOINFER_S14F_DIR and GOINFER_S14F_OUT")
	}
	dir := os.Getenv("GOINFER_WHISPER_DIR")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, "models", "whisper-small")
	}
	raw, err := os.ReadFile(filepath.Join(data, "refs.json"))
	if err != nil {
		t.Fatal(err)
	}
	var refs map[string]string
	if err := json.Unmarshal(raw, &refs); err != nil {
		t.Fatal(err)
	}
	ids := make([]string, 0, len(refs))
	for id := range refs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	if lim := os.Getenv("GOINFER_S14F_LIMIT"); lim != "" {
		var n int
		fmt.Sscanf(lim, "%d", &n)
		if n > 0 && n < len(ids) {
			ids = ids[:n]
		}
	}
	tr, err := LoadTranscriber(dir)
	if err != nil {
		t.Fatal(err)
	}
	type row struct {
		Text string `json:"text"`
		Lang string `json:"lang"`
	}
	res := map[string]row{}
	t0 := time.Now()
	for i, id := range ids {
		wav, err := os.ReadFile(filepath.Join(data, id+".wav"))
		if err != nil {
			t.Fatal(err)
		}
		samples, err := multimodal.DecodeWAVAnyRate(wav)
		if err != nil {
			t.Fatal(err)
		}
		text, r, err := tr.Transcribe(samples, "", "transcribe")
		if err != nil {
			t.Fatalf("%s: %v", id, err)
		}
		res[id] = row{text, r.Language}
		fmt.Fprintf(os.Stderr, "[s14f %5.0fs] %d/%d %s: %s %q\n", time.Since(t0).Seconds(), i+1, len(ids), id, r.Language, text)
	}
	b, _ := json.MarshalIndent(res, "", " ")
	if err := os.WriteFile(out, b, 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("%d clips in %.0f s, wrote %s", len(ids), time.Since(t0).Seconds(), out)
}
