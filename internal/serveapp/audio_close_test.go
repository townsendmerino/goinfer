package serveapp

import (
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/audio"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/multimodal"
)

// TestUnload_closesTheAudioEncoders: across load, a first clip and an unload, the drain closes each lazily loaded audio
// encoder once (Voxtral's on voxtral-tiny, Qwen3-ASR's on qwen3asr-tiny), and none for an entry whose clip never came. The
// unload goes through the admin route and the real drain, around a sentinel model whose Close is a no-op.
func TestUnload_closesTheAudioEncoders(t *testing.T) {
	vox, err := filepath.Abs(filepath.Join("..", "..", "testdata", "voxtral-tiny"))
	if err != nil {
		t.Fatal(err)
	}
	qasr, err := filepath.Abs(filepath.Join("..", "..", "testdata", "qwen3asr-tiny"))
	if err != nil {
		t.Fatal(err)
	}
	clip, _ := multimodal.DecodeWAVAnyRate(sineWAV(32000, 16000))
	firstClip := map[string]func(lm *loadedModel) error{
		"voxtral": func(lm *loadedModel) error {
			enc, err := lm.voxtralEncoder()
			if err != nil {
				return err
			}
			wins, err := audio.WhisperFeaturesWindows(clip, 128)
			if err != nil {
				return err
			}
			_, err = enc.Embed(wins)
			return err
		},
		"qwenASR": func(lm *loadedModel) error {
			enc, err := lm.qwenASREncoder()
			if err != nil {
				return err
			}
			f, err := audio.QwenASRFeatures(clip)
			if err != nil {
				return err
			}
			_, _, err = enc.Forward(f.Data, f.T, f.Valid)
			return err
		},
	}
	for _, c := range []struct {
		name  string
		clip  string // the encoder a first clip loads, "" for none
		wantV int
		wantQ int
	}{
		{"a Voxtral clip", "voxtral", 1, 0},
		{"a Qwen3-ASR clip", "qwenASR", 0, 1},
		{"no clip", "", 0, 0},
	} {
		t.Run(c.name, func(t *testing.T) {
			closed := map[string]int{}
			prev := closeAudioEncoder
			closeAudioEncoder = func(field string, enc interface{ Close() error }) error { closed[field]++; return prev(field, enc) }
			t.Cleanup(func() { closeAudioEncoder = prev })

			m := &decoder.Model{}
			lm := &loadedModel{name: "t", model: m, voxtralDir: vox, qwenASRDir: qasr}
			s := &server{models: map[string]*loadedModel{"t": lm}, liveness: map[*decoder.Model]*modelLiveness{m: {refs: 1}},
				draining: map[string]struct{}{}, cfg: config{allowAdmin: true, unloadDrainWait: 5 * time.Second}}
			if c.clip != "" {
				if err := firstClip[c.clip](lm); err != nil {
					t.Fatalf("first clip: %v", err)
				}
			}
			w := httptest.NewRecorder()
			s.handleAdminUnload(w, httptest.NewRequest("POST", "/admin/models/unload", strings.NewReader(`{"name":"t"}`)))
			if w.Code != http.StatusOK {
				t.Fatalf("unload: %d %s", w.Code, w.Body.String())
			}
			if closed["voxtral"] != c.wantV || closed["qwenASR"] != c.wantQ {
				t.Errorf("Close calls %v; want voxtral %d, qwenASR %d", closed, c.wantV, c.wantQ)
			}
			if lm.voxtral != nil || lm.qwenASR != nil {
				t.Error("the entry still holds an encoder after its unload")
			}
		})
	}
}
