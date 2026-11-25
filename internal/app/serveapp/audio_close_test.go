package serveapp

import (
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/internal/audio"
	"github.com/townsendmerino/goinfer/multimodal"
)

// TestUnload_closesTheAudioEncoders: across load, a first clip and an unload, the drain closes each lazily loaded audio
// encoder once (Voxtral's on voxtral-tiny, Qwen3-ASR's on qwen3asr-tiny), and none for an entry whose clip never came. The
// unload goes through the admin route and the real drain, around a sentinel model whose Close is a no-op.
func TestUnload_closesTheAudioEncoders(t *testing.T) {
	vox, err := filepath.Abs(filepath.Join("../../../testdata/voxtral-tiny"))
	if err != nil {
		t.Fatal(err)
	}
	qasr, err := filepath.Abs(filepath.Join("../../../testdata/qwen3asr-tiny"))
	if err != nil {
		t.Fatal(err)
	}
	clip, _ := multimodal.DecodeWAVAnyRate(sineWAV(32000, 16000))
	firstClip := map[string]func(lm *loadedModel) error{
		"voxtral": func(lm *loadedModel) error {
			enc, err := lm.audio.(*voxtralAudio).encoder(lm.name)
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
			enc, err := lm.audio.(*qwenASRAudio).encoder()
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
		tower func() audioTower
		clip  string // the encoder a first clip loads, "" for none
		wantV int
		wantQ int
	}{
		{"a Voxtral clip", func() audioTower { return &voxtralAudio{dir: vox} }, "voxtral", 1, 0},
		{"a Qwen3-ASR clip", func() audioTower { return &qwenASRAudio{dir: qasr} }, "qwenASR", 0, 1},
		{"no Voxtral clip", func() audioTower { return &voxtralAudio{dir: vox} }, "", 0, 0},
		{"no Qwen3-ASR clip", func() audioTower { return &qwenASRAudio{dir: qasr} }, "", 0, 0},
	} {
		t.Run(c.name, func(t *testing.T) {
			closed := map[string]int{}
			prev := closeAudioEncoder
			closeAudioEncoder = func(field string, enc interface{ Close() error }) error { closed[field]++; return prev(field, enc) }
			t.Cleanup(func() { closeAudioEncoder = prev })

			m := &fakeCloser{}
			lm := &loadedModel{name: "t", closer: m, audio: c.tower()}
			s := &server{models: map[string]*loadedModel{"t": lm}, liveness: map[io.Closer]*modelLiveness{m: {refs: 1}},
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
			if v, ok := lm.audio.(*voxtralAudio); ok && (v.enc != nil || v.device != nil) {
				t.Error("the entry still holds an encoder after its unload")
			}
			if q, ok := lm.audio.(*qwenASRAudio); ok && q.enc != nil {
				t.Error("the entry still holds an encoder after its unload")
			}
		})
	}
}

// fakeWhisperDevice stands in for a backend's device tower: an output of the encoder's shape until Close, an error after.
type fakeWhisperDevice struct {
	n      int
	closed bool
}

func (d *fakeWhisperDevice) Forward([]float32) ([]float32, error) {
	if d.closed {
		return nil, errors.New("fake device: used after Close")
	}
	return make([]float32, d.n), nil
}

func (d *fakeWhisperDevice) Close() error { d.closed = true; return nil }

// TestUnload_closesOnlyItsOwnAudioDevice: two Voxtral entries on one device backend each get their own device from the
// backend's factory (multimodal.UseWhisperDevice). Unloading one closes its device, and the other still encodes a clip on
// its own. The factory here builds a device per encoder, as the CUDA one does (newAudioTower, one Device each).
func TestUnload_closesOnlyItsOwnAudioDevice(t *testing.T) {
	vox, err := filepath.Abs(filepath.Join("../../../testdata/voxtral-tiny"))
	if err != nil {
		t.Fatal(err)
	}
	const backend = "test-audio-device"
	multimodal.RegisterWhisperDevice(backend, func(enc *audio.WhisperEncoder) (audio.WhisperDevice, error) {
		return &fakeWhisperDevice{n: enc.Cfg.MaxPositions * enc.Cfg.DModel}, nil
	})
	clip, _ := multimodal.DecodeWAVAnyRate(sineWAV(32000, 16000))
	wins, err := audio.WhisperFeaturesWindows(clip, 128)
	if err != nil {
		t.Fatal(err)
	}
	s := &server{models: map[string]*loadedModel{}, liveness: map[io.Closer]*modelLiveness{}, draining: map[string]struct{}{},
		cfg: config{allowAdmin: true, unloadDrainWait: 5 * time.Second}}
	entry := func(name string) *loadedModel {
		m := &fakeCloser{}
		lm := &loadedModel{name: name, closer: m, audio: &voxtralAudio{dir: vox, backend: backend}}
		s.models[name], s.liveness[m] = lm, &modelLiveness{refs: 1}
		enc, err := lm.audio.(*voxtralAudio).encoder(name)
		if err != nil {
			t.Fatal(err)
		}
		if lm.audio.(*voxtralAudio).device == nil {
			t.Fatalf("%s: the tower did not go on the test device", name)
		}
		if _, err := enc.Embed(wins); err != nil {
			t.Fatalf("%s: a clip on the device: %v", name, err)
		}
		return lm
	}
	a, b := entry("a"), entry("b")
	t.Cleanup(b.closeEntryNatives)
	dev := a.audio.(*voxtralAudio).device.(*fakeWhisperDevice)
	w := httptest.NewRecorder()
	s.handleAdminUnload(w, httptest.NewRequest("POST", "/admin/models/unload", strings.NewReader(`{"name":"a"}`)))
	if w.Code != http.StatusOK {
		t.Fatalf("unload: %d %s", w.Code, w.Body.String())
	}
	if !dev.closed {
		t.Error("the unloaded entry's device was not closed")
	}
	enc, err := b.audio.(*voxtralAudio).encoder("b")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := enc.Embed(wins); err != nil {
		t.Errorf("the other entry, after the first one's unload: %v", err)
	}
}
