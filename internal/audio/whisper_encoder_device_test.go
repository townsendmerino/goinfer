package audio

import (
	"errors"
	"testing"
)

type fakeWhisperDevice struct {
	calls int
	err   error
}

func (f *fakeWhisperDevice) Forward(feats []float32) ([]float32, error) {
	f.calls++
	return []float32{float32(len(feats))}, f.err
}
func (f *fakeWhisperDevice) Close() error { return nil }

// A device set on the encoder takes Forward, returns its error as is, and a planted defect or SetDevice(nil) goes back to the CPU computation.
func TestWhisperEncoder_deviceTakesForward(t *testing.T) {
	cfg := WhisperEncoderConfig{DModel: 8, Layers: 1, Heads: 2, FFN: 16, MelBins: 80, MaxPositions: WhisperFrames / 2}
	e := &WhisperEncoder{Cfg: cfg}
	feats := make([]float32, 80*WhisperFrames)
	d := &fakeWhisperDevice{}
	e.SetDevice(d)
	out, err := e.Forward(feats)
	if err != nil || d.calls != 1 || len(out) != 1 || out[0] != float32(len(feats)) {
		t.Fatalf("device path: out %v, err %v, calls %d", out, err, d.calls)
	}
	d.err = errors.New("device lost")
	if _, err := e.Forward(feats); !errors.Is(err, d.err) {
		t.Fatalf("a device error must come back as is, got %v", err)
	}
	if _, err := e.Forward(feats[:10]); err == nil || d.calls != 2 {
		t.Fatalf("a wrong-length input must be refused before the device sees it (err %v, calls %d)", err, d.calls)
	}
	e.SetDefectForTest(whisperEncDefectNoPos)
	func() {
		defer func() { _ = recover() }() // the empty weights make the CPU path panic: all that matters is that the device was not called
		if _, err := e.Forward(feats); err != nil {
			t.Log(err)
		}
	}()
	if d.calls != 2 {
		t.Fatalf("a planted defect must run the CPU computation, but the device was called (%d)", d.calls)
	}
}

// Weights exports the encoder's own slices, one layer entry per layer.
func TestWhisperEncoder_weightsView(t *testing.T) {
	e := &WhisperEncoder{Cfg: WhisperEncoderConfig{Layers: 2}, conv1W: []float32{1}, pos: []float32{2}, lnW: []float32{3}, layers: make([]whisperLayer, 2)}
	e.layers[1].kW = []float32{4}
	w := e.Weights()
	if len(w.Layers) != 2 || &w.Conv1W[0] != &e.conv1W[0] || &w.Pos[0] != &e.pos[0] || &w.FinalLNW[0] != &e.lnW[0] || &w.Layers[1].KW[0] != &e.layers[1].kW[0] {
		t.Fatalf("Weights does not alias the encoder's tensors: %+v", w)
	}
	if w.LayerNormEps != 1e-5 {
		t.Fatalf("eps %v", w.LayerNormEps)
	}
}
