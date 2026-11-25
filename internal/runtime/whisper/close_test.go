package whisper

import (
	"context"
	"errors"
	"testing"

	"github.com/townsendmerino/goinfer/internal/audio"
)

// TestTranscriber_closeReleasesAndRefuses: Close releases the encoder's and the decoder's checkpoint mappings, a second Close
// does nothing, and every entry point after it returns an error naming the closed transcriber or decoder, never a read of
// the released mapping (the decoder's views are dropped, so a read would panic, not return).
func TestTranscriber_closeReleasesAndRefuses(t *testing.T) {
	tr, _ := tinyTS(t)
	clip := tsClip(MaxSamples/3, 3, 7)
	enc, err := tr.Encode(clip)
	if err != nil {
		t.Fatalf("before Close: %v", err)
	}
	if _, err := tr.TranscribeWith(clip, TranscribeOptions{}); err != nil {
		t.Fatalf("before Close: %v", err)
	}
	if err := tr.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := tr.Close(); err != nil {
		t.Errorf("a second Close: %v, want nil", err)
	}
	if err := tr.Dec.Close(); err != nil {
		t.Errorf("a second Decoder.Close: %v, want nil", err)
	}
	if tr.Dec.embed != nil || tr.Dec.pos != nil || tr.Dec.layers != nil || tr.Dec.lnW != nil {
		t.Error("the decoder still holds views into the released mapping")
	}
	if _, err := tr.Encode(clip); !errors.Is(err, errTranscriberClosed) {
		t.Errorf("Encode after Close: %v, want %v", err, errTranscriberClosed)
	}
	if _, err := tr.TranscribeWith(clip, TranscribeOptions{}); !errors.Is(err, errTranscriberClosed) {
		t.Errorf("TranscribeWith after Close: %v, want %v", err, errTranscriberClosed)
	}
	if _, err := tr.Dec.NewState(enc); !errors.Is(err, errDecoderClosed) {
		t.Errorf("Decoder.NewState after Close: %v, want %v", err, errDecoderClosed)
	}
	if _, err := tr.Dec.Generate(enc, tr.Gen, "en", "transcribe"); !errors.Is(err, errDecoderClosed) {
		t.Errorf("Decoder.Generate after Close: %v, want %v", err, errDecoderClosed)
	}
	if _, err := tr.Enc.Forward(make([]float32, tr.Mels*3000)); err == nil {
		t.Error("the encoder's Forward after Close returned no error")
	}
}

// cancelOnForward runs the encoder on a second CPU copy, counts the windows it is asked for, and cancels a context on the first.
type cancelOnForward struct {
	enc    *audio.WhisperEncoder
	cancel context.CancelFunc
	calls  int
}

func (d *cancelOnForward) Forward(feats []float32) ([]float32, error) {
	d.calls++
	if d.calls == 1 {
		d.cancel()
	}
	return d.enc.Forward(feats)
}

func (d *cancelOnForward) Close() error { return nil }

// TestTranscribeWithContext_stopsAtCancel: a context cancelled while the first 30 s window runs ends a multi-window clip with the
// context's error before the next window's encoder pass, and an already-cancelled context runs no window at all.
func TestTranscribeWithContext_stopsAtCancel(t *testing.T) {
	tr, g := tinyTS(t)
	c, ok := g.Cases["long68s"]
	if !ok {
		t.Fatal("the golden has no long68s case")
	}
	clip := tsClip(c.Samples, c.A, c.B)
	cpu, err := audio.LoadWhisperEncoder(tsDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { cpu.Close() })
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	dev := &cancelOnForward{enc: cpu, cancel: cancel}
	tr.Enc.SetDevice(dev)
	defer tr.Enc.SetDevice(nil)

	if _, err := tr.TranscribeWithContext(ctx, clip, TranscribeOptions{}); !errors.Is(err, context.Canceled) {
		t.Errorf("cancelled during the first window: err = %v, want %v", err, context.Canceled)
	}
	if dev.calls != 1 {
		t.Errorf("the encoder ran %d windows after the context was cancelled in the first; want 1", dev.calls)
	}
	dev.calls = 0
	if _, err := tr.TranscribeWithContext(ctx, clip, TranscribeOptions{}); !errors.Is(err, context.Canceled) {
		t.Errorf("an already-cancelled context: err = %v, want %v", err, context.Canceled)
	}
	if dev.calls != 0 {
		t.Errorf("an already-cancelled context ran %d windows, want 0", dev.calls)
	}
}
