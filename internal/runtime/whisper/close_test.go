package whisper

import (
	"errors"
	"testing"
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
