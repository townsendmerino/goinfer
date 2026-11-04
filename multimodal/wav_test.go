package multimodal

import "testing"

// Samples is the price a caller refuses a clip by, so it must be exactly the length Decode returns: every rate class
// (equal, up by an integer, up and down by a ratio, down), channel count, and length, with torn frames at the end of the file.
func TestWAV_samplesIsDecodesLength(t *testing.T) {
	for _, rate := range []int{4000, 8000, 11025, 16000, 22050, 44100, 48000, 96000, 352800, 384000} {
		for _, ch := range []int{1, 2, 8} {
			for _, frames := range []int{0, 1, 2, 3, 159, 160, 161, 4096} {
				whole := wavBytes(make([]float32, frames*ch), ch, rate, false)
				for _, torn := range []int{0, 1, 3} {
					if torn >= len(whole)-44 && torn > 0 {
						continue
					}
					data := whole[:len(whole)-torn]
					w, err := ParseWAV(data)
					if err != nil {
						t.Fatalf("%d Hz, %d channels, %d frames, %d torn: %v", rate, ch, frames, torn, err)
					}
					got := w.Decode()
					if w.Samples() != len(got) {
						t.Errorf("%d Hz, %d channels, %d frames, %d torn: Samples %d, Decode returned %d", rate, ch, frames, torn, w.Samples(), len(got))
					}
					direct, err := DecodeWAVAnyRate(data)
					if err != nil || len(direct) != len(got) {
						t.Errorf("%d Hz, %d channels, %d frames, %d torn: DecodeWAVAnyRate gave %d samples, %v", rate, ch, frames, torn, len(direct), err)
					}
				}
			}
		}
	}
}

// Parsing touches the header only: a megabyte clip costs nothing until Decode.
func TestWAV_parseDecodesNothing(t *testing.T) {
	data := wavBytes(make([]float32, 1<<19), 1, 4000, false)
	if n := testing.AllocsPerRun(10, func() {
		w, err := ParseWAV(data)
		if err != nil || w.Samples() != 1<<21 {
			t.Fatalf("Samples %d, %v", w.Samples(), err)
		}
	}); n != 0 {
		t.Errorf("ParseWAV and Samples allocated %v times", n)
	}
}

// What DecodeWAVAnyRate refuses, ParseWAV refuses, with the same words.
func TestWAV_refusesWhatDecodeRefuses(t *testing.T) {
	good := wavBytes(make([]float32, 8), 1, 16000, false)
	badRate := wavBytes(make([]float32, 8), 1, 383999, false)
	eightBit := append([]byte(nil), good...)
	eightBit[34] = 8
	noFmt := append(append([]byte(nil), good[:12]...), good[36:]...)
	for name, data := range map[string][]byte{
		"empty": nil, "not riff": []byte("ID3 not a wav at all"), "header only": good[:12],
		"filter bomb rate": badRate, "8-bit": eightBit, "data before fmt": noFmt,
	} {
		_, parseErr := ParseWAV(data)
		_, decodeErr := DecodeWAVAnyRate(data)
		if parseErr == nil || decodeErr == nil || parseErr.Error() != decodeErr.Error() {
			t.Errorf("%s: ParseWAV %v, DecodeWAVAnyRate %v", name, parseErr, decodeErr)
		}
	}
}
