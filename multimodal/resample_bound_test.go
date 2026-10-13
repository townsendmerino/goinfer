package multimodal

import (
	"encoding/binary"
	"testing"
	"time"
)

// Every rate a recording is made at resamples; a header written to make the filter enormous is refused before any tap is built.
func TestResampleSupported_commonRatesYesHostileRatesNo(t *testing.T) {
	for _, r := range []int{4000, 8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 47250, 48000, 64000, 88200, 96000, 176400, 192000, 352800, 384000} {
		if !ResampleSupported(r, 16000) {
			t.Errorf("%d Hz must be supported", r)
		}
	}
	for _, r := range []int{383999, 383993, 262147, 100003} {
		if ResampleSupported(r, 16000) {
			t.Errorf("%d Hz (coprime to 16000, a %d-tap filter) must be refused", r, 20*r)
		}
	}
}

// The same through the decoder, from a header: a 44-byte-plus WAV declaring 383999 Hz and one sample returns an error quickly instead of
// building a 7.7-million-tap filter.
func TestDecodeWAVAnyRate_refusesAFilterBombRate(t *testing.T) {
	wav := func(rate uint32) []byte {
		b := make([]byte, 44+2)
		copy(b[0:], "RIFF")
		binary.LittleEndian.PutUint32(b[4:], uint32(len(b)-8))
		copy(b[8:], "WAVEfmt ")
		binary.LittleEndian.PutUint32(b[16:], 16)
		binary.LittleEndian.PutUint16(b[20:], 1)
		binary.LittleEndian.PutUint16(b[22:], 1)
		binary.LittleEndian.PutUint32(b[24:], rate)
		binary.LittleEndian.PutUint32(b[28:], rate*2)
		binary.LittleEndian.PutUint16(b[32:], 2)
		binary.LittleEndian.PutUint16(b[34:], 16)
		copy(b[36:], "data")
		binary.LittleEndian.PutUint32(b[40:], 2)
		return b
	}
	start := time.Now()
	if _, err := DecodeWAVAnyRate(wav(383999)); err == nil {
		t.Error("383999 Hz decoded: the filter-bomb rate was accepted")
	}
	if d := time.Since(start); d > time.Second {
		t.Errorf("refusing it took %v: the filter was built first", d)
	}
	if _, err := DecodeWAVAnyRate(wav(44100)); err != nil {
		t.Errorf("44100 Hz refused: %v", err)
	}
}
