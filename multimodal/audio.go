package multimodal

import (
	"encoding/binary"
	"fmt"

	"github.com/townsendmerino/goinfer/internal/audio"
)

// WAV rates DecodeWAVAnyRate takes. Anything in range is resampled to 16 kHz; outside it is far more likely a broken
// header than audio.
const (
	wavMinRate = 4000
	wavMaxRate = 384000
	wavMaxCh   = 8
)

// DecodeWAV reads a RIFF WAV of 16-bit PCM, mono, at 16 kHz into samples in [-1, 1] (s/32768). Any other format, channel count or
// rate is refused; DecodeWAVAnyRate takes other rates and channel counts.
func DecodeWAV(data []byte) ([]float32, error) { return decodeWAV(data, false) }

// DecodeWAVAnyRate reads a RIFF WAV of 16-bit PCM (plain, or WAVE_FORMAT_EXTENSIBLE with a PCM subformat), 1 to 8
// channels at 4 to 384 kHz, into 16 kHz mono samples in [-1, 1]: the channels averaged (Downmix), then resampled
// (Resample) unless already at 16 kHz. A 16 kHz mono file comes out exactly as DecodeWAV's.
func DecodeWAVAnyRate(data []byte) ([]float32, error) { return decodeWAV(data, true) }

// WAV is a checked RIFF WAV whose samples are not yet decoded, so a caller can refuse a clip over its limit from the
// header, before paying for the samples and for a resample that can multiply them. It aliases the bytes it was parsed from.
type WAV struct {
	pcm      []byte // the data chunk
	ch, rate int
}

// ParseWAV checks data as DecodeWAVAnyRate does and returns the clip undecoded.
func ParseWAV(data []byte) (WAV, error) { return parseWAV(data, true) }

// Samples is the length of Decode's result.
func (w WAV) Samples() int {
	return resampledLen(len(w.pcm)/2/w.ch, w.rate, audio.Gemma4SampleRate)
}

// Decode returns the clip as DecodeWAVAnyRate does.
func (w WAV) Decode() []float32 {
	n := len(w.pcm) / 2 / w.ch * w.ch // whole frames only
	out := make([]float32, n)
	for i := range n {
		out[i] = float32(int16(binary.LittleEndian.Uint16(w.pcm[2*i:]))) / 32768
	}
	return Resample(Downmix(out, w.ch), w.rate, audio.Gemma4SampleRate)
}

func decodeWAV(data []byte, anyRate bool) ([]float32, error) {
	w, err := parseWAV(data, anyRate)
	if err != nil {
		return nil, err
	}
	return w.Decode(), nil
}

func parseWAV(data []byte, anyRate bool) (WAV, error) {
	if len(data) < 12 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return WAV{}, fmt.Errorf("audio: not a RIFF/WAVE file")
	}
	var fmtOK bool
	var ch, rate int
	for off := 12; off+8 <= len(data); {
		id, size := string(data[off:off+4]), int(binary.LittleEndian.Uint32(data[off+4:off+8]))
		body := off + 8
		if body+size > len(data) {
			size = len(data) - body
		}
		switch id {
		case "fmt ":
			if size < 16 {
				return WAV{}, fmt.Errorf("audio: WAV fmt chunk of %d bytes", size)
			}
			f := data[body : body+size]
			format, c := binary.LittleEndian.Uint16(f[0:2]), binary.LittleEndian.Uint16(f[2:4])
			r, bits := binary.LittleEndian.Uint32(f[4:8]), binary.LittleEndian.Uint16(f[14:16])
			if format == 0xFFFE && size >= 40 { // WAVE_FORMAT_EXTENSIBLE: the real format is the subformat GUID's first two bytes
				format = binary.LittleEndian.Uint16(f[24:26])
			}
			if !anyRate && (format != 1 || bits != 16 || c != 1 || r != audio.Gemma4SampleRate) {
				return WAV{}, fmt.Errorf("audio: WAV is format %d, %d-bit, %d channel(s) at %d Hz; want 16-bit PCM mono at 16000 Hz (no resampling is done)", format, bits, c, r)
			}
			if format != 1 || bits != 16 || c < 1 || c > wavMaxCh || r < wavMinRate || r > wavMaxRate {
				return WAV{}, fmt.Errorf("audio: WAV is format %d, %d-bit, %d channel(s) at %d Hz; want 16-bit PCM, 1-%d channels, %d-%d Hz", format, bits, c, r, wavMaxCh, wavMinRate, wavMaxRate)
			}
			if !ResampleSupported(int(r), audio.Gemma4SampleRate) {
				return WAV{}, fmt.Errorf("audio: a WAV at %d Hz is not supported (resampling it to %d Hz would need an impractically long filter); resample it to %d Hz first", r, audio.Gemma4SampleRate, audio.Gemma4SampleRate)
			}
			ch, rate, fmtOK = int(c), int(r), true
		case "data":
			if !fmtOK {
				return WAV{}, fmt.Errorf("audio: WAV data before its fmt chunk")
			}
			return WAV{pcm: data[body : body+size], ch: ch, rate: rate}, nil
		}
		off = body + size + size&1
	}
	return WAV{}, fmt.Errorf("audio: WAV has no data chunk")
}
