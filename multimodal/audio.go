package multimodal

import (
	"encoding/binary"
	"fmt"

	"github.com/townsendmerino/aikit/audio"
)

// WAV rates DecodeWAVAnyRate takes. Anything in range is resampled to 16 kHz; outside it is far more likely a broken
// header than audio.
const (
	wavMinRate = 4000
	wavMaxRate = 384000
	wavMaxCh   = 8
)

// DecodeWAV reads a RIFF WAV of 16-bit PCM, mono, at 16 kHz into samples in [-1, 1] (s/32768). Any other format,
// channel count or rate is refused: serve does not resample yet. DecodeWAVAnyRate does, but G-S5e
// (docs/tasks/task-multimodal-support-2026-10.md) read FAIL against scipy's resampler, so it is not wired in; the
// owner decides.
func DecodeWAV(data []byte) ([]float32, error) { return decodeWAV(data, false) }

// DecodeWAVAnyRate reads a RIFF WAV of 16-bit PCM (plain, or WAVE_FORMAT_EXTENSIBLE with a PCM subformat), 1 to 8
// channels at 4 to 384 kHz, into 16 kHz mono samples in [-1, 1]: the channels averaged (Downmix), then resampled
// (Resample) unless already at 16 kHz. A 16 kHz mono file comes out exactly as DecodeWAV's.
func DecodeWAVAnyRate(data []byte) ([]float32, error) { return decodeWAV(data, true) }

func decodeWAV(data []byte, anyRate bool) ([]float32, error) {
	if len(data) < 12 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, fmt.Errorf("audio: not a RIFF/WAVE file")
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
				return nil, fmt.Errorf("audio: WAV fmt chunk of %d bytes", size)
			}
			f := data[body : body+size]
			format, c := binary.LittleEndian.Uint16(f[0:2]), binary.LittleEndian.Uint16(f[2:4])
			r, bits := binary.LittleEndian.Uint32(f[4:8]), binary.LittleEndian.Uint16(f[14:16])
			if format == 0xFFFE && size >= 40 { // WAVE_FORMAT_EXTENSIBLE: the real format is the subformat GUID's first two bytes
				format = binary.LittleEndian.Uint16(f[24:26])
			}
			if !anyRate && (format != 1 || bits != 16 || c != 1 || r != audio.Gemma4SampleRate) {
				return nil, fmt.Errorf("audio: WAV is format %d, %d-bit, %d channel(s) at %d Hz; want 16-bit PCM mono at 16000 Hz (no resampling is done)", format, bits, c, r)
			}
			if format != 1 || bits != 16 || c < 1 || c > wavMaxCh || r < wavMinRate || r > wavMaxRate {
				return nil, fmt.Errorf("audio: WAV is format %d, %d-bit, %d channel(s) at %d Hz; want 16-bit PCM, 1-%d channels, %d-%d Hz", format, bits, c, r, wavMaxCh, wavMinRate, wavMaxRate)
			}
			ch, rate, fmtOK = int(c), int(r), true
		case "data":
			if !fmtOK {
				return nil, fmt.Errorf("audio: WAV data before its fmt chunk")
			}
			n := size / 2 / ch * ch // whole frames only
			out := make([]float32, n)
			for i := range n {
				out[i] = float32(int16(binary.LittleEndian.Uint16(data[body+2*i:]))) / 32768
			}
			return Resample(Downmix(out, ch), rate, audio.Gemma4SampleRate), nil
		}
		off = body + size + size&1
	}
	return nil, fmt.Errorf("audio: WAV has no data chunk")
}
