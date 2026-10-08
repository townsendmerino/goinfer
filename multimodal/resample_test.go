package multimodal

import (
	"bytes"
	"encoding/binary"
	"math"
	"slices"
	"strings"
	"testing"
)

// wavBytes builds a 16-bit PCM WAV of interleaved frames (plain fmt chunk, or WAVE_FORMAT_EXTENSIBLE).
func wavBytes(x []float32, ch, rate int, extensible bool) []byte {
	var b bytes.Buffer
	w := func(v any) { _ = binary.Write(&b, binary.LittleEndian, v) }
	fmtLen := 16
	if extensible {
		fmtLen = 40
	}
	b.WriteString("RIFF")
	w(uint32(4 + 8 + fmtLen + 8 + 2*len(x)))
	b.WriteString("WAVEfmt ")
	w(uint32(fmtLen))
	if extensible {
		w(uint16(0xFFFE))
	} else {
		w(uint16(1))
	}
	w(uint16(ch))
	w(uint32(rate))
	w(uint32(rate * ch * 2))
	w(uint16(ch * 2))
	w(uint16(16))
	if extensible {
		w(uint16(22))             // cbSize
		w(uint16(16))             // valid bits
		w(uint32(0))              // channel mask
		w(uint16(1))              // the PCM subformat GUID's first two bytes
		b.Write(make([]byte, 14)) // the rest of the GUID
	}
	b.WriteString("data")
	w(uint32(2 * len(x)))
	for _, v := range x {
		w(int16(math.Round(float64(v) * 32767)))
	}
	return b.Bytes()
}

func sine(n, rate int, hz, amp float64) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(amp * math.Sin(2*math.Pi*hz*float64(i)/float64(rate)))
	}
	return x
}

func TestResample_passthroughAndLength(t *testing.T) {
	x := sine(1000, 16000, 440, 0.5)
	if y := Resample(x, 16000, 16000); &y[0] != &x[0] {
		t.Error("equal rates must return the input itself")
	}
	for _, c := range []struct{ from, n, want int }{{44100, 44100, 16000}, {48000, 4801, 1600}, {8000, 100, 200}, {22050, 7, 5}} {
		if got := len(Resample(make([]float32, c.n), c.from, 16000)); got != c.want {
			t.Errorf("%d samples at %d Hz: %d out, want %d", c.n, c.from, got, c.want)
		}
	}
}

// In the passband a tone comes through at its amplitude and phase; above the 16 kHz Nyquist it is gone; DC has unit gain.
func TestResample_toneAliasDC(t *testing.T) {
	for _, from := range []int{44100, 48000, 22050, 8000, 96000, 44056} { // 44056: a rate with no small common factor (the direct, untabled path for up*taps over the cap is exercised below)
		n := from // one second
		x := sine(n, from, 1000, 0.5)
		y := Resample(x, from, 16000)
		want := sine(len(y), 16000, 1000, 0.5)
		worst := 0.0
		for i := 200; i < len(y)-200; i++ { // away from the edges, where the kernel runs off the signal
			worst = math.Max(worst, math.Abs(float64(y[i]-want[i])))
		}
		if worst > 2e-3 {
			t.Errorf("%d Hz -> 16 kHz, a 1 kHz tone: max error %.2e against the ideal tone", from, worst)
		}
		if from > 24000 {
			z := Resample(sine(n, from, 12000, 0.5), from, 16000) // 12 kHz: above the output's Nyquist
			var e float64
			for i := 200; i < len(z)-200; i++ {
				e = math.Max(e, math.Abs(float64(z[i])))
			}
			if e > 5e-4 { // about -60 dB against the 0.5 input
				t.Errorf("%d Hz -> 16 kHz, a 12 kHz tone: residual %.2e, want it filtered out", from, e)
			}
		}
		dc := make([]float32, n)
		for i := range dc {
			dc[i] = 0.25
		}
		d := Resample(dc, from, 16000)
		if v := d[len(d)/2]; math.Abs(float64(v)-0.25) > 1e-6 {
			t.Errorf("%d Hz: DC 0.25 came out %v", from, v)
		}
	}
}

func TestDownmix(t *testing.T) {
	got := Downmix([]float32{1, 3, -2, 2, 0.5, 0.25}, 2)
	want := []float32{2, 0, 0.375}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Downmix = %v, want %v", got, want)
		}
	}
}

// DecodeWAVAnyRate takes stereo 48 kHz (plain or extensible) as the mean of its channels resampled, and still refuses
// what it cannot read; DecodeWAV still refuses anything but 16 kHz mono.
func TestDecodeWAV_resamplesAndDownmixes(t *testing.T) {
	s := sine(4800, 48000, 700, 0.4)
	d := sine(4800, 48000, 3100, 0.2)
	lr := make([]float32, 2*len(s))
	for i := range s {
		lr[2*i], lr[2*i+1] = s[i]+d[i], s[i]-d[i]
	}
	want := Resample(s, 48000, 16000)
	for _, ext := range []bool{false, true} {
		got, err := DecodeWAVAnyRate(wavBytes(lr, 2, 48000, ext))
		if err != nil {
			t.Fatalf("extensible=%v: %v", ext, err)
		}
		if len(got) != len(want) {
			t.Fatalf("extensible=%v: %d samples, want %d", ext, len(got), len(want))
		}
		worst := 0.0
		for i := range want {
			worst = math.Max(worst, math.Abs(float64(got[i]-want[i])))
		}
		if worst > 1e-4 { // 16-bit rounding of each channel
			t.Errorf("extensible=%v: max |diff| %.2e against the resampled mean", ext, worst)
		}
	}
	mono := sine(1600, 16000, 300, 0.3)
	got, err := DecodeWAVAnyRate(wavBytes(mono, 1, 16000, false))
	if err != nil || len(got) != len(mono) {
		t.Fatalf("16 kHz mono: %d samples, %v", len(got), err)
	}
	strict, err := DecodeWAV(wavBytes(mono, 1, 16000, false))
	if err != nil || !slices.Equal(strict, got) {
		t.Errorf("16 kHz mono: DecodeWAV and DecodeWAVAnyRate disagree (%v)", err)
	}
	if _, err := DecodeWAV(wavBytes(lr, 2, 48000, false)); err == nil || !strings.Contains(err.Error(), "16000 Hz") {
		t.Errorf("DecodeWAV took 48 kHz stereo (%v); it stays strict until G-S5e passes", err)
	}
	bad := wavBytes(mono, 1, 16000, false)
	binary.LittleEndian.PutUint16(bad[34:36], 24) // 24-bit
	if _, err := DecodeWAVAnyRate(bad); err == nil || !strings.Contains(err.Error(), "24-bit") {
		t.Errorf("a 24-bit WAV: %v, want a refusal naming it", err)
	}
	if _, err := DecodeWAVAnyRate(wavBytes(mono, 1, 1000, false)); err == nil {
		t.Error("a 1 kHz WAV was taken")
	}
}

// A rate pair whose polyphase table would pass resampleMaxTable computes its kernel per sample; it must agree with the
// tabled path's arithmetic exactly.
func TestResample_untabledPathMatches(t *testing.T) {
	x := sine(3000, 47999, 900, 0.5) // gcd(47999, 16000) = 1: 16000 phases
	tabled := Resample(x, 47999, 16000)
	save := resampleMaxTableVar
	resampleMaxTableVar = 0
	direct := Resample(x, 47999, 16000)
	resampleMaxTableVar = save
	for i := range tabled {
		if tabled[i] != direct[i] {
			t.Fatalf("sample %d: tabled %v, direct %v", i, tabled[i], direct[i])
		}
	}
}
