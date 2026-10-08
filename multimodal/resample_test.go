package multimodal

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
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
	for _, c := range []struct{ from, n, want int }{{44100, 44100, 16000}, {48000, 4801, 1601}, {8000, 100, 200}, {22050, 7, 6}} {
		if got := len(Resample(make([]float32, c.n), c.from, 16000)); got != c.want { // ceil(n·up/down), as resample_poly
			t.Errorf("%d samples at %d Hz: %d out, want %d", c.n, c.from, got, c.want)
		}
	}
}

// TestResample_matchesResamplePoly is G-S5e option (b)'s filter check: scipy.signal.resample_poly's output on a deterministic
// signal (a 440 Hz and a 7.9 kHz tone, a 12 kHz one above the output's Nyquist, noise) at 44.1, 48, 22.05 and 8 kHz, pinned
// by scripts/pin_resample_poly_golden.py. Every sample within 1e-6.
func TestResample_matchesResamplePoly(t *testing.T) {
	raw, err := os.ReadFile("../testdata/resample_poly_golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var g struct {
		Scipy string
		Cases []struct {
			Rate int
			X, Y []float64
		}
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	for _, c := range g.Cases {
		x := make([]float32, len(c.X))
		for i, v := range c.X {
			x[i] = float32(v)
		}
		y := Resample(x, c.Rate, 16000)
		if len(y) != len(c.Y) {
			t.Fatalf("%d Hz: %d samples, scipy %d", c.Rate, len(y), len(c.Y))
		}
		worst := 0.0
		for i := range y {
			worst = math.Max(worst, math.Abs(float64(y[i])-c.Y[i]))
		}
		t.Logf("%d Hz -> 16 kHz: %d samples, max |diff| against scipy %s's resample_poly %.2e", c.Rate, len(y), g.Scipy, worst)
		if worst > 1e-6 {
			t.Errorf("%d Hz: max |diff| %.2e against resample_poly, over 1e-6", c.Rate, worst)
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
