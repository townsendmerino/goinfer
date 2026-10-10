//go:build realckpt

package multimodal

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/audio"
)

// TestGemma4AudioResample is G-S5e of docs/tasks/task-multimodal-support-2026-10.md (S5's follow-up): the LibriSpeech clip at 44.1 kHz mono and at 48 kHz stereo (made offline by
// scripts/make_resample_clips.py with ffmpeg and numpy), through DecodeWAVAnyRate's downmix and resampler and the real E2B
// audio tower (GOINFER_GEMMA4_E2B, default ~/models/gemma-4-E2B-unq), soft tokens against the 16 kHz original's. The
// reference is the same files brought back to 16 kHz by scipy's resample_poly. PASS: goinfer's worst-token and mean
// cosine each >= the reference's - 0.001; 0.001-0.005 below is ambiguous (parked); worse fails. Two planted defects
// must each miss the PASS bar: the 44.1 kHz clip resampled as if it were 48 kHz, and the stereo clip read from its left
// channel only. CPU, float32.
func TestGemma4AudioResample(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	dir := os.Getenv("GOINFER_GEMMA4_E2B")
	if dir == "" {
		dir = filepath.Join(home, "models", "gemma-4-E2B-unq")
	}
	if strings.HasPrefix(dir, "/Volumes/") || strings.HasPrefix(dir, "/srv/models") {
		t.Fatalf("%s is on the archive (CLAUDE.md)", dir)
	}
	enc, err := audio.LoadGemma4AudioEncoder(dir)
	if err != nil {
		t.Skipf("no E2B audio tower: %v", err)
	}
	const base = "../testdata/speech/librispeech-1272-128104-0000"
	read := func(p string) []byte {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	f32 := func(p string) []float32 {
		b := read(p)
		out := make([]float32, len(b)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		}
		return out
	}
	// pcm reads a 16-bit PCM WAV's interleaved samples and channel count, untouched (for the planted defects).
	pcm := func(b []byte) ([]float32, int) {
		ch := int(binary.LittleEndian.Uint16(b[22:24]))
		for off := 12; off+8 <= len(b); {
			id, size := string(b[off:off+4]), int(binary.LittleEndian.Uint32(b[off+4:off+8]))
			if id == "data" {
				out := make([]float32, size/2)
				for i := range out {
					out[i] = float32(int16(binary.LittleEndian.Uint16(b[off+8+2*i:]))) / 32768
				}
				return out, ch
			}
			off += 8 + size + size&1
		}
		t.Fatal("no data chunk")
		return nil, 0
	}
	soft := func(samples []float32) []float32 {
		mel, T, err := audio.Gemma4Features(samples)
		if err != nil {
			t.Fatal(err)
		}
		out, err := enc.Forward(mel, T)
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	d := enc.TextHiddenSize
	cmp := func(got, want []float32) (worst, mean float64, n int) {
		n = min(len(got), len(want)) / d
		worst = 1
		for r := range n {
			var dot, na, nb float64
			for j := range d {
				x, y := float64(got[r*d+j]), float64(want[r*d+j])
				dot, na, nb = dot+x*y, na+x*x, nb+y*y
			}
			c := dot / math.Sqrt(na*nb)
			worst = math.Min(worst, c)
			mean += c / float64(n)
		}
		return worst, mean, n
	}

	orig, err := DecodeWAV(read(base + ".wav"))
	if err != nil {
		t.Fatal(err)
	}
	want := soft(orig)
	type arm struct{ name, refFile, wavFile string }
	verdict := func(gw, gm, rw, rm float64) string {
		switch {
		case gw >= rw-0.001 && gm >= rm-0.001:
			return "PASS"
		case gw >= rw-0.005 && gm >= rm-0.005:
			return "ambiguous (parked)"
		}
		return "FAIL"
	}
	refs := map[string][2]float64{}
	for _, a := range []arm{{"44.1 kHz mono", "-44k1.ref16k.f32", "-44k1.wav"}, {"48 kHz stereo", "-48k-stereo.ref16k.f32", "-48k-stereo.wav"}} {
		got, err := DecodeWAVAnyRate(read(base + a.wavFile))
		if err != nil {
			t.Fatal(err)
		}
		gw, gm, gn := cmp(soft(got), want)
		rw, rm, rn := cmp(soft(f32(base+a.refFile)), want)
		refs[a.name] = [2]float64{rw, rm}
		v := verdict(gw, gm, rw, rm)
		fmt.Fprintf(os.Stderr, "[G-S5e] %s: goinfer worst %.6f mean %.6f (%d tokens, %d samples) | scipy worst %.6f mean %.6f (%d tokens) -> %s\n",
			a.name, gw, gm, gn, len(got), rw, rm, rn, v)
		if gn != len(want)/d {
			t.Errorf("%s: %d soft tokens, the original %d", a.name, gn, len(want)/d)
		}
		if v != "PASS" {
			t.Errorf("%s: G-S5e %s", a.name, v)
		}
	}

	x44, _ := pcm(read(base + "-44k1.wav"))
	lr, ch := pcm(read(base + "-48k-stereo.wav"))
	left := make([]float32, len(lr)/ch)
	for i := range left {
		left[i] = lr[i*ch]
	}
	for _, p := range []struct {
		name, against string
		samples       []float32
	}{
		{"(1) 44.1 kHz resampled as 48 kHz", "44.1 kHz mono", Resample(x44, 48000, 16000)},
		{"(2) stereo, left channel only", "48 kHz stereo", Resample(left, 48000, 16000)},
	} {
		gw, gm, _ := cmp(soft(p.samples), want)
		r := refs[p.against]
		v := verdict(gw, gm, r[0], r[1])
		fmt.Fprintf(os.Stderr, "[G-S5e] planted %s: worst %.6f mean %.6f -> %s\n", p.name, gw, gm, v)
		if v == "PASS" {
			t.Errorf("planted defect %s passed the bar", p.name)
		}
	}
}
