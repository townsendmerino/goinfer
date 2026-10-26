package audio

import (
	"math"
	"math/cmplx"
	"testing"
)

// The valid-frame counts are the sums of transformers 5.15.0's WhisperFeatureExtractor attention mask (goinfer's scripts/pin_whisper_features.py).
func TestWhisperValidFrames(t *testing.T) {
	for _, c := range []struct{ n, want int }{
		{1, 1}, {159, 1}, {160, 1}, {161, 2}, {800, 5}, {16000, 100}, {93680, 586}, {480000, 3000}, {500000, 3000}, {0, 0},
	} {
		if got := WhisperValidFrames(c.n); got != c.want {
			t.Errorf("%d samples: %d valid frames, want %d", c.n, got, c.want)
		}
	}
}

// The mixed-radix FFT against a direct DFT.
func TestWhisperFFT_matchesDFT(t *testing.T) {
	in := make([]complex128, whisperNFFT)
	x := uint64(88172645463325252)
	for i := range in {
		x ^= x << 13
		x ^= x >> 7
		x ^= x << 17
		in[i] = complex(float64(x%2000)/1000-1, 0)
	}
	want := make([]complex128, whisperNFFT)
	for k := range want {
		for n, v := range in {
			want[k] += v * cmplx.Exp(complex(0, -2*math.Pi*float64(k*n)/whisperNFFT))
		}
	}
	buf, out, tmp := append([]complex128(nil), in...), make([]complex128, whisperNFFT), make([]complex128, whisperNFFT)
	whisperFFT(buf, out, tmp)
	for k := range want {
		if d := cmplx.Abs(out[k] - want[k]); d > 1e-9 {
			t.Fatalf("bin %d: FFT %v, DFT %v (|diff| %.3g)", k, out[k], want[k], d)
		}
	}
}

func TestWhisperFeatures_edges(t *testing.T) {
	if _, _, err := WhisperFeatures(nil, 128); err == nil {
		t.Error("an empty clip was accepted")
	}
	if _, _, err := WhisperFeatures(make([]float32, 100), 64); err == nil {
		t.Error("64 mels was accepted")
	}
	// Silence: every log value hits the floor, so after the clamp and the scale the whole array is (-10 + 4) / 4 = -1.5, and nothing is NaN.
	f, valid, err := WhisperFeatures(make([]float32, 32000), 128)
	if err != nil || valid != 200 || len(f) != 128*WhisperFrames {
		t.Fatalf("silence: len %d, valid %d, err %v", len(f), valid, err)
	}
	for i, v := range f {
		if v != -1.5 {
			t.Fatalf("silence value %d = %v, want -1.5", i, v)
		}
	}
	// A clip past 30 s is truncated to it: the features equal those of its first 30 s.
	long := make([]float32, WhisperChunkLen+8000)
	for i := range long {
		long[i] = float32(0.4 * math.Sin(float64(i)*0.05))
	}
	a, va, _ := WhisperFeatures(long, 80)
	b, vb, _ := WhisperFeatures(long[:WhisperChunkLen], 80)
	if va != WhisperFrames || vb != WhisperFrames {
		t.Errorf("valid frames %d and %d, want %d", va, vb, WhisperFrames)
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("truncation changed value %d: %v against %v", i, a[i], b[i])
		}
	}
}

// Every planted defect changes the output on a speech-like input (the gate in goinfer measures by how much; this only proves the seams are wired).
func TestWhisperFeatures_plantedDefectsMove(t *testing.T) {
	x := make([]float32, 16000)
	for i := range x {
		x[i] = float32(0.3*math.Sin(float64(i)*0.07) + 0.1*math.Sin(float64(i)*0.31))
	}
	base, _, _ := WhisperFeatures(x, 128)
	for d := 1; d <= WhisperDefectCountForTest; d++ {
		got, _, err := WhisperFeaturesDefectForTest(x, 128, d)
		if err != nil {
			t.Fatal(err)
		}
		var worst float64
		for i := range got {
			worst = math.Max(worst, math.Abs(float64(got[i]-base[i])))
		}
		if worst < 1e-3 {
			t.Errorf("planted defect %d moved the features by only %.3g", d, worst)
		}
	}
}
