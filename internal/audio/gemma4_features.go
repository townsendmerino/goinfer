// Package audio is aikit's audio towers: Gemma 4's (gemma4_audio, which EmbeddingGemma 2 shares) log-mel front end
// and conformer encoder, in pure Go.
//
// Parity reference: transformers 5.19.0's Gemma4AudioFeatureExtractor and Gemma4AudioModel, read and probed in
// goinfer's docs/measurements/embeddinggemma2-2026-10-06/audio-gate0/spec.md (the section numbers below are that
// spec's).
package audio

import (
	"fmt"
	"math"
	"math/cmplx"
)

// Gemma 4's feature extractor constants (preprocessor_config.json; the extractor recomputes the frame sizes from
// 20 ms / 10 ms at 16 kHz and the FFT size from the frame, and they come out equal).
const (
	Gemma4SampleRate = 16000
	gemma4Frame      = 320
	gemma4Hop        = 160
	gemma4FFT        = 512
	gemma4Mels       = 128
	gemma4MelFloor   = 0.001
	gemma4MaxSamples = 480000 // 30 s: the extractor truncates here
)

// Gemma4ValidFrames is the number of valid log-mel frames for n samples (spec §1.9): frame i is valid iff its whole
// 321-sample window (the 160-sample left pad included) lies in real audio.
func Gemma4ValidFrames(n int) int {
	n = min(n, gemma4MaxSamples)
	if n <= gemma4Hop {
		return 0
	}
	return (n-161)/gemma4Hop + 1
}

// Gemma4SoftTokens is the number of soft tokens the tower emits for t valid frames: two stride-2 convs, each
// ceil(t/2) (spec §3).
func Gemma4SoftTokens(t int) int {
	t = (t + 1) / 2
	return (t + 1) / 2
}

var (
	gemma4Window    [gemma4Frame]float32
	gemma4MelFilter [][]float64 // [257][128]
)

func init() {
	// np.hanning(321)[:-1] cast to float32: the periodic Hann, 0.5 - 0.5 cos(2πn/320) (spec §1.5).
	for n := range gemma4Window {
		gemma4Window[n] = float32(0.5 - 0.5*math.Cos(2*math.Pi*float64(n)/float64(gemma4Frame)))
	}
	gemma4MelFilter = htkMelFilterBank(gemma4FFT/2+1, gemma4Mels, 0, 8000, Gemma4SampleRate)
}

func hzToMelHTK(f float64) float64 { return 2595 * math.Log10(1+f/700) }
func melToHzHTK(m float64) float64 { return 700 * (math.Pow(10, m/2595) - 1) }

// htkMelFilterBank is transformers' mel_filter_bank(..., norm=None, mel_scale="htk") with triangles in Hz:
// [bins][mels] (spec §1.8).
func htkMelFilterBank(bins, mels int, fmin, fmax float64, sr int) [][]float64 {
	mmin, mmax := hzToMelHTK(fmin), hzToMelHTK(fmax)
	ff := make([]float64, mels+2) // filter_freqs
	for i := range ff {
		m := mmin + (mmax-mmin)*float64(i)/float64(mels+1)
		ff[i] = melToHzHTK(m)
	}
	fb := make([][]float64, bins)
	for b := range fb {
		fft := float64(sr/2) * float64(b) / float64(bins-1)
		fb[b] = make([]float64, mels)
		for m := range mels {
			down := -(ff[m] - fft) / (ff[m+1] - ff[m])
			up := (ff[m+2] - fft) / (ff[m+2] - ff[m+1])
			fb[b][m] = math.Max(0, math.Min(down, up))
		}
	}
	return fb
}

// Gemma4Features is the extractor's log-mel for 16 kHz mono samples in [-1, 1]: [T, 128] float32 over the T valid
// frames only (Gemma4ValidFrames), which is all the tower reads (spec §3). Samples past 30 s are dropped, as the
// extractor truncates. There is no resampling: the caller supplies 16 kHz.
func Gemma4Features(samples []float32) ([]float32, int, error) {
	samples = samples[:min(len(samples), gemma4MaxSamples)]
	T := Gemma4ValidFrames(len(samples))
	if T == 0 {
		return nil, 0, fmt.Errorf("audio: %d samples is under one frame (more than %d needed at 16 kHz)", len(samples), gemma4Hop)
	}
	out := make([]float32, T*gemma4Mels)
	buf := make([]complex128, gemma4FFT)
	mag := make([]float64, gemma4FFT/2+1)
	for t := range T {
		// Frame t covers samples [t*hop - 160, t*hop + 160) of the original (the 160-sample left pad is zeros).
		for n := range buf {
			buf[n] = 0
		}
		for n := range gemma4Frame {
			i := t*gemma4Hop + n - gemma4Frame/2
			var s float32
			if i >= 0 {
				s = samples[i]
			}
			buf[n] = complex(float64(s*gemma4Window[n]), 0) // the product in f32, as the extractor does (§1.5)
		}
		fft512(buf)
		for b := range mag {
			mag[b] = cmplx.Abs(buf[b]) // magnitude, not power
		}
		row := out[t*gemma4Mels : (t+1)*gemma4Mels]
		for m := range gemma4Mels {
			var s float64
			for b, v := range mag {
				s += v * gemma4MelFilter[b][m]
			}
			row[m] = float32(math.Log(s + gemma4MelFloor))
		}
	}
	return out, T, nil
}

// fft512 is an in-place radix-2 FFT of length 512 in float64.
func fft512(a []complex128) {
	n := len(a)
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			a[i], a[j] = a[j], a[i]
		}
	}
	for size := 2; size <= n; size <<= 1 {
		for start := 0; start < n; start += size {
			for k := range size / 2 {
				wk := cmplx.Rect(1, -2*math.Pi*float64(k)/float64(size))
				u, v := a[start+k], a[start+k+size/2]*wk
				a[start+k], a[start+k+size/2] = u+v, u-v
			}
		}
	}
}
