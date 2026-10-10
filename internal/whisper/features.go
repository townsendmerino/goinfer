package whisper

import (
	"fmt"
	"math"
	"sync"
)

// The log-mel front end of transformers' WhisperFeatureExtractor as long-form generate calls it (truncation=False, padding="longest"): no padding to 30 s, the log-mel of the whole clip, so the centre
// padding reflects at the clip's true ends and the clamp's maximum is the whole clip's; the last STFT frame dropped, so T = floor(n / 160) frames. aikit's audio.WhisperFeatures pads the waveform to 30 s and
// audio.WhisperFeaturesWindows to a multiple of 30 s (Voxtral's processor), so neither is this one; the arithmetic below is aikit's, ported (the NumPy path of the extractor, float64 with the float32
// roundings the reference makes), and held to the extractor by G-S14g1. If aikit ever exports the unpadded form this file should go.
const (
	// SampleRate is Whisper's 16 kHz.
	SampleRate = 16000
	// WindowFrames is the frames one encoder window takes (30 s).
	WindowFrames = 3000
	nfft         = 400
	hop          = 160
	bins         = nfft/2 + 1
	melFloor     = 1e-10
)

// planted defects of features_test.go: the zero value is the correct extractor.
const (
	featNone          = iota
	featPadTo30s      // the waveform zero-padded to 30 s first (aikit's WhisperFeatures)
	featPadToMultiple // zero-padded to a multiple of 30 s (aikit's WhisperFeaturesWindows)
	featKeepLast      // the last STFT frame kept
	featNoClamp       // no clamp at the maximum minus 8
	featWindowMax     // the clamp's maximum taken per 30 s window, not over the clip
)

// LongestFeatures returns the [mels][T] log-mel features of a clip of 16 kHz mono samples in [-1, 1] (row-major, mel-major) and T = floor(len(samples) / 160).
func LongestFeatures(samples []float32, mels int) ([]float32, int, error) {
	return longestFeatures(samples, mels, featNone)
}

func longestFeatures(samples []float32, mels, defect int) ([]float32, int, error) {
	if mels != 80 && mels != 128 {
		return nil, 0, fmt.Errorf("whisper: features take 80 or 128 mels, not %d", mels)
	}
	n := len(samples)
	if n < nfft/2+1 {
		return nil, 0, fmt.Errorf("whisper: a %d-sample clip is too short for the front end (it needs more than %d)", n, nfft/2)
	}
	switch defect {
	case featPadTo30s:
		n = 30 * SampleRate
	case featPadToMultiple:
		n = (n + 30*SampleRate - 1) / (30 * SampleRate) * 30 * SampleRate
	}
	const half = nfft / 2
	x := make([]float64, n+nfft)
	for i := range min(len(samples), n) {
		x[half+i] = float64(samples[i])
	}
	if defect == featNone || defect == featKeepLast || defect == featNoClamp || defect == featWindowMax {
		for i := 1; i <= half; i++ {
			x[half-i] = x[half+i]
			x[half+n-1+i] = x[half+n-1-i]
		}
	}
	logmel, frames := logMel(x, mels)
	T := frames - 1 // the last STFT frame is dropped
	if defect == featKeepLast {
		T = frames
	}
	maxOver := func(f0, f1 int) float32 {
		mx := float32(math.Inf(-1))
		for m := range mels {
			for f := f0; f < f1; f++ {
				if v := logmel[m*frames+f]; v > mx {
					mx = v
				}
			}
		}
		return mx
	}
	clipMax := maxOver(0, T)
	out := make([]float32, mels*T)
	for f := range T {
		mx := clipMax
		if defect == featWindowMax {
			w0 := f / WindowFrames * WindowFrames
			mx = maxOver(w0, min(w0+WindowFrames, T))
		}
		thr := mx - 8
		for m := range mels {
			v := logmel[m*frames+f]
			if defect != featNoClamp && v < thr {
				v = thr
			}
			out[m*T+f] = (v + 4) / 4
		}
	}
	return out, T, nil
}

// Window is features [frames seek, seek+3000) of feats ([mels][T]) as one encoder input [mels][3000]: a window that runs past the end is padded with ZEROS in the feature domain, as generate's
// _get_input_segment does (not with the features of silence).
func Window(feats []float32, mels, T, seek, numFrames int) []float32 {
	out := make([]float32, mels*WindowFrames)
	for m := range mels {
		copy(out[m*WindowFrames:m*WindowFrames+numFrames], feats[m*T+seek:m*T+seek+numFrames])
	}
	return out
}

func logMel(x []float64, mels int) ([]float32, int) {
	window := hann()
	bank := melBank(mels)
	frames := 1 + (len(x)-nfft)/hop
	spec := make([]float64, bins)
	logmel := make([]float32, mels*frames)
	buf := make([]complex128, nfft)
	out := make([]complex128, nfft)
	tmp := make([]complex128, nfft)
	for f := range frames {
		fr := x[f*hop : f*hop+nfft]
		for i := range buf {
			buf[i] = complex(fr[i]*window[i], 0)
		}
		fft(buf, out, tmp)
		for k := range bins {
			re, im := float64(float32(real(out[k]))), float64(float32(imag(out[k])))
			h := math.Hypot(re, im)
			spec[k] = h * h
		}
		for m := range mels {
			row := bank[m*bins : (m+1)*bins]
			var s float64
			for k, v := range row {
				s += v * spec[k]
			}
			if s < melFloor {
				s = melFloor
			}
			logmel[m*frames+f] = float32(math.Log10(s))
		}
	}
	return logmel, frames
}

// hann is np.hanning(401)[:-1], the periodic Hann window the extractor uses.
func hann() []float64 {
	w := make([]float64, nfft)
	for i := range w {
		w[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(nfft))
	}
	return w
}

var (
	banks   = map[int][]float64{}
	banksMu sync.Mutex
)

// melBank is transformers' mel_filter_bank(201, mels, 0, 8000, 16000, norm="slaney", mel_scale="slaney"), laid out [mel][bin].
func melBank(mels int) []float64 {
	banksMu.Lock()
	defer banksMu.Unlock()
	if b, ok := banks[mels]; ok {
		return b
	}
	pts := linspace(slaneyHzToMel(0), slaneyHzToMel(8000), mels+2)
	freqs := make([]float64, mels+2)
	for i, m := range pts {
		freqs[i] = slaneyMelToHz(m)
	}
	fftFreqs := linspace(0, 8000, bins)
	bank := make([]float64, mels*bins)
	for m := range mels {
		enorm := 2.0 / (freqs[m+2] - freqs[m])
		for k, ff := range fftFreqs {
			down := -(freqs[m] - ff) / (freqs[m+1] - freqs[m])
			up := (freqs[m+2] - ff) / (freqs[m+2] - freqs[m+1])
			bank[m*bins+k] = math.Max(0, math.Min(down, up)) * enorm
		}
	}
	banks[mels] = bank
	return bank
}

func slaneyHzToMel(f float64) float64 {
	const minLogHz, minLogMel = 1000.0, 15.0
	if f >= minLogHz {
		return minLogMel + math.Log(f/minLogHz)*(27.0/math.Log(6.4))
	}
	return 3.0 * f / 200.0
}

func slaneyMelToHz(m float64) float64 {
	const minLogHz, minLogMel = 1000.0, 15.0
	if m >= minLogMel {
		return minLogHz * math.Exp(math.Log(6.4)/27.0*(m-minLogMel))
	}
	return 200.0 * m / 3.0
}

// linspace is np.linspace(a, b, n): arange(n)*step + a, the last point exactly b.
func linspace(a, b float64, n int) []float64 {
	y := make([]float64, n)
	step := (b - a) / float64(n-1)
	for i := range y {
		y[i] = float64(i)*step + a
	}
	y[n-1] = b
	return y
}

var (
	twiddle   [nfft]complex128
	twiddleOn sync.Once
)

// fft is a 400-point complex DFT by mixed-radix decimation in time (400 = 2^4 * 5^2), float64 throughout. in is clobbered as scratch; out receives the spectrum.
func fft(in, out, tmp []complex128) {
	twiddleOn.Do(func() {
		for k := range twiddle {
			s, c := math.Sincos(-2 * math.Pi * float64(k) / nfft)
			twiddle[k] = complex(c, s)
		}
	})
	fftRec(in, 1, nfft, out, tmp)
}

func fftRec(in []complex128, stride, n int, out, tmp []complex128) {
	if n == 1 {
		out[0] = in[0]
		return
	}
	p := 2
	for n%p != 0 {
		p++
	}
	m := n / p
	for r := range p {
		fftRec(in[r*stride:], stride*p, m, out[r*m:(r+1)*m], tmp[r*m:(r+1)*m])
	}
	scale := nfft / n
	for k := range m {
		for q := range p {
			idx := k + q*m
			var s complex128
			for r := range p {
				s += twiddle[(r*idx*scale)%nfft] * out[r*m+k]
			}
			tmp[idx] = s
		}
	}
	copy(out[:n], tmp[:n])
}
