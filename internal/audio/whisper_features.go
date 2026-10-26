package audio

import (
	"fmt"
	"math"
	"sync"
)

// The Whisper log-mel front end: transformers' WhisperFeatureExtractor, which Whisper, Voxtral, Qwen3-ASR and Qwen3-Omni all use. It is NOT Gemma 4's (gemma4_features.go):
// a 400-point FFT (not a power of two), the power spectrum, a Slaney mel bank with Slaney area normalisation, log10 clamped to each clip's maximum minus 8, and the last STFT
// frame dropped.
//
// Parity reference: transformers 5.15.0's WhisperFeatureExtractor, its NumPy path (_np_extract_fbank_features, float64 arithmetic) and its torch path (float32, what __call__ uses).
// The two differ by up to 3.1e-05 on real inputs; this port follows the NumPy path, including the places it rounds to float32: the FFT result is stored as complex64, and the log
// is cast to float32 before the clamp and the final (x + 4) / 4 (docs/tasks/task-multimodal-support-2026-10.md, G-S14a, in goinfer).
const (
	WhisperSampleRate = 16000
	WhisperChunkLen   = 480000 // 30 s: the extractor pads to this and truncates past it
	WhisperFrames     = 3000   // frames per 30 s chunk, the extractor's nb_max_frames
	whisperNFFT       = 400
	whisperHop        = 160
	whisperBins       = whisperNFFT/2 + 1
	whisperMelFloor   = 1e-10
)

// WhisperValidFrames is the number of non-padding feature frames for n samples: the extractor's attention mask, ceil(n/160), capped at one chunk.
func WhisperValidFrames(n int) int {
	if n <= 0 {
		return 0
	}
	return min((n+whisperHop-1)/whisperHop, WhisperFrames)
}

// Planted defects for the gate's red runs (G-S14a). Zero is the shipped computation.
const (
	whisperDefectNone = iota
	whisperDefectHTK
	whisperDefectMagnitude
	whisperDefectNoClamp
	whisperDefectSymmetricHann
	whisperDefectZeroPad
	whisperDefectNoScale
)

// WhisperDefectCountForTest is the number of planted defects WhisperFeaturesDefectForTest accepts (1..N); 0 means none.
const WhisperDefectCountForTest = whisperDefectNoScale

// WhisperFeatures returns the [mels x 3000] log-mel features of a clip of 16 kHz mono samples in [-1, 1], row-major (mel-major), exactly as WhisperFeatureExtractor does with its defaults:
// the clip zero-padded to 30 s, or truncated to 30 s. mels is 80 or 128. The second result is WhisperValidFrames of the clip.
func WhisperFeatures(samples []float32, mels int) ([]float32, int, error) {
	return whisperFeatures(samples, mels, whisperDefectNone)
}

// WhisperFeaturesDefectForTest is WhisperFeatures with one planted defect (1..WhisperDefectCountForTest), for gates that must see themselves go red. Never call it in production.
func WhisperFeaturesDefectForTest(samples []float32, mels, defect int) ([]float32, int, error) {
	return whisperFeatures(samples, mels, defect)
}

func whisperFeatures(samples []float32, mels, defect int) ([]float32, int, error) {
	if mels != 80 && mels != 128 {
		return nil, 0, fmt.Errorf("audio: Whisper features take 80 or 128 mels, not %d", mels)
	}
	if len(samples) == 0 {
		return nil, 0, fmt.Errorf("audio: empty clip")
	}
	valid := WhisperValidFrames(len(samples))
	// Zero-pad to 30 s (or truncate): the extractor does this to the waveform BEFORE the STFT, so the centre padding below reflects the zeros, not the clip.
	x := make([]float64, WhisperChunkLen+whisperNFFT)
	n := min(len(samples), WhisperChunkLen)
	const half = whisperNFFT / 2
	for i := range n {
		x[half+i] = float64(samples[i])
	}
	// Centre padding by reflection (np.pad mode "reflect": the edge sample is not repeated).
	if defect != whisperDefectZeroPad {
		for i := 1; i <= half; i++ {
			x[half-i] = x[half+i]
			x[half+WhisperChunkLen-1+i] = x[half+WhisperChunkLen-1-i]
		}
	}
	logmel, frames := whisperLogMel(x, mels, defect)
	// Drop the last frame, clamp to the clip's maximum minus 8, scale.
	feat := make([]float32, mels*WhisperFrames)
	mx := float32(math.Inf(-1))
	for m := range mels {
		for f := range WhisperFrames {
			if v := logmel[m*frames+f]; v > mx {
				mx = v
			}
		}
	}
	thr := mx - 8
	for m := range mels {
		for f := range WhisperFrames {
			v := logmel[m*frames+f]
			if defect != whisperDefectNoClamp && v < thr {
				v = thr
			}
			if defect != whisperDefectNoScale {
				v = (v + 4) / 4
			}
			feat[m*WhisperFrames+f] = v
		}
	}
	return feat, valid, nil
}

// WhisperFeaturesWindows is the feature computation of Voxtral's processor (VoxtralProcessor._retrieve_input_features): the clip is zero-padded UP to a multiple of 480,000 samples (never
// truncated; a clip already a whole number of windows is not padded further, and an empty one is an error), ONE log-mel is computed over the whole padded signal (so the reflect padding sits at the
// true ends and the clamp's maximum is the whole signal's, not each window's), and the result is split along time into 3000-frame windows. Each window is [mels x 3000] row-major, exactly
// WhisperFeatures' layout. For a clip of up to 30 s it returns one window equal to WhisperFeatures'.
func WhisperFeaturesWindows(samples []float32, mels int) ([][]float32, error) {
	if mels != 80 && mels != 128 {
		return nil, fmt.Errorf("audio: Whisper features take 80 or 128 mels, not %d", mels)
	}
	if len(samples) == 0 {
		return nil, fmt.Errorf("audio: empty clip")
	}
	windows := (len(samples) + WhisperChunkLen - 1) / WhisperChunkLen
	padded := windows * WhisperChunkLen
	const half = whisperNFFT / 2
	x := make([]float64, padded+whisperNFFT)
	for i, v := range samples {
		x[half+i] = float64(v)
	}
	for i := 1; i <= half; i++ {
		x[half-i] = x[half+i]
		x[half+padded-1+i] = x[half+padded-1-i]
	}
	logmel, frames := whisperLogMel(x, mels, whisperDefectNone)
	total := windows * WhisperFrames // the last STFT frame is dropped
	mx := float32(math.Inf(-1))
	for m := range mels {
		for f := range total {
			if v := logmel[m*frames+f]; v > mx {
				mx = v
			}
		}
	}
	thr := mx - 8
	out := make([][]float32, windows)
	for w := range windows {
		feat := make([]float32, mels*WhisperFrames)
		for m := range mels {
			for f := range WhisperFrames {
				v := logmel[m*frames+w*WhisperFrames+f]
				if v < thr {
					v = thr
				}
				feat[m*WhisperFrames+f] = (v + 4) / 4
			}
		}
		out[w] = feat
	}
	return out, nil
}

// whisperLogMel is the STFT, power spectrum, mel projection and log10 of an already centre-padded signal x (len >= 400): [mels][frames] float32, frames = 1 + (len(x)-400)/160, the
// last of them not yet dropped. It is the part Whisper's extractor and Qwen3-ASR's share; the padding around it differs.
func whisperLogMel(x []float64, mels, defect int) ([]float32, int) {
	window := whisperHann(defect == whisperDefectSymmetricHann)
	bank := whisperMelBank(mels, defect == whisperDefectHTK)
	frames := 1 + (len(x)-whisperNFFT)/whisperHop
	spec := make([]float64, whisperBins)
	logmel := make([]float32, mels*frames)
	buf := make([]complex128, whisperNFFT)
	out := make([]complex128, whisperNFFT)
	tmp := make([]complex128, whisperNFFT)
	for f := range frames {
		fr := x[f*whisperHop : f*whisperHop+whisperNFFT]
		for i := range buf {
			buf[i] = complex(fr[i]*window[i], 0)
		}
		whisperFFT(buf, out, tmp)
		for k := range whisperBins {
			// complex64 storage in the reference's NumPy path: round both parts to float32, then |.|^2 in float64 via the modulus.
			re, im := float64(float32(real(out[k]))), float64(float32(imag(out[k])))
			h := math.Hypot(re, im)
			if defect == whisperDefectMagnitude {
				spec[k] = h
			} else {
				spec[k] = h * h
			}
		}
		for m := range mels {
			row := bank[m*whisperBins : (m+1)*whisperBins]
			var s float64
			for k, v := range row {
				s += v * spec[k]
			}
			if s < whisperMelFloor {
				s = whisperMelFloor
			}
			logmel[m*frames+f] = float32(math.Log10(s)) // cast to float32 here, as np.asarray(spectrogram, float32) does
		}
	}
	return logmel, frames
}

// whisperHann is np.hanning(401)[:-1], the periodic Hann the extractor uses (window_function(400, "hann")), or the symmetric np.hanning(400) for the planted defect.
func whisperHann(symmetric bool) []float64 {
	w := make([]float64, whisperNFFT)
	den := float64(whisperNFFT)
	if symmetric {
		den = float64(whisperNFFT - 1)
	}
	for i := range w {
		w[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/den)
	}
	return w
}

var (
	whisperBanks   = map[[2]int][]float64{} // (mels, htk) -> [mels][201]
	whisperBanksMu sync.Mutex
)

// whisperMelBank is transformers' mel_filter_bank(201, mels, 0, 8000, 16000, norm="slaney", mel_scale="slaney") (HTK scale for the planted defect), laid out [mel][bin].
func whisperMelBank(mels int, htk bool) []float64 {
	key := [2]int{mels, 0}
	if htk {
		key[1] = 1
	}
	whisperBanksMu.Lock()
	defer whisperBanksMu.Unlock()
	if b, ok := whisperBanks[key]; ok {
		return b
	}
	toMel, toHz := slaneyHzToMel, slaneyMelToHz
	if htk {
		toMel = func(f float64) float64 { return 2595.0 * math.Log10(1.0+f/700.0) }
		toHz = func(m float64) float64 { return 700.0 * (math.Pow(10, m/2595.0) - 1.0) }
	}
	lo, hi := toMel(0), toMel(8000)
	pts := numpyLinspace(lo, hi, mels+2)
	freqs := make([]float64, mels+2)
	for i, m := range pts {
		freqs[i] = toHz(m)
	}
	fft := numpyLinspace(0, 8000, whisperBins)
	bank := make([]float64, mels*whisperBins)
	for m := range mels {
		enorm := 2.0 / (freqs[m+2] - freqs[m])
		for k, ff := range fft {
			down := -(freqs[m] - ff) / (freqs[m+1] - freqs[m])
			up := (freqs[m+2] - ff) / (freqs[m+2] - freqs[m+1])
			v := math.Max(0, math.Min(down, up))
			bank[m*whisperBins+k] = v * enorm
		}
	}
	whisperBanks[key] = bank
	return bank
}

func slaneyHzToMel(f float64) float64 {
	const minLogHz, minLogMel = 1000.0, 15.0
	logstep := 27.0 / math.Log(6.4)
	if f >= minLogHz {
		return minLogMel + math.Log(f/minLogHz)*logstep
	}
	return 3.0 * f / 200.0
}

func slaneyMelToHz(m float64) float64 {
	const minLogHz, minLogMel = 1000.0, 15.0
	logstep := math.Log(6.4) / 27.0
	if m >= minLogMel {
		return minLogHz * math.Exp(logstep*(m-minLogMel))
	}
	return 200.0 * m / 3.0
}

// numpyLinspace is np.linspace(a, b, n): arange(n)*step + a, with the last point set to b exactly.
func numpyLinspace(a, b float64, n int) []float64 {
	y := make([]float64, n)
	step := (b - a) / float64(n-1)
	for i := range y {
		y[i] = float64(i)*step + a
	}
	y[n-1] = b
	return y
}

var (
	whisperTw   [whisperNFFT]complex128
	whisperTwOn sync.Once
)

// whisperFFT is a 400-point complex DFT by mixed-radix decimation in time (400 = 2^4 * 5^2), float64 throughout. in is clobbered as scratch; out receives the spectrum.
func whisperFFT(in, out, tmp []complex128) {
	whisperTwOn.Do(func() {
		for k := range whisperTw {
			s, c := math.Sincos(-2 * math.Pi * float64(k) / whisperNFFT)
			whisperTw[k] = complex(c, s)
		}
	})
	fftRec(in, 1, whisperNFFT, out, tmp)
}

// fftRec transforms the n points in[0], in[stride], ... into out[:n].
func fftRec(in []complex128, stride, n int, out, tmp []complex128) {
	if n == 1 {
		out[0] = in[0]
		return
	}
	p := 2
	for n%p != 0 {
		p++ // 2, then 3, 4 (never reached: 2 divides first), 5
	}
	m := n / p
	for r := range p {
		fftRec(in[r*stride:], stride*p, m, out[r*m:(r+1)*m], tmp[r*m:(r+1)*m])
	}
	scale := whisperNFFT / n
	for k := range m {
		for q := range p {
			idx := k + q*m
			var s complex128
			for r := range p {
				s += whisperTw[(r*idx*scale)%whisperNFFT] * out[r*m+k]
			}
			tmp[idx] = s
		}
	}
	copy(out[:n], tmp[:n])
}
