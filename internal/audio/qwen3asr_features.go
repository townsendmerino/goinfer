package audio

import (
	"fmt"
	"math"
)

// Qwen3-ASR's front end: transformers 5.15.0's Qwen3ASRFeatureExtractor as the processor calls it (padding=True, truncation=False, a single clip). The arithmetic is Whisper's
// (whisper_features.go), the framing is not: the clip is NOT padded to 30 s and NOT truncated, so the centre padding reflects real audio at the clip's end; a clip under 8,000 samples is
// zero-padded to 8,000 first (min_length; its mask is deliberately not adjusted, as in the original library); the mel axis is then right-padded with zeros to a multiple of 100 (2 * n_window)
// and the mask with zeros likewise. The reference has only a torch float32 path, so a port that works in float64 differs from it by float32 rounding (the Whisper paths differ by up to 3e-05).
const (
	QwenASRMels        = 128
	QwenASRChunkFrames = 100 // 2 * n_window
	qwenASRMinSamples  = 8000
)

// QwenASRFeats is the extractor's output for one clip.
type QwenASRFeats struct {
	Data  []float32 // [128][T], mel-major, T a multiple of 100
	T     int       // padded frame count
	Valid int       // real frames: floor(max(n, 8000) / 160)
	Mask  []uint8   // T entries, 1 for a real frame
}

// Planted defects for G-S14b1's red runs. Zero is the shipped computation.
const (
	qwenASRDefectNone = iota
	qwenASRDefectZeroPadEnd
	qwenASRDefectPadTo30s
	qwenASRDefectTruncate30s
	qwenASRDefectNoMultiple
	qwenASRDefectNoMinLength
	qwenASRDefectMaskOnes
)

// QwenASRDefectCountForTest is the number of planted defects QwenASRFeaturesDefectForTest accepts (1..N).
const QwenASRDefectCountForTest = qwenASRDefectMaskOnes

// QwenASRFeatures computes the encoder input for one 16 kHz mono clip in [-1, 1].
func QwenASRFeatures(samples []float32) (*QwenASRFeats, error) { return qwenASRFeatures(samples, 0) }

// QwenASRFeaturesDefectForTest is QwenASRFeatures with one planted defect, for gates that must see themselves go red. Never call it in production.
func QwenASRFeaturesDefectForTest(samples []float32, defect int) (*QwenASRFeats, error) {
	return qwenASRFeatures(samples, defect)
}

func qwenASRFeatures(samples []float32, defect int) (*QwenASRFeats, error) {
	if len(samples) == 0 {
		return nil, fmt.Errorf("audio: empty clip")
	}
	n := len(samples)
	pad := n
	if defect != qwenASRDefectNoMinLength && pad < qwenASRMinSamples {
		pad = qwenASRMinSamples
	}
	if defect == qwenASRDefectPadTo30s && pad < WhisperChunkLen {
		pad = WhisperChunkLen
	}
	if defect == qwenASRDefectTruncate30s && pad > WhisperChunkLen {
		pad = WhisperChunkLen
		n = min(n, pad)
	}
	const half = whisperNFFT / 2
	if pad <= half {
		return nil, fmt.Errorf("audio: a clip of %d samples is too short to reflect-pad", pad)
	}
	x := make([]float64, pad+whisperNFFT)
	for i := range min(n, pad) {
		x[half+i] = float64(samples[i])
	}
	for i := 1; i <= half; i++ {
		x[half-i] = x[half+i]
		if defect != qwenASRDefectZeroPadEnd {
			x[half+pad-1+i] = x[half+pad-1-i]
		}
	}
	logmel, frames := whisperLogMel(x, QwenASRMels, 0)
	F := frames - 1 // the last STFT frame is dropped
	mx := float32(math.Inf(-1))
	for m := range QwenASRMels {
		for f := range F {
			if v := logmel[m*frames+f]; v > mx {
				mx = v
			}
		}
	}
	thr := mx - 8
	T := F
	if defect != qwenASRDefectNoMultiple {
		T = (F + QwenASRChunkFrames - 1) / QwenASRChunkFrames * QwenASRChunkFrames
	}
	out := &QwenASRFeats{Data: make([]float32, QwenASRMels*T), T: T, Valid: F, Mask: make([]uint8, T)}
	for m := range QwenASRMels {
		for f := range F {
			v := logmel[m*frames+f]
			if v < thr {
				v = thr
			}
			out.Data[m*T+f] = (v + 4) / 4
		}
	}
	for f := range T {
		if f < F || defect == qwenASRDefectMaskOnes {
			out.Mask[f] = 1
		}
	}
	return out, nil
}
