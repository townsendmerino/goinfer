package multimodal

import "math"

// Audio towers take 16 kHz mono (S5's follow-up, docs/tasks/task-multimodal-support-2026-10.md, G-S5e). A WAV at another rate
// or with several channels is brought to that here: the channels averaged per frame (librosa's to_mono convention), then a
// windowed-sinc resampler. G-S5e gates it against scipy's resample_poly through the real E2B audio tower.

// resampleZeros is the kernel's half-width in zero crossings of the lower rate's sinc, and resampleBeta its Kaiser window's
// beta (about 80 dB of stopband). The cutoff is the lower Nyquist itself (resampleCutoff 1, scipy's resample_poly
// convention): Gemma 4's log-mel filterbank runs to exactly 8 kHz, and G-S5e's first reading, with the cutoff at 0.97 of
// Nyquist, lost its top bins (mel bin 127 mean |d log-mel| 0.96 against scipy's 0.58) and failed the gate.
const (
	resampleZeros  = 16
	resampleBeta   = 8.0
	resampleCutoff = 1.0
)

// resampleMaxTableVar caps the polyphase table (phases x taps floats). A rate pair over it computes its kernel per output
// sample instead: slower, same arithmetic (a var so a test can force that path).
var resampleMaxTableVar = 1 << 21

// Downmix averages interleaved frames of ch channels into mono. ch == 1 returns x itself.
func Downmix(x []float32, ch int) []float32 {
	if ch <= 1 {
		return x
	}
	n := len(x) / ch
	out := make([]float32, n)
	for i := range n {
		var s float64
		for c := range ch {
			s += float64(x[i*ch+c])
		}
		out[i] = float32(s / float64(ch))
	}
	return out
}

// Resample converts mono samples at from Hz to to Hz with a Kaiser-windowed sinc low-pass at resampleCutoff of the lower of
// the two Nyquist rates. Equal rates return x itself. The output has round(len(x)·to/from) samples; output n sits at input
// position n·from/to.
func Resample(x []float32, from, to int) []float32 {
	if from == to || len(x) == 0 {
		return x
	}
	g := gcd(from, to)
	up, down := to/g, from/g // output n sits at input position n·down/up
	fc := resampleCutoff * math.Min(1, float64(to)/float64(from))
	half := float64(resampleZeros) / fc // kernel half-width in input samples
	i0 := int(math.Ceil(half))
	taps := 2*i0 + 2 // offsets -i0..i0+1 from the output's floor position: the window's whole support at any phase in [0, 1)
	nOut := int(math.Round(float64(len(x)) * float64(to) / float64(from)))
	out := make([]float32, nOut)

	kernel := func(phase float64, h []float64) { // h[j] weighs x[k-i0+j] for an output at input position k+phase
		var sum float64
		for j := range h {
			t := float64(j-i0) - phase
			v := 0.0
			if a := math.Abs(t); a <= half {
				v = fc * sinc(fc*t) * kaiser(t/half)
			}
			h[j] = v
			sum += v
		}
		for j := range h { // unit gain at DC for every phase
			h[j] /= sum
		}
	}
	apply := func(n int, h []float64) {
		k := n * down / up
		var acc float64
		for j, w := range h {
			if i := k - i0 + j; i >= 0 && i < len(x) {
				acc += w * float64(x[i])
			}
		}
		out[n] = float32(acc)
	}

	if up*taps <= resampleMaxTableVar {
		table := make([][]float64, up)
		for p := range up {
			table[p] = make([]float64, taps)
			kernel(float64(p)/float64(up), table[p])
		}
		for n := range nOut {
			apply(n, table[n*down%up])
		}
		return out
	}
	h := make([]float64, taps)
	for n := range nOut {
		kernel(float64(n*down%up)/float64(up), h)
		apply(n, h)
	}
	return out
}

func sinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	return math.Sin(math.Pi*x) / (math.Pi * x)
}

// kaiser is the Kaiser window at u in [-1, 1].
func kaiser(u float64) float64 {
	return besselI0(resampleBeta*math.Sqrt(math.Max(0, 1-u*u))) / besselI0(resampleBeta)
}

// besselI0 is the modified Bessel function of the first kind, order 0, by its power series (converges fast for the betas
// used here).
func besselI0(x float64) float64 {
	sum, term, q := 1.0, 1.0, x*x/4
	for k := 1; k < 64; k++ {
		term *= q / float64(k*k)
		sum += term
		if term < sum*1e-17 {
			break
		}
	}
	return sum
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}
