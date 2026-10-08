package multimodal

import "math"

// Audio towers take 16 kHz mono (S5's follow-up, docs/tasks/task-multimodal-support-2026-10.md, G-S5e). A WAV at another rate
// or with several channels is brought to that here: the channels averaged per frame (librosa's to_mono convention), then
// resampled with scipy.signal.resample_poly's own filter, built the same way (owner decision 2026-10-07, G-S5e option (b)):
// up and down reduced by their gcd, a firwin low-pass at 1/max(up, down) of the upsampled Nyquist with unit DC gain, a Kaiser
// window of beta 5.0 over 20·max(up, down)+1 taps, centred and scaled by up, zero padding, ceil(n·up/down) outputs.
// TestResample_matchesResamplePoly holds it to scipy's output (testdata/resample_poly_golden.json).

// resampleBeta is resample_poly's default Kaiser beta, and resampleHalf its half-length in units of max(up, down).
const (
	resampleBeta = 5.0
	resampleHalf = 10
)

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

// Resample converts mono samples at from Hz to to Hz as scipy.signal.resample_poly(x, up, down) does with its defaults.
// Equal rates return x itself.
func Resample(x []float32, from, to int) []float32 {
	if from == to || len(x) == 0 {
		return x
	}
	g := gcd(from, to)
	up, down := to/g, from/g
	h := resamplePolyFilter(up, down)
	last := len(h) - 1
	half := last / 2
	nOut := (len(x)*up + down - 1) / down // ceil(n·up/down)
	out := make([]float32, nOut)
	for n := range nOut {
		// y[n] = sum over k of x[k]·h[n·down − k·up + half]: the upsampled-then-filtered signal at n·down, filter centred.
		t := n*down + half
		kLo := 0
		if t > last {
			kLo = (t - last + up - 1) / up // the smallest k with t − k·up <= last
		}
		kHi := min(t/up, len(x)-1)
		var acc float64
		for k := kLo; k <= kHi; k++ {
			acc += float64(x[k]) * h[t-k*up]
		}
		out[n] = float32(acc)
	}
	return out
}

// resamplePolyFilter is resample_poly's filter: firwin(2·half+1, 1/max(up, down), window=('kaiser', 5.0)), a windowed sinc
// normalised to unit sum, then multiplied by up.
func resamplePolyFilter(up, down int) []float64 {
	maxRate := max(up, down)
	fc := 1 / float64(maxRate)
	n := 2*resampleHalf*maxRate + 1
	h := make([]float64, n)
	var sum float64
	for i := range h {
		m := float64(i) - float64(n-1)/2
		h[i] = fc * sinc(fc*m) * kaiserSym(i, n)
		sum += h[i]
	}
	for i := range h {
		h[i] = h[i] / sum * float64(up)
	}
	return h
}

func sinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	return math.Sin(math.Pi*x) / (math.Pi * x)
}

// kaiserSym is scipy's symmetric Kaiser window (get_window(('kaiser', beta), n, fftbins=False)) at sample i of n.
func kaiserSym(i, n int) float64 {
	if n == 1 {
		return 1
	}
	u := 2*float64(i)/float64(n-1) - 1
	return besselI0(resampleBeta*math.Sqrt(math.Max(0, 1-u*u))) / besselI0(resampleBeta)
}

// besselI0 is the modified Bessel function of the first kind, order 0, by its power series (converges fast for beta 5).
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
