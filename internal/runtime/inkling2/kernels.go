// Package inkling2 holds the per-token numerics of the Inkling 2 decoder (inkling2, inkling2_loop) that no other family
// shares: the L2 clip, log scaling, the depth mix of the attention residuals, the cross-pass message controllers, the
// static and dynamic short convolutions, the attention with a relative-position bias and a per-token sink, and the SITU
// expert activation. Every function works on one position over plain slices; decoder owns the weights' matmuls and the KV
// cache, and calls these. The reference is transformers' modeling_inkling2_loop.py and llama.cpp's inkling2.cpp.
package inkling2

import "math"

// UnitRMS writes x / rms(x) (no gain) into dst, computed in float64.
func UnitRMS(dst, x []float32, eps float64) {
	var s float64
	for _, v := range x {
		s += float64(v) * float64(v)
	}
	inv := 1 / math.Sqrt(s/float64(len(x))+eps)
	for i, v := range x {
		dst[i] = float32(float64(v) * inv)
	}
}

// Clip multiplies each of the rows of width d in x by scale (when non-nil, one value per element of x) and then scales a row
// down to an L2 norm of thr when its norm is larger: x·min(1, thr/(|x|+1e-6)), transformers' Inkling2LoopL2Clip.
func Clip(x, scale []float32, d int, thr float64) {
	for r := 0; r+d <= len(x); r += d {
		row := x[r : r+d]
		var s float64
		for i := range row {
			if scale != nil {
				row[i] = float32(float64(row[i]) * float64(scale[r+i]))
			}
			s += float64(row[i]) * float64(row[i])
		}
		f := thr / (math.Sqrt(s) + 1e-6)
		if f < 1 {
			for i := range row {
				row[i] = float32(float64(row[i]) * f)
			}
		}
	}
}

// LogScale is the full-attention layers' position scale 1 + alpha·ln(max(1, (pos+1)/floor)).
func LogScale(pos, floor int, alpha float64) float64 {
	r := float64(pos+1) / float64(floor)
	if r < 1 {
		return 1
	}
	return 1 + alpha*math.Log(r)
}

// Mix is the attention over depth that forms a layer's inputs from the attention residuals: each of the nq queries
// (queries, [nq, hidden]) attends over the unit-RMS-normalized states (normed, each [hidden], already normalized with
// UnitRMS) at scale hidden^-0.5, and out[q] is the weighted sum of the normalized states. out is [nq, hidden].
func Mix(out, queries []float32, nq int, normed [][]float32, hidden int) {
	scale := 1 / math.Sqrt(float64(hidden))
	w := make([]float64, len(normed))
	for q := range nq {
		qv := queries[q*hidden : (q+1)*hidden]
		mx := math.Inf(-1)
		for j, s := range normed {
			var d float64
			for i, v := range s {
				d += float64(qv[i]) * float64(v)
			}
			w[j] = d * scale
			mx = max(mx, w[j])
		}
		var sum float64
		for j := range w {
			w[j] = math.Exp(w[j] - mx)
			sum += w[j]
		}
		o := out[q*hidden : (q+1)*hidden]
		for i := range o {
			var acc float64
			for j, s := range normed {
				acc += w[j] * float64(s[i])
			}
			o[i] = float32(acc / sum)
		}
	}
}

// matvec computes dst = W·x for a row-major W of len(dst) rows, in float64.
func matvec(dst, w, x []float32) {
	n := len(x)
	for r := range dst {
		row := w[r*n : (r+1)*n]
		var s float64
		for i, v := range row {
			s += float64(v) * float64(x[i])
		}
		dst[r] = float32(s)
	}
}

// Control mixes a cross-pass message m into a loop layer's input x in place: g = down·[unitRMS(x); unitRMS(m)], then
// x = (1 - tanh(forget·g))·x + tanh(gain·g)·m. down is [rank, 2*hidden], gain and forget [hidden, rank].
func Control(x, m, down, gain, forget []float32, rank int, eps float64) {
	hidden := len(x)
	cat := make([]float32, 2*hidden)
	UnitRMS(cat[:hidden], x, eps)
	UnitRMS(cat[hidden:], m, eps)
	g := make([]float32, rank)
	matvec(g, down, cat)
	gn := make([]float32, hidden)
	fg := make([]float32, hidden)
	matvec(gn, gain, g)
	matvec(fg, forget, g)
	for i := range x {
		f := math.Tanh(float64(fg[i]))
		a := math.Tanh(float64(gn[i]))
		x[i] = float32((1-f)*float64(x[i]) + a*float64(m[i]))
	}
}

// ConvWindow is one short convolution's history: the K-1 inputs before the current position, oldest first, zero before the
// sequence starts.
type ConvWindow struct {
	rows [][]float32 // at most K-1, oldest first
}

// Reset empties the window.
func (w *ConvWindow) Reset() { w.rows = w.rows[:0] }

// push appends x and keeps the last k-1 rows.
func (w *ConvWindow) push(x []float32, k int) {
	if len(w.rows) == k-1 && k > 1 {
		old := w.rows[0]
		copy(w.rows, w.rows[1:])
		w.rows[k-2] = append(old[:0], x...)
		return
	}
	w.rows = append(w.rows, append([]float32(nil), x...))
}

// back returns the input lag positions before the current one, or nil before the sequence start.
func (w *ConvWindow) back(lag int) []float32 {
	i := len(w.rows) - lag
	if i < 0 {
		return nil
	}
	return w.rows[i]
}

// Conv runs the causal depthwise short convolution with its residual on one position, in place on x ([channels]), and
// pushes x's input into the window. taps is [channels, k] with tap k-1 weighting the current position. dyn, when non-nil, is
// the dynamic taps ([heads, k], tap lag weighting the input lag positions back), shared by the channels/heads channels of each
// head: the output is then (static + dynamic + x)/sqrt(2k+1), else (static + x)/sqrt(k+1).
func Conv(x []float32, taps []float32, k int, dyn []float32, win *ConvWindow) {
	ch := len(x)
	in := append([]float32(nil), x...)
	div := math.Sqrt(float64(k + 1))
	var heads, hw int
	if dyn != nil {
		div = math.Sqrt(float64(2*k + 1))
		heads = len(dyn) / k
		hw = ch / heads
	}
	for c := range ch {
		s := float64(taps[c*k+k-1]) * float64(in[c])
		if dyn != nil {
			s += float64(dyn[(c/hw)*k]) * float64(in[c])
		}
		for lag := 1; lag < k; lag++ {
			p := win.back(lag)
			if p == nil {
				break
			}
			s += float64(taps[c*k+k-1-lag]) * float64(p[c])
			if dyn != nil {
				s += float64(dyn[(c/hw)*k+lag]) * float64(p[c])
			}
		}
		x[c] = float32((s + float64(in[c])) / div)
	}
	win.push(in, k)
}

// Situ is the experts' activation, in place on gate: betaG·tanh(g/betaG)·sigmoid(g) · betaU·tanh(u/betaU).
func Situ(gate, up []float32, betaG, betaU float64) {
	for i, g := range gate {
		gf, uf := float64(g), float64(up[i])
		a := betaG * math.Tanh(gf/betaG) / (1 + math.Exp(-gf))
		gate[i] = float32(a * betaU * math.Tanh(uf/betaU))
	}
}

// Sigmoid returns 1/(1+e^-x).
func Sigmoid(x float32) float32 { return float32(1 / (1 + math.Exp(-float64(x)))) }

// AttendParams is one visit's attention geometry for Attend.
type AttendParams struct {
	Heads, KVHeads, HeadDim, DRel int
	Extent                        int     // the bias is zero at a backward distance >= Extent
	Window                        int     // keys at a backward distance >= Window are not attended; 0 attends all
	Scale                         float64 // q·k multiplier
}

// Attend writes the attention context of one query position into ctx ([Heads*HeadDim]). keys and vals hold nKeys rows of
// [KVHeads*HeadDim] at row stride, for positions base..base+nKeys-1, the last of which is the query's own position pos. The
// logit of key j is q·k·Scale + bias, bias = rel[h]·relBias[pos-j] inside the extent and 0 past it; sinks[h] is one more logit
// that carries no value. rel is [Heads*DRel], relBias [Extent, DRel].
func Attend(p AttendParams, ctx, q, keys, vals []float32, stride, nKeys, base, pos int, rel, relBias, sinks []float32) {
	hd := p.HeadDim
	group := p.Heads / p.KVHeads
	lo := 0
	if p.Window > 0 && pos-base-lo >= p.Window {
		lo = pos - base - p.Window + 1
	}
	logits := make([]float64, nKeys)
	for h := range p.Heads {
		kh := h / group
		qh := q[h*hd : (h+1)*hd]
		rh := rel[h*p.DRel : (h+1)*p.DRel]
		mx := float64(sinks[h])
		for j := lo; j < nKeys; j++ {
			kr := keys[j*stride+kh*hd : j*stride+(kh+1)*hd]
			var d float64
			for i, v := range qh {
				d += float64(v) * float64(kr[i])
			}
			d *= p.Scale
			if dist := pos - (base + j); dist >= 0 && dist < p.Extent {
				rb := relBias[dist*p.DRel : (dist+1)*p.DRel]
				var b float64
				for i, v := range rh {
					b += float64(v) * float64(rb[i])
				}
				d += b
			}
			logits[j] = d
			mx = max(mx, d)
		}
		sum := math.Exp(float64(sinks[h]) - mx)
		for j := lo; j < nKeys; j++ {
			logits[j] = math.Exp(logits[j] - mx)
			sum += logits[j]
		}
		out := ctx[h*hd : (h+1)*hd]
		acc := make([]float64, hd)
		for j := lo; j < nKeys; j++ {
			vr := vals[j*stride+kh*hd : j*stride+(kh+1)*hd]
			wj := logits[j]
			for i, v := range vr {
				acc[i] += wj * float64(v)
			}
		}
		for i := range out {
			out[i] = float32(acc[i] / sum)
		}
	}
}
