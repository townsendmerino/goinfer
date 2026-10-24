//go:build cuda

package cuda

import (
	"math"
	"sync"
	"sync/atomic"
)

// Host replicas of the MoE expert GEMVs (cuda/moe.cu's gemv_w4a8_moe and the product gemv_w4a8_moe_wacc accumulates), for
// experts the CPU computes instead of copying them to the device (docs/tasks/task-multimodal-support-2026-10.md, S20 step 4).
// The result must be bit-identical to the kernel's, so the float operations follow the committed PTX (cuda/testdata/moe.ptx)
// one for one: per lane, facc + fma(s0, p0, round(s1*p1)) over each pair of words 32 apart, one fma per word of the ragged
// tail, the warp's shuffle tree, then the activation scale. The integer products are exact in any order.

// moeCPUDefect plants one arithmetic defect for the gate that must see it (G-S20h); the zero value is the kernel's arithmetic.
// Each changes the ORDER of float additions: a scale times a word's integer product (11 by at most 13 significant bits) is
// exact in float32, so how the multiplies pair with the adds cannot change a result, and a defect there would be invisible.
type moeCPUDefect struct {
	unfoldedPair  bool // facc = (facc + s0·p0) + s1·p1 instead of facc + (s0·p0 + s1·p1)
	blocksReverse bool // each lane walks its 64-word blocks last to first
	serialTree    bool // the 32 lanes summed in lane order instead of the shuffle tree
}

// fmaf32 is a float32 fused multiply-add, rounded once: the float64 product of two float32 values is exact, TwoSum gives the
// float64 sum's error, and a sum that lands exactly on a float32 midpoint is nudged toward that error before the one rounding
// to float32 (the only case where rounding twice differs from rounding once).
func fmaf32(a, b, c float32) float32 {
	p := float64(a) * float64(b)
	cc := float64(c)
	s := p + cc
	bb := s - p
	e := (p - (s - bb)) + (cc - bb)
	if e != 0 && isF32Midpoint(s) {
		s = math.Nextafter(s, math.Copysign(math.Inf(1), e))
	}
	return float32(s)
}

// isF32Midpoint reports whether the float64 s lies exactly halfway between two adjacent float32 values.
func isF32Midpoint(s float64) bool {
	r := float64(float32(s))
	if r == s || math.IsInf(r, 0) {
		return false
	}
	var lo, hi float64
	if r < s {
		lo, hi = r, float64(math.Nextafter32(float32(r), float32(math.Inf(1))))
	} else {
		lo, hi = float64(math.Nextafter32(float32(r), float32(math.Inf(-1)))), r
	}
	return s-lo == hi-s
}

// f16ToF32 is __half2float: exact.
func f16ToF32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	man := uint32(h) & 0x3ff
	switch {
	case exp == 0 && man == 0:
		return math.Float32frombits(sign)
	case exp == 0: // subnormal half: normalize
		e := int32(-14)
		for man&0x400 == 0 {
			man <<= 1
			e--
		}
		man &= 0x3ff
		return math.Float32frombits(sign | uint32(e+127)<<23 | man<<13)
	case exp == 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | man<<13)
	}
	return math.Float32frombits(sign | (exp-15+127)<<23 | man<<13)
}

// w4a8WordDot is the kernel's two chained __dp4a over one packed word: its low nibbles (less 8) against the four int8 of
// aLo, its high nibbles against aHi.
func w4a8WordDot(w uint32, aLo, aHi int32) int32 {
	var p int32
	for b := 0; b < 4; b++ {
		lo := int32((w>>(8*b))&0xf) - 8
		hi := int32((w>>(8*b+4))&0xf) - 8
		p += lo*int32(int8(aLo>>(8*b))) + hi*int32(int8(aHi>>(8*b)))
	}
	return p
}

// w4a8RowRef is one weight row of gemv_w4a8_moe before its activation scale: w the row's packed words (Kwords), s its f16
// group scales (one per four words), a the activation packed four int8 per int32 (2*Kwords).
func w4a8RowRef(w []uint32, s []uint16, a []int32, d moeCPUDefect) float32 {
	kw := len(w)
	var lane [32]float32
	full := kw / 64 * 64
	for i := 0; i < full; i += 64 {
		base := i
		if d.blocksReverse {
			base = full - 64 - i
		}
		for l := 0; l < 32; l++ {
			wi, wj := base+l, base+l+32
			p0 := float32(w4a8WordDot(w[wi], a[2*wi], a[2*wi+1]))
			p1 := float32(w4a8WordDot(w[wj], a[2*wj], a[2*wj+1]))
			s0, s1 := f16ToF32(s[wi>>2]), f16ToF32(s[wj>>2])
			if d.unfoldedPair {
				lane[l] = float32(float32(lane[l]+float32(s0*p0)) + float32(s1*p1))
				continue
			}
			lane[l] = float32(lane[l] + fmaf32(s0, p0, float32(s1*p1)))
		}
	}
	for base := full; base < kw; base += 32 {
		for l := 0; l < 32; l++ {
			wi := base + l
			if wi >= kw {
				continue
			}
			p := float32(w4a8WordDot(w[wi], a[2*wi], a[2*wi+1]))
			lane[l] = fmaf32(f16ToF32(s[wi>>2]), p, lane[l])
		}
	}
	if d.serialTree {
		var acc float32
		for l := 0; l < 32; l++ {
			acc = float32(acc + lane[l])
		}
		return acc
	}
	for off := 16; off > 0; off >>= 1 {
		for l := 0; l < off; l++ {
			lane[l] = float32(lane[l] + lane[l+off])
		}
	}
	return lane[0]
}

// w4a8MoEGemvRef is gemv_w4a8_moe for one expert on the host: rows [0, n) of the expert whose packed rows start at w (Kwords
// words per row) and scales at s (Kgroups per row), dst[r] = row result × aScale.
func w4a8MoEGemvRef(w []uint32, s []uint16, a []int32, aScale float32, n, kwords, kgroups int, dst []float32, d moeCPUDefect) {
	for r := 0; r < n; r++ {
		dst[r] = float32(w4a8RowRef(w[r*kwords:(r+1)*kwords], s[r*kgroups:(r+1)*kgroups], a, d) * aScale)
	}
}

// moeActTables is one int8 activation (2*kwords int32, four int8 each) laid out for w4a8RowsAVX2: lo[w] and hi[w] are the quads
// word w's low and high nibbles multiply, and sum8[w] is 8 times the sum of those eight int8, which takes out the nibbles'
// offset of 8 (the kernel subtracts it per nibble; the integer total is the same).
type moeActTables struct {
	a            []int32
	lo, hi, sum8 []int32
	kwords       int
}

func newMoEActTables(a []int32) moeActTables {
	kw := len(a) / 2
	t := moeActTables{a: a, lo: make([]int32, kw), hi: make([]int32, kw), sum8: make([]int32, kw), kwords: kw}
	for w := 0; w < kw; w++ {
		t.lo[w], t.hi[w] = a[2*w], a[2*w+1]
		var sum int32
		for b := 0; b < 4; b++ {
			sum += int32(int8(a[2*w]>>(8*b))) + int32(int8(a[2*w+1]>>(8*b)))
		}
		t.sum8[w] = 8 * sum
	}
	return t
}

// rows computes len(out) consecutive weight rows (w and s start at the first row) before the activation scale, on the AVX2
// kernel when this CPU has it and the row length allows, else on the reference.
func (t moeActTables) rows(w []uint32, s []uint16, out []float32) {
	n, kw := len(out), t.kwords
	if n == 0 {
		return
	}
	if moeCPUAsm && kw%8 == 0 {
		_ = w[n*kw-1]
		_ = s[n*kw/4-1]
		w4a8RowsAVX2(&w[0], &s[0], &t.lo[0], &t.hi[0], &t.sum8[0], kw, n, &out[0])
		return
	}
	kg := kw / 4
	for r := 0; r < n; r++ {
		out[r] = w4a8RowRef(w[r*kw:(r+1)*kw], s[r*kg:(r+1)*kg], t.a, moeCPUDefect{})
	}
}

// moeCPUTask is one run of weight rows for the host: rows len(out) of a projection whose rows start at w and s, against the
// activation tab, results before the activation scale.
type moeCPUTask struct {
	tab *moeActTables
	w   []uint32
	s   []uint16
	out []float32
}

// moeCPUChunkRows is the rows one worker takes at a time: small enough that eight workers share a 704-row projection, large
// enough that the hand-off is a fraction of the chunk's memory time.
const moeCPUChunkRows = 64

// moeCPUPool runs host expert rows on n persistent goroutines. run hands every task's rows out in chunks and returns when all
// are done; one run at a time (the resident's executor is its only caller).
type moeCPUPool struct {
	n      int
	start  []chan struct{}
	done   sync.WaitGroup
	tasks  []moeCPUTask
	chunks []moeCPUChunk
	next   atomic.Int64
}

type moeCPUChunk struct{ task, r0, r1 int }

func newMoECPUPool(n int) *moeCPUPool {
	p := &moeCPUPool{n: n, start: make([]chan struct{}, n)}
	for i := range p.start {
		p.start[i] = make(chan struct{}, 1)
		go p.worker(p.start[i])
	}
	return p
}

func (p *moeCPUPool) worker(start chan struct{}) {
	for range start {
		for {
			c := p.next.Add(1) - 1
			if c >= int64(len(p.chunks)) {
				break
			}
			ch := p.chunks[c]
			t := p.tasks[ch.task]
			kw := t.tab.kwords
			t.tab.rows(t.w[ch.r0*kw:ch.r1*kw], t.s[ch.r0*kw/4:ch.r1*kw/4], t.out[ch.r0:ch.r1])
		}
		p.done.Done()
	}
}

func (p *moeCPUPool) run(tasks []moeCPUTask) {
	p.tasks = tasks
	p.chunks = p.chunks[:0]
	for i, t := range tasks {
		for r := 0; r < len(t.out); r += moeCPUChunkRows {
			p.chunks = append(p.chunks, moeCPUChunk{task: i, r0: r, r1: min(r+moeCPUChunkRows, len(t.out))})
		}
	}
	p.next.Store(0)
	p.done.Add(p.n)
	for _, s := range p.start {
		s <- struct{}{}
	}
	p.done.Wait()
}

// close stops the workers.
func (p *moeCPUPool) close() {
	for _, s := range p.start {
		close(s)
	}
}
