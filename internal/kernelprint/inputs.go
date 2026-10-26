package kernelprint

// gen is splitmix64. The suite's inputs come only from it and from exact float32 arithmetic (sums, and products by powers
// of two inside explicit conversions, which Go may not fuse), so every architecture feeds the kernels the same bits.
type gen struct{ s uint64 }

func newGen(seed uint64) *gen { return &gen{s: seed} }

func (g *gen) next() uint64 {
	g.s += 0x9e3779b97f4a7c15
	z := g.s
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// unit returns a value in [-1, 1) on a 2^-23 grid.
func (g *gen) unit() float32 {
	return float32(float32(int32(g.next()>>40)-(1<<23)) * (1.0 / (1 << 23)))
}

// bell returns the sum of four unit values: a hump in (-4, 4).
func (g *gen) bell() float32 {
	return g.unit() + g.unit() + g.unit() + g.unit()
}

// vec returns n bell values times scale (a power of two). Every 13th value is an outlier eight times larger.
func (g *gen) vec(n int, scale float32) []float32 {
	v := make([]float32, n)
	for i := range v {
		x := float32(g.bell() * scale)
		if i%13 == 5 {
			x = float32(x * 8)
		}
		v[i] = x
	}
	return v
}

// mat returns rows×cols values as vec does, with row 1 all zero and row 2 constant when there are enough rows, so the
// quantizers meet a zero maximum and a flat row.
func (g *gen) mat(rows, cols int, scale float32) []float32 {
	m := g.vec(rows*cols, scale)
	if rows > 2 {
		clear(m[cols : 2*cols])
		c := float32(g.bell() * scale)
		for j := range cols {
			m[2*cols+j] = c
		}
	}
	return m
}

// ties returns rows whose maximum magnitude is exactly 127, so the int8 scale is 1 and x.5 values sit exactly on a
// rounding tie.
func ties(cols int) []float32 {
	v := make([]float32, cols)
	for j := range v {
		v[j] = float32(j%254-127) + 0.5
	}
	v[0] = 127
	return v
}

// probes32 is a set of float32 bit patterns for the 16-bit conversions: every exponent from far below binary16's
// subnormals to past its overflow, each with mantissas at the round-to-nearest-even boundaries of a 10-bit and a 7-bit
// target, both signs, plus zeros and infinities.
func probes32() []uint32 {
	mants := []uint32{0, 1, 0x0fff, 0x1000, 0x1001, 0x2000, 0x3000, 0x7fff, 0x8000, 0x8001, 0x18000, 0x7fe000, 0x7fefff, 0x7ff000, 0x7fffff}
	var out []uint32
	for e := uint32(90); e <= 160; e++ {
		for _, m := range mants {
			out = append(out, e<<23|m, 1<<31|e<<23|m)
		}
	}
	return append(out, 0, 1<<31, 0x7f800000, 0xff800000, 1, 0x00400000)
}

// half16 returns n finite binary16 bit patterns with exponents in [lo, hi] (biased, at most 30).
func (g *gen) half16(n int, lo, hi uint16) []uint16 {
	u := make([]uint16, n)
	for i := range u {
		r := g.next()
		e := lo + uint16(r%uint64(hi-lo+1))
		u[i] = uint16(r>>8)&0x8000 | e<<10 | uint16(r>>16)&0x3ff
	}
	return u
}

// bf16 returns n finite bfloat16 bit patterns with exponents in [lo, hi] (biased).
func (g *gen) bf16(n int, lo, hi uint16) []uint16 {
	u := make([]uint16, n)
	for i := range u {
		r := g.next()
		e := lo + uint16(r%uint64(hi-lo+1))
		u[i] = uint16(r>>8)&0x8000 | e<<7 | uint16(r>>16)&0x7f
	}
	return u
}

// bytes returns n random bytes.
func (g *gen) bytes(n int) []byte {
	b := make([]byte, n)
	for i := 0; i < n; i += 8 {
		r := g.next()
		for j := 0; j < 8 && i+j < n; j++ {
			b[i+j] = byte(r >> (8 * j))
		}
	}
	return b
}
