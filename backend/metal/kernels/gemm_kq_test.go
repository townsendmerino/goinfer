//go:build darwin

package kernels

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"
)

// kqKinds are gemm_kq_tile's weight kinds, in its KIND order.
var kqKinds = []string{"q4k", "q5k", "q6k", "int8"}

// kqTiles are the tiles gemm_kq_tile is instantiated at, the int4 GEMM's set (gemmTileSize picks among them).
var kqTiles = []struct {
	name   string
	tm, tn int
}{{"m64n64", 64, 64}, {"m32n64", 32, 64}, {"m64n32", 64, 32}, {"m32n32", 32, 32}, {"m16n32", 16, 32}}

// kqSegment is one segment's device bytes and, per weight, the value the q4k lane's decode kernels compute with (float64,
// exact for these formats).
type kqSegment struct {
	kind   string
	raw    []byte    // the segment buffer as q4kLaneBufOn lays it out
	scales []float32 // int8: one scale a row; nil for the block kinds
	w      []float64 // N*K, row-major
}

// kqScaleMin is ggml's get_scale_min_k4.
func kqScaleMin(j int, q []byte) (sc, m float64) {
	if j < 4 {
		return float64(q[j] & 63), float64(q[j+4] & 63)
	}
	return float64(q[j+4]&15 | (q[j-4]>>6)<<4), float64(q[j+4]>>4 | (q[j]>>6)<<4)
}

// kqMakeSegment builds an N x K segment of the kind with random contents and its reference weights: Q4_K and Q5_K as GGUF
// super-blocks (ggml's dequantize_row_q4_K / _q5_K), Q6_K in the lane's planes (ql, qh, scales, d over the segment's T
// blocks; ggml's dequantize_row_q6_K), int8 as row-major codes with an f32 scale a row.
func kqMakeSegment(rng *rand.Rand, kind string, N, K int) kqSegment {
	NB := K / 256
	f16 := func(lo, hi float64) (uint16, float64) {
		h := F32ToF16(float32(lo + (hi-lo)*rng.Float64()))
		return h, float64(f16ToF32(h))
	}
	s := kqSegment{kind: kind, w: make([]float64, N*K)}
	switch kind {
	case "q4k", "q5k":
		bb := map[string]int{"q4k": 144, "q5k": 176}[kind]
		qs := map[string]int{"q4k": 16, "q5k": 48}[kind]
		s.raw = make([]byte, N*NB*bb)
		for g := range N * NB {
			blk := s.raw[g*bb : (g+1)*bb]
			for i := 4; i < bb; i++ {
				blk[i] = byte(rng.Intn(256))
			}
			if g%5 == 0 { // the 6-bit range ends: every scale and min at 63
				for i := 4; i < 16; i++ {
					blk[i] = 0xFF
				}
			}
			hd, d := f16(0.0005, 0.02)
			hm, dmin := f16(0.0002, 0.01)
			blk[0], blk[1], blk[2], blk[3] = byte(hd), byte(hd>>8), byte(hm), byte(hm>>8)
			n, sb := g/NB, g%NB
			for e := range 256 {
				j, l := e/32, e%32
				q := int(blk[qs+(j/2)*32+l]>>(4*(j&1))) & 15
				if kind == "q5k" && blk[16+l]>>j&1 == 1 {
					q += 16
				}
				sc, m := kqScaleMin(j, blk[4:16])
				s.w[n*K+sb*256+e] = d*sc*float64(q) - dmin*m
			}
		}
	case "q6k":
		T := N * NB
		s.raw = make([]byte, T*210)
		for i := range T * 208 {
			s.raw[i] = byte(rng.Intn(256))
		}
		ds := make([]float64, T)
		for g := range T {
			h, d := f16(0.0002, 0.004)
			s.raw[T*208+2*g], s.raw[T*208+2*g+1], ds[g] = byte(h), byte(h>>8), d
		}
		for g := range T {
			n, sb := g/NB, g%NB
			ql, qh, sc := s.raw[g*128:], s.raw[T*128+g*64:], s.raw[T*192+g*16:]
			for e := range 256 {
				half, qq, l := e/128, (e%128)/32, e%32
				q := int(ql[half*64+(qq&1)*32+l]>>(4*(qq>>1)))&15 | int(qh[half*32+l]>>(2*qq))&3<<4
				s.w[n*K+sb*256+e] = ds[g] * float64(int8(sc[half*8+l/16+2*qq])) * float64(q-32)
			}
		}
	case "int8":
		s.raw, s.scales = make([]byte, N*K), make([]float32, N)
		for n := range N {
			s.scales[n] = float32(1+rng.Intn(1000)) * 2e-5
			for k := range K {
				c := int8(rng.Intn(255) - 127)
				s.raw[n*K+k] = byte(c)
				s.w[n*K+k] = float64(float32(c) * s.scales[n])
			}
		}
	}
	return s
}

// kqRNE is a float32 rounded to the nearest half, ties to even: Metal's half(float). (f16.Bits rounds ties up.)
func kqRNE(f float32) uint16 {
	b := math.Float32bits(f)
	sign, a := uint16(b>>16)&0x8000, b&0x7fffffff
	switch {
	case a > 0x7f800000:
		return sign | 0x7e00
	case a < 0x38800000: // below 2^-14: a half subnormal, in units of 2^-24
		return sign | uint16(math.RoundToEven(float64(math.Float32frombits(a))*(1<<24)))
	}
	h := (a>>23-112)<<10 | (a&0x7fffff)>>13
	if rem := a & 0x1fff; rem > 0x1000 || rem == 0x1000 && h&1 == 1 {
		h++
	}
	if h >= 0x7c00 {
		return sign | 0x7c00
	}
	return sign | uint16(h)
}

// TestKQRNE_matchesMetal holds kqRNE to the device's own conversion on every float32 bit pattern of a sweep that crosses the
// subnormal, normal and overflow ranges and every tie, so the staging test below compares against Metal's rounding.
func TestKQRNE_matchesMetal(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseObjects()
	defer d.ReleaseAll()
	lib, err := d.CompileLibrary(`
#include <metal_stdlib>
using namespace metal;
kernel void to_half(device const float* x[[buffer(0)]], device half* y[[buffer(1)]], uint i[[thread_position_in_grid]]) {
    y[i] = half(x[i]);
}`, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	p, err := d.NewComputePipeline(lib, "to_half")
	if err != nil {
		t.Fatal(err)
	}
	var xs []float32
	for e := uint32(90); e <= 145; e++ { // exponents from deep below the half range to past its top
		for m := uint32(0); m < 1<<23; m += 0x7ff {
			xs = append(xs, math.Float32frombits(e<<23|m), -math.Float32frombits(e<<23|m))
			xs = append(xs, math.Float32frombits(e<<23|m&^0x1fff|0x1000)) // a tie at the normal cut
		}
	}
	x, y := NewBufferFloats(d, xs), NewBufferU16s(d, make([]uint16, len(xs)))
	cq := d.NewCommandQueue()
	e := cq.Begin()
	e.Dispatch(p, len(xs), 64, x, y)
	e.End()
	if err := e.Err(); err != nil {
		t.Fatal(err)
	}
	got := y.U16s()
	bad := 0
	for i, f := range xs {
		if w := kqRNE(f); got[i] != w {
			if bad++; bad <= 5 {
				t.Errorf("half(%g = %#08x): device %#04x, kqRNE %#04x", f, math.Float32bits(f), got[i], w)
			}
		}
	}
	if bad > 0 {
		t.Fatalf("%d of %d conversions differ", bad, len(xs))
	}
	t.Logf("%d conversions equal", len(xs))
}

// kqDispatch runs one gemm_kq tile over segment s: rows activation rows of A (f16, K a row) into columns [coff, coff+N) of C
// (ldc a row), in the given epilogue mode.
func kqDispatch(t *testing.T, d *Device, cq Queue, p Pipeline, tm, tn int, s kqSegment, a Buffer, rows, N, K int,
	c Buffer, ldc, coff int, bias Buffer, mode uint32) {
	t.Helper()
	w := NewBufferInt8(d, asInt8(s.raw))
	ws := w
	if s.scales != nil {
		ws = NewBufferFloats(d, s.scales)
	}
	e := cq.Begin()
	e.Dispatch2D(p, (N+tn-1)/tn, (rows+tm-1)/tm, 128, 1, a, w, ws, c, NewBufferU32(d, uint32(rows)), NewBufferU32(d, uint32(N)),
		NewBufferU32(d, uint32(K)), bias, NewBufferU32(d, mode), NewBufferU32(d, uint32(ldc)), NewBufferU32(d, uint32(coff)))
	e.End()
	if err := e.Err(); err != nil {
		t.Fatalf("%s %dx%d: %v", s.kind, N, K, err)
	}
}

// TestGemmKQ_stagesDecodeWeights: gemm_kq_tile stages each weight as the q4k lane's decode kernels compute with it, rounded once
// to half. Driven with one-hot activation rows (row m is 1 at k = m), the GEMM's output row m is the staged weight column k = m
// (one product by 1.0 and zeros, stored as half), so every staged weight is read back bit for bit and compared with the
// reference rounded to nearest even: d*sc*q - dmin*m for Q4_K and Q5_K, d*sc*(q-32) for Q6_K in the lane's planes, q*scale for
// int8 (-0 and +0 read back alike, as they weigh alike). Every kind at every tile, N not a multiple of the tile, K two super-blocks.
func TestGemmKQ_stagesDecodeWeights(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseObjects()
	defer d.ReleaseAll()
	lib, err := d.CompileLibrary(PrefillKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	cq := d.NewCommandQueue()
	rng := rand.New(rand.NewSource(8))
	const N, K = 72, 512
	onehot := make([]uint16, K*K)
	for m := range K {
		onehot[m*K+m] = F32ToF16(1)
	}
	a, bias := NewBufferU16s(d, onehot), NewBufferFloats(d, make([]float32, N))
	checked := 0
	for _, kind := range kqKinds {
		s := kqMakeSegment(rng, kind, N, K)
		for _, tl := range kqTiles {
			p, err := d.NewComputePipeline(lib, fmt.Sprintf("gemm_kq_%s_%s", kind, tl.name))
			if err != nil {
				t.Fatalf("pipeline: %v", err)
			}
			c := NewBufferU16s(d, make([]uint16, K*N))
			kqDispatch(t, d, cq, p, tl.tm, tl.tn, s, a, K, N, K, c, N, 0, bias, 0)
			got := c.U16s()
			bad := 0
			for k := range K {
				for n := range N {
					want := kqRNE(float32(s.w[n*K+k]))
					if want == 0x8000 { // the accumulator starts at +0, so a staged -0 reads back as +0
						want = 0
					}
					if g := got[k*N+n]; g != want {
						if bad++; bad <= 3 {
							t.Errorf("%s %s: weight [%d,%d] staged %#04x (%g), decode's value %g rounds to %#04x", kind, tl.name, n, k,
								g, f16ToF32(g), s.w[n*K+k], want)
						}
					}
				}
			}
			if bad > 0 {
				t.Fatalf("%s %s: %d of %d staged weights differ", kind, tl.name, bad, N*K)
			}
			checked += N * K
		}
	}
	t.Logf("%d staged weights equal to decode's, rounded to half", checked)
}

// TestGemmKQ_segmentsMatchReference: two segments of different kinds side by side in one output, as a fused group's segments
// write it (ldc the group's width, each at its column offset), against a float64 reference over the same f16 activations and
// half weights, in the three epilogue modes. Bound: one half rounding of the result plus f32 accumulation over K products
// (TestGemmW8Tile_matchesReference's); every tile equals m64n64 bit for bit, and no column outside a segment is written.
func TestGemmKQ_segmentsMatchReference(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	defer d.ReleaseObjects()
	defer d.ReleaseAll()
	lib, err := d.CompileLibrary(PrefillKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	cq := d.NewCommandQueue()
	rng := rand.New(rand.NewSource(9))
	for _, pair := range [][2]string{{"q5k", "q4k"}, {"q4k", "q6k"}, {"int8", "q5k"}, {"q6k", "int8"}} {
		const K = 768
		N1, N2 := 136, 72
		ldc := N1 + N2 + 8 // a guard band past the second segment, never written
		segs := []kqSegment{kqMakeSegment(rng, pair[0], N1, K), kqMakeSegment(rng, pair[1], N2, K)}
		offs, widths := []int{0, N1}, []int{N1, N2}
		biasV := make([]float32, ldc)
		for i := range biasV {
			biasV[i] = float32(rng.NormFloat64())
		}
		bias := NewBufferFloats(d, biasV)
		for _, rows := range []int{9, 40, 72} {
			av := make([]uint16, rows*K)
			for i := range av {
				av[i] = F32ToF16(float32(rng.NormFloat64()))
			}
			resid := make([]uint16, rows*ldc)
			for i := range resid {
				resid[i] = F32ToF16(float32(rng.NormFloat64()))
			}
			a := NewBufferU16s(d, av)
			for mode := range uint32(3) {
				var first []uint16
				for _, tl := range kqTiles {
					c := NewBufferU16s(d, resid)
					for si, s := range segs {
						p, err := d.NewComputePipeline(lib, fmt.Sprintf("gemm_kq_%s_%s", s.kind, tl.name))
						if err != nil {
							t.Fatalf("pipeline: %v", err)
						}
						kqDispatch(t, d, cq, p, tl.tm, tl.tn, s, a, rows, widths[si], K, c, ldc, offs[si], bias, mode)
					}
					got := slices.Clone(c.U16s())
					if first != nil {
						for i := range got {
							if got[i] != first[i] {
								t.Fatalf("%v rows %d mode %d: %s differs from m64n64 at %d", pair, rows, mode, tl.name, i)
							}
						}
						continue
					}
					first = got
					for m := range rows {
						for col := N1 + N2; col < ldc; col++ {
							if got[m*ldc+col] != resid[m*ldc+col] {
								t.Fatalf("%v rows %d mode %d: guard column %d of row %d written", pair, rows, mode, col, m)
							}
						}
						for si, s := range segs {
							for n := range widths[si] {
								col := offs[si] + n
								var ref, abs float64
								for k := range K {
									x := float64(f16ToF32(av[m*K+k])) * float64(f16ToF32(kqRNE(float32(s.w[n*K+k]))))
									ref += x
									abs += math.Abs(x)
								}
								switch mode {
								case 1:
									ref += float64(biasV[col])
								case 2:
									ref += float64(f16ToF32(resid[m*ldc+col]))
								}
								g := float64(f16ToF32(got[m*ldc+col]))
								bound := math.Abs(ref)*math.Pow(2, -10) + 4*math.Sqrt(K)*math.Pow(2, -24)*abs + 1e-6
								if diff := math.Abs(g - ref); diff > bound {
									t.Fatalf("%v rows %d mode %d [%d,%d] (%s): kernel %g, reference %g (diff %g > %g)", pair, rows, mode,
										m, col, s.kind, g, ref, diff, bound)
								}
							}
						}
					}
				}
			}
		}
	}
}
