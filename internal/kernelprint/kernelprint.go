package kernelprint

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"hash"
	"math"
	"runtime"
	"sort"
	"strings"
	"testing/fstest"

	"github.com/townsendmerino/aikit/embed"
	"github.com/townsendmerino/aikit/linalg"
)

// Config is the dispatch configuration a fingerprint holds for: two builds that agree on it run the same kernels.
type Config struct {
	Arch   string   `json:"arch"`
	Active []string `json:"active"`
	Forced []string `json:"forced"`
}

func (c Config) String() string {
	s := c.Arch + "[" + strings.Join(c.Active, ",") + "]"
	if len(c.Forced) > 0 {
		s += " forced=" + strings.Join(c.Forced, ",")
	}
	return s
}

// CurrentConfig reads this process's dispatch configuration from linalg.
func CurrentConfig() Config {
	k := linalg.ActiveKernels()
	return Config{Arch: runtime.GOARCH, Active: k.Active, Forced: k.Forced}
}

// Report is one Compute: a fingerprint ("sha256:<hex>") per kernel name.
type Report struct {
	Config  Config            `json:"config"`
	Kernels map[string]string `json:"kernels"`
}

// Digest is one SHA-256 over the configuration and every kernel's fingerprint, in name order: what a key mixes in
// until it can name the kernels a family's operations use (VK4).
func (r Report) Digest() string {
	names := make([]string, 0, len(r.Kernels))
	for n := range r.Kernels {
		names = append(names, n)
	}
	sort.Strings(names)
	h := sha256.New()
	h.Write([]byte(r.Config.String() + "\n"))
	for _, n := range names {
		h.Write([]byte(n + "=" + r.Kernels[n] + "\n"))
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}

// Kernels returns the kernel names in suite order.
func Kernels() []string {
	out := make([]string, len(suite))
	for i, k := range suite {
		out[i] = k.name
	}
	return out
}

// Compute runs every kernel over its inputs and returns the fingerprints.
func Compute() Report {
	r := Report{Config: CurrentConfig(), Kernels: make(map[string]string, len(suite))}
	for _, k := range suite {
		r.Kernels[k.name] = run(k)
	}
	return r
}

func run(k kernel) string {
	s := &sink{h: sha256.New()}
	k.run(s, newGen(seedOf(k.name)))
	return "sha256:" + hex.EncodeToString(s.h.Sum(nil))
}

// seedOf derives a kernel's input seed from its name, so adding or reordering kernels never changes another's inputs.
func seedOf(name string) uint64 {
	d := sha256.Sum256([]byte(name))
	return binary.LittleEndian.Uint64(d[:8])
}

// sink is what a kernel writes its outputs to. tag separates calls, so two shapes cannot hash alike by concatenation.
type sink struct {
	h   hash.Hash
	buf []byte
}

func (s *sink) tag(name string, ints ...int) {
	s.buf = append(s.buf[:0], name...)
	for _, v := range ints {
		s.buf = binary.LittleEndian.AppendUint64(s.buf, uint64(v))
	}
	s.h.Write(s.buf)
}

func (s *sink) f32(v []float32) {
	s.buf = s.buf[:0]
	for _, x := range v {
		s.buf = binary.LittleEndian.AppendUint32(s.buf, math.Float32bits(x))
	}
	s.h.Write(s.buf)
}

func (s *sink) u16(v []uint16) { s.h.Write(u16bytes(v)) }

func (s *sink) i8(v []int8) {
	s.buf = s.buf[:0]
	for _, x := range v {
		s.buf = append(s.buf, byte(x))
	}
	s.h.Write(s.buf)
}

func (s *sink) bytes(v []byte) { s.h.Write(v) }

func (s *sink) ok(b bool) {
	if b {
		s.h.Write([]byte{1})
	} else {
		s.h.Write([]byte{0})
	}
}

type kernel struct {
	name string
	run  func(s *sink, g *gen)
}

// Shapes. M=1 is decode; M>1 crosses aikit's 2-row, 4-row and 32-row tiles with remainders; K and N cross the SIMD
// widths with tails, the f32 kernel's 768 k-tile and its 2048 packing threshold.
var (
	mList = []int{1, 2, 3, 4, 5, 8, 9, 16, 17, 33}
	nList = []int{1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 33}
	kF32  = []int{1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 65, 100, 767, 768, 769, 2048, 2304}
	kQ8   = []int{1, 7, 16, 17, 32, 33, 64, 100, 128, 256, 768, 2048}
	// kG32 are the K of per-32 work (int4 weights, per-32 activations): goinfer takes any cols, so a ragged last group
	// and K below one group occur.
	kG32   = []int{16, 32, 48, 64, 96, 100, 128, 256, 768, 2048}
	mLarge = []int{1, 2, 5, 17}
	nLarge = []int{1, 4, 9, 33}
	// thresholds are the Workspace fan-out thresholds: -1 leaves the zero Workspace at aikit's default, 1 always fans out,
	// and the other two are goinfer's decode values (DefaultDecodeParallelThreshold and int4ParThreshold).
	thresholds = []int{-1, 1, 300_000, 1 << 20}
	// groups are the activation-quantization groups goinfer runs: per row and per 32.
	groups = []int{0, 32}
)

// shapes calls fn for each (M, K, N) over ks, with fewer M and N at K >= 768 so the suite stays seconds.
func shapes(ks []int, fn func(M, K, N int)) {
	for _, K := range ks {
		ms, ns := mList, nList
		if K >= 768 {
			ms, ns = mLarge, nLarge
		}
		for _, M := range ms {
			for _, N := range ns {
				fn(M, K, N)
			}
		}
	}
}

func workspace(thr, group int) *linalg.Workspace {
	ws := &linalg.Workspace{}
	if thr >= 0 {
		ws.SetThreshold(thr)
	}
	ws.SetActQuantGroup(group)
	return ws
}

// int4Layouts builds the int4 layouts goinfer loads for one weight: canonical, row4 repacked beside canonical (heap),
// row4 in place (the safetensors and GGUF loaders), row4 from the .giw emit, and split-half (amd64, AVX2 without
// VNNI+VL). A layout the configuration refuses is reported with ok=false and no weight.
type int4Layout struct {
	name string
	wm   linalg.WeightMat
	ok   bool
}

func int4Layouts(w []float32, N, K int) []int4Layout {
	const group = 32
	canon := linalg.QuantizeInt4(w, N, K, group)
	heap := linalg.QuantizeInt4(w, N, K, group)
	heapOK := heap.RepackInt4Row4()
	q4, q4s := int4F16(w, N, K)
	inPlace, inPlaceOK := linalg.RepackInt4Row4InPlaceF16(q4, q4s, N, K, group)
	// The .giw reader wraps what the emit repacked (row4Emit), and declines a shape or core that cannot run row4.
	q4g, q4sg := int4F16(w, N, K)
	if linalg.Int4Row4Usable(N, K, group) {
		q4g, q4sg = row4Emit(q4g, q4sg, N, K, group)
	}
	giw, giwOK := linalg.WrapInt4Row4OnlyF16(q4g, q4sg, N, K, group)
	split := linalg.QuantizeInt4(w, N, K, group)
	splitOK := split.RepackInt4SplitHalf()
	return []int4Layout{{"canonical", canon, true}, {"row4-heap", heap, heapOK}, {"row4-inplace", inPlace, inPlaceOK}, {"row4-giw", giw, giwOK}, {"split-half", split, splitOK}}
}

var suite = []kernel{
	// f32
	{"linalg.MatmulBT", func(s *sink, g *gen) {
		shapes(kF32, func(M, K, N int) {
			a, b, dst := g.vec(M*K, 1), g.mat(N, K, 0.25), make([]float32, M*N)
			linalg.MatmulBT(a, b, dst, M, K, N)
			s.tag("mm", M, K, N)
			s.f32(dst)
		})
		// One shape past the process-wide fan-out threshold (16.78M MACs).
		M, K, N := 64, 2304, 128
		a, b, dst := g.vec(M*K, 1), g.mat(N, K, 0.25), make([]float32, M*N)
		linalg.MatmulBT(a, b, dst, M, K, N)
		s.tag("mm-par", M, K, N)
		s.f32(dst)
	}},
	{"linalg.MatmulBTInto", func(s *sink, g *gen) {
		shapes(kF32, func(M, K, N int) {
			a, b, dst := g.vec(M*K, 1), g.mat(N, K, 0.25), make([]float32, M*N)
			linalg.MatmulBTInto(dst, a, b, M, K, N)
			s.tag("mm", M, K, N)
			s.f32(dst)
		})
	}},
	{"linalg.Dot", func(s *sink, g *gen) {
		for _, n := range dotLens() {
			a, b := g.vec(n, 1), g.vec(n, 0.5)
			s.tag("dot", n)
			s.f32([]float32{linalg.Dot(a, b)})
		}
	}},

	// 16-bit conversions and weights
	{"linalg.F16ToF32", func(s *sink, g *gen) {
		out := make([]float32, 1<<16)
		for h := range out {
			out[h] = linalg.F16ToF32(uint16(h))
		}
		s.f32(out)
	}},
	{"linalg.F16ToF32Slice", func(s *sink, g *gen) {
		all := make([]uint16, 1<<16)
		for h := range all {
			all[h] = uint16(h)
		}
		for _, n := range []int{1, 3, 4, 7, 8, 9, 15, 16, 17, 33} {
			for i := 0; i+n <= 4096; i += n {
				dst := make([]float32, n)
				linalg.F16ToF32Slice(dst, all[i:i+n])
				s.f32(dst)
			}
		}
		dst := make([]float32, len(all))
		linalg.F16ToF32Slice(dst, all)
		s.f32(dst)
	}},
	{"linalg.BF16ToF32", func(s *sink, g *gen) {
		out := make([]float32, 1<<16)
		for h := range out {
			out[h] = linalg.BF16ToF32(uint16(h))
		}
		s.f32(out)
	}},
	{"linalg.F32ToF16", func(s *sink, g *gen) {
		p := probes32()
		out := make([]uint16, len(p))
		for i, b := range p {
			out[i] = linalg.F32ToF16(math.Float32frombits(b))
		}
		s.u16(out)
	}},
	{"linalg.F32ToF16Slice", func(s *sink, g *gen) {
		p := probes32()
		src := make([]float32, len(p))
		for i, b := range p {
			src[i] = math.Float32frombits(b)
		}
		for _, n := range []int{1, 3, 4, 7, 8, 9, 15, 16, 17, 33, len(src)} {
			for i := 0; i+n <= len(src); i += n {
				dst := make([]uint16, n)
				linalg.F32ToF16Slice(dst, src[i:i+n])
				s.u16(dst)
			}
		}
	}},
	{"linalg.F32ToBF16", func(s *sink, g *gen) {
		p := probes32()
		out := make([]uint16, len(p))
		for i, b := range p {
			out[i] = linalg.F32ToBF16(math.Float32frombits(b))
		}
		s.u16(out)
	}},
	{"linalg.QuantizeBF16", func(s *sink, g *gen) {
		for _, rc := range [][2]int{{1, 1}, {3, 17}, {4, 64}, {5, 100}} {
			wm := linalg.QuantizeBF16(g.mat(rc[0], rc[1], 1), rc[0], rc[1])
			u, bf, ok := wm.Half16()
			s.tag("q", rc[0], rc[1])
			s.ok(bf && ok)
			s.u16(u)
		}
	}},
	{"linalg.QuantizeF16", func(s *sink, g *gen) {
		for _, rc := range [][2]int{{1, 1}, {3, 17}, {4, 64}, {5, 100}} {
			wm := linalg.QuantizeF16(g.mat(rc[0], rc[1], 1), rc[0], rc[1])
			u, bf, ok := wm.Half16()
			s.tag("q", rc[0], rc[1])
			s.ok(!bf && ok)
			s.u16(u)
		}
	}},
	{"linalg.WeightMat(bf16).MatmulBTInto", func(s *sink, g *gen) {
		half16Matmul(s, g, true)
	}},
	{"linalg.WeightMat(f16).MatmulBTInto", func(s *sink, g *gen) {
		half16Matmul(s, g, false)
	}},

	// int8
	{"linalg.QuantizeRowsInt8", func(s *sink, g *gen) {
		for _, rows := range []int{1, 3, 4, 5} {
			for _, cols := range quantCols() {
				q, sc := linalg.QuantizeRowsInt8(g.mat(rows, cols, 1), rows, cols)
				s.tag("q", rows, cols)
				s.i8(q)
				s.f32(sc)
			}
		}
		for _, cols := range quantCols() {
			q, sc := linalg.QuantizeRowsInt8(ties(cols), 1, cols)
			s.tag("ties", cols)
			s.i8(q)
			s.f32(sc)
		}
	}},
	{"linalg.QuantizeRowInt8", func(s *sink, g *gen) {
		for _, cols := range quantCols() {
			for _, row := range [][]float32{g.vec(cols, 1), ties(cols), make([]float32, cols)} {
				q := make([]int8, cols)
				sc := linalg.QuantizeRowInt8(row, q)
				s.tag("q", cols)
				s.i8(q)
				s.f32([]float32{sc})
			}
		}
	}},
	{"linalg.QuantizeInt8", func(s *sink, g *gen) {
		for _, w8a8 := range []bool{false, true} {
			for _, cols := range quantCols() {
				wm := linalg.QuantizeInt8(g.mat(4, cols, 1), 4, cols, w8a8)
				q, sc, isW8A8, ok := wm.Int8()
				s.tag("q", cols)
				s.ok(isW8A8 == w8a8 && ok)
				s.i8(q)
				s.f32(sc)
			}
		}
	}},
	{"linalg.DequantizeRowsInt8Into", func(s *sink, g *gen) {
		for _, rows := range []int{1, 3, 4} {
			for _, cols := range quantCols() {
				q, sc := int8Rows(g, rows, cols)
				dst := make([]float32, rows*cols)
				linalg.DequantizeRowsInt8Into(dst, q, sc, rows, cols)
				s.tag("dq", rows, cols)
				s.f32(dst)
			}
		}
	}},
	{"linalg.DequantizeRowInt8", func(s *sink, g *gen) {
		for _, cols := range quantCols() {
			q, sc := int8Rows(g, 1, cols)
			dst := make([]float32, cols)
			linalg.DequantizeRowInt8(q, sc[0], dst)
			s.tag("dq", cols)
			s.f32(dst)
		}
	}},
	{"linalg.DotI8", func(s *sink, g *gen) {
		for _, n := range dotLens() {
			a, _ := int8Rows(g, 1, n)
			b, _ := int8Rows(g, 1, n)
			if n > 2 {
				a[0], b[0], a[1], b[1] = -128, -128, 127, -128
			}
			s.tag("dot", n)
			v := linalg.DotI8(a, b)
			s.bytes(binary.LittleEndian.AppendUint32(nil, uint32(v)))
		}
	}},
	{"linalg.MatmulBTQ8Into", func(s *sink, g *gen) {
		for _, thr := range thresholds {
			shapes(kQ8, func(M, K, N int) {
				a := g.vec(M*K, 1)
				q, sc := linalg.QuantizeRowsInt8(g.mat(N, K, 0.25), N, K)
				dst := make([]float32, M*N)
				linalg.MatmulBTQ8Into(workspace(thr, 0), a, q, sc, dst, M, K, N)
				s.tag("mm", thr, M, K, N)
				s.f32(dst)
			})
		}
	}},
	{"linalg.MatmulBTQ8", func(s *sink, g *gen) {
		shapes(kQ8, func(M, K, N int) {
			a := g.vec(M*K, 1)
			q, sc := linalg.QuantizeRowsInt8(g.mat(N, K, 0.25), N, K)
			dst := make([]float32, M*N)
			linalg.MatmulBTQ8(a, q, sc, dst, M, K, N)
			s.tag("mm", M, K, N)
			s.f32(dst)
		})
	}},
	{"linalg.MatmulBTW8A8Into", func(s *sink, g *gen) {
		for _, group := range groups {
			ks := kQ8
			if group > 0 {
				ks = kG32
			}
			for _, thr := range thresholds {
				shapes(ks, func(M, K, N int) {
					a := g.vec(M*K, 1)
					q, sc := linalg.QuantizeRowsInt8(g.mat(N, K, 0.25), N, K)
					dst := make([]float32, M*N)
					linalg.MatmulBTW8A8Into(workspace(thr, group), a, q, sc, dst, M, K, N)
					s.tag("mm", group, thr, M, K, N)
					s.f32(dst)
				})
			}
		}
	}},
	{"linalg.MatmulBTW8A8", func(s *sink, g *gen) {
		shapes(kQ8, func(M, K, N int) {
			a := g.vec(M*K, 1)
			q, sc := linalg.QuantizeRowsInt8(g.mat(N, K, 0.25), N, K)
			dst := make([]float32, M*N)
			linalg.MatmulBTW8A8(a, q, sc, dst, M, K, N)
			s.tag("mm", M, K, N)
			s.f32(dst)
		})
	}},
	{"linalg.MatmulBTW8A8Batch", func(s *sink, g *gen) {
		for _, group := range groups {
			for _, thr := range thresholds {
				for _, K := range kG32 {
					for _, M := range mLarge {
						a := g.vec(M*K, 1)
						var ops []linalg.W8A8Op
						for _, N := range []int{8, 4, 12} {
							q, sc := linalg.QuantizeRowsInt8(g.mat(N, K, 0.25), N, K)
							ops = append(ops, linalg.W8A8Op{BQ: q, Scales: sc, Dst: make([]float32, M*N), N: N})
						}
						linalg.MatmulBTW8A8Batch(workspace(thr, group), a, M, K, ops)
						s.tag("batch", group, thr, M, K)
						for _, op := range ops {
							s.f32(op.Dst)
						}
					}
				}
			}
		}
	}},

	// int4
	{"linalg.QuantizeGroupInt4Row", func(s *sink, g *gen) {
		for _, group := range []int{32, 64} {
			for _, cols := range []int{64, 128, 256, 2048} {
				for _, row := range [][]float32{g.vec(cols, 1), ties(cols), make([]float32, cols)} {
					packed, sc := make([]byte, cols/2), make([]float32, cols/group)
					linalg.QuantizeGroupInt4Row(row, cols, group, packed, sc)
					s.tag("q", group, cols)
					s.bytes(packed)
					s.f32(sc)
				}
			}
		}
	}},
	{"linalg.QuantizeInt4", func(s *sink, g *gen) {
		for _, rc := range [][2]int{{1, 32}, {3, 64}, {4, 256}, {8, 2048}} {
			wm := linalg.QuantizeInt4(g.mat(rc[0], rc[1], 1), rc[0], rc[1], 32)
			q4, q4s, group, ok := wm.Int4F16()
			s.tag("q", rc[0], rc[1], group)
			s.ok(ok)
			s.bytes(q4)
			s.u16(q4s)
		}
	}},
	{"linalg.QuantizeActQ", func(s *sink, g *gen) {
		for _, group := range groups {
			ks := []int{1, 7, 17, 33, 100, 256}
			if group > 0 {
				ks = kG32
			}
			for _, K := range ks {
				for _, M := range mList {
					var q linalg.ActQ
					a := g.vec(M*K, 1)
					if M > 1 {
						copy(a[K:], ties(K))
					}
					linalg.QuantizeActQ(a, M, K, group, &q)
					s.tag("q", group, M, K)
					s.i8(q.Q)
					s.f32(q.S)
				}
			}
		}
	}},
	{"linalg.WeightMat(int4).MatmulBTW4A8Into", func(s *sink, g *gen) {
		for _, group := range groups {
			for _, thr := range thresholds {
				shapes(kG32, func(M, K, N int) {
					a, w := g.vec(M*K, 1), g.mat(N, K, 0.25)
					for _, l := range int4Layouts(w, N, K) {
						s.tag(l.name, group, thr, M, K, N)
						s.ok(l.ok)
						if !l.ok {
							continue
						}
						dst := make([]float32, M*N)
						l.wm.SetActQuantGroup(group)
						l.wm.MatmulBTW4A8Into(workspace(thr, group), a, dst, M)
						s.f32(dst)
					}
				})
			}
		}
	}},
	{"linalg.WeightMat(int4).MatmulBTW4A8PreInto", func(s *sink, g *gen) {
		for _, group := range groups {
			shapes(kG32, func(M, K, N int) {
				a, w := g.vec(M*K, 1), g.mat(N, K, 0.25)
				for _, l := range int4Layouts(w, N, K) {
					s.tag(l.name, group, M, K, N)
					s.ok(l.ok)
					if !l.ok {
						continue
					}
					ws := workspace(1<<20, group)
					l.wm.SetActQuantGroup(group)
					var q linalg.ActQ
					l.wm.QuantizeActW4A8(ws, a, M, &q)
					dst := make([]float32, M*N)
					l.wm.MatmulBTW4A8PreInto(ws, &q, dst, M)
					s.f32(dst)
				}
			})
		}
	}},
	{"linalg.MatmulBTW4A8F16Into", func(s *sink, g *gen) {
		for _, group := range groups {
			shapes(kG32, func(M, K, N int) {
				a := g.vec(M*K, 1)
				q4, q4s := int4F16(g.mat(N, K, 0.25), N, K)
				dst := make([]float32, M*N)
				linalg.MatmulBTW4A8F16Into(workspace(1<<20, group), a, q4, q4s, dst, M, K, N, 32)
				s.tag("mm", group, M, K, N)
				s.f32(dst)
			})
		}
	}},
	{"linalg.MatmulBTW4A8F16Pre", func(s *sink, g *gen) {
		for _, group := range groups {
			shapes(kG32, func(M, K, N int) {
				a := g.vec(M*K, 1)
				q4, q4s := int4F16(g.mat(N, K, 0.25), N, K)
				var q linalg.ActQ
				linalg.QuantizeActQ(a, M, K, group, &q)
				dst := make([]float32, M*N)
				linalg.MatmulBTW4A8F16Pre(workspace(1<<20, group), &q, q4, q4s, dst, M, K, N, 32)
				s.tag("mm", group, M, K, N)
				s.f32(dst)
			})
		}
	}},
	{"linalg.MatmulBTW4A8Batch", func(s *sink, g *gen) {
		for _, group := range groups {
			for _, thr := range thresholds {
				for _, K := range kG32 {
					for _, M := range mLarge {
						a := g.vec(M*K, 1)
						for _, repack := range []bool{false, true} {
							var ops []linalg.W4A8Op
							for _, N := range []int{8, 4, 12} {
								wm := linalg.QuantizeInt4(g.mat(N, K, 0.25), N, K, 32)
								if repack {
									wm.RepackInt4Row4()
								}
								q4, q4s, _, _ := wm.Int4F16()
								row4, row4s, _ := wm.Int4Row4F16()
								ops = append(ops, linalg.W4A8Op{W4: q4, ScalesF16: q4s, Row4: row4, Row4ScalesF16: row4s, Dst: make([]float32, M*N), N: N})
							}
							linalg.MatmulBTW4A8Batch(workspace(thr, group), a, M, K, 32, ops)
							s.tag("batch", group, thr, M, K)
							s.ok(repack)
							for _, op := range ops {
								s.f32(op.Dst)
							}
						}
					}
				}
			}
		}
	}},
	{"linalg.WeightMat(q4k).MatmulBTInto", func(s *sink, g *gen) {
		q4k := ggmlTypeByName("Q4_K")
		for _, thr := range thresholds {
			for _, K := range []int{256, 512, 768, 2304} {
				for _, M := range mLarge {
					for _, N := range []int{1, 2, 3, 4, 5, 8, 9, 17} {
						wm, err := linalg.WrapQ4K(g.blocks(q4k, N*K/256), N, K)
						if err != nil {
							panic(err)
						}
						a, dst := g.vec(M*K, 1), make([]float32, M*N)
						wm.MatmulBTInto(workspace(thr, 0), a, dst, M)
						s.tag("mm", thr, M, K, N)
						s.f32(dst)
					}
				}
			}
		}
	}},

	// attention
	{"linalg.MatmulQKAcc64", func(s *sink, g *gen) {
		for _, hd := range headDims() {
			for _, nKeys := range []int{1, 2, 7, 64, 513} {
				for _, M := range []int{1, 2, 5, 17} {
					kvDim := 2*hd + 3
					a, keys, dst := g.vec(M*hd, 1), g.vec(nKeys*kvDim, 1), make([]float32, M*nKeys)
					linalg.MatmulQKAcc64(a, keys, dst, M, hd, nKeys, hd, kvDim)
					s.tag("qk", hd, nKeys, M)
					s.f32(dst)
				}
			}
		}
	}},
	{"linalg.MatmulQKAcc64Group", func(s *sink, g *gen) {
		for _, hd := range headDims() {
			for _, nKeys := range []int{1, 2, 7, 64, 513} {
				for _, G := range []int{1, 2, 4, 6, 8} {
					kvDim := 2*hd + 3
					a, keys, dst := g.vec(G*hd, 1), g.vec(nKeys*kvDim, 1), make([]float32, G*nKeys)
					linalg.MatmulQKAcc64Group(a, keys, dst, G, hd, nKeys, hd, kvDim)
					s.tag("qk", hd, nKeys, G)
					s.f32(dst)
				}
			}
		}
	}},
	{"linalg.MatmulAVAcc64", func(s *sink, g *gen) {
		for _, hd := range headDims() {
			for _, nKeys := range []int{1, 2, 7, 64, 513} {
				for _, M := range []int{1, 2, 5, 17} {
					kvDim := 2*hd + 3
					scores, vals := g.probs(M, nKeys), g.vec(nKeys*kvDim, 1)
					dst, acc := make([]float32, M*hd), make([]float64, hd)
					linalg.MatmulAVAcc64(scores, vals, dst, acc, M, nKeys, hd, hd, kvDim)
					s.tag("av", hd, nKeys, M)
					s.f32(dst)
				}
			}
		}
	}},
	{"linalg.MatmulAVAcc64Group", func(s *sink, g *gen) {
		for _, hd := range headDims() {
			for _, nKeys := range []int{1, 2, 7, 64, 513} {
				for _, G := range []int{1, 2, 4, 6, 8} {
					kvDim := 2*hd + 3
					scores, vals := g.probs(G, nKeys), g.vec(nKeys*kvDim, 1)
					dst, acc := make([]float32, G*hd), make([]float64, G*hd)
					linalg.MatmulAVAcc64Group(scores, vals, dst, acc, G, nKeys, hd, hd, kvDim)
					s.tag("av", hd, nKeys, G)
					s.f32(dst)
				}
			}
		}
	}},
	{"linalg.SoftmaxRowContractInto", func(s *sink, g *gen) {
		for _, n := range []int{1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 64, 100, 511, 512, 513, 1000} {
			for masked := range 2 {
				src := g.vec(n, 4)
				if masked == 1 {
					for j := n/2 + 1; j < n; j++ {
						src[j] = float32(math.Inf(-1))
					}
				}
				dst := make([]float32, n)
				linalg.SoftmaxRowContractInto(dst, src)
				s.tag("sm", n, masked)
				s.f32(dst)
				// In place, as internal/whisper calls it.
				linalg.SoftmaxRowContractInto(src, src)
				s.f32(src)
			}
		}
	}},
	{"linalg.GatherVBlockMajor", func(s *sink, g *gen) {
		for _, hd := range []int{16, 64} {
			for _, nKeys := range []int{1, 5, 512, 513, 1100} {
				kvDim := 2 * hd
				keys, vals := g.vec(nKeys*kvDim, 1), g.vec(nKeys*kvDim, 1)
				kh, vBlk := make([]float32, nKeys*hd), make([]float32, nKeys*hd)
				linalg.GatherVBlockMajor(kh, vBlk, keys, vals, 1, hd, kvDim, nKeys)
				s.tag("g", hd, nKeys)
				s.f32(kh)
				s.f32(vBlk)
			}
		}
	}},
	{"linalg.AttendTileFused", func(s *sink, g *gen) {
		for _, hd := range []int{16, 64, 80} {
			for _, nKeys := range []int{1, 5, 512, 513, 1100} {
				for _, kt := range []int{1, 3, 8} {
					if kt > nKeys {
						continue
					}
					for _, window := range []int{0, 600} {
						kvDim := hd
						keys, vals := g.vec(nKeys*kvDim, 1), g.vec(nKeys*kvDim, 1)
						kh, vBlk := make([]float32, nKeys*hd), make([]float32, nKeys*hd)
						linalg.GatherVBlockMajor(kh, vBlk, keys, vals, 0, hd, kvDim, nKeys)
						qh, ch := g.vec(kt*hd, 1), make([]float32, kt*hd)
						lo, hi := make([]int, kt), make([]int, kt)
						for i := range kt {
							hi[i] = nKeys - kt + i
							if window > 0 {
								lo[i] = max(0, hi[i]-window)
							}
						}
						ok := linalg.AttendTileFused(linalg.MatmulBT, qh, kh, vBlk, ch, linalg.NewFusedAttnScratch(kt, hd), kt, hd, nKeys, 1/math.Sqrt(float64(hd)), lo, hi)
						s.tag("fa", hd, nKeys, kt, window)
						s.ok(ok)
						s.f32(ch)
					}
				}
			}
		}
	}},
}

func init() {
	// embed: every GGUF type the reader dequantizes, read through GGUFFile.Tensor as goinfer's loader does.
	for _, t := range append([]ggmlType{{"F32", 0, 1, 4, nil, -1}, {"F16", 1, 1, 2, nil, -1}, {"BF16", 30, 1, 2, nil, -1}}, ggmlTypes...) {
		suite = append(suite, kernel{"embed.GGUFFile(" + t.name + ").Tensor", func(s *sink, g *gen) { ggufTensor1(s, g, t) }})
	}
	suite = append(suite,
		kernel{"embed.DequantMXFP4Blocks", func(s *sink, g *gen) {
			mx := ggmlTypeByName("MXFP4")
			for _, n := range []int{1, 2, 7, 64} {
				dst := make([]float32, n*32)
				if err := embed.DequantMXFP4Blocks(g.blocks(mx, n), n, dst); err != nil {
					panic(err)
				}
				s.tag("mx", n)
				s.f32(dst)
			}
		}},
		kernel{"embed.DequantMXFP4Split", func(s *sink, g *gen) {
			mx := ggmlTypeByName("MXFP4")
			for _, n := range []int{1, 2, 7, 64} {
				raw := g.blocks(mx, n)
				blocks, scales := make([]byte, 0, 16*n), make([]byte, 0, n)
				for b := range n {
					scales = append(scales, raw[b*17])
					blocks = append(blocks, raw[b*17+1:b*17+17]...)
				}
				dst := make([]float32, n*32)
				if err := embed.DequantMXFP4Split(blocks, scales, n, dst); err != nil {
					panic(err)
				}
				s.tag("mx", n)
				s.f32(dst)
			}
		}},
	)
	for _, dtype := range []string{"F32", "F16", "BF16"} {
		suite = append(suite,
			kernel{"embed.SafetensorsFile(" + dtype + ").TensorF32", func(s *sink, g *gen) { safetensorsRead(s, g, dtype, false) }},
			kernel{"embed.Tensor(" + dtype + ").SubF32Transposed", func(s *sink, g *gen) { safetensorsRead(s, g, dtype, true) }},
		)
	}
}

func ggmlTypeByName(name string) ggmlType {
	for _, t := range ggmlTypes {
		if t.name == name {
			return t
		}
	}
	panic("kernelprint: no ggml type " + name)
}

func ggufTensor1(s *sink, g *gen, t ggmlType) {
	for _, nb := range []int{1, 3} {
		for _, rows := range []int{1, 3} {
			cols := nb * max(t.elems, 32)
			var data []byte
			switch t.name {
			case "F32":
				data = make([]byte, 0, 4*rows*cols)
				for _, v := range g.vec(rows*cols, 1) {
					data = binary.LittleEndian.AppendUint32(data, math.Float32bits(v))
				}
			case "F16":
				data = u16bytes(g.half16(rows*cols, 0, 30))
			case "BF16":
				data = u16bytes(g.bf16(rows*cols, 100, 150))
			default:
				data = g.blocks(t, rows*cols/t.elems)
			}
			f, err := embed.OpenGGUFBytes(buildGGUF([]ggufTensor{{"w", []uint64{uint64(cols), uint64(rows)}, t.id, data}}))
			if err != nil {
				panic(fmt.Sprintf("kernelprint: GGUF %s: %v", t.name, err))
			}
			dims, out, err := f.Tensor("w")
			if err != nil {
				panic(fmt.Sprintf("kernelprint: GGUF %s tensor: %v", t.name, err))
			}
			s.tag("t", nb, rows, len(dims))
			s.f32(out)
		}
	}
}

func safetensorsRead(s *sink, g *gen, dtype string, transposed bool) {
	for _, rc := range [][2]int{{1, 1}, {3, 17}, {4, 64}, {5, 100}} {
		rows, cols := rc[0], rc[1]
		var data []byte
		switch dtype {
		case "F32":
			for _, v := range g.vec(rows*cols, 1) {
				data = binary.LittleEndian.AppendUint32(data, math.Float32bits(v))
			}
		case "F16":
			data = u16bytes(g.half16(rows*cols, 0, 30))
		case "BF16":
			data = u16bytes(g.bf16(rows*cols, 100, 150))
		}
		f, err := embed.OpenSafetensorsFromFS(fstest.MapFS{"m.safetensors": {Data: buildSafetensors([]stTensor{{"w", dtype, []int{rows, cols}, data}})}}, "m.safetensors")
		if err != nil {
			panic(fmt.Sprintf("kernelprint: safetensors %s: %v", dtype, err))
		}
		var out []float32
		if transposed {
			t, terr := f.Tensor("w")
			if terr != nil {
				panic(terr)
			}
			out, err = t.SubF32Transposed(0, rows, cols)
		} else {
			out, err = f.TensorF32("w")
		}
		if err != nil {
			panic(fmt.Sprintf("kernelprint: safetensors %s read: %v", dtype, err))
		}
		s.tag("st", rows, cols)
		s.f32(out)
	}
}

func half16Matmul(s *sink, g *gen, bf bool) {
	for _, thr := range thresholds {
		shapes(kQ8, func(M, K, N int) {
			a, w := g.vec(M*K, 1), g.mat(N, K, 0.25)
			wm := linalg.QuantizeF16(w, N, K)
			if bf {
				wm = linalg.QuantizeBF16(w, N, K)
			}
			dst := make([]float32, M*N)
			wm.MatmulBTInto(workspace(thr, 0), a, dst, M)
			s.tag("mm", thr, M, K, N)
			s.f32(dst)
		})
	}
	for _, rc := range [][2]int{{3, 64}, {8, 100}} {
		u := g.half16(rc[0]*rc[1], 5, 20)
		wm := linalg.WrapF16(u, rc[0], rc[1])
		if bf {
			wm = linalg.WrapBF16(g.bf16(rc[0]*rc[1], 110, 135), rc[0], rc[1])
		}
		a, dst := g.vec(2*rc[1], 1), make([]float32, 2*rc[0])
		wm.MatmulBTInto(workspace(-1, 0), a, dst, 2)
		s.tag("wrap", rc[0], rc[1])
		s.f32(dst)
	}
}

// int4F16 quantizes w (rows×cols) to int4 at goinfer's group of 32 and returns the canonical nibbles and binary16 scales.
func int4F16(w []float32, rows, cols int) ([]byte, []uint16) {
	wm := linalg.QuantizeInt4(w, rows, cols, 32)
	q4, q4s, _, _ := wm.Int4F16()
	return q4, q4s
}

func dotLens() []int {
	var out []int
	for n := range 71 {
		out = append(out, n)
	}
	return append(out, 127, 128, 129, 255, 256, 257, 1000, 1024)
}

func quantCols() []int {
	return []int{1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 128, 257, 1000}
}

func headDims() []int { return []int{1, 15, 16, 17, 32, 33, 64, 80, 128} }

// int8Rows returns rows×cols random int8 codes (including -128 and 127) and positive per-row scales.
func int8Rows(g *gen, rows, cols int) ([]int8, []float32) {
	b := g.bytes(rows * cols)
	q := make([]int8, len(b))
	for i, v := range b {
		q[i] = int8(v)
	}
	sc := make([]float32, rows)
	for i := range sc {
		sc[i] = float32(g.unit()*(1.0/64)) + (1.0 / 32)
	}
	return q, sc
}

// probs returns rows of nKeys non-negative weights, as attention scores after softmax are.
func (g *gen) probs(rows, nKeys int) []float32 {
	p := make([]float32, rows*nKeys)
	for i := range p {
		p[i] = float32(g.unit()+1) * (1.0 / 512)
	}
	return p
}
