//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"
)

// attnPrefillCase is one attention_prefill dispatch: M query rows at startPos over startPos+M keys of a GQA cache.
type attnPrefillCase struct {
	nH, nKV, hd, startPos, M, window int
}

// attnPrefillInputs builds random f16 q|k|v for c.
func attnPrefillInputs(c attnPrefillCase, seed int64) (qkv, kc, vc []uint16) {
	rng := rand.New(rand.NewSource(seed))
	qStride := (c.nH + 2*c.nKV) * c.hd
	qkv = make([]uint16, c.M*qStride)
	kc = make([]uint16, (c.startPos+c.M)*c.nKV*c.hd)
	vc = make([]uint16, len(kc))
	for i := range qkv {
		qkv[i] = f32ToF16(rng.Float32()*2 - 1)
	}
	for i := range kc {
		kc[i], vc[i] = f32ToF16(rng.Float32()*2-1), f32ToF16(rng.Float32()*2-1)
	}
	return qkv, kc, vc
}

// prefillRig compiles the prefill kernels once: a pipeline by name, and a 1-D dispatch.
type prefillRig struct {
	d    *Device
	pipe func(string) Pipeline
	run  func(p Pipeline, n, tg int, args ...Buffer)
}

// runAttnPrefill dispatches kernel (attention_prefill, or attention_prefill_tiled with tile keys per tile) at production's
// tgReduceAttn threads and returns the raw f16 output.
func runAttnPrefill(rig prefillRig, kernel string, c attnPrefillCase, qkv, kc, vc []uint16, tile int) []uint16 {
	d := rig.d
	out := NewBufferU16s(d, make([]uint16, c.M*c.nH*c.hd))
	args := []Buffer{NewBufferU16s(d, qkv), NewBufferU16s(d, kc), NewBufferU16s(d, vc), out, NewBufferU32(d, uint32(c.nH)),
		NewBufferU32(d, uint32(c.nKV)), NewBufferU32(d, uint32(c.hd)), NewBufferU32(d, uint32(c.startPos)),
		NewBufferFloats(d, []float32{float32(1 / math.Sqrt(float64(c.hd)))}), NewBufferU32(d, uint32((c.nH+2*c.nKV)*c.hd)),
		NewBufferU32(d, uint32(c.window))}
	if tile > 0 {
		args = append(args, NewBufferU32(d, uint32(tile)))
	}
	rig.run(rig.pipe(kernel), c.M*c.nH*tgReduceAttn, tgReduceAttn, args...)
	return append([]uint16(nil), out.U16s()[:c.M*c.nH*c.hd]...)
}

func prefillLib(t *testing.T) prefillRig {
	t.Helper()
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(prefillKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile prefillKernels: %v", err)
	}
	q := d.NewCommandQueue()
	pipes := map[string]Pipeline{}
	return prefillRig{d: d,
		pipe: func(n string) Pipeline {
			if p, ok := pipes[n]; ok {
				return p
			}
			p, err := d.NewComputePipeline(lib, n)
			if err != nil {
				t.Fatalf("pipeline %s: %v", n, err)
			}
			pipes[n] = p
			return p
		},
		run: func(p Pipeline, n, tg int, args ...Buffer) { q.Run1D(p, n, tg, args...) }}
}

// TestAttentionPrefillTiled_bitIdenticalToTheExactKernel: within attention_prefill's 4,096 keys, attention_prefill_tiled's
// output does not depend on the tile size at all: tiles of 1, 32, 100, 128, 1,000 and 4,096 keys give the same f16 bits,
// including tiles smaller than the threadgroup and ones that do not divide it, with a sliding window and without. Each thread
// owns the same keys in the same order across tiles, and the sums are reduced the same way. Against attention_prefill itself
// the output is bit-identical at head dims up to 256 and within 1 f16 ulp at 512, where the two kernels' own reductions over
// a thread's four output dims compile differently under fast math (the same count at every tile size, so not the tiling).
// A pass that fits 4,096 keys still runs attention_prefill (exactAttnTile).
func TestAttentionPrefillTiled_bitIdenticalToTheExactKernel(t *testing.T) {
	rig := prefillLib(t)
	for _, c := range []attnPrefillCase{
		{nH: 4, nKV: 2, hd: 16, startPos: 0, M: 40},
		{nH: 2, nKV: 1, hd: 256, startPos: 900, M: 6},
		{nH: 2, nKV: 1, hd: 512, startPos: 1500, M: 4},
		{nH: 2, nKV: 1, hd: 256, startPos: 3000, M: 4, window: 512},
		{nH: 2, nKV: 1, hd: 256, startPos: 4088, M: 8}, // the last row reaches exactly 4,096 keys
		{nH: 2, nKV: 1, hd: 512, startPos: 4088, M: 8},
	} {
		qkv, kc, vc := attnPrefillInputs(c, int64(c.hd+c.startPos))
		exact := runAttnPrefill(rig, "attention_prefill", c, qkv, kc, vc, 0)
		ref := runAttnPrefill(rig, "attention_prefill_tiled", c, qkv, kc, vc, prefillExactAttnMaxKeys)
		for _, tile := range []int{1, 32, 100, 128, 1000} {
			got := runAttnPrefill(rig, "attention_prefill_tiled", c, qkv, kc, vc, tile)
			if diff := countDiff(got, ref, 0); diff != 0 {
				t.Errorf("%+v tile %d: %d of %d outputs differ from the same kernel at one 4,096-key tile", c, tile, diff, len(ref))
			}
		}
		ulps := 0
		if c.hd > 256 {
			ulps = 1
		}
		if diff := countDiff(ref, exact, ulps); diff != 0 {
			t.Errorf("%+v: %d of %d outputs differ from attention_prefill by more than %d f16 ulp", c, diff, len(ref), ulps)
		}
	}
}

// countDiff counts the f16 values of a and b more than ulps apart.
func countDiff(a, b []uint16, ulps int) int {
	n := 0
	for i := range a {
		d := int(a[i]) - int(b[i])
		if d < 0 {
			d = -d
		}
		if d > ulps || (a[i]^b[i])&0x8000 != 0 && a[i] != b[i] {
			n++
		}
	}
	return n
}

// TestAttentionPrefillTiled_pastTheExactKernelsBound: past 4,096 keys, where attention_prefill cannot run (its score buffer
// would overflow), attention_prefill_tiled at its production tile matches a float64 reference of the same attention within
// 4e-3 (f16 output), for full and sliding-window attention at Gemma 4's head dims.
func TestAttentionPrefillTiled_pastTheExactKernelsBound(t *testing.T) {
	rig := prefillLib(t)
	for _, c := range []attnPrefillCase{
		{nH: 2, nKV: 1, hd: 256, startPos: 6136, M: 8},
		{nH: 2, nKV: 1, hd: 512, startPos: 8184, M: 8},
		{nH: 2, nKV: 1, hd: 256, startPos: 8184, M: 8, window: 1024},
	} {
		qkv, kc, vc := attnPrefillInputs(c, int64(7*c.hd+c.startPos))
		got := runAttnPrefill(rig, "attention_prefill_tiled", c, qkv, kc, vc, prefillExactAttnMaxKeys)
		qStride, kvDim := (c.nH+2*c.nKV)*c.hd, c.nKV*c.hd
		scale := 1 / math.Sqrt(float64(c.hd))
		worst := 0.0
		for m := range c.M {
			nKeys := c.startPos + m + 1
			lo := 0
			if c.window > 0 && nKeys > c.window {
				lo = nKeys - c.window
			}
			for h := range c.nH {
				kvh := h / (c.nH / c.nKV)
				sc := make([]float64, nKeys)
				mx := math.Inf(-1)
				for s := lo; s < nKeys; s++ {
					var a float64
					for dd := range c.hd {
						a += float64(f16ToF32(qkv[m*qStride+h*c.hd+dd])) * float64(f16ToF32(kc[s*kvDim+kvh*c.hd+dd]))
					}
					sc[s] = a * scale
					mx = math.Max(mx, sc[s])
				}
				var sum float64
				for s := lo; s < nKeys; s++ {
					sc[s] = math.Exp(sc[s] - mx)
					sum += sc[s]
				}
				for dd := range c.hd {
					var a float64
					for s := lo; s < nKeys; s++ {
						a += sc[s] * float64(f16ToF32(vc[s*kvDim+kvh*c.hd+dd]))
					}
					worst = math.Max(worst, math.Abs(a/sum-float64(f16ToF32(got[m*c.nH*c.hd+h*c.hd+dd]))))
				}
			}
		}
		if worst > 4e-3 {
			t.Errorf("%+v: max abs diff %.2e from the float64 reference, over 4e-3", c, worst)
		}
	}
}
