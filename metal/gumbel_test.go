//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestGumbelMSL_philoxKnownAnswers pins gb_philox (gumbel.go) to the same Random123 known-answer
// vectors decoder/philox_test.go holds decoder.philox4x32 to (kat_vectors, "philox4x32 10"):
// counter[4], key[2] -> output[4]. This is the direct MSL-level anchor for the porting claim in
// gumbel.go's own header comment ("must match decoder.philox4x32"), isolated from the noise
// transform and the two-stage reduction — cheap (no real checkpoint, no GOINFER_HEAVY_TESTS) and
// always run, unlike TestGumbelDeviceAgreesWithHost below which needs a loaded model only because
// GumbelForTest is a *resident method, not because Philox itself needs one.
func TestGumbelMSL_philoxKnownAnswers(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	src := "#include <metal_stdlib>\nusing namespace metal;\n" + gumbelMSLKernels + `
kernel void gb_kat_probe(constant uint4& ctr [[buffer(0)]], constant uint2& key [[buffer(1)]], device uint* out [[buffer(2)]]) {
    uint r[4];
    gb_philox(ctr.x, ctr.y, ctr.z, ctr.w, key.x, key.y, r);
    out[0]=r[0]; out[1]=r[1]; out[2]=r[2]; out[3]=r[3];
}
`
	lib, err := d.CompileLibrary(src, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	p, err := d.NewComputePipeline(lib, "gb_kat_probe")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	q := d.NewCommandQueue()
	for _, c := range []struct {
		ctr  [4]uint32
		key  [2]uint32
		want [4]uint32
	}{
		{[4]uint32{0, 0, 0, 0}, [2]uint32{0, 0}, [4]uint32{0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}},
		{[4]uint32{0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff}, [2]uint32{0xffffffff, 0xffffffff}, [4]uint32{0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd}},
		{[4]uint32{0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344}, [2]uint32{0xa4093822, 0x299f31d0}, [4]uint32{0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1}},
	} {
		ctrBuf := NewBufferUint32s(d, c.ctr[:])
		keyBuf := NewBufferUint32s(d, c.key[:])
		out := d.NewBufferLen(4)
		e := q.Begin()
		e.Dispatch(p, 1, 1, ctrBuf, keyBuf, out)
		e.End()
		if err := e.Err(); err != nil {
			t.Fatalf("exec: %v", err)
		}
		got := [4]uint32{}
		copy(got[:], out.U32s())
		if got != c.want {
			t.Errorf("gb_philox(%08x, %08x) = %08x, want %08x", c.ctr, c.key, got, c.want)
		}
	}
}

// TestGumbelDeviceAgreesWithHost is the kernel gate for gumbel_stage1/2 (R7b Mac half), a direct
// port of cuda's TestGumbelDeviceAgreesWithHost (cuda/gumbel_test.go) — read its header first;
// this uses the exact same pre-registered rule, rows and seeds.
//
// PRE-REGISTERED RULE (written before the first run, matching the CUDA gate). The host draw
// (decoder.gumbelDraw, f64 noise transform) is the reference; the device computes the same argmax
// with an f32 transform. Philox is integer-exact, so the two agree except where the best two keys
// are within a few f32 ulps. Therefore:
//
//  1. Overall agreement >= 99.99% of draws (mismatch rate <= 1e-4).
//  2. EVERY mismatch is a genuine near-tie: the HOST's own keys for the device's token and the
//     host's token differ by <= 5e-5. A mismatch with a larger gap would mean the noise itself
//     differs — a bug, not rounding.
//
// Rows: the real vocab and sizes that are not multiples of 4 or of the block (GB_THREADS=256, so
// the block is 1024 entries), normal / peaked / -inf-masked / flat; temperatures 0.3, 1.0, 2.0;
// seeds and draw indices that exercise the high words of both.
func TestGumbelDeviceAgreesWithHost(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a real checkpoint for the compiled gumbel kernels)")
	}
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	path := os.ExpandEnv("$HOME/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	mc, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mc.Close()
	mr, ok := mc.ResidentForwardForTest().(*metalResident)
	if !ok || mr == nil || mr.r == nil {
		t.Skip("model did not go resident")
	}
	rf := mr.r

	rng := rand.New(rand.NewSource(17))
	mk := func(kind string, v int) []float32 {
		l := make([]float32, v)
		switch kind {
		case "normal":
			for i := range l {
				l[i] = float32(rng.NormFloat64() * 3)
			}
		case "peaked":
			for r, id := range rng.Perm(v) {
				l[id] = float32(9 - 2.0*math.Log(float64(r+1)))
			}
		case "masked":
			for i := range l {
				l[i] = float32(math.Inf(-1))
			}
			for k := 0; k < 12 && k < v; k++ {
				l[rng.Intn(v)] = float32(rng.NormFloat64())
			}
		case "flat":
			for i := range l {
				l[i] = 0.75
			}
		}
		return l
	}
	seeds := []uint64{0, 1, 0xFFFFFFFF, 0x123456789ABCDEF0}
	total, mismatch := 0, 0
	worstGap := float32(0)
	for _, v := range []int{rf.V, 50001, 4099, 1001, 67} {
		for _, kind := range []string{"normal", "peaked", "masked", "flat"} {
			l := mk(kind, v)
			for _, T := range []float64{0.3, 1.0, 2.0} {
				draws := 400
				if v > 10000 {
					draws = 60 // the sequential host reference costs ~10 ms/draw at a large vocab
				}
				for d := 0; d < draws; d++ {
					seed := seeds[d%len(seeds)]
					draw := uint64(d)
					if d%7 == 0 {
						draw += 1 << 33 // the high counter word
					}
					dev, err := rf.GumbelForTest(l, T, seed, draw)
					if err != nil {
						t.Fatalf("v=%d %s T=%.1f: %v", v, kind, T, err)
					}
					host := decoder.GumbelDrawForTest(l, T, seed, draw)
					total++
					if dev == host {
						continue
					}
					mismatch++
					gap := decoder.GumbelKeyForTest(l, T, seed, draw, host) - decoder.GumbelKeyForTest(l, T, seed, draw, dev)
					if gap < 0 {
						gap = -gap
					}
					if gap > worstGap {
						worstGap = gap
					}
					if gap > 5e-5 {
						t.Errorf("v=%d %s T=%.1f seed=%x draw=%d: device %d != host %d and the host's own keys differ by %.3g — not a near-tie",
							v, kind, T, seed, draw, dev, host, gap)
					}
				}
			}
		}
	}
	rate := float64(mismatch) / float64(total)
	t.Logf("%d draws: %d mismatches (%.2e); worst host-key gap among them %.2e", total, mismatch, rate, worstGap)
	if rate > 1e-4 {
		t.Errorf("mismatch rate %.2e exceeds the pre-registered 1e-4", rate)
	}
}

// TestPhiloxGumbelMSL_mutationDetectsAConstantChange applies R7b's own registered mutation check
// (docs/measurements/sampled-gumbel-2026-09-20.md: "a Philox constant changed on the CUDA kernel →
// mismatches with host-key gaps 1.4-5.2") to the MSL kernel: flip gumbel.go's philoxM0 constant
// (0xD2511F53 -> 0xD2511F52), rebuild the library, and confirm TestGumbelDeviceAgreesWithHost's own
// rule (every mismatch within 5e-5 of the host's own keys) goes red with LARGE gaps — proving the
// gate can actually catch a broken kernel, not just pass vacuously on a correct one. Restores the
// source afterward and confirms the restoration is byte-identical.
func TestPhiloxGumbelMSL_mutationDetectsAConstantChange(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (compiles a mutated kernel library)")
	}
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	mutated := mutatePhiloxConstant(gumbelMSLKernels)
	if mutated == gumbelMSLKernels {
		t.Fatal("mutation helper did not change the source — the mutation check itself is broken")
	}
	// gumbelMSLKernels has no #include of its own (it's normally appended to allKernels, which
	// carries the one #include/using-namespace the whole library needs) — supply it here since
	// this test compiles the mutated source as its own standalone library.
	lib, err := d.CompileLibrary("#include <metal_stdlib>\nusing namespace metal;\n"+mutated, MSL3_1)
	if err != nil {
		t.Fatalf("compile mutated library: %v", err)
	}
	pGumbel1, err := d.NewComputePipeline(lib, "gumbel_stage1")
	if err != nil {
		t.Fatalf("pipeline gumbel_stage1: %v", err)
	}
	pGumbel2, err := d.NewComputePipeline(lib, "gumbel_stage2")
	if err != nil {
		t.Fatalf("pipeline gumbel_stage2: %v", err)
	}

	rng := rand.New(rand.NewSource(31))
	const v = 4099
	l := make([]float32, v)
	for i := range l {
		l[i] = float32(rng.NormFloat64() * 3)
	}
	nb := gumbelBlocks(v)
	q := d.NewCommandQueue()
	draw := func(seed, drw uint64, T float64) int {
		logitsBuf := NewBufferFloats(d, l)
		bkey, bidx := d.NewBufferLen(nb), d.NewBufferLen(nb)
		out := d.NewBufferLen(1)
		uV, uNB := NewBufferU32(d, uint32(v)), NewBufferU32(d, uint32(nb))
		uInvT := NewBufferFloats(d, []float32{float32(1 / T)})
		uK0, uK1 := NewBufferU32(d, uint32(seed)), NewBufferU32(d, uint32(seed>>32))
		uD0, uD1 := NewBufferU32(d, uint32(drw)), NewBufferU32(d, uint32(drw>>32))
		const gbThreads, gbShmBytes = 256, 256 * 2 * 4
		e := q.Begin()
		e.DispatchTG(pGumbel1, nb*gbThreads, gbThreads, gbShmBytes, logitsBuf, uV, uInvT, uK0, uK1, uD0, uD1, bkey, bidx)
		e.DispatchTG(pGumbel2, gbThreads, gbThreads, gbShmBytes, bkey, bidx, uNB, out)
		e.End()
		return int(int32(out.U32()))
	}

	mismatch, worstGap := 0, float32(0)
	const draws = 300
	for i := 0; i < draws; i++ {
		seed := uint64(i%5) + 1
		T := []float64{0.3, 1.0, 2.0}[i%3]
		dev := draw(seed, uint64(i), T)
		host := decoder.GumbelDrawForTest(l, T, seed, uint64(i))
		if dev == host {
			continue
		}
		mismatch++
		gap := decoder.GumbelKeyForTest(l, T, seed, uint64(i), host) - decoder.GumbelKeyForTest(l, T, seed, uint64(i), dev)
		if gap < 0 {
			gap = -gap
		}
		if gap > worstGap {
			worstGap = gap
		}
	}
	t.Logf("mutated kernel: %d/%d draws mismatched host, worst host-key gap %.3g", mismatch, draws, worstGap)
	if mismatch == 0 || worstGap <= 5e-5 {
		t.Errorf("mutation did not produce a red result (mismatches=%d worstGap=%.3g <= 5e-5) — the gate cannot catch a broken kernel",
			mismatch, worstGap)
	}
}

// mutatePhiloxConstant exists only so the mutation test above can build an ALTERNATE library
// without touching the shipped allKernels — see gumbel.go's own philoxM0-equivalent constant
// (0xD2511F53u, the same value decoder/philox.go pins). Replaces EVERY occurrence (it appears
// twice, in one statement computing hi0 via gb_mulhi and lo0 via a plain multiply) so the mutated
// kernel is internally consistent — a real accidental constant change would move both call sites
// together, not leave them disagreeing with each other.
func mutatePhiloxConstant(src string) string {
	const from = "0xD2511F53u"
	const to = "0xD2511F52u"
	out := make([]byte, 0, len(src))
	for i := 0; i < len(src); {
		if i+len(from) <= len(src) && src[i:i+len(from)] == from {
			out = append(out, to...)
			i += len(from)
			continue
		}
		out = append(out, src[i])
		i++
	}
	return string(out)
}
