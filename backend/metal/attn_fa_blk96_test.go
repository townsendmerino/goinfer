//go:build darwin

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/ebitengine/purego/objc"
)

// The hd = 96 block kernel's bar against float64 attention: the registration's gate 1
// (docs/measurements/metal-fa-hd96-phi3-2026-11-23/README.md), the CUDA hd-96 lane's bar.
const blk96Cos, blk96MaxAbs = 0.9999, 1e-2

// blk96Case is one synthetic input: nKV heads of one query head each (MHA), nKeys keys, split splits.
type blk96Case struct {
	nKV, nKeys, split int
	peaked            bool
}

func blk96Cases() []blk96Case {
	var cs []blk96Case
	for _, nKV := range []int{4, 32} {
		for _, split := range []int{4, attnFABlkSplit} {
			for _, nKeys := range []int{attnFADepthFloor, attnFADepthFloor + 1, 3900, 4100} {
				for _, peaked := range []bool{false, true} {
					cs = append(cs, blk96Case{nKV, nKeys, split, peaked})
				}
			}
		}
	}
	return cs
}

// blk96Inputs builds q (f32, [nKV*96]) and f16 K/V ([nKeys][nKV][96]); peaked aligns three keys with each head's query, so the
// online softmax rescales across blocks and splits.
func blk96Inputs(c blk96Case) (q []float32, kh, vh []uint16) {
	const hd = 96
	rng := rand.New(rand.NewSource(int64(c.nKeys*100 + c.nKV*10 + c.split)))
	q = make([]float32, c.nKV*hd)
	for i := range q {
		q[i] = float32(rng.NormFloat64())
	}
	kh, vh = make([]uint16, c.nKeys*c.nKV*hd), make([]uint16, c.nKeys*c.nKV*hd)
	for i := range kh {
		kh[i] = f32ToF16(float32(rng.NormFloat64() * 0.5))
		vh[i] = f32ToF16(float32(rng.NormFloat64()))
	}
	if c.peaked {
		for _, s := range []int{0, c.nKeys / 3, c.nKeys - 1} {
			for h := range c.nKV {
				for d := range hd {
					kh[(s*c.nKV+h)*hd+d] = f32ToF16(q[h*hd+d] * 0.35)
				}
			}
		}
	}
	return q, kh, vh
}

// blk96Ref is MHA attention in float64 from the same f32 q and f16 K/V.
func blk96Ref(c blk96Case, q []float32, kh, vh []uint16) []float64 {
	const hd = 96
	scale := float64(float32(1 / math.Sqrt(hd)))
	ref := make([]float64, c.nKV*hd)
	sc := make([]float64, c.nKeys)
	for h := range c.nKV {
		mx := math.Inf(-1)
		for s := range c.nKeys {
			var a float64
			for d := range hd {
				a += float64(q[h*hd+d]) * float64(f16ToF32(kh[(s*c.nKV+h)*hd+d]))
			}
			sc[s] = a * scale
			mx = math.Max(mx, sc[s])
		}
		var sum float64
		for s := range sc {
			sc[s] = math.Exp(sc[s] - mx)
			sum += sc[s]
		}
		for d := range hd {
			var a float64
			for s := range c.nKeys {
				a += sc[s] * float64(f16ToF32(vh[(s*c.nKV+h)*hd+d]))
			}
			ref[h*hd+d] = a / sum
		}
	}
	return ref
}

// blk96Run dispatches attention_fa_blk96_g1 and attention_fa_combine from lib the way the resident does and returns ctx.
func blk96Run(t *testing.T, d *Device, cq Queue, lib objc.ID, c blk96Case, q []float32, kh, vh []uint16) []float32 {
	t.Helper()
	const hd, G = 96, 1
	pipe, err := d.NewComputePipeline(lib, "attention_fa_blk96_g1")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	comb, err := d.NewComputePipeline(lib, "attention_fa_combine")
	if err != nil {
		t.Fatalf("combine: %v", err)
	}
	kb, vb := d.NewBufferBytes(len(kh)*2), d.NewBufferBytes(len(vh)*2)
	copy(kb.U16s()[:len(kh)], kh)
	copy(vb.U16s()[:len(vh)], vh)
	partial := d.NewBufferLen(c.nKV * c.split * G * (hd + 2))
	out := d.NewBufferLen(c.nKV * hd)
	uG, uSplit := NewBufferU32(d, G), NewBufferU32(d, uint32(c.split))
	e := cq.Begin()
	e.DispatchTG(pipe, c.nKV*c.split*128, 128, 128*6*G*4, NewBufferFloats(d, q), kb, vb, partial,
		NewBufferU32(d, uint32(c.nKV)), uG, NewBufferU32(d, uint32(c.nKeys)), NewBufferFloats(d, []float32{float32(1 / math.Sqrt(hd))}),
		NewBufferU32(d, 0), uSplit, kvIdentityTable(d))
	e.Dispatch(comb, c.nKV*hd, hd, partial, out, uG, NewBufferU32(d, hd), uSplit)
	e.End()
	if err := e.Err(); err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	return append([]float32(nil), out.Floats()[:c.nKV*hd]...)
}

// blk96Misses runs every case on lib and returns how many heads fall outside the bar, with the worst cosine and max abs error.
func blk96Misses(t *testing.T, lib objc.ID, report func(c blk96Case, h int, cos, maxAbs float64), record bool) (misses int, worstCos, worstAbs float64) {
	t.Helper()
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	cq := d.NewCommandQueue()
	worstCos = 1
	for _, c := range blk96Cases() {
		q, kh, vh := blk96Inputs(c)
		got := blk96Run(t, d, cq, lib, c, q, kh, vh)
		if record {
			recordAttnDigest(t, "attention_fa_blk96_g1,attention_fa_combine", got)
		}
		ref := blk96Ref(c, q, kh, vh)
		for h := range c.nKV {
			var dot, na, nb, mx float64
			for i := h * 96; i < (h+1)*96; i++ {
				g := float64(got[i])
				dot, na, nb = dot+g*ref[i], na+g*g, nb+ref[i]*ref[i]
				mx = math.Max(mx, math.Abs(g-ref[i]))
			}
			cos := dot / math.Sqrt(na*nb)
			worstCos, worstAbs = math.Min(worstCos, cos), math.Max(worstAbs, mx)
			if !(cos >= blk96Cos && mx <= blk96MaxAbs) {
				misses++
				if report != nil {
					report(c, h, cos, mx)
				}
			}
		}
	}
	return misses, worstCos, worstAbs
}

// TestAttnFABlk96MatchesFloat64 checks the shipped attention_fa_blk96_g1 with attention_fa_combine against float64 attention on
// synthetic MHA inputs, no checkpoint: 4 and 32 KV heads, 4 and the production 16 splits, the depth floor, a one-key chunk tail and
// two depths past 3,900, near-uniform and peaked weights.
func TestAttnFABlk96MatchesFloat64(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile allKernels: %v", err)
	}
	misses, worstCos, worstAbs := blk96Misses(t, lib, func(c blk96Case, h int, cos, mx float64) {
		t.Errorf("nKV=%d nKeys=%d split=%d peaked=%v head %d: cosine %.6f, max abs %.3g (bar cosine >= %g, max abs <= %g)",
			c.nKV, c.nKeys, c.split, c.peaked, h, cos, mx, blk96Cos, blk96MaxAbs)
	}, true)
	t.Logf("%d cases; worst cosine %.7f, worst max abs %.3g; %d heads outside the bar", len(blk96Cases()), worstCos, worstAbs, misses)
}

// blk96PlantPV is the plant: the P·V step drops the third of each lane's three dims.
const blk96PV, blk96PlantPV = "acc[g] += simd_shuffle(p[g], j) * v3;", "acc[g] += simd_shuffle(p[g], j) * float3(v3.xy, 0.0f);"

// TestAttnFABlk96_plantGoesRed: TestAttnFABlk96MatchesFloat64's bar catches a kernel that drops one lane dimension of P·V, so its
// pass says the kernel attends over all 96 dims.
func TestAttnFABlk96_plantGoesRed(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	if n := strings.Count(allKernels, blk96PV); n != 2 {
		t.Fatalf("attention_fa_blk96's P·V statement appears %d times in allKernels, want 2 (the full block and the tail)", n)
	}
	lib, err := d.CompileLibrary(strings.ReplaceAll(allKernels, blk96PV, blk96PlantPV), MSL3_1)
	if err != nil {
		t.Fatalf("compile the planted library: %v", err)
	}
	misses, worstCos, _ := blk96Misses(t, lib, nil, false)
	if misses == 0 {
		t.Fatalf("the planted kernel passes the bar on every case: the bar cannot see a dropped lane dimension")
	}
	t.Logf("planted kernel: %d heads outside the bar, worst cosine %.4f", misses, worstCos)
}

// TestAttnFAHeadDimOK_hd96: an hd = 96 layer takes attention_fa only at G = 1 and only under Options.FlashDecodeHD96; the head dims
// that took it before are unchanged by the option.
func TestAttnFAHeadDimOK_hd96(t *testing.T) {
	for _, c := range []struct {
		hd, nH, nKV int
		hd96, want  bool
	}{
		{96, 32, 32, true, true},
		{96, 32, 32, false, false},
		{96, 32, 16, true, false}, // G = 2: no instantiation
		{96, 32, 0, true, false},
		{128, 32, 8, false, true},
		{128, 32, 8, true, true},
		{64, 14, 2, false, attnFABlk64On},
		{64, 14, 2, true, attnFABlk64On},
		{80, 32, 32, true, false},
	} {
		if got := attnFAHeadDimOK(c.hd, c.nH, c.nKV, c.hd96); got != c.want {
			t.Errorf("attnFAHeadDimOK(hd=%d, nH=%d, nKV=%d, hd96=%v) = %v, want %v", c.hd, c.nH, c.nKV, c.hd96, got, c.want)
		}
		if got := attnFAPartialFloats(c.hd, c.nH, c.nKV, c.hd96) > 0; got != (c.want && attnFAGroupOK(c.nH, c.nKV)) {
			t.Errorf("attnFAPartialFloats(hd=%d, nH=%d, nKV=%d, hd96=%v) > 0 = %v, want %v", c.hd, c.nH, c.nKV, c.hd96, got, c.want)
		}
	}
}

// TestFlashDecodeHD96_metalSelection builds the MC3 fixture at an MHA head-dim-96 geometry with and without the option: with it,
// the resident sizes the partial buffer, selects the hd = 96 block kernel at the block split and lets every layer take
// attention_fa past the floor; without it, none of that.
func TestFlashDecodeHD96_metalSelection(t *testing.T) {
	for _, on := range []bool{false, true} {
		t.Run(fmt.Sprintf("FlashDecodeHD96=%v", on), func(t *testing.T) {
			prevW, prevO := mc3FixtureWriter, mc3TestHD96
			defer func() { mc3FixtureWriter, mc3TestHD96 = prevW, prevO }()
			mc3FixtureWriter = func(t *testing.T, maxPositions int) string { return writeMC3FixtureGeom(t, maxPositions, 4, 4, 96) }
			mc3TestHD96 = on
			_, r := mc3FixtureResident(t, 2, 4096)
			if g := r.layers[0].geom; g.hd != 96 || r.nH != g.nKV {
				t.Fatalf("fixture geometry hd=%d nH=%d nKV=%d, want hd 96 MHA", g.hd, r.nH, g.nKV)
			}
			if r.faHD96 != on {
				t.Fatalf("resident faHD96 = %v, want %v", r.faHD96, on)
			}
			if got := r.attnFANKV > 0 && r.attnFAPartial != (Buffer{}) && r.attnFABlkSplit == attnFABlkSplit; got != on {
				t.Fatalf("attention_fa sized and on the block split = %v (attnFANKV %d, split %d), want %v", got, r.attnFANKV, r.attnFABlkSplit, on)
			}
			r.setPosAheadForTest(attnFADepthFloor + 10)
			for l := range r.nL {
				if got := r.canUseAttnFA(l); got != on {
					t.Fatalf("layer %d: canUseAttnFA at %d keys = %v, want %v", l, r.curNKeys, got, on)
				}
			}
		})
	}
}
