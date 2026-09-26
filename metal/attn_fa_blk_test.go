//go:build darwin

package metal

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

// r17Kernels is R17's step-2 prototype (docs/tasks/red-october.md; docs/measurements/metal-decode-attn-r17-2026-09-25.md)
// — attention_fa's first pass restructured in the shape of llama.cpp's flash_attn_ext_vec, as graded on 2026-09-25
// (fidelity decision on set B, confirmation run 3.50x). It SHIPPED as attention_fa_blk in kernels.go (allKernels);
// this is that shipped text rebuilt into the standalone library form the R17 harnesses compile (the prototype's own
// macro name, its own #include), so the tests grade exactly what production dispatches.
// TestAttnFABlkIsTheGradedKernel pins the rebuilt text to the SHA-256 of the source that was graded.
var r17Kernels = r17GradedKernelSource()

// r17GradedKernelSourceSHA256 is the SHA-256 of the prototype source graded 2026-09-25 (r17Kernels as committed in
// 39545bd6, the commit the pre-registered decision run was built from).
const r17GradedKernelSourceSHA256 = "ae72a2b060007abee88c940ac60308bb5650edaf8c761a273287ce873520ab4e"

// r17GradedKernelSource rebuilds the graded prototype's standalone source from the shipped attention_fa_blk section
// of allKernels (between its begin/end markers, minus the header comment).
func r17GradedKernelSource() string {
	const begin, end = "// ---- attention_fa_blk begin", "// ---- attention_fa_blk end ----"
	b, e := strings.Index(allKernels, begin), strings.Index(allKernels, end)
	if b < 0 || e < b {
		panic("attention_fa_blk markers not found in allKernels")
	}
	lines := strings.Split(allKernels[b:e], "\n")
	n := 0
	for n < len(lines) && (strings.HasPrefix(lines[n], "//") || lines[n] == "") {
		n++
	}
	body := strings.Join(lines[n:], "\n")
	return "#include <metal_stdlib>\nusing namespace metal;\n" + strings.ReplaceAll(body, "ATTN_FA_BLK_UNROLL", "R17_UNROLL")
}

// TestAttnFABlkIsTheGradedKernel fails if the shipped attention_fa_blk kernel is edited: a change to it needs the
// R17 grading (fidelity gate, kernel accuracy, speed) re-run — then update the pin with the new record's commit.
func TestAttnFABlkIsTheGradedKernel(t *testing.T) {
	sum := sha256.Sum256([]byte(r17Kernels))
	if got := hex.EncodeToString(sum[:]); got != r17GradedKernelSourceSHA256 {
		t.Fatalf("shipped attention_fa_blk differs from the kernel graded 2026-09-25 (sha256 %s, graded %s) — re-grade before changing the pin", got, r17GradedKernelSourceSHA256)
	}
}

// blkRun dispatches one attention_fa first pass (pipe) + attention_fa_combine on the given q (f32, [nH*hd]) and
// f16 K/V ([nKeys][nKV][hd]), the way the resident does (model.go's attention_fa dispatch), and returns ctx.
func blkRun(t *testing.T, d *Device, cq Queue, pipe, comb Pipeline, G, nKV, nKeys, split int, q []float32, kh, vh []uint16) []float32 {
	t.Helper()
	const hd = 128
	nH := G * nKV
	kb, vb := d.NewBufferBytes(len(kh)*2), d.NewBufferBytes(len(vh)*2)
	copy(kb.U16s()[:len(kh)], kh)
	copy(vb.U16s()[:len(vh)], vh)
	partial := d.NewBufferLen(nKV * split * G * (hd + 2))
	out := d.NewBufferLen(nH * hd)
	uG, uSplit := NewBufferU32(d, uint32(G)), NewBufferU32(d, uint32(split))
	e := cq.Begin()
	e.DispatchTG(pipe, nKV*split*128, 128, 128*6*G*4, NewBufferFloats(d, q), kb, vb, partial,
		NewBufferU32(d, uint32(nKV)), uG, NewBufferU32(d, uint32(nKeys)), NewBufferFloats(d, []float32{float32(1 / math.Sqrt(hd))}),
		NewBufferU32(d, 0), uSplit)
	e.Dispatch(comb, nH*hd, hd, partial, out, uG, NewBufferU32(d, hd), uSplit)
	e.End()
	if err := e.Err(); err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	return append([]float32(nil), out.Floats()[:nH*hd]...)
}

// blkInputs builds q and f16 K/V; peaked plants a sink-like key (position 0) and a few strong keys, so the online
// softmax's rescaling across blocks and splits is exercised, not only near-uniform weights.
func blkInputs(seed int64, G, nKV, nKeys int, peaked bool) (q []float32, kh, vh []uint16) {
	const hd = 128
	rng := rand.New(rand.NewSource(seed))
	nH := G * nKV
	q = make([]float32, nH*hd)
	for i := range q {
		q[i] = float32(rng.NormFloat64())
	}
	kh, vh = make([]uint16, nKeys*nKV*hd), make([]uint16, nKeys*nKV*hd)
	for i := range kh {
		kh[i] = f32ToF16(float32(rng.NormFloat64() * 0.5))
		vh[i] = f32ToF16(float32(rng.NormFloat64()))
	}
	if peaked {
		for _, s := range []int{0, nKeys / 3, nKeys - 1} {
			for kvh := 0; kvh < nKV; kvh++ {
				for dd := 0; dd < hd; dd++ { // align key s with the group's first query head: a large score
					kh[(s*nKV+kvh)*hd+dd] = f32ToF16(q[(kvh*G)*hd+dd] * 0.35)
				}
			}
		}
	}
	return q, kh, vh
}

// blkRef is attention in float64 from the same f32 q and f16 K/V.
func blkRef(G, nKV, nKeys int, q []float32, kh, vh []uint16) []float64 {
	const hd = 128
	nH := G * nKV
	scale := float64(float32(1 / math.Sqrt(hd)))
	ref := make([]float64, nH*hd)
	sc := make([]float64, nKeys)
	for h := 0; h < nH; h++ {
		kvh := h / G
		mx := math.Inf(-1)
		for s := 0; s < nKeys; s++ {
			var a float64
			for dd := 0; dd < hd; dd++ {
				a += float64(q[h*hd+dd]) * float64(f16ToF32(kh[(s*nKV+kvh)*hd+dd]))
			}
			sc[s] = a * scale
			mx = math.Max(mx, sc[s])
		}
		var sum float64
		for s := range sc {
			sc[s] = math.Exp(sc[s] - mx)
			sum += sc[s]
		}
		for dd := 0; dd < hd; dd++ {
			var a float64
			for s := 0; s < nKeys; s++ {
				a += sc[s] * float64(f16ToF32(vh[(s*nKV+kvh)*hd+dd]))
			}
			ref[h*hd+dd] = a / sum
		}
	}
	return ref
}

// TestAttnFABlkMatchesFloat64 checks the SHIPPED attention_fa_blk (both instantiations, at the production split
// count) against float64 attention on synthetic inputs — no checkpoint. Key counts cover a chunk tail of one key
// (1537), several rounds per simdgroup (3900, 4100), and the depth floor; inputs cover near-uniform and peaked
// (sink-like) weights. The bound is a correctness bound, orders of magnitude above f32 rounding: a wrong key range,
// scale, rescale or merge gives relative errors of 1e-3 and up. The kernel's measured error on real inputs is
// ~2e-7 median (metal-decode-attn-r17-2026-09-25.md).
func TestAttnFABlkMatchesFloat64(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile allKernels: %v", err)
	}
	comb, err := d.NewComputePipeline(lib, "attention_fa_combine")
	if err != nil {
		t.Fatalf("combine pipeline: %v", err)
	}
	cq := d.NewCommandQueue()
	const bound = 1e-4
	worst := 0.0
	for _, gc := range []struct{ G, nKV int }{{6, 2}, {7, 4}} {
		pipe, err := d.NewComputePipeline(lib, fmt.Sprintf("attention_fa_blk_g%d", gc.G))
		if err != nil {
			t.Fatalf("pipeline g%d: %v", gc.G, err)
		}
		for _, nKeys := range []int{attnFADepthFloor, attnFADepthFloor + 1, 3900, 4100} {
			for _, peaked := range []bool{false, true} {
				q, kh, vh := blkInputs(int64(nKeys*10+gc.G), gc.G, gc.nKV, nKeys, peaked)
				got := blkRun(t, d, cq, pipe, comb, gc.G, gc.nKV, nKeys, attnFABlkSplit, q, kh, vh)
				ref := blkRef(gc.G, gc.nKV, nKeys, q, kh, vh)
				for h := 0; h < gc.G*gc.nKV; h++ {
					var num, den float64
					for dd := 0; dd < 128; dd++ {
						x := float64(got[h*128+dd]) - ref[h*128+dd]
						num, den = num+x*x, den+ref[h*128+dd]*ref[h*128+dd]
					}
					rel := math.Sqrt(num / den)
					worst = math.Max(worst, rel)
					if !(rel <= bound) {
						t.Errorf("G=%d nKV=%d nKeys=%d peaked=%v head %d: relative L2 error %.3g vs float64 (bound %g)", gc.G, gc.nKV, nKeys, peaked, h, rel, bound)
					}
				}
			}
		}
	}
	t.Logf("worst per-head relative L2 error vs float64: %.3g (bound %g)", worst, bound)
}

// TestAttnFABlkSelection pins which kernel production dispatches as attention_fa's first pass: the legacy
// attention_fa for a group size the block kernel was not graded at (the committed llama-attnfa-tiny fixture, G=2),
// and — with GOINFER_HEAVY_TESTS=1 and the checkpoints in ~/models — attention_fa_blk for Qwen2.5-1.5B (G=6) and
// -7B (G=7), checked by running the resident's own r.pAttnFA on synthetic inputs and requiring output BIT-IDENTICAL to
// the graded kernel source (r17Kernels) compiled on its own.
func TestAttnFABlkSelection(t *testing.T) {
	check := func(t *testing.T, path string, wantBlk bool) {
		m, err := decoder.Load(path, decoder.Options{Quant: "int4", ResidentContext: 4096})
		if err != nil {
			t.Fatalf("load %s: %v", path, err)
		}
		r, err := buildResident(m)
		if err != nil {
			t.Fatalf("resident: %v", err)
		}
		defer r.Close()
		if r.attnFANKV == 0 {
			t.Fatalf("%s: attention_fa is not eligible on this model", path)
		}
		G := r.nH / r.attnFANKV
		if got := r.attnFABlkSplit > 0; got != wantBlk {
			t.Fatalf("%s (G=%d): block kernel selected=%v, want %v", path, G, got, wantBlk)
		}
		wantSplit := (2*attnFACoreCount + r.attnFANKV - 1) / r.attnFANKV
		if wantBlk {
			wantSplit = attnFABlkSplit
		}
		if got := r.attnFASplitFor(3901, r.attnFANKV); got != wantSplit {
			t.Fatalf("%s: split count at 3901 keys = %d, want %d", path, got, wantSplit)
		}
		if !wantBlk {
			return
		}
		lib, err := r.d.CompileLibrary(r17Kernels, MSL3_1)
		if err != nil {
			t.Fatalf("compile graded source: %v", err)
		}
		graded, err := r.d.NewComputePipeline(lib, fmt.Sprintf("attention_fa_blk_g%d", G))
		if err != nil {
			t.Fatalf("graded pipeline: %v", err)
		}
		cq := r.d.NewCommandQueue()
		q, kh, vh := blkInputs(7, G, r.attnFANKV, 3900, true)
		a := blkRun(t, r.d, cq, r.pAttnFA, r.pAttnFACombine, G, r.attnFANKV, 3900, attnFABlkSplit, q, kh, vh)
		b := blkRun(t, r.d, cq, graded, r.pAttnFACombine, G, r.attnFANKV, 3900, attnFABlkSplit, q, kh, vh)
		for i := range a {
			if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
				t.Fatalf("%s: the resident's attention_fa pipeline is not the graded kernel (output %d differs)", path, i)
			}
		}
	}
	t.Run("G=2 keeps attention_fa", func(t *testing.T) {
		dir := "../testdata/llama-attnfa-tiny"
		if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
			t.Skipf("fixture not present: %v", err)
		}
		check(t, dir, false)
	})
	for _, mm := range []string{"qwen2.5-coder-1.5b-instruct-q4_k_m.int4.metal.giw", "qwen2.5-7b-instruct-q4_k_m.int4.metal.giw"} {
		t.Run(mm, func(t *testing.T) {
			requireHeavyModel(t)
			path := os.ExpandEnv("$HOME/models/" + mm)
			if _, err := os.Stat(path); err != nil {
				t.Skipf("checkpoint not present: %s", path)
			}
			check(t, path, true)
		})
	}
}
