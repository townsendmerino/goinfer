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
	"github.com/townsendmerino/goinfer/metal/kernels"
)

// r17Kernels is the shipped attention_fa_blk (kernels.go, allKernels) rebuilt into the standalone library form the R17
// harnesses compile (the prototype's own macro name, its own #include), so the tests grade exactly what production
// dispatches. The prototype is R17's step 2 (docs/tasks/red-october.md;
// docs/measurements/metal-decode-attn-r17-2026-09-25.md): attention_fa's first pass restructured in the shape of
// llama.cpp's flash_attn_ext_vec.
// TestAttnFABlkIsTheGradedKernel pins the rebuilt text to the SHA-256 of the source that was graded.
var r17Kernels = r17WithoutTable(r17GradedKernelSource())

// r17TableKernels is the same source as it ships, reading the block table, for a test that installs it in a resident.
var r17TableKernels = strings.Replace(r17GradedKernelSource(), "using namespace metal;\n", "using namespace metal;\n"+kernels.KVRowMSL, 1)

// r17GradedKernelSourceSHA256 is the SHA-256 of the prototype source that was graded: r17Kernels as built from the
// commit the pre-registered decision run used: docs/code-notes/metal.md#r17GradedKernelSourceSHA256
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

// r17WithoutTable is source with its block-table reads taken back out: the kernel as graded, addressing a contiguous cache. Paged KV
// adds a lookup per key and nothing else, and the pin below holds that to the byte.
func r17WithoutTable(source string) string {
	source = strings.ReplaceAll(source, "kv_row(kt,b0 + j)*kvDim", "(b0 + j)*kvDim")
	return strings.ReplaceAll(source, ", device const uint* kt[[buffer(10)]],", ",")
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
		NewBufferU32(d, 0), uSplit, kvIdentityTable(d))
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
			for kvh := range nKV {
				for dd := range hd { // align key s with the group's first query head: a large score
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
	for h := range nH {
		kvh := h / G
		mx := math.Inf(-1)
		for s := range nKeys {
			var a float64
			for dd := range hd {
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
		for dd := range hd {
			var a float64
			for s := range nKeys {
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
// scale, rescale or merge gives relative errors of 1e-3 and up. The kernel's error on real inputs is in
// docs/measurements/metal-decode-attn-r17-2026-09-25.md.
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
				recordAttnDigest(t, fmt.Sprintf("attention_fa_blk_g%d,attention_fa_combine", gc.G), got)
				ref := blkRef(gc.G, gc.nKV, nKeys, q, kh, vh)
				for h := 0; h < gc.G*gc.nKV; h++ {
					var num, den float64
					for dd := range 128 {
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

// TestAttnFABlkSelection pins which kernel production dispatches as attention_fa's first pass. On the committed
// llama-attnfa-tiny fixture (G=2) that is the block kernel exactly when attnFABlkAnyG is on (it is on by default); the G=2
// instantiation sits outside the graded, hash-pinned region, and
// TestAttnFABlk_anyGMatchesG7 is what pins it head for head against g7. With GOINFER_HEAVY_TESTS=1 and the checkpoints
// in ~/models, attention_fa_blk for Qwen2.5-1.5B (G=6) and -7B (G=7) is checked by running the resident's own r.pAttnFA
// on synthetic inputs and requiring output BIT-IDENTICAL to the graded kernel source (r17Kernels) compiled on its own.
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
		if !wantBlk || (G != 6 && G != 7) {
			return // the graded source instantiates G = 6 and 7 only
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
	t.Run("G=2 follows attnFABlkAnyG", func(t *testing.T) {
		dir := "../testdata/llama-attnfa-tiny"
		if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
			t.Skipf("fixture not present: %v", err)
		}
		check(t, dir, attnFABlkAnyG)
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

// TestKVPages_fusedKernelsReadTheTable: every first pass of the fused attention reads a key through its block table. Rows laid out
// block by block in a scrambled order, with the table that names them, give the partials of the contiguous layout bit for bit, and a
// table one block off does not. 300 keys cross several block edges, and the block kernels' unaligned 32-key runs straddle them.
func TestKVPages_fusedKernelsReadTheTable(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatal(err)
	}
	cq := d.NewCommandQueue()
	const nKeys, blocks = 300, 6
	for _, c := range []struct {
		kernel          string
		nKV, G, hd, nSp int
	}{
		{"attention_fa", 2, 2, 128, 4},
		{"attention_fa_blk_g2", 2, 2, 128, attnFABlkSplit},
		{"attention_fa_blk_g7", 1, 7, 128, attnFABlkSplit},
		{"attention_fa_blk64_g7", 1, 7, 64, attnFABlkSplit},
	} {
		t.Run(c.kernel, func(t *testing.T) {
			pipe, err := d.NewComputePipeline(lib, c.kernel)
			if err != nil {
				t.Fatal(err)
			}
			kvDim, nH := c.nKV*c.hd, c.nKV*c.G
			rng := rand.New(rand.NewSource(int64(c.G*100 + c.hd)))
			rows := func() []uint16 { // the logical rows, one block's worth past the keys
				out := make([]uint16, blocks*64*kvDim)
				for i := range out {
					out[i] = f32ToF16(float32(rng.NormFloat64()))
				}
				return out
			}
			kl, vl := rows(), rows()
			q := make([]float32, nH*c.hd)
			for i := range q {
				q[i] = float32(rng.NormFloat64())
			}
			// perm[i] is the physical block of logical block i, past a null block.
			run := func(perm []uint32, lie uint32) []float32 {
				phys := func(l []uint16) Buffer {
					b := d.NewBufferBytes((blocks + 1) * 64 * kvDim * 2)
					for i, at := range perm {
						copy(b.U16s()[int(at)*64*kvDim:(int(at)+1)*64*kvDim], l[i*64*kvDim:(i+1)*64*kvDim])
					}
					return b
				}
				tab := make([]uint32, len(perm))
				for i, at := range perm {
					tab[i] = at + lie
				}
				partial := d.NewBufferLen(c.nKV * c.nSp * c.G * (c.hd + 2))
				e := cq.Begin()
				e.DispatchTG(pipe, c.nKV*c.nSp*128, 128, 128*6*c.G*4, NewBufferFloats(d, q), phys(kl), phys(vl), partial,
					NewBufferU32(d, uint32(c.nKV)), NewBufferU32(d, uint32(c.G)), NewBufferU32(d, nKeys), NewBufferFloats(d, []float32{float32(1 / math.Sqrt(float64(c.hd)))}),
					NewBufferU32(d, 0), NewBufferU32(d, uint32(c.nSp)), NewBufferUint32s(d, tab))
				e.End()
				if err := e.Err(); err != nil {
					t.Fatal(err)
				}
				return append([]float32(nil), partial.Floats()...)
			}
			inOrder, scrambled := []uint32{1, 2, 3, 4, 5, 6}, []uint32{6, 3, 5, 1, 4, 2}
			differ := func(a, b []float32) int {
				n := 0
				for i := range a {
					if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
						n++
					}
				}
				return n
			}
			want := run(inOrder, 0)
			if n := differ(want, run(scrambled, 0)); n != 0 {
				t.Errorf("%d of %d partial values differ between scrambled and in-order blocks", n, len(want))
			}
			if n := differ(want, run([]uint32{1, 2, 3, 4, 5, 5}, 1)); n == 0 {
				t.Error("a table one block off moved no partial value: the kernel does not read the table")
			}
		})
	}
}

// TestKVPages_fusedAttentionDepthThroughTheTable: past attnFADepthFloor, decode and the batched step run the fused attention
// through the table: a real resident's scrambled blocks read as its in-order ones, on attention_fa and on the block kernel, on slots
// bound in turn, with the step's rows dispatched one at a time and as one. No prefill runs, which does not read the table yet. Both
// arms share any defect that sends a row to the wrong slot, so each row is also held to a decode of the same token on its own slot.
func TestKVPages_fusedAttentionDepthThroughTheTable(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}
	const fx = "../testdata/llama-attnfa-tiny"
	if _, err := os.Stat(fx + "/model.safetensors"); err != nil {
		t.Skipf("no fixture: %v", err)
	}
	const depth = attnFADepthFloor + 8
	for _, c := range []struct {
		name      string
		blk, rows bool
	}{{"attention_fa", false, false}, {"attention_fa_blk", true, false}, {"attention_fa_rows", false, true}, {"attention_fa_blk_rows", true, true}} {
		blk := c.blk
		t.Run(c.name, func(t *testing.T) {
			defer func(v, w bool) { attnFABlkAnyG, mc3FARowsOn = v, w }(attnFABlkAnyG, mc3FARowsOn)
			attnFABlkAnyG, mc3FARowsOn = blk, c.rows
			run := func(scramble bool) [][]float32 {
				kvScrambleForTest = scramble
				defer func() { kvScrambleForTest = false }()
				m, err := decoder.Load(fx, decoder.Options{Quant: "int4", ResidentContext: depth + 64, ResidentKVSlots: 2})
				if err != nil {
					t.Fatal(err)
				}
				defer m.Close()
				r, err := buildResident(m)
				if err != nil {
					t.Fatal(err)
				}
				defer r.Close()
				if r.attnFANKV == 0 || (r.attnFABlkSplit > 0) != blk {
					t.Fatalf("attention_fa's layer count %d, block kernel %v; want the block kernel %v", r.attnFANKV, r.attnFABlkSplit > 0, blk)
				}
				if r.batch == nil {
					t.Fatalf("no batched step: %s", r.batchIneligible())
				}
				var out [][]float32
				for s := range 2 {
					if err := r.useKVSlot(s); err != nil {
						t.Fatal(err)
					}
					for pos := range depth {
						l := r.ForwardEmb(mc3Emb(r, (pos*7+s*13+1)%r.V), pos)
						if pos >= attnFADepthFloor {
							out = append(out, append([]float32(nil), l...))
						}
					}
				}
				seqs := []batchSeq{{slot: 0, pos: depth, emb: mc3Emb(r, 3)}, {slot: 1, pos: depth, emb: mc3Emb(r, 5)}}
				decoded := make([][]float32, len(seqs))
				for _, q := range seqs {
					if err := r.useKVSlot(q.slot); err != nil {
						t.Fatal(err)
					}
					decoded[q.slot] = append([]float32(nil), r.ForwardEmb(q.emb, q.pos)...)
				}
				for m, l := range mc3Step(t, r, seqs) {
					for j := range l {
						if math.Float32bits(l[j]) != math.Float32bits(decoded[seqs[m].slot][j]) {
							t.Fatalf("scrambled %v: step row %d logit %d is %v, its slot's decode %v", scramble, m, j, l[j], decoded[seqs[m].slot][j])
						}
					}
					out = append(out, append([]float32(nil), l...))
				}
				return out
			}
			a, b := run(false), run(true)
			n := 0
			for i := range a {
				for j := range a[i] {
					if math.Float32bits(a[i][j]) != math.Float32bits(b[i][j]) {
						n++
					}
				}
			}
			if n != 0 {
				t.Errorf("%d logits differ between scrambled and in-order blocks", n)
			}
		})
	}
}
