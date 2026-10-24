//go:build darwin && goinfer_testhooks

package metal

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/hostmem"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// kso1Q6KGGUF is n rows of random GGUF Q6_K blocks over K (random codes and int8 scales, small positive f16 d).
func kso1Q6KGGUF(rng *rand.Rand, n, K int) []byte {
	raw := make([]byte, n*(K/256)*210)
	for b := 0; b < len(raw); b += 210 {
		for i := b; i < b+208; i++ {
			raw[i] = byte(rng.Intn(256))
		}
		h := f32ToF16(float32(0.0005 + 0.002*rng.Float64()))
		raw[b+208], raw[b+209] = byte(h), byte(h>>8)
	}
	return raw
}

// kso1Q6KPlanes lays N rows of GGUF Q6_K blocks over K out as the lane uploads one segment (q6kPlanarRow).
func kso1Q6KPlanes(raw []byte, N, K int) []byte {
	nb, rb := K/256, K/256*210
	out := make([]byte, len(raw))
	for r := range N {
		q6kPlanarRow(out, N*nb, r*nb, raw[r*rb:(r+1)*rb])
	}
	return out
}

// kso1NativeWorst runs the kind's three variants at R = 4 (the shipped R) from src over random blocks at N x K and returns the
// worst |err| / sum|w*a| against the float64 reference over every row and variant.
func kso1NativeWorst(t *testing.T, d *Device, src, kind string, N, K int, rng *rand.Rand) float64 {
	lib, err := d.CompileLibrary(src, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	q := d.NewCommandQueue()
	bb, raw, ref := 176, kso1Q5KRaw(rng, N, K), kso1DequantQ5K
	if kind == "q6k" {
		bb, raw, ref = 210, kso1Q6KGGUF(rng, N, K), kso1DequantQ6K
	}
	rb := K / 256 * bb
	up := raw
	if kind == "q6k" {
		up = kso1Q6KPlanes(raw, N, K)
	}
	w := d.NewBufferBytes(len(up))
	iv := w.Int8s()[:len(up)]
	for i, c := range up {
		iv[i] = int8(c)
	}
	a, bias, init := make([]float32, K), make([]float32, N), make([]float32, N)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
		if i%997 == 0 {
			a[i] = float32(300 * rng.NormFloat64())
		}
	}
	for i := range bias {
		bias[i], init[i] = float32(rng.NormFloat64()), float32(rng.NormFloat64())
	}
	want, mag := make([]float64, N), make([]float64, N)
	for r := range N {
		wr := ref(raw[r*rb:(r+1)*rb], K)
		for i := range K {
			want[r] += wr[i] * float64(a[i])
			mag[r] += math.Abs(wr[i] * float64(a[i]))
		}
	}
	ba, bbias, uK, uN := NewBufferFloats(d, a), NewBufferFloats(d, bias), NewBufferU32(d, uint32(K)), NewBufferU32(d, uint32(N))
	worst := 0.0
	for _, v := range []string{"", "_resid", "_bias"} {
		p, err := d.NewComputePipeline(lib, fmt.Sprintf("gemv_%s_f32%s_rows4", kind, v))
		if err != nil {
			t.Fatal(err)
		}
		out := NewBufferFloats(d, slices.Clone(init))
		args := []Buffer{w, ba, out, uK, uN}
		if v == "_bias" {
			args = append(args, bbias)
		}
		e := q.Begin()
		e.Dispatch(p, (N+3)/4*32, 256, args...)
		e.End()
		if err := e.Err(); err != nil {
			t.Fatalf("%s%s: dispatch: %v", kind, v, err)
		}
		got := out.Floats()[:N]
		for r := range N {
			x := float64(got[r])
			switch v {
			case "_resid":
				x -= float64(init[r])
			case "_bias":
				x -= float64(bias[r])
			}
			worst = max(worst, math.Abs(x-want[r])/mag[r])
		}
	}
	return worst
}

// TestKSO1_nativeKernels is O1-G1 (docs/tasks/task-kernel-survey-2026-10.md, "KS-O1 on Metal: gates"): the q4k lane's native
// gemv_q5k_f32 and gemv_q6k_f32 rows kernels (plain, residual, bias at R = 4) against the float64 reference that dequantizes
// the same blocks, at a partial threadgroup over 1, 2, 6, 19 and 35 super-blocks a row and at Phi-3's shapes (qkv, 12, and
// down, 32, for both kinds; the head for Q6_K); random
// blocks, so the 6-bit scales and mins reach both range ends. Each row within G-Q1's bar (1e-5 of sum|w*a|), and each planted
// defect red.
func TestKSO1_nativeKernels(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	src := allKernels + mc3RowsKernels
	rng := rand.New(rand.NewSource(79))
	// NB = K/256 of 1, 2, 6, 19 and 35 (the 1.5B's attn_v, the 0.5B's and the 1.5B's ffn_down) besides Phi-3's 12 and 32: an odd
	// or non-multiple-of-4 block count is where a row-strided Q6_K layout would misalign.
	odd := [][2]int{{1, 256}, {37, 512}, {37, 1536}, {37, 4864}, {37, 8960}}
	shapes := map[string][][2]int{"q5k": append(slices.Clone(odd), [2]int{9216, 3072}, [2]int{3072, 8192}), "q6k": append(slices.Clone(odd), [2]int{3072, 8192}, [2]int{32064, 3072})}
	for _, kind := range []string{"q5k", "q6k"} {
		for _, sh := range shapes[kind] {
			worst := kso1NativeWorst(t, d, src, kind, sh[0], sh[1], rng)
			fmt.Fprintf(os.Stderr, "[O1-G1] %s N=%d K=%d: worst |err|/sum|w*a| %.2e\n", kind, sh[0], sh[1], worst)
			if worst > 1e-5 {
				t.Errorf("%s N=%d K=%d: worst %.2e over 1e-5", kind, sh[0], sh[1], worst)
			}
		}
		d.ReleaseAll()
	}
	for _, p := range []struct{ name, kind, from, to string }{
		{"(1) Q5_K's high bit dropped", "q5k", "uint lo = (q[i] & 0x0F0F0F0Fu) | (((qh[i] >> (2u*j)) & 0x01010101u) << 4u);", "uint lo = (q[i] & 0x0F0F0F0Fu);"},
		{"(2) Q6_K's -32 offset dropped", "q6k", "inline float4 q6k_s6(uint t) { return float4(as_type<char4>(((t | 0x80808080u) - 0x20202020u) ^ 0x80808080u)); }",
			"inline float4 q6k_s6(uint t) { return float4(as_type<uchar4>(t)); }"},
		{"(3) a Q6_K group read with its neighbour's scale", "q6k", "(float(sc[0])*d0 + float(sc[2])*d1)", "(float(sc[1])*d0 + float(sc[2])*d1)"},
		{"(4) a super-block offset of one", "q5k", "acc[r] += q5k_blk(wq + (ulong)(row0 + r)*(ulong)NB*176ul + b*176u,", "acc[r] += q5k_blk(wq + (ulong)(row0 + r)*(ulong)NB*176ul + ((b+1u)%NB)*176u,"},
	} {
		if strings.Count(src, p.from) != 1 {
			t.Fatalf("plant %s: its source text is not in the library once", p.name)
		}
		worst := kso1NativeWorst(t, d, strings.Replace(src, p.from, p.to, 1), p.kind, 37, 512, rng)
		fmt.Fprintf(os.Stderr, "[O1-G1] planted %s: worst %.2e\n", p.name, worst)
		if worst <= 1e-5 {
			t.Errorf("planted %s left the bar green (%.2e)", p.name, worst)
		}
		d.ReleaseAll()
	}
}

// kso1ReferenceCheck is O1-G1's reference check on one GGUF: the float64 Q5_K and Q6_K dequantization the kernels are graded
// against, on the file's real blocks, within 1e-6 of each row's largest weight of aikit's RowDequantizer, on three rows of every
// such tensor. It returns how many tensors of each kind it checked.
func kso1ReferenceCheck(t *testing.T, path string) map[string]int {
	checked := map[string]int{}
	g, err := decoder.OpenGGUFMmap(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	for _, name := range g.Names() {
		for _, k := range []struct {
			kind string
			bb   int
			get  func(string) ([]int, []byte, bool, error)
			ref  func([]byte, int) []float64
		}{{"q5k", 176, g.Q5KRaw, kso1DequantQ5K}, {"q6k", 210, g.Q6KRaw, kso1DequantQ6K}} {
			dims, raw, ok, err := k.get(name)
			if err != nil {
				t.Fatalf("%s %s: %v", filepath.Base(path), name, err)
			}
			if !ok {
				continue
			}
			in, out := dims[0], dims[1]
			_, into, err := g.RowDequantizer(name)
			if err != nil {
				t.Fatal(err)
			}
			want := make([]float32, in)
			rb := in / 256 * k.bb
			for _, r := range []int{0, out / 2, out - 1} {
				if err := into(r*in, want); err != nil {
					t.Fatal(err)
				}
				got := k.ref(raw[r*rb:(r+1)*rb], in)
				big := 0.0
				for _, v := range want {
					big = max(big, math.Abs(float64(v)))
				}
				for i := range want {
					if math.Abs(got[i]-float64(want[i])) > 1e-6*big {
						t.Fatalf("%s %s row %d [%d]: reference %v, aikit %v", filepath.Base(path), name, r, i, got[i], want[i])
					}
				}
			}
			checked[k.kind]++
		}
	}
	fmt.Fprintf(os.Stderr, "[O1-G1] %s: reference within 1e-6 of aikit's dequantizer on %d Q5_K and %d Q6_K tensors\n", filepath.Base(path), checked["q5k"], checked["q6k"])
	return checked
}

// TestKSO1_referenceMatchesAikit is kso1ReferenceCheck on Qwen2.5-Coder-0.5B q4_k_m, which must hold Q6_K tensors.
func TestKSO1_referenceMatchesAikit(t *testing.T) {
	home, _ := os.UserHomeDir()
	path := filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no GGUF at %s", path)
	}
	if c := kso1ReferenceCheck(t, path); c["q6k"] == 0 {
		t.Fatalf("no Q6_K tensor in %s", path)
	}
}

// TestKSO1_referenceMatchesAikitPhi3 is kso1ReferenceCheck on Phi-3 mini (the registered GOINFER_PHI3_GGUF asset), which must
// hold both kinds.
func TestKSO1_referenceMatchesAikitPhi3(t *testing.T) {
	if c := kso1ReferenceCheck(t, decoder.AssetPathForTest(t, "GOINFER_PHI3_GGUF")); c["q5k"] == 0 || c["q6k"] == 0 {
		t.Fatalf("Phi-3 holds %v, want both kinds", c)
	}
}

// TestKSO1_stepMatchesDecode is O1-G2's kernel half: the batched step's mc3_gemv_q5k_f32_rows<B> and mc3_gemv_q6k_f32_rows<B>
// (B = 1, 2, 4, 8; modes 0, 1, 2) against decode's rows kernels at R = 4 on the same weights and each activation row alone, at
// 1, 6, 12, 19 and 35 super-blocks a row and an odd row count, in the resident's fast-math library. Byte-equal.
func TestKSO1_stepMatchesDecode(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels+mc3RowsKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe := func(n string) Pipeline {
		p, e := d.NewComputePipeline(lib, n)
		if e != nil {
			t.Fatalf("%s: %v", n, e)
		}
		return p
	}
	q := d.NewCommandQueue()
	rng := rand.New(rand.NewSource(83))
	const N = 997
	for _, kc := range []struct {
		kind string
		K    int
	}{{"q5k", 256}, {"q5k", 1536}, {"q5k", 3072}, {"q5k", 4864}, {"q5k", 8960}, {"q6k", 256}, {"q6k", 1536}, {"q6k", 3072}, {"q6k", 4864}, {"q6k", 8960}} {
		kind, K := kc.kind, kc.K
		raw := kso1Q5KRaw(rng, N, K)
		if kind == "q6k" {
			raw = kso1Q6KPlanes(kso1Q6KGGUF(rng, N, K), N, K)
		}
		w := d.NewBufferBytes(len(raw))
		iv := w.Int8s()[:len(raw)]
		for i, c := range raw {
			iv[i] = int8(c)
		}
		bias, init, act := make([]float32, N), make([]float32, 8*N), make([]float32, 8*K)
		for i := range bias {
			bias[i] = float32(rng.NormFloat64())
		}
		for i := range init {
			init[i] = float32(rng.NormFloat64())
		}
		for i := range act {
			act[i] = float32(rng.NormFloat64())
		}
		bB, bA, uK, uN, uLda, uLdo := NewBufferFloats(d, bias), NewBufferFloats(d, act), NewBufferU32(d, uint32(K)), NewBufferU32(d, N), NewBufferU32(d, uint32(K)), NewBufferU32(d, N)
		for mode := range 3 {
			for _, B := range []int{1, 2, 4, 8} {
				step := NewBufferFloats(d, slices.Clone(init[:B*N]))
				e := q.Begin()
				e.Dispatch(pipe(fmt.Sprintf("mc3_gemv_%s_f32_rows%d", kind, B)), N*32, 256, w, bA, step, uK, uN, uLda, uLdo, bB, NewBufferU32(d, uint32(mode)))
				e.End()
				if err := e.Err(); err != nil {
					t.Fatalf("step %s B=%d: dispatch: %v", kind, B, err)
				}
				diff := 0
				for b := range B {
					out := NewBufferFloats(d, slices.Clone(init[b*N:(b+1)*N]))
					variant := []string{"", "_bias", "_resid"}[mode]
					args := []Buffer{w, bA.At(b * K * 4), out, uK, uN}
					if mode == 1 {
						args = append(args, bB)
					}
					e := q.Begin()
					e.Dispatch(pipe(fmt.Sprintf("gemv_%s_f32%s_rows4", kind, variant)), (N+3)/4*32, 256, args...)
					e.End()
					if err := e.Err(); err != nil {
						t.Fatalf("decode %s%s: dispatch: %v", kind, variant, err)
					}
					for i, v := range out.U32s()[:N] {
						if v != step.U32s()[b*N+i] {
							diff++
						}
					}
				}
				fmt.Fprintf(os.Stderr, "[O1-G2k] %s K=%d mode %d B=%d: %d of %d differ\n", kind, K, mode, B, diff, B*N)
				if diff != 0 {
					t.Errorf("%s K=%d mode %d B=%d: %d outputs differ between the step and decode", kind, K, mode, B, diff)
				}
			}
		}
	}
}

// kso1LaneBytes is the device bytes of a resident's q4k-lane weight segments (each segment's int8 or block weight buffer and,
// for int8, its float32 scales).
func kso1LaneBytes(r *resident) (total int, byKind map[string]int) {
	byKind = map[string]int{}
	for _, L := range r.layers {
		for _, segs := range [][]q4kSeg{L.qkvSegs, L.oSegs, L.guSegs, L.dSegs} {
			for _, sg := range segs {
				n := sg.w.Len() + 4*sg.s.Len() // Len counts elements: int8 weights, float32 scales
				total += n
				byKind[sg.kind] += n
			}
		}
	}
	return total, byKind
}

// kso1LaneBytesCheck is O1-G3 on one q4k GGUF: decoder.Load on the Metal backend, which builds the resident from the load's own
// mapping, with the int8 copies (q4kNativeOffForTest, set before the load, so that arm never opens the reader; the decoder still
// records provenance in both arms) and with the native kinds, after one discarded warm-up load so neither arm reads the file
// cold. Native must upload exactly the file's layer Q5_K and Q6_K bytes in place of their int8 rows and scales, and layer 0's
// native segments must hold the file's bytes (kso1CheckLayer0Blocks); the fit pricing's saving
// (metalKQuantSavings) must equal the device bytes it saves; and the whole load must allocate no more Go heap than the int8 arm
// plus kso1HeapSlack.
func kso1LaneBytesCheck(t *testing.T, path, label string) {
	build := func(off bool) (int, map[string]int, uint64, int64) {
		q4kNativeOffForTest = off
		defer func() { q4kNativeOffForTest = false }()
		var m *decoder.Model
		var err error
		heap := heapAllocDuring(func() {
			m, err = decoder.Load(path, decoder.Options{Quant: "q4k", Backend: "metal", ResidentContext: 256})
		})
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		mr, ok := m.ResidentForwardForTest().(*metalResident)
		if !ok || !mr.r.q4kLane {
			t.Fatalf("%s: the load did not build a q4k-lane Metal resident (%T)", label, m.ResidentForwardForTest())
		}
		tot, by := kso1LaneBytes(mr.r)
		if !off {
			kso1CheckLayer0Blocks(t, m, mr.r, path, label)
		}
		return tot, by, heap, m.KQuantDeviceSavingsForTest()
	}
	build(false)
	oldTot, oldBy, oldHeap, _ := build(true)
	newTot, newBy, newHeap, saved := build(false)
	g, err := decoder.OpenGGUFMmap(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	file := map[string]int{}
	for _, name := range g.Names() {
		if !strings.HasPrefix(name, "blk.") {
			continue
		}
		for kind, get := range map[string]func(string) ([]int, []byte, bool, error){"q5k": g.Q5KRaw, "q6k": g.Q6KRaw} {
			_, raw, ok, err := get(name)
			if err != nil {
				t.Fatalf("%s %s: %v", label, name, err)
			}
			if ok {
				file[kind] += len(raw)
			}
		}
	}
	fmt.Fprintf(os.Stderr, "[O1-G3] %s lane device bytes: int8 copies %d (%v), native %d (%v), the file's layer blocks %v; fit pricing saves %d; load heap %d -> %d bytes\n",
		label, oldTot, oldBy, newTot, newBy, file, saved, oldHeap, newHeap)
	if newBy["q5k"] != file["q5k"] || newBy["q6k"] != file["q6k"] || newBy["q4k"] != oldBy["q4k"] || newTot >= oldTot {
		t.Errorf("%s: native q5k %d / q6k %d bytes against the file's %v, q4k %d against %d, total %d against %d", label, newBy["q5k"], newBy["q6k"], file, newBy["q4k"], oldBy["q4k"], newTot, oldTot)
	}
	if saved != int64(oldTot-newTot) {
		t.Errorf("%s: the fit pricing saves %d bytes, the device %d", label, saved, oldTot-newTot)
	}
	if newHeap > oldHeap+kso1HeapSlack {
		t.Errorf("%s: the native load allocated %d heap bytes against %d (+%d slack)", label, newHeap, oldHeap, kso1HeapSlack)
	}
}

// kso1HeapSlack is O1-G3's allowance for the Go runtime's own allocations during a load (GC is off while heapAllocDuring
// counts, but background goroutines still allocate), registered in docs/tasks/task-kernel-survey-2026-10.md, "Review fixes for
// #215, second pass".
const kso1HeapSlack = 1 << 20

// kso1CheckLayer0Blocks compares the device bytes of layer 0's native segments with the file's own tensors, read straight from
// the GGUF (not through KQuantReader or q6kPlanarRow): a group's rows come from its fused tensor when that tensor has the group's
// row count (Phi-3's attn_qkv, ffn_up = gate|up), else from one tensor a part (attn_q/k/v, ffn_gate/ffn_up), in order. Q5_K rows
// are compared verbatim, Q6_K through kso1Q6KPlanesRef. These files store no RoPE-permuted rows. At least one segment must be
// checked.
func kso1CheckLayer0Blocks(t *testing.T, m *decoder.Model, r *resident, path, label string) {
	g, err := decoder.OpenGGUFMmap(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	L, lw := &r.layers[0], &m.Weights().Layers[0]
	checked := 0
	for _, gr := range []struct {
		segs  []q4kSeg
		parts []*linalg.WeightMat
		fused string
		names []string
	}{{L.qkvSegs, []*linalg.WeightMat{&lw.QProj, &lw.KProj, &lw.VProj}, "attn_qkv", []string{"attn_q", "attn_k", "attn_v"}},
		{L.oSegs, []*linalg.WeightMat{&lw.OProj}, "", []string{"attn_output"}},
		{L.guSegs, []*linalg.WeightMat{&lw.GateProj, &lw.UpProj}, "ffn_up", []string{"ffn_gate", "ffn_up"}},
		{L.dSegs, []*linalg.WeightMat{&lw.DownProj}, "", []string{"ffn_down"}}} {
		rows := 0
		for _, w := range gr.parts {
			rows += w.Rows()
		}
		// src[r] is the group's row r as (tensor, row in it).
		type at struct {
			name string
			row  int
		}
		var src []at
		if dims, ok := g.Dims("blk.0." + gr.fused + ".weight"); gr.fused != "" && ok && len(dims) == 2 && dims[1] == rows {
			for r := range rows {
				src = append(src, at{"blk.0." + gr.fused + ".weight", r})
			}
		} else {
			for pi, w := range gr.parts {
				for r := range w.Rows() {
					src = append(src, at{"blk.0." + gr.names[pi] + ".weight", r})
				}
			}
		}
		for _, sg := range gr.segs {
			if sg.kind != "q5k" && sg.kind != "q6k" {
				continue
			}
			get, bb := g.Q5KRaw, 176
			if sg.kind == "q6k" {
				get, bb = g.Q6KRaw, 210
			}
			var raw []byte
			K := 0
			for r := sg.off; r < sg.off+sg.rows; r++ {
				dims, blocks, ok, err := get(src[r].name)
				if err != nil || !ok {
					t.Fatalf("%s: %s is not %s in the file (err %v)", label, src[r].name, sg.kind, err)
				}
				K = dims[0]
				rb := K / 256 * bb
				raw = append(raw, blocks[src[r].row*rb:(src[r].row+1)*rb]...)
			}
			want := raw
			if sg.kind == "q6k" {
				want = kso1Q6KPlanesRef(raw, sg.rows, K)
			}
			got := sg.w.Int8s()[:sg.w.Len()]
			if len(got) != len(want) {
				t.Fatalf("%s layer 0 %s segment at row %d: %d bytes on the device, %d in the file", label, sg.kind, sg.off, len(got), len(want))
			}
			for i, v := range got {
				if byte(v) != want[i] {
					t.Fatalf("%s layer 0 %s segment at row %d: byte %d is %d, the file's %d", label, sg.kind, sg.off, i, byte(v), want[i])
				}
			}
			checked++
		}
	}
	if checked == 0 {
		t.Fatalf("%s: layer 0 holds no native segment to check", label)
	}
	fmt.Fprintf(os.Stderr, "[O1-G3] %s: layer 0's %d native segments hold the file's blocks byte for byte\n", label, checked)
}

// kso1Q6KPlanesRef is the Q6_K segment layout written out byte by byte, independent of q6kPlanarRow: for block g = row*NB + b
// of a segment of T blocks, ql[i] at g*128+i, qh[i] at T*128 + g*64+i, scale i at T*192 + g*16+i, and d's two bytes at T*208 +
// g*2.
func kso1Q6KPlanesRef(raw []byte, N, K int) []byte {
	nb := K / 256
	T := N * nb
	out := make([]byte, len(raw))
	for r := range N {
		for b := range nb {
			g := r*nb + b
			blk := raw[g*210:]
			for i := range 128 {
				out[g*128+i] = blk[i]
			}
			for i := range 64 {
				out[T*128+g*64+i] = blk[128+i]
			}
			for i := range 16 {
				out[T*192+g*16+i] = blk[192+i]
			}
			out[T*208+g*2], out[T*208+g*2+1] = blk[208], blk[209]
		}
	}
	return out
}

// TestKSO1_laneBytes is O1-G3 on the Qwen2.5-Coder-0.5B q4_k_m (Q6_K ffn_down layers; no Q5_K).
func TestKSO1_laneBytes(t *testing.T) {
	home, _ := os.UserHomeDir()
	path := filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no GGUF at %s", path)
	}
	kso1LaneBytesCheck(t, path, "0.5B")
}

// TestKSO1_laneBytesPhi3 is O1-G3 on Phi-3 mini (the registered GOINFER_PHI3_GGUF asset), whose Q5_K attn_qkv the 0.5B lacks.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -tags goinfer_testhooks -run '^TestKSO1_laneBytesPhi3$' -v ./metal/
func TestKSO1_laneBytesPhi3(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	kso1LaneBytesCheck(t, decoder.AssetPathForTest(t, "GOINFER_PHI3_GGUF"), "Phi-3 mini")
}

// kso1KL is KL(p || q) in nats for the softmaxes of logits p and q, in float64.
func kso1KL(p, q []float32) float64 {
	lse := func(x []float32) float64 {
		m := math.Inf(-1)
		for _, v := range x {
			m = max(m, float64(v))
		}
		s := 0.0
		for _, v := range x {
			s += math.Exp(float64(v) - m)
		}
		return m + math.Log(s)
	}
	lp, lq := lse(p), lse(q)
	kl := 0.0
	for i := range p {
		a := float64(p[i]) - lp
		kl += math.Exp(a) * (a - (float64(q[i]) - lq))
	}
	return kl
}

// TestKSO1_towardTheFile is O1-G6: Qwen2.5-Coder-0.5B q4_k_m (Q6_K ffn_down in half its layers), the CPU at f32 on the GGUF (the
// file's weights exactly) as the reference. Over G3's eight prompts, it takes the reference's 32-token greedy continuation and runs
// it teacher-forced through two q4k Metal residents: the int8 copies (q4kNativeOffForTest) and the native lane. At every position it
// takes KL(reference || resident) of the logits. PASS iff the native lane's mean KL is at most the int8 lane's; up to 1.10x is
// ambiguous (parked); above that, FAIL.
//
//	GOINFER_HEAVY_TESTS=1 go test -count=1 -tags goinfer_testhooks -run '^TestKSO1_towardTheFile$' -v ./metal/
func TestKSO1_towardTheFile(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") != "1" {
		t.Skip("heavy-checkpoint test: set GOINFER_HEAVY_TESTS=1")
	}
	home, _ := os.UserHomeDir()
	gguf := filepath.Join(home, "models", "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf")
	tk, err := tokenizer.LoadGGUF(gguf)
	if err != nil {
		t.Skipf("no tokenizer: %v", err)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatal(err)
	}
	mc, err := decoder.Load(gguf, decoder.Options{Backend: "cpu", Quant: "f32"})
	if err != nil {
		t.Fatal(err)
	}
	defer mc.Close()
	mg, err := decoder.Load(gguf, decoder.Options{Backend: "cpu", Quant: "q4k", ResidentContext: 512})
	if err != nil {
		t.Fatal(err)
	}
	defer mg.Close()
	build := func(off bool) *resident {
		q4kNativeOffForTest = off
		defer func() { q4kNativeOffForTest = false }()
		r, err := buildResident(mg)
		if err != nil {
			t.Fatal(err)
		}
		if !r.q4kLane {
			t.Fatal("the resident did not take the q4k lane")
		}
		return r
	}
	rOld, rNew := build(true), build(false)
	defer rOld.Close()
	defer rNew.Close()
	if _, by := kso1LaneBytes(rNew); by["q6k"] == 0 {
		t.Fatal("the native resident holds no Q6_K segment")
	}
	var klOld, klNew float64
	n, newBelow := 0, 0
	for pi, text := range g3Prompts {
		ids, err := tk.Encode(tmpl.Render("", []chat.Turn{{Role: "user", Content: text}}), false)
		if err != nil {
			t.Fatal(err)
		}
		cc := mc.NewCache(len(ids) + g3NewTokens)
		seq := append([]int(nil), ids...)
		var po, pn float64
		for i := 0; i < len(ids)+g3NewTokens-1; i++ {
			cl, err := mc.ForwardForTest(seq[i], cc)
			if err != nil {
				t.Fatal(err)
			}
			cl = slices.Clone(cl)
			if i >= len(ids)-1 {
				seq = append(seq, argmaxF(cl))
			}
			emb := mg.EmbedResidentForTest(seq[i])
			o, nw := kso1KL(cl, rOld.ForwardEmb(emb, i)), kso1KL(cl, rNew.ForwardEmb(emb, i))
			po, pn = po+o, pn+nw
			if nw < o {
				newBelow++
			}
			n++
		}
		klOld, klNew = klOld+po, klNew+pn
		fmt.Fprintf(os.Stderr, "[O1-G6] prompt %d (%d positions): mean KL int8 lane %.5f, native %.5f\n", pi+1, len(ids)+g3NewTokens-1,
			po/float64(len(ids)+g3NewTokens-1), pn/float64(len(ids)+g3NewTokens-1))
	}
	ratio := klNew / klOld
	fmt.Fprintf(os.Stderr, "[O1-G6] %d positions: mean KL(f32 || int8 lane) %.5f, KL(f32 || native) %.5f, native/int8 %.4f; native lower at %d of %d positions\n",
		n, klOld/float64(n), klNew/float64(n), ratio, newBelow, n)
	switch {
	case ratio <= 1.0:
		fmt.Fprintf(os.Stderr, "[O1-G6] PASS\n")
	case ratio <= 1.10:
		t.Errorf("O1-G6 AMBIGUOUS (parked): native/int8 mean KL %.4f", ratio)
	default:
		t.Errorf("O1-G6 FAIL: native/int8 mean KL %.4f", ratio)
	}
}

// kso1Speed is O1-G7's A/B on one q4k GGUF. It builds one resident on the native lane (from a CPU-backend load, so only one
// resident exists), plus int8 copies of every projection group (qkv, o, gate|up, down) that holds a Q5_K or Q6_K segment
// (q4kLaneProj with no reader), after checking the copies fit: available memory must cover them plus the 2 GB margin, and the
// GPU's working set must hold them beside the resident. Each run swaps every layer's groups between the two and decodes 128
// greedy tokens after a 32-token prompt through ForwardEmb, A B B A over 10 pairs after 2 warm runs each, with each pair's start
// load printed. A fit refusal prints FIT-REFUSED. It returns the median native/int8 per-token time and how many pairs native was
// faster.
func kso1Speed(t *testing.T, path, label string) (float64, int) {
	mg, err := decoder.Load(path, decoder.Options{Backend: "cpu", Quant: "q4k", ResidentContext: 256})
	if err != nil {
		if strings.Contains(err.Error(), "NOT loaded") {
			fmt.Fprintf(os.Stderr, "[O1-G7] %s FIT-REFUSED: %v\n", label, err)
		}
		t.Fatal(err)
	}
	defer mg.Close()
	if why := residentMemoryDecline(mg); why != "" {
		fmt.Fprintf(os.Stderr, "[O1-G7] %s FIT-REFUSED: %s\n", label, why)
		t.Fatalf("Metal's fit guard declines %s: %s", label, why)
	}
	r, err := buildResident(mg)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if !r.q4kLane {
		t.Fatal("the resident did not take the q4k lane")
	}
	hasNative := func(ss []q4kSeg) bool {
		for _, s := range ss {
			if s.kind == "q5k" || s.kind == "q6k" {
				return true
			}
		}
		return false
	}
	partBytes := func(w *linalg.WeightMat) int64 {
		if raw, ok := w.Q4K(); ok {
			return int64(len(raw))
		}
		q8, sc, _, _ := w.Int8()
		return int64(len(q8)) + 4*int64(len(sc))
	}
	type group struct {
		segs  *[]q4kSeg
		parts []*linalg.WeightMat
	}
	var rebuild []group
	var extra int64
	for l := range r.layers {
		L, lw := &r.layers[l], &mg.Weights().Layers[l]
		for _, g := range []group{{&L.qkvSegs, []*linalg.WeightMat{&lw.QProj, &lw.KProj, &lw.VProj}}, {&L.oSegs, []*linalg.WeightMat{&lw.OProj}},
			{&L.guSegs, []*linalg.WeightMat{&lw.GateProj, &lw.UpProj}}, {&L.dSegs, []*linalg.WeightMat{&lw.DownProj}}} {
			if hasNative(*g.segs) {
				rebuild = append(rebuild, g)
				for _, w := range g.parts {
					extra += partBytes(w)
				}
			}
		}
	}
	if len(rebuild) == 0 {
		t.Fatalf("%s: no projection took a native kind", label)
	}
	const margin = 2 << 30
	avail := hostmem.Available()
	devNow := residentNeedBytes(mg) - mg.ResidentHostCopyBytes(metalMoESlotsFromEnv(mg))
	switch wsCap := metalWorkingSetCap(); {
	case avail > 0 && extra+margin > avail:
		fmt.Fprintf(os.Stderr, "[O1-G7] %s FIT-REFUSED: the int8 arm's copies need %d bytes + the 2 GB margin, %d available\n", label, extra, avail)
		t.Fatalf("%s: the int8 arm's %d bytes do not fit %d available", label, extra, avail)
	case wsCap > 0 && devNow+extra > wsCap:
		fmt.Fprintf(os.Stderr, "[O1-G7] %s FIT-REFUSED: resident %d + int8 copies %d bytes over the GPU working set %d\n", label, devNow, extra, wsCap)
		t.Fatalf("%s: the int8 arm does not fit the GPU working set", label)
	}
	native, int8s := make([][]q4kSeg, len(rebuild)), make([][]q4kSeg, len(rebuild))
	for i, g := range rebuild {
		native[i], int8s[i] = *g.segs, q4kLaneProj(r.d, nil, g.parts...)
	}
	fmt.Fprintf(os.Stderr, "[O1-G7] %s: %d projection groups native; the int8 arm adds %d bytes (%d available)\n", label, len(rebuild), extra, avail)
	nb, ob := kso1LaneBytes(r)
	fmt.Fprintf(os.Stderr, "[O1-G7] %s: lane device bytes %d (%v)\n", label, nb, ob)
	use := func(a [][]q4kSeg) {
		for i, g := range rebuild {
			*g.segs = a[i]
		}
	}
	defer use(native)
	const prompt, gen = 32, 128
	_, _, _, _, _, _, vocab := mg.Dims()
	toks := map[bool][]int{}
	run := func(isNative bool) float64 {
		a := int8s
		if isNative {
			a = native
		}
		use(a)
		for pos := range prompt {
			r.ForwardEmb(mg.EmbedResidentForTest((pos*7919+13)%vocab), pos)
		}
		tok := 13
		seq := make([]int, 0, gen)
		t0 := time.Now()
		for pos := prompt; pos < prompt+gen; pos++ {
			lg := r.ForwardEmb(mg.EmbedResidentForTest(tok), pos)
			tok = 0
			for i, v := range lg {
				if v > lg[tok] {
					tok = i
				}
			}
			seq = append(seq, tok)
		}
		s := time.Since(t0).Seconds() / gen
		if prev, ok := toks[isNative]; ok && !slices.Equal(prev, seq) {
			t.Errorf("%s: the native=%v arm's tokens changed between runs", label, isNative)
		}
		toks[isNative] = seq
		return s
	}
	for range 2 {
		run(false)
		run(true)
	}
	var ratios []float64
	below := 0
	for i := range 10 {
		la, _ := exec.Command("sysctl", "-n", "vm.loadavg").Output()
		var o, n float64
		if i%2 == 0 {
			o, n = run(false), run(true)
		} else {
			n, o = run(true), run(false)
		}
		ratios = append(ratios, n/o)
		if n < o {
			below++
		}
		fmt.Fprintf(os.Stderr, "[O1-G7] %s pair %d (load %s): int8 %.2f ms/token (%.1f tok/s), native %.2f ms/token (%.1f tok/s), native/int8 %.4f\n",
			label, i, la[:len(la)-1], o*1e3, 1/o, n*1e3, 1/n, n/o)
	}
	slices.Sort(ratios)
	med := (ratios[4] + ratios[5]) / 2
	same := 0
	for i := range toks[true] {
		if toks[true][i] == toks[false][i] {
			same++
		} else {
			break
		}
	}
	fmt.Fprintf(os.Stderr, "[O1-G7] %s median native/int8 %.4f, %d of 10 pairs below 1; the arms' greedy tokens agree for the first %d of %d\n",
		label, med, below, same, gen)
	return med, below
}

// TestKSO1_speed is O1-G7: kso1Speed on Phi-3 mini 4k at --quant q4k (the registered GOINFER_PHI3_GGUF asset), then on
// Qwen2.5-Coder-1.5B q4_k_m. SHIP iff Phi-3's median native/int8 is <= 0.98 with at least 9 of 10 pairs faster and the 1.5B's
// median is <= 1.00; PARK if Phi-3 is between 0.98 and 1.00 or has fewer than 9 pairs faster; KILL if Phi-3 or the 1.5B is above
// 1.00. A timed run: hold the timing lock (GOINFER_KSO1_TIMING=1).
func TestKSO1_speed(t *testing.T) {
	if os.Getenv("GOINFER_KSO1_TIMING") == "" {
		t.Skip("set GOINFER_KSO1_TIMING=1 (a timed run under the timing lock)")
	}
	pm, pb := kso1Speed(t, decoder.AssetPathForTest(t, "GOINFER_PHI3_GGUF"), "Phi-3 mini")
	home, _ := os.UserHomeDir()
	qm, qb := kso1Speed(t, filepath.Join(home, "models", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"), "Qwen2.5-Coder-1.5B")
	verdict := "PARK"
	switch {
	case pm > 1.00 || qm > 1.00:
		verdict = "KILL"
	case pm <= 0.98 && pb >= 9:
		verdict = "SHIP"
	}
	fmt.Fprintf(os.Stderr, "[O1-G7] Phi-3 median %.4f (%d of 10), 1.5B median %.4f (%d of 10): %s\n", pm, pb, qm, qb, verdict)
	if verdict != "SHIP" {
		t.Errorf("O1-G7 %s: Phi-3 median %.4f (%d of 10 pairs faster), the 1.5B's %.4f", verdict, pm, pb, qm)
	}
}
