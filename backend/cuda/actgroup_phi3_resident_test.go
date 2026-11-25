//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	gc "github.com/eitamring/gocudrv/cuda"
	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/testgate"
	"github.com/townsendmerino/goinfer/tokenizer"
)

func cosine64(a, b []float32) float64 {
	var ab, aa, bb float64
	for i := range a {
		ab += float64(a[i]) * float64(b[i])
		aa += float64(a[i]) * float64(a[i])
		bb += float64(b[i]) * float64(b[i])
	}
	return ab / math.Sqrt(aa*bb)
}

// TestActGroup_phi3ResidentMatchesCPU: Phi-3-mini resident on CUDA at int8int8 with per-32 activation
// scales (queue-engineering.md H2's configuration that passed the quality gate) — admitted despite
// the activation-quantization hazard, it runs the actgroup.cu kernels; its logits over a ~130-token
// chat prompt (filler, the regime that broke per-vector scales) match the CPU per-32 path to kernel
// accumulation order, and hold the gate's quality bar against goinfer's own f32.
//
//	GOINFER_PHI3_MINI_GGUF=~/models/phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf \
//	  go test -tags 'cuda goinfer_testhooks' ./cuda -run TestActGroup_phi3Resident -v
func TestActGroup_phi3ResidentMatchesCPU(t *testing.T) {
	path := os.Getenv("GOINFER_PHI3_MINI_GGUF")
	if path == "" {
		t.Skip("set GOINFER_PHI3_MINI_GGUF")
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatal(err)
	}
	tmpl, err := chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
	if err != nil {
		t.Fatal(err)
	}
	text := strings.Repeat("the ", 110) + "\n\nThe text above is only filler. Now write a long, detailed essay about the history of the printing press."
	ids, err := tk.EncodeSegments(tmpl.RenderSegments("", []chat.Turn{{Role: "user", Content: text}}), false)
	if err != nil {
		t.Fatal(err)
	}

	// ResidentContext 512: 4096 positions of f32 KV plus the int8 weights exceed an 8 GB card.
	gpu, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "int8int8", ActQuantGroup: 32, ResidentContext: 512})
	if err != nil {
		t.Fatalf("cuda load: %v", err)
	}
	defer gpu.Close()
	rf := gpu.ResidentForwardForTest()
	if rf == nil {
		t.Fatalf("resident did not engage: %s", gpu.ResidentDecline())
	}
	if r, ok := rf.(*cudaResident); !ok || !r.actG32 {
		t.Fatal("resident engaged without the per-32 kernels")
	}
	var gl [][]float32
	for p, id := range ids {
		lg, err := rf.Forward(gpu.EmbedResidentForTest(id), p)
		if err != nil {
			t.Fatalf("resident forward pos %d: %v", p, err)
		}
		gl = append(gl, slices.Clone(lg))
	}

	cpuLogits := func(opts decoder.Options) [][]float32 {
		m, err := decoder.Load(path, opts)
		if err != nil {
			t.Fatalf("cpu load %+v: %v", opts, err)
		}
		defer m.Close()
		c := m.NewCache(len(ids) + 1)
		out := make([][]float32, len(ids))
		for p, id := range ids {
			lg, err := m.ForwardForTest(id, c)
			if err != nil {
				t.Fatal(err)
			}
			out[p] = slices.Clone(lg)
		}
		return out
	}
	stats := func(name string, ref [][]float32) (p10, median float64) {
		cs := make([]float64, len(ids))
		for p := range ids {
			cs[p] = cosine64(gl[p], ref[p])
		}
		slices.Sort(cs)
		t.Logf("%s: %d positions, cosine min %.6f p10 %.6f median %.6f", name, len(ids), cs[0], cs[len(cs)/10], cs[len(cs)/2])
		return cs[len(cs)/10], cs[len(cs)/2]
	}
	// The kernels are responsible for matching the CPU per-32 path, not for the prompt's own difficulty:
	// on this prompt the CPU per-32 path itself sits below the gate's quality on the gate's own prompt. So
	// the bars are relative: median agreement with CPU per-32, and quality no worse than CPU per-32's on
	// the same prompt. Figures: docs/code-notes/cuda.md#TestActGroup_phi3ResidentMatchesCPU.bars.
	cpuG, f32 := cpuLogits(decoder.Options{Quant: "int8int8", ActQuantGroup: 32}), cpuLogits(decoder.Options{})
	_, agreeMed := stats("cuda per-32 vs cpu per-32", cpuG)
	cudaQ, _ := stats("cuda per-32 vs cpu f32", f32)
	gl = cpuG
	cpuQ, _ := stats("cpu per-32 vs cpu f32 (same prompt, the baseline)", f32)
	if agreeMed < 0.9995 {
		t.Errorf("CUDA per-32 vs CPU per-32: median cosine %.6f, want >= 0.9995 (same algorithm; differs only by kernel rounding)", agreeMed)
	}
	if cudaQ < cpuQ-0.01 {
		t.Errorf("CUDA per-32 quality p10 %.4f is worse than the CPU per-32 path's %.4f on the same prompt", cudaQ, cpuQ)
	}
}

// TestActCfg_spreadsOnlyPer32 pins actCfg's grid: a per-32 activation kernel gets a block for every blockDim/32 groups (Phi-3's
// shapes: the 3,072-wide norm and quant on 12 blocks of 256, the 8,192-wide GLU on 8 of 1,024), a per-vector one keeps one block
// (its absmax spans the vector). Bit-identity at every grid is the kernels' own test
// (kernels/actgroup_kernel_test.go:TestActGroupKernels_warpQuantizersEqualSerial).
func TestActCfg_spreadsOnlyPer32(t *testing.T) {
	for _, c := range []struct {
		g32            bool
		block, n, want int
	}{
		{true, 256, 3072, 12}, {true, glueQuantThreads, 8192, 8}, {true, 256, 64, 1}, {true, 256, 288, 2},
		{false, 256, 3072, 1}, {false, glueQuantThreads, 8192, 1},
	} {
		r := &cudaResident{actG32: c.g32}
		if got := r.actCfg(c.block, 4, c.n).GridX; int(got) != c.want {
			t.Errorf("actCfg(block %d, n %d) with actG32 %v: grid %d, want %d", c.block, c.n, c.g32, got, c.want)
		}
	}
}

// prefillG32Rig is phi3-tiny resident at int8int8 with per-32 activations: every per-32 decode kernel and prefill_g32.cu's
// batched kernels are bound, so kernel-level checks can run on synthetic shapes. It skips without a device.
func prefillG32Rig(t *testing.T) (*decoder.Model, *cudaResident) {
	t.Helper()
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	m, err := decoder.Load("../../testdata/phi3-tiny", decoder.Options{Backend: "cuda", Quant: "int8int8", ActQuantGroup: 32})
	if err != nil {
		t.Fatalf("load phi3-tiny: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	rf, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok || !rf.actG32 {
		t.Fatalf("phi3-tiny at int8int8/per-32 is not a per-32 CUDA resident (resident %T)", m.ResidentForwardForTest())
	}
	return m, rf
}

// g32TestWeight builds an N x K weight of the given kind in the layout the resident uploads: Q4_K's 144-byte and Q5_K's
// 176-byte blocks with sane f16 d/dmin, Q6_K in q6kPlanes' planes with sane f16 d, int8 words with f32 row scales.
func g32TestWeight(rf *cudaResident, rng *rand.Rand, kind string, N, K int) cudaWQ {
	nSB := K / 256
	half := func() uint32 { return uint32(0x1400 + rng.Intn(0x0c00)) } // positive f16 in about [0.001, 0.03]
	words := func(n int) []uint32 {
		w := make([]uint32, n)
		for i := range w {
			w[i] = rng.Uint32()
		}
		return w
	}
	wt := cudaWQ{kind: kind, N: N, K: K}
	switch kind {
	case "q4k", "q5k":
		per := map[string]int{"q4k": 36, "q5k": 44}[kind]
		w := words(N * nSB * per)
		for b := 0; b < N*nSB; b++ {
			w[b*per] = half() | half()<<16
		}
		wt.W = rf.tupu32(w)
	case "q6k":
		T := N * nSB
		w := words((210*T + 3) / 4)
		for g := range T {
			byteOff := 208*T + 2*g
			sh := uint(byteOff%4) * 8
			w[byteOff/4] = w[byteOff/4]&^(0xffff<<sh) | half()<<sh
		}
		wt.W = rf.tupu32(w)
	case "int8":
		wt.W = rf.tupu32(words(N * K / 4))
		ws := make([]float32, N)
		for i := range ws {
			ws[i] = float32(rng.Float64()*0.02 + 0.001)
		}
		wt.ws = rf.tup32(ws)
	}
	return wt
}

func g32Floats(rng *rand.Rand, n int, scale float64) []float32 {
	f := make([]float32, n)
	for i := range f {
		f[i] = float32(rng.NormFloat64() * scale)
		if rng.Intn(97) == 0 {
			f[i] *= 40 // Phi-3-like outliers, so groups' scales differ widely
		}
	}
	return f
}

func sameBits32(a, b []float32) int {
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return i
		}
	}
	return -1
}

// TestPrefillG32_rowsMatchDecode: each per-32 rows GEMV (prefill_g32.cu) equals its decode kernel (doG, one launch per activation
// row) bit for bit, for Q4_K, Q5_K, Q6_K and int8, at M = 1..33 (inside and across the MT tile), N with a ragged last warp and
// block, K from one super-block to the FFN width, with and without a bias, writing and accumulating. The batched quantizer
// produces both arms' activations, so a quantizer difference cannot hide here; TestPrefillG32_quantizersMatchDecode checks it.
func TestPrefillG32_rowsMatchDecode(t *testing.T) {
	_, rf := prefillG32Rig(t)
	rng := rand.New(rand.NewSource(7))
	const maxM = 33
	cases := 0
	for _, K := range []int{256, 768, 3072, 8192} {
		x := g32Floats(rng, maxM*K, 1)
		for _, N := range []int{1, 15, 1001} {
			bias := g32Floats(rng, N, 0.5)
			init := g32Floats(rng, maxM*N, 1)
			for _, kind := range []string{"q4k", "q5k", "q6k", "int8"} {
				err := rf.do(func() error {
					wt := g32TestWeight(rf, rng, kind, N, K)
					xb, aq, as := rf.tup32(x), rf.ti(maxM*K/4), rf.tf(maxM*rf.actScaleLen(K))
					if e := rf.launch(rf.bQuant, LaunchConfig{GridX: maxM, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
						Arg(xb), gpu.ArgValue(int32(K)), Arg(aq), Arg(as), gpu.ArgValue(int32(maxM))); e != nil {
						return e
					}
					bb := rf.tup32(bias)
					for _, M := range []int{1, 2, 7, 8, 9, 17, 33} {
						for _, withBias := range []bool{false, true} {
							for _, accum := range []int32{0, 1} {
								ba := ArgNull()
								if withBias {
									ba = Arg(bb)
								}
								want, got := rf.tup32(init[:M*N]), rf.tup32(init[:M*N])
								for m := range M {
									if e := rf.doG(wt, aq.At(m*K), as.At(m*rf.actScaleLen(K)*4), ba, want.At(m*N*4), accum); e != nil {
										return e
									}
								}
								if e := rf.bGemvRowsG32(wt, aq, as, ba, got, M, accum); e != nil {
									return e
								}
								if e := rf.stream.Sync(); e != nil {
									return e
								}
								w, g := make([]float32, M*N), make([]float32, M*N)
								if e := gpu.Download(want, w); e != nil {
									return e
								}
								if e := gpu.Download(got, g); e != nil {
									return e
								}
								if i := sameBits32(w, g); i >= 0 {
									t.Errorf("%s K=%d N=%d M=%d bias=%v accum=%d: element (m=%d, n=%d) decode %v, rows kernel %v",
										kind, K, N, M, withBias, accum, i/N, i%N, w[i], g[i])
								}
								cases++
							}
						}
					}
					return nil
				})
				if err != nil {
					t.Fatalf("%s K=%d N=%d: %v", kind, K, N, err)
				}
			}
		}
	}
	t.Logf("%d cases, every output bit-equal to the decode kernel's", cases)
}

// TestPrefillG32_quantizersMatchDecode: the batched per-32 quantizers bound into bRms, bQuant and bSw equal the decode ones (fRms,
// fQ, fSw at their decode launch shapes), row by row, in the int8 codes and in both halves of the scale row (the group scales and
// the group sums gemv_q4k_g32 reads). The rmsnorm runs with and without (1 + w); both glu activations run, and the SiLU glu
// also at a nonzero uOff (the up rows read from uOff on, row stride I, as decode reads u[uOff + k]). bSw's product overwrites
// its gate rows, so each arm gets a copy.
func TestPrefillG32_quantizersMatchDecode(t *testing.T) {
	_, rf := prefillG32Rig(t)
	rng := rand.New(rand.NewSource(11))
	const M = 9
	for _, K := range []int{64, 3072, 8192} {
		const uOffTest = 64
		x, w, u := g32Floats(rng, M*K, 2), g32Floats(rng, K, 1), g32Floats(rng, M*K+uOffTest, 1)
		sl := rf.actScaleLen(K)
		for _, op := range []string{"rms", "rms-addone", "quant", "glu-silu", "glu-gelu", "glu-uoff"} {
			var wq, gq []int32
			var ws, gs []float32
			err := rf.do(func() error {
				qd, sd, qb, sb := rf.ti(M*K/4), rf.tf(M*sl), rf.ti(M*K/4), rf.tf(M*sl)
				xb, wb, ub := rf.tup32(x), rf.tup32(w), rf.tup32(u)
				act, addOne, uOff := int32(1), int32(0), int32(0)
				switch op {
				case "glu-gelu":
					act = 0
				case "rms-addone":
					addOne = 1
				case "glu-uoff":
					uOff = uOffTest
				}
				scratch := rf.tf(K)
				for m := range M {
					xm, qm, sm := xb.At(m*K*4), qd.At(m*K), sd.At(m*sl*4)
					var e error
					switch op {
					case "rms", "rms-addone":
						e = rf.launch(rf.fRms, onecfg(256, (K+256)*4), Arg(xm), Arg(wb), gpu.ArgValue(int32(K)), gpu.ArgValue(float32(1e-5)), gpu.ArgValue(addOne), Arg(qm), Arg(sm))
					case "quant":
						e = rf.launch(rf.fQ, onecfg(256, 256*4), Arg(xm), gpu.ArgValue(int32(K)), Arg(qm), Arg(sm))
					default:
						e = rf.launch(rf.fSw, onecfg(glueQuantThreads, glueQuantThreads*4), Arg(xm), Arg(ub.At(m*K*4)), gpu.ArgValue(int32(0)), gpu.ArgValue(uOff),
							gpu.ArgValue(int32(K)), gpu.ArgValue(act), Arg(qm), Arg(sm), Arg(scratch))
					}
					if e != nil {
						return e
					}
				}
				var e error
				cfg := LaunchConfig{GridX: M, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4}
				switch op {
				case "rms", "rms-addone":
					e = rf.launch(rf.bRms, LaunchConfig{GridX: M, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((256 + K) * 4)},
						Arg(xb), Arg(wb), gpu.ArgValue(int32(K)), gpu.ArgValue(float32(1e-5)), gpu.ArgValue(addOne), Arg(qb), Arg(sb))
				case "quant":
					e = rf.launch(rf.bQuant, cfg, Arg(xb), gpu.ArgValue(int32(K)), Arg(qb), Arg(sb), gpu.ArgValue(int32(M)))
				default:
					gb := rf.tup32(x) // bSw writes its product over the gate rows
					e = rf.launch(rf.bSw, cfg, Arg(gb), Arg(ub), gpu.ArgValue(uOff), gpu.ArgValue(int32(K)), gpu.ArgValue(act), Arg(qb), Arg(sb), gpu.ArgValue(int32(M)))
				}
				if e != nil {
					return e
				}
				if e := rf.stream.Sync(); e != nil {
					return e
				}
				wq, gq, ws, gs = make([]int32, M*K/4), make([]int32, M*K/4), make([]float32, M*sl), make([]float32, M*sl)
				for _, d := range []error{gpu.Download(qd, wq), gpu.Download(qb, gq), gpu.Download(sd, ws), gpu.Download(sb, gs)} {
					if d != nil {
						return d
					}
				}
				return nil
			})
			if err != nil {
				t.Fatalf("%s K=%d: %v", op, K, err)
			}
			for i := range wq {
				if wq[i] != gq[i] {
					t.Fatalf("%s K=%d: code word %d (row %d) decode %#x, batched %#x", op, K, i, i/(K/4), wq[i], gq[i])
				}
			}
			if i := sameBits32(ws, gs); i >= 0 {
				t.Fatalf("%s K=%d: scale %d (row %d, slot %d of %d) decode %v, batched %v", op, K, i, i/sl, i%sl, sl, ws[i], gs[i])
			}
		}
	}
}

// TestPrefillG32_tinyPassMatchesSequential: a per-32 load is admitted to the batched prefill (PrefillPath reads batched), and its
// pass is bit-identical to the sequential path it replaces: PrefillLast's logits equal the last Forward of a per-token loop, and
// four greedy decode steps after each (which read the KV the prefill wrote) equal each other, and HiddenLast equals the per-32
// dequant of the hidden state decode quantized for its head, at prompt lengths inside one MT tile, across tiles and across
// several passes (GOINFER_PREFILL_CHUNK 16).
func TestPrefillG32_tinyPassMatchesSequential(t *testing.T) {
	m, rf := prefillG32Rig(t)
	if ok, why := rf.PrefillPath(); !ok || !strings.HasPrefix(why, "batched") {
		t.Fatalf("PrefillPath on a per-32 load: %v, %q; want batched", ok, why)
	}
	_, _, _, _, _, _, vocab := m.Dims()
	embs := make([][]float32, 300)
	for i := range embs {
		embs[i] = slices.Clone(m.EmbedResidentForTest((i*37 + 11) % vocab))
	}
	argmax := func(l []float32) int {
		b := 0
		for i, v := range l {
			if v > l[b] {
				b = i
			}
		}
		return b
	}
	// Forward's logits slice is reused by the next call, so every step is copied.
	tail := func(seed []float32, n int) [][]float32 {
		out := [][]float32{slices.Clone(seed)}
		for s := range 4 {
			l, e := rf.Forward(m.EmbedResidentForTest(argmax(out[len(out)-1])), n+s)
			if e != nil {
				t.Fatalf("decode step %d: %v", s, e)
			}
			out = append(out, slices.Clone(l))
		}
		return out
	}
	for _, chunk := range []string{"", "16"} {
		if chunk != "" {
			defer decoder.SetKnobForTest(m, "GOINFER_PREFILL_CHUNK", chunk)()
		}
		for _, n := range []int{1, 5, 8, 9, 33, 70, 300} {
			rf.Reset()
			for i := 0; i < n-1; i++ {
				if e := rf.ForwardNoLogits(embs[i], i); e != nil {
					t.Fatalf("n=%d seq pos %d: %v", n, i, e)
				}
			}
			s, e := rf.Forward(embs[n-1], n-1)
			if e != nil {
				t.Fatalf("n=%d seq last: %v", n, e)
			}
			// The last Forward's final norm left the quantized hidden state in r.aq/r.aSc: HiddenLast must return its per-32 dequant.
			hq, hs := make([]int8, rf.hidden), make([]float32, rf.actScaleLen(rf.hidden))
			if e := rf.do(func() error {
				if e := gpu.Download(rf.aq, hq); e != nil {
					return e
				}
				return gpu.Download(rf.aSc, hs)
			}); e != nil {
				t.Fatal(e)
			}
			wantH := make([]float32, rf.hidden)
			for i := range wantH {
				wantH[i] = float32(hq[i]) * hs[i/32]
			}
			seq := tail(s, n)
			rf.Reset()
			b, e := rf.PrefillLast(context.Background(), embs[:n], 0)
			if e != nil {
				t.Fatalf("n=%d chunk=%q PrefillLast: %v", n, chunk, e)
			}
			bat := tail(b, n)
			for step := range seq {
				if i := sameBits32(seq[step], bat[step]); i >= 0 {
					t.Fatalf("n=%d chunk=%q step %d: logit %d sequential %v, batched %v", n, chunk, step, i, seq[step][i], bat[step][i])
				}
			}
			rf.Reset()
			h, e := rf.HiddenLast(context.Background(), embs[:n], 0)
			if e != nil {
				t.Fatalf("n=%d chunk=%q HiddenLast: %v", n, chunk, e)
			}
			if i := sameBits32(wantH, h); i >= 0 {
				t.Fatalf("n=%d chunk=%q HiddenLast[%d] = %v, the decode's quantized hidden dequantizes to %v", n, chunk, i, h[i], wantH[i])
			}
		}
	}
}

// op6cLoad loads OP6-C's registered model (docs/measurements/op6c-q4k-prefill-2026-11-23/README.md) at its default CUDA load,
// --quant q4k with serve's f16 KV, and the first n prompts of prefill-gate set A at k tokens. GOINFER_OP6C_MODEL overrides the
// path (a Qwen2.5 q4_k_m checks the lane on a second family; the registered cells are Phi-3's).
func op6cLoad(t *testing.T, k, n int) (*decoder.Model, *cudaResident, [][]int, string) {
	t.Helper()
	testgate.RequireHeavy(t)
	path := os.Getenv("GOINFER_OP6C_MODEL")
	if path == "" {
		path = modelPath("phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no model at %s", path)
	}
	was := kvF16Auto.Swap(true) // serve's auto KV (f16 for phi3), not the suite's f32
	t.Cleanup(func() { kvF16Auto.Store(was) })
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: "q4k", ResidentContext: k + 64})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	rf, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok || !rf.actG32 {
		t.Fatalf("%s at q4k is not a per-32 CUDA resident", path)
	}
	if ok, why := rf.PrefillPath(); !ok {
		t.Fatalf("PrefillPath: %s", why)
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	var prompts [][]int
	for _, f := range decoder.PrefillGatePromptSetFor("a")[:n] {
		prompts = append(prompts, decoder.PrefillGateProseIDsForTest(t, tk, f, k)[:k])
	}
	return m, rf, prompts, path
}

// op6cSeq is the sequential arm: one decode forward per prompt token, the route main takes for this load. It returns the seed logits.
func op6cSeq(t *testing.T, rf *cudaResident, embs [][]float32) []float32 {
	t.Helper()
	rf.Reset()
	for i := 0; i < len(embs)-1; i++ {
		if e := rf.ForwardNoLogits(embs[i], i); e != nil {
			t.Fatalf("sequential pos %d: %v", i, e)
		}
	}
	l, e := rf.Forward(embs[len(embs)-1], len(embs)-1)
	if e != nil {
		t.Fatalf("sequential last: %v", e)
	}
	return slices.Clone(l)
}

// op6cBatched is the batched arm: PrefillLast, which must run the pass (the load's PrefillPath reads batched).
func op6cBatched(t *testing.T, rf *cudaResident, embs [][]float32) []float32 {
	t.Helper()
	rf.Reset()
	l, e := rf.PrefillLast(context.Background(), embs, 0)
	if e != nil {
		t.Fatalf("PrefillLast: %v", e)
	}
	return l
}

// TestOP6C_q4kPassBitIdentity is OP6-C's gate 2: on Phi-3 mini at q4k, set A's first three prompts at K = 128 and 3,900, the
// batched arm's seed logits and 16 greedy decode steps (each step's full logits, reading the KV the prefill wrote) against the
// sequential arm's. PASS is every value bit-equal.
//
//	GOINFER_HEAVY_TESTS=1 go test -tags 'cuda goinfer_testhooks' -run TestOP6C_q4kPassBitIdentity -v -timeout 30m .
func TestOP6C_q4kPassBitIdentity(t *testing.T) {
	ks := []int{128, 3900}
	m, rf, prompts, path := op6cLoad(t, ks[len(ks)-1], 3)
	t0 := time.Now()
	argmax := func(l []float32) int {
		b := 0
		for i, v := range l {
			if v > l[b] {
				b = i
			}
		}
		return b
	}
	steps := func(seed []float32, K int) [][]float32 {
		out := [][]float32{slices.Clone(seed)}
		for s := range 16 {
			l, e := rf.Forward(m.EmbedResidentForTest(argmax(out[len(out)-1])), K+s)
			if e != nil {
				t.Fatalf("decode step %d: %v", s, e)
			}
			out = append(out, slices.Clone(l))
		}
		return out
	}
	values, fails := 0, 0
	for _, K := range ks {
		for pi, ids := range prompts {
			embs := make([][]float32, K)
			for i, id := range ids[:K] {
				embs[i] = m.EmbedResidentForTest(id)
			}
			fmt.Fprintf(os.Stderr, "[op6c gate2] K=%d prompt %d/%d: sequential arm (elapsed %s)\n", K, pi+1, len(prompts), time.Since(t0).Round(time.Second))
			a := steps(op6cSeq(t, rf, embs), K)
			fmt.Fprintf(os.Stderr, "[op6c gate2] K=%d prompt %d/%d: batched arm (elapsed %s)\n", K, pi+1, len(prompts), time.Since(t0).Round(time.Second))
			b := steps(op6cBatched(t, rf, embs), K)
			for s := range a {
				values += len(a[s])
				if i := sameBits32(a[s], b[s]); i >= 0 {
					fails++
					t.Errorf("K=%d prompt %d step %d: logit %d sequential %v, batched %v (argmax %d vs %d)", K, pi, s, i, a[s][i], b[s][i], argmax(a[s]), argmax(b[s]))
				}
			}
			fmt.Fprintf(os.Stderr, "OP6C_IDENTITY model=%s K=%d prompt=%d steps=%d differing_steps_so_far=%d\n", filepath.Base(path), K, pi, len(a), fails)
		}
	}
	t.Logf("OP6C gate 2: %d logits compared over %d prompts x %v depths x 17 rows; %d rows differ (elapsed %s)", values, len(prompts), ks, fails, time.Since(t0).Round(time.Second))
}

// op6cHostOK is the void rule's reading at a cell's start (docs/measurements/noise-registry.md §8): the 1-minute load average
// and MemAvailable. A cell is void when the load is above half nobara's 16 cores or available memory is under the load's fit
// need plus 2 GB; the fit need is taken as the checkpoint's size, all a resident Phi-3 maps on the host.
func op6cHostOK(needBytes int64) (load float64, availMB int64, ok bool) {
	if b, err := os.ReadFile("/proc/loadavg"); err == nil {
		fmt.Sscanf(string(b), "%f", &load)
	}
	if b, err := os.ReadFile("/proc/meminfo"); err == nil {
		for ln := range strings.SplitSeq(string(b), "\n") {
			if strings.HasPrefix(ln, "MemAvailable:") {
				var kb int64
				fmt.Sscanf(strings.TrimPrefix(ln, "MemAvailable:"), "%d", &kb)
				availMB = kb / 1024
			}
		}
	}
	return load, availMB, load <= 8.0 && availMB*(1<<20) >= needBytes+2<<30
}

// TestOP6C_q4kPassSpeed is OP6-C's gate 3, an instrument: it logs every cell and pair, the median and the pairs above 1, and
// asserts nothing about speed; the registration's ADOPT / KILL rule grades the log. It times the sequential arm (A) against the
// batched arm (B) on one load, P = 128 and 3,900 from position 0 on set A's first prompt: per P one untimed call of each arm, then
// 10 pairs alternating A B / B A (5 A B B A blocks). A cell started under heavy load is void and its pair re-run in order, at most
// 4 a P. The metric is the median per-pair A / B. Run under the exclusive timing lock.
//
//	GOINFER_HEAVY_TESTS=1 python3 scripts/timing_lock.py run --label op6c-gate3 -- go test -tags 'cuda goinfer_testhooks' -run TestOP6C_q4kPassSpeed -v -timeout 40m ./backend/cuda/
func TestOP6C_q4kPassSpeed(t *testing.T) {
	ps := []int{128, 3900}
	m, rf, prompts, path := op6cLoad(t, ps[len(ps)-1], 1)
	fi, _ := os.Stat(path)
	need := fi.Size()
	t.Logf("OP6C pass scratch at %d rows: %.1f MiB; prefill path: %s", rf.prefillPassRows(), float64(rf.PrefillScratchBytes(rf.prefillPassRows()))/(1<<20),
		func() string { _, w := rf.PrefillPath(); return w }())
	t0 := time.Now()
	ratios := map[int][]float64{}
	for _, P := range ps {
		embs := make([][]float32, P)
		for i, id := range prompts[0][:P] {
			embs[i] = m.EmbedResidentForTest(id)
		}
		arm := func(name string) (time.Duration, bool) {
			load, avail, ok := op6cHostOK(need)
			s := time.Now()
			if name == "A" {
				op6cSeq(t, rf, embs)
			} else {
				op6cBatched(t, rf, embs)
			}
			d := time.Since(s)
			fmt.Fprintf(os.Stderr, "OP6C_CELL P=%d arm=%s ms=%.1f load1=%.2f availMB=%d void=%v (elapsed %s)\n", P, name, float64(d.Microseconds())/1e3, load, avail, !ok, time.Since(t0).Round(time.Second))
			return d, ok
		}
		arm("A")
		arm("B")
		voids := 0
		for i := 0; i < 10; {
			order := []string{"A", "B"}
			if i%2 == 1 {
				order = []string{"B", "A"}
			}
			d := map[string]time.Duration{}
			good := true
			for _, a := range order {
				var ok bool
				d[a], ok = arm(a)
				good = good && ok
			}
			if !good {
				voids++
				fmt.Fprintf(os.Stderr, "OP6C_VOID P=%d pair=%d (re-run %d of 4)\n", P, i, voids)
				if voids > 4 {
					t.Fatalf("P=%d: more than 4 void pairs; the registration repeats the run", P)
				}
				continue
			}
			r := float64(d["A"]) / float64(d["B"])
			ratios[P] = append(ratios[P], r)
			fmt.Fprintf(os.Stderr, "OP6C_PAIR P=%d pair=%d order=%s A_ms=%.1f B_ms=%.1f ratio=%.3f\n", P, i, strings.Join(order, ""), float64(d["A"].Microseconds())/1e3, float64(d["B"].Microseconds())/1e3, r)
			i++
		}
	}
	for _, P := range ps {
		r := slices.Clone(ratios[P])
		slices.Sort(r)
		above := 0
		for _, v := range r {
			if v > 1 {
				above++
			}
		}
		med := (r[(len(r)-1)/2] + r[len(r)/2]) / 2
		t.Logf("OP6C_SUMMARY P=%d pairs=%d median_A_over_B=%.3f pairs_above_1=%d ratios=%v", P, len(r), med, above, ratios[P])
	}
}
