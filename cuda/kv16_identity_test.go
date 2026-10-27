//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	gpu "github.com/townsendmerino/goinfer/cuda/internal/gpudev"
	"github.com/townsendmerino/goinfer/decoder"
)

// The exact-input identity of docs/tasks/task-cuda-f16-kv-2026-10.md: on K and V that are already f16-representable, a _kv16 attention reader
// widens each value exactly and folds in the f32 original's order, so its output must equal the f32 original's BIT FOR BIT. A bar of cosine 0.99
// could not tell a reader that is right from one that is slightly wrong; this can.

// f16BitsToF32 widens a binary16 pattern (exact).
func f16BitsToF32(h uint16) float32 {
	sign := float32(1)
	if h&0x8000 != 0 {
		sign = -1
	}
	e, m := int(h>>10)&0x1F, float32(h&0x3FF)
	switch e {
	case 0:
		return sign * m * float32(math.Ldexp(1, -24))
	case 0x1F:
		if m == 0 {
			return float32(math.Inf(int(sign)))
		}
		return float32(math.NaN())
	}
	return sign * (1 + m/1024) * float32(math.Ldexp(1, e-15))
}

// kv16Representable returns n pseudo-random values that survive the f16 round trip unchanged, spread over a few magnitudes so
// a reader that truncated or mis-widened them could not hide.
func kv16Representable(n int, seed uint32) []float32 {
	out := make([]float32, n)
	s := seed
	for i := range out {
		s = s*1664525 + 1013904223
		v := float32(int32(s>>8)%4000-2000) / 700 // about ±2.9, with f32 mantissa bits an f16 cannot hold
		if i%97 == 0 {
			v *= 40 // an outlier channel
		}
		out[i] = f16BitsToF32(f32ToF16RNE(v))
	}
	return out
}

// TestKV16_exactInputIdentity_decodeReaders drives every decode attention reader on the f32 resident and the f16 resident with the SAME
// f16-representable K/V (uploaded through UploadKV, which rounds the f16 arm's copy: exact on these values) and the same query, and requires
// the attention context to match bit for bit. The readers are called directly, so which one the dispatcher would pick does not matter.
func TestKV16_exactInputIdentity_decodeReaders(t *testing.T) {
	t.Setenv("GOINFER_CUDA_FLASH_DECODE", "16")
	t.Setenv("GOINFER_CUDA_FLASH_DECODE_MIN_KEYS", "0")
	for _, model := range []string{kv16Model, "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"} { // hd 64, hd 128
		t.Run(model, func(t *testing.T) {
			const ctx = 3072
			_, _, r32, r16 := kv16PairOf(t, model, ctx)
			Ly := &r32.layers[0]
			if !r16.layers[0].kvHalf || Ly.kvHalf {
				t.Fatalf("arms are not f32/f16: f32 arm kvHalf=%v, f16 arm kvHalf=%v", Ly.kvHalf, r16.layers[0].kvHalf)
			}
			if r32.skScores == (Pipeline{}) || r16.skScores == (Pipeline{}) || r32.faSplit == 0 || r16.faSplit == 0 || r32.bAttn == (Pipeline{}) {
				t.Fatalf("a reader did not load: skScores %v/%v faSplit %d/%d bAttn %v",
					r32.skScores != (Pipeline{}), r16.skScores != (Pipeline{}), r32.faSplit, r16.faSplit, r32.bAttn != (Pipeline{}))
			}
			if !r16.faEligible(0) {
				t.Fatal("the flash-decode lane is not eligible on layer 0")
			}
			nH, nKV, hd, kvDim := r32.nH, Ly.nKV, Ly.hd, Ly.kvDim
			q := kv16Representable(nH*hd, 7)

			type reader struct {
				name string
				run  func(r *cudaResident, l, pos int) error
			}
			readers := []reader{
				{"flash-decode partial+combine", func(r *cudaResident, l, pos int) error { return r.flashDecodeAttn(l, pos) }},
				{"split-KV scores+softmax+vsum", func(r *cudaResident, l, pos int) error { return r.splitKVAttnDecode(l, pos) }},
				{"attn_batched M=1", func(r *cudaResident, l, pos int) error {
					nWin := pos + 1
					return r.launch(r.bAttn, LaunchConfig{GridX: uint32(r.nH), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((nWin + 128) * 4)},
						Arg(r.qB), Arg(r.kvK(l)), Arg(r.kvV(l)), Arg(r.kvT(l)), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(r.layers[l].nKV)), gpu.ArgValue(int32(r.layers[l].hd)),
						gpu.ArgValue(int32(pos)), gpu.ArgValue(r.attnScale), gpu.ArgValue(r.layers[l].window), gpu.ArgValue(int32(1)), Arg(r.cctx), r.sinkArg(l))
				}},
				{"glue attention", func(r *cudaResident, l, pos int) error {
					nWin := pos + 1
					return r.launch(r.fAttn, LaunchConfig{GridX: uint32(r.nH), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((nWin + 128) * 4)},
						Arg(r.qB), Arg(r.kvK(l)), Arg(r.kvV(l)), Arg(r.kvT(l)), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(r.layers[l].nKV)), gpu.ArgValue(int32(r.layers[l].hd)),
						gpu.ArgValue(int32(nWin)), gpu.ArgValue(r.attnScale), gpu.ArgValue(r.layers[l].window), Arg(r.cctx))
				}},
			}
			for _, nKeys := range []int{1, 2, 37, 128, 129, 300, 1025, 3000} {
				keys := kv16Representable(nKeys*kvDim, uint32(11+nKeys))
				vals := kv16Representable(nKeys*kvDim, uint32(23+nKeys))
				for _, r := range []*cudaResident{r32, r16} {
					if err := r.UploadKV(0, 0, keys, vals); err != nil {
						t.Fatalf("UploadKV nKeys=%d: %v", nKeys, err)
					}
					if err := r.do(func() error { return gpu.Upload(r.qB, q) }); err != nil {
						t.Fatalf("upload q: %v", err)
					}
				}
				for _, rd := range readers {
					out := [2][]float32{}
					for i, r := range []*cudaResident{r32, r16} {
						out[i] = make([]float32, nH*hd)
						if err := r.do(func() error {
							if e := rd.run(r, 0, nKeys-1); e != nil {
								return e
							}
							if e := r.stream.Sync(); e != nil {
								return e
							}
							return gpu.Download(r.cctx, out[i])
						}); err != nil {
							t.Fatalf("%s nKeys=%d arm %d: %v", rd.name, nKeys, i, err)
						}
					}
					diff, nz := 0, 0
					for i := range out[0] {
						if math.Float32bits(out[0][i]) != math.Float32bits(out[1][i]) {
							diff++
						}
						if out[0][i] != 0 {
							nz++
						}
					}
					if nz == 0 {
						t.Errorf("%s nKeys=%d: the f32 arm's context is all zero, so the comparison is vacuous", rd.name, nKeys)
					}
					if diff != 0 {
						t.Errorf("%s nKeys=%d (nH=%d nKV=%d hd=%d): %d/%d context elements differ between the f16 reader and the f32 original on f16-representable K/V",
							rd.name, nKeys, nH, nKV, hd, diff, len(out[0]))
					}
				}
			}
		})
	}
}

// TestKV16_exactInputIdentity_prefillReaders is the same bar for the batched-prefill attention kernels: attn_batched, attn_img_batched and attn_span_batched (with an image
// block inside the rows) and every fused tile that serves the head dimension, over a fresh prompt and a warm cache. The fused kernels already round K/V to f16 in the f32 build, so on representable inputs
// the two builds' fused kernels must also agree to the bit.
func TestKV16_exactInputIdentity_prefillReaders(t *testing.T) {
	for _, model := range []string{kv16Model, "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"} {
		t.Run(model, func(t *testing.T) {
			const ctx = 1024
			_, _, r32, r16 := kv16PairOf(t, model, ctx)
			Ly := &r32.layers[0]
			nH, hd, kvDim, nKV := r32.nH, Ly.hd, Ly.kvDim, Ly.nKV
			type kernel struct {
				name     string
				p32, p16 Pipeline
				fused    bool
				bm, bn   int   // query-tile rows and key-tile rows; 0 for the 64x64 kernel
				img      bool  // attn_img_batched and attn_span_batched: take the image-block table after the sinks
				window   int32 // a sliding window to pass instead of the layer's, so a span reaches back past it; 0 keeps the layer's
			}
			var ks []kernel
			ks = append(ks, kernel{name: "attn_batched", p32: r32.bAttn, p16: r16.bAttn},
				kernel{name: "attn_img_batched", p32: r32.bAttnImg, p16: r16.bAttnImg, img: true},
				kernel{name: "attn_span_batched", p32: r32.bAttnSpan, p16: r16.bAttnSpan, img: true},
				// A 32-key window, shorter than every span below, so a span row's window reaches back to its span start: the branch
				// attn_span_batched has and attn_img_batched does not.
				kernel{name: "attn_span_batched window 32", p32: r32.bAttnSpan, p16: r16.bAttnSpan, img: true, window: 32},
				kernel{name: "attn_img_batched window 32", p32: r32.bAttnImg, p16: r16.bAttnImg, img: true, window: 32})
			if hd == 64 {
				ks = append(ks, kernel{name: "attn_fused hd64", p32: r32.bAttnFused64, p16: r16.bAttnFused64, fused: true},
					kernel{name: "attn_fused_bm32x64 hd64", p32: r32.bAttnBM32x64hd64, p16: r16.bAttnBM32x64hd64, fused: true, bm: 32, bn: 64},
					kernel{name: "attn_fused_bm32x32 hd64", p32: r32.bAttnBM32x32hd64, p16: r16.bAttnBM32x32hd64, fused: true, bm: 32, bn: 32},
					kernel{name: "attn_fused_bm128 hd64", p32: r32.bAttnBM128hd64, p16: r16.bAttnBM128hd64, fused: true, bm: 128, bn: 64})
			} else if hd == 128 {
				ks = append(ks, kernel{name: "attn_fused hd128", p32: r32.bAttnFused128, p16: r16.bAttnFused128, fused: true},
					kernel{name: "attn_fused_bm32x64 hd128", p32: r32.bAttnBM32x64hd128, p16: r16.bAttnBM32x64hd128, fused: true, bm: 32, bn: 64},
					kernel{name: "attn_fused_bm32x32 hd128", p32: r32.bAttnBM32x32hd128, p16: r16.bAttnBM32x32hd128, fused: true, bm: 32, bn: 32},
					kernel{name: "attn_fused_bm128 hd128", p32: r32.bAttnBM128hd128, p16: r16.bAttnBM128hd128, fused: true, bm: 128, bn: 64})
			} else {
				t.Fatalf("no prefill kernel table for hd=%d", hd)
			}
			for _, k := range ks {
				if k.p32 == (Pipeline{}) || k.p16 == (Pipeline{}) {
					t.Fatalf("%s did not load on both arms (f32 %v, f16 %v): a missing kernel is not a pass", k.name, k.p32 != (Pipeline{}), k.p16 != (Pipeline{}))
				}
			}
			for _, tc := range []struct{ startPos, M int }{{0, 70}, {0, 300}, {37, 200}, {100, 129}} {
				nTot := tc.startPos + tc.M
				keys := kv16Representable(nTot*kvDim, uint32(5+nTot))
				vals := kv16Representable(nTot*kvDim, uint32(9+nTot))
				q := kv16Representable(tc.M*nH*hd, uint32(13+tc.M))
				for _, r := range []*cudaResident{r32, r16} {
					if err := r.UploadKV(0, 0, keys, vals); err != nil {
						t.Fatal(err)
					}
				}
				f32Out := map[string][]float32{} // each kernel's f32 output, so the windowed span case can be shown to differ from the plain one
				for _, k := range ks {
					var out [2][]float32
					for i, r := range []*cudaResident{r32, r16} {
						p := k.p32
						if i == 1 {
							p = k.p16
						}
						out[i] = make([]float32, tc.M*nH*hd)
						if err := r.do(func() error {
							qB := gpu.NewBufferLenOf[float32](r.dev, len(q))
							cB := gpu.NewBufferLenOf[float32](r.dev, len(q))
							if e := gpu.Upload(qB, q); e != nil {
								return e
							}
							win := Ly.window
							if k.window != 0 {
								win = k.window
							}
							args := []gpu.KernelArg{Arg(qB), Arg(r.kvK(0)), Arg(r.kvV(0)), Arg(r.kvT(0)), gpu.ArgValue(int32(nH)), gpu.ArgValue(int32(nKV)),
								gpu.ArgValue(int32(hd)), gpu.ArgValue(int32(tc.startPos)), gpu.ArgValue(r.attnScale), gpu.ArgValue(win),
								gpu.ArgValue(int32(tc.M)), Arg(cB), r.sinkArg(0)}
							if k.img { // one image block over rows [startPos+10, startPos+M-20) of the M
								bB := gpu.NewBufferLenOf[int32](r.dev, 2)
								if e := gpu.Upload(bB, []int32{int32(tc.startPos + 10), int32(nTot - 20)}); e != nil {
									return e
								}
								args = append(args, Arg(bB), gpu.ArgValue(int32(1)))
							}
							cfg := LaunchConfig{GridX: uint32(nH), GridY: uint32(tc.M), GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((nTot + 128) * 4)}
							if k.fused {
								cfg = LaunchConfig{GridX: uint32(nH), GridY: uint32((tc.M + attnFusedBM - 1) / attnFusedBM), GridZ: 1,
									BlockX: attnFusedThreads, BlockY: 1, BlockZ: 1, SharedMemBytes: attnFusedShmem(hd)}
								if k.bm > 0 {
									cfg.GridY = uint32((tc.M + k.bm - 1) / k.bm)
									cfg.BlockX = uint32(k.bm / 16 * 32)
									cfg.SharedMemBytes = uint32(2 * (k.bn*(hd+attnFusedKPAD) + hd*(k.bn+attnFusedKPAD)))
								}
							}
							if e := r.launch(p, cfg, args...); e != nil {
								return e
							}
							if e := r.stream.Sync(); e != nil {
								return e
							}
							return gpu.Download(cB, out[i])
						}); err != nil {
							t.Fatalf("%s startPos=%d M=%d arm %d: %v", k.name, tc.startPos, tc.M, i, err)
						}
					}
					diff, nz := 0, 0
					for i := range out[0] {
						if math.Float32bits(out[0][i]) != math.Float32bits(out[1][i]) {
							diff++
						}
						if out[0][i] != 0 {
							nz++
						}
					}
					if nz == 0 {
						t.Errorf("%s startPos=%d M=%d: the f32 arm's output is all zero, so the comparison is vacuous", k.name, tc.startPos, tc.M)
					}
					if diff != 0 {
						t.Errorf("%s startPos=%d M=%d hd=%d: %d/%d elements differ between the f16 reader and the f32 original", k.name, tc.startPos, tc.M, hd, diff, len(out[0]))
					}
					f32Out[k.name] = out[0]
				}
				// The window must change what the span kernel computes, and the span kernel must differ from the image kernel under it
				// (its window reaches back to the span start, min(window start, span start)), or the windowed case is vacuous.
				if a, b := f32Out["attn_span_batched"], f32Out["attn_span_batched window 32"]; len(a) > 0 && slices.Equal(a, b) {
					t.Errorf("startPos=%d M=%d: attn_span_batched gives the same output with a 32-key window as with none: the window case is vacuous", tc.startPos, tc.M)
				}
				if a, b := f32Out["attn_img_batched window 32"], f32Out["attn_span_batched window 32"]; len(a) > 0 && slices.Equal(a, b) {
					t.Errorf("startPos=%d M=%d: under a 32-key window attn_span_batched equals attn_img_batched: its reach-back to the span start did not fire", tc.startPos, tc.M)
				}
			}
		})
	}
}

// TestKV16_pairwiseMRoPENeverF16 holds the invariant behind GLM-OCR's f32-only pairwise m-RoPE kernel (rope_kv_mrope_batched_pw writes f32
// rows): a resident built with an f16 cache for it, forced here past decoder.CUDAKVF16Decline, does not load that kernel, and its batched
// m-RoPE prefill declines. Without the force the same load takes the kernel, so the f16 arm's decline is the guard's doing.
func TestKV16_pairwiseMRoPENeverF16(t *testing.T) {
	requireCUDADevice(t)
	ckpt := filepath.Join("..", "testdata", "glm-ocr-tiny")
	if _, err := os.Stat(ckpt); err != nil {
		t.Skipf("no fixture: %v", err)
	}
	load := func(forceF16 bool) *cudaResident {
		if forceF16 {
			restore := decoder.SetResolveKVHookForTest(func(backend string, _ decoder.Options) (string, string, bool) {
				return "f16", "forced by the test", backend == "cuda"
			})
			defer restore()
		}
		m, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: "int8int8"})
		if err != nil {
			t.Fatalf("load (force f16 %v): %v", forceF16, err)
		}
		t.Cleanup(func() { m.Close() })
		r, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok {
			t.Fatalf("not CUDA-resident (force f16 %v): %s", forceF16, m.ResidentDecline())
		}
		if !r.pairwiseRoPE {
			t.Fatal("glm-ocr-tiny does not rotate pairwise: this test checks nothing")
		}
		return r
	}
	r32 := load(false)
	if r32.kvF16 || !r32.mropePrefillReady {
		t.Fatalf("f32 control: kvF16 %v, m-RoPE prefill ready %v; want the pairwise m-RoPE kernel loaded", r32.kvF16, r32.mropePrefillReady)
	}
	r16 := load(true)
	if !r16.kvF16 {
		t.Fatal("the forced load did not build an f16 cache: the guard went unexercised")
	}
	if r16.mropePrefillReady || r16.bRopeKVMRoPE != (Pipeline{}) {
		t.Fatal("an f16 cache loaded the f32-only pairwise m-RoPE kernel")
	}
	row := [][]float32{make([]float32, r16.hidden)}
	if _, err := r32.PrefillMRoPELast(context.Background(), row, 0, [][3]int{{0, 0, 0}}); err != nil { // the same call runs on the f32 cache
		t.Fatalf("PrefillMRoPELast on the f32 control: %v", err)
	}
	if _, err := r16.PrefillMRoPELast(context.Background(), row, 0, [][3]int{{0, 0, 0}}); !errors.Is(err, errPrefillDeclined) {
		t.Fatalf("PrefillMRoPELast on the f16 cache: %v, want it declined", err)
	}
}
