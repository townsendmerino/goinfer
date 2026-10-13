//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"math"
	"os"
	"strings"
	"testing"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// The f16 KV cache (docs/tasks/task-cuda-f16-kv-2026-10.md): CUDA's resident K/V as __half behind the _kv16 kernel twins.

const kv16Model = "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"

// kv16Pair loads the same checkpoint twice, once per cache precision, and returns the two residents. Both arms run the same binary, the same
// weights and the same context; the only difference is the cache.
func kv16Pair(t *testing.T, ctx int) (m32, m16 *decoder.Model, r32, r16 *cudaResident) {
	return kv16PairOf(t, kv16Model, ctx)
}

func kv16PairOf(t *testing.T, model string, ctx int) (m32, m16 *decoder.Model, r32, r16 *cudaResident) {
	t.Helper()
	requireCUDADevice(t)
	path := modelPath(model)
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s", path)
	}
	load := func(kv string) (*decoder.Model, *cudaResident) {
		m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: kv16Quant(model), ActQuantGroup: kv16ActGroup(model), KVPrecision: kv, ResidentContext: ctx})
		if err != nil {
			t.Fatalf("load (-kv %s): %v", kv, err)
		}
		r, ok := m.ResidentForwardForTest().(*cudaResident)
		if !ok {
			m.Close()
			t.Fatalf("-kv %s: no CUDA resident: %s", kv, m.ResidentDecline())
		}
		return m, r
	}
	m32, r32 = load("f32")
	m16, r16 = load("f16")
	t.Cleanup(func() { m32.Close(); m16.Close() })
	return
}

// kv16Quant is the weight quant the served bench loads a model at: int4, and int8int8 for Phi-3 (its activation outliers decline the int4 resident).
func kv16Quant(model string) string {
	if strings.Contains(model, "hi3") {
		return "int8int8"
	}
	return "int4"
}

// kv16ActGroup is the activation-scale group the served bench's Phi-3 load uses (per-32), 0 for everything else.
func kv16ActGroup(model string) int {
	if strings.Contains(model, "hi3") {
		return 32
	}
	return 0
}

func kv16Prompt(m *decoder.Model, n int) [][]float32 {
	embs := make([][]float32, n)
	for i := range embs {
		embs[i] = m.EmbedResidentForTest(1000 + 37*i%5000)
	}
	return embs
}

// TestKV16_cacheIsHalfAndTheKnobIsHonoured: -kv f16 on CUDA now does what it says. The f16 arm's per-layer buffers are half the f32 arm's,
// the precision it reports is f16, decoder.Plan and the build price the same bytes, and -kv f32 stays f32.
func TestKV16_cacheIsHalfAndTheKnobIsHonoured(t *testing.T) {
	m32, m16, r32, r16 := kv16Pair(t, 4096)
	if got := m32.ResidentKVPrecision(); got != "f32" {
		t.Errorf("-kv f32 reports %q", got)
	}
	if got := m16.ResidentKVPrecision(); got != "f16" {
		t.Errorf("-kv f16 reports %q: the knob is still ignored", got)
	}
	if r32.kvF16 || !r16.kvF16 {
		t.Fatalf("resident flags: f32 arm kvF16=%v, f16 arm kvF16=%v", r32.kvF16, r16.kvF16)
	}
	for l := range r16.layers {
		if r16.kc[l].Len() != r32.kc[l].Len() || r16.vc[l].Len() != r32.vc[l].Len() {
			t.Fatalf("layer %d: element counts differ (%d/%d vs %d/%d)", l, r16.kc[l].Len(), r16.vc[l].Len(), r32.kc[l].Len(), r32.vc[l].Len())
		}
		if !r16.layers[l].kvHalf || r32.layers[l].kvHalf {
			t.Fatalf("layer %d: kvHalf f16 arm %v, f32 arm %v", l, r16.layers[l].kvHalf, r32.layers[l].kvHalf)
		}
	}
	b32, b16 := kvBytesForCap(r32.ctxCap, r32.layers), kvBytesForCap(r16.ctxCap, r16.layers)
	if r32.ctxCap != r16.ctxCap || b16*2 != b32 {
		t.Errorf("KV bytes at the same capacity %d: f32 %d, f16 %d (want exactly half)", r32.ctxCap, b32, b16)
	}
	if want := m16.ResidentKVBytes("cuda", r16.ctxCap, true, false); want != b16 {
		t.Errorf("decoder prices the f16 cache at %d bytes, the build allocates %d", want, b16)
	}
	if want := m32.ResidentKVBytes("cuda", r32.ctxCap, false, false); want != b32 {
		t.Errorf("decoder prices the f32 cache at %d bytes, the build allocates %d", want, b32)
	}
}

// TestKV16_layer0StoreIsTheRoundedF32Store: layer 0's K and V are computed from the same embedding by the same f32 arithmetic in both arms (the
// cache is read only by the attention that comes after the store), so the f16 arm's layer-0 cache must equal the f32 arm's rounded to nearest
// even, bit for bit, at every position. That checks the f16 store kernels on the batched prefill path and on the decode path.
func TestKV16_layer0StoreIsTheRoundedF32Store(t *testing.T) {
	_, m16, r32, r16 := kv16Pair(t, 1024)
	const nPrompt, nDecode = 70, 6
	embs := kv16Prompt(m16, nPrompt)
	ctx := context.Background()
	if _, err := r32.PrefillLast(ctx, embs, 0); err != nil {
		t.Fatalf("f32 prefill: %v", err)
	}
	if _, err := r16.PrefillLast(ctx, embs, 0); err != nil {
		t.Fatalf("f16 prefill: %v", err)
	}
	for i := 0; i < nDecode; i++ {
		e := m16.EmbedResidentForTest(2000 + 11*i)
		if _, err := r32.Forward(e, nPrompt+i); err != nil {
			t.Fatalf("f32 decode %d: %v", i, err)
		}
		if _, err := r16.Forward(e, nPrompt+i); err != nil {
			t.Fatalf("f16 decode %d: %v", i, err)
		}
	}
	n := (nPrompt + nDecode) * r32.layers[0].kvDim
	for _, c := range []struct {
		name   string
		b32    Buffer
		b16    Buffer
		layer0 int
	}{{"K", r32.kc[0], r16.kc[0], 0}, {"V", r32.vc[0], r16.vc[0], 0}} {
		f := make([]float32, c.b32.Len())
		h := make([]uint16, c.b16.Len())
		if e := gpu.Download(c.b32, f); e != nil {
			t.Fatal(e)
		}
		if e := gpu.Download(c.b16, h); e != nil {
			t.Fatal(e)
		}
		bad, nonzero := 0, 0
		for i := 0; i < n; i++ {
			if want := f32ToF16RNE(f[i]); h[i] != want {
				if bad < 3 {
					t.Errorf("layer 0 %s[%d]: f16 cache %#04x, f32 %g rounds to %#04x", c.name, i, h[i], f[i], want)
				}
				bad++
			}
			if f[i] != 0 {
				nonzero++
			}
		}
		if bad > 0 {
			t.Errorf("layer 0 %s: %d of %d elements differ from the rounded f32 store", c.name, bad, n)
		}
		if nonzero < n/2 {
			t.Errorf("layer 0 %s: only %d of %d elements are nonzero: the comparison is vacuous", c.name, nonzero, n)
		}
	}
}

// TestKV16_uploadKVRoundsLikeTheKernels: the CPU-prefill bridge writes the same f16 values the store kernels would.
func TestKV16_uploadKVRoundsLikeTheKernels(t *testing.T) {
	_, _, _, r16 := kv16Pair(t, 256)
	kvDim := r16.layers[3].kvDim
	const n = 5
	keys, vals := make([]float32, n*kvDim), make([]float32, n*kvDim)
	for i := range keys {
		keys[i] = float32(math.Sin(float64(i)*0.37)) * 7
		vals[i] = float32(math.Cos(float64(i)*0.11)) * 0.013
	}
	if e := r16.UploadKV(3, 2, keys, vals); e != nil {
		t.Fatal(e)
	}
	got := make([]uint16, r16.kc[3].Len())
	if e := gpu.Download(r16.kc[3], got); e != nil {
		t.Fatal(e)
	}
	for i := range keys {
		if want := f32ToF16RNE(keys[i]); got[2*kvDim+i] != want {
			t.Fatalf("K[%d]: cache %#04x, want %#04x", i, got[2*kvDim+i], want)
		}
	}
	for _, i := range []int{0, 1, kvDim - 1} { // positions 0 and 1 were not written
		if got[i] != 0 {
			t.Errorf("element %d below the upload base was disturbed: %#04x", i, got[i])
		}
	}
}

// TestKV16_everyAttentionPathStaysCloseToF32 drives both arms over real prose and compares last-position logits step by step on each attention path
// an f16 twin exists for: the coalesced M=1 decode attention (attn_batched), split-KV, the flash-decode lane, and the prefill attention both
// fused (attn_fused, the 128-row-tile variant on hd 128) and unfused (attn_batched). Two models, head dim 64 and 128. The bar is a smoke bar; the
// registered fidelity gate is the verdict. What this catches is a twin that reads the cache wrongly (a cosine far from 1) or a path that never ran
// (a launch counter at 0: the cell would be vacuous).
func TestKV16_everyAttentionPathStaysCloseToF32(t *testing.T) {
	type cell struct {
		name  string
		env   map[string]string
		check func(t *testing.T, r *cudaResident)
	}
	cells := []cell{
		{"decode attn_batched M=1", map[string]string{"GOINFER_CUDA_FLASH_DECODE": "0", "GOINFER_SPLITKV_ATTN": "0"},
			func(t *testing.T, r *cudaResident) {
				if r.faLaunches != 0 || r.skLaunches != 0 {
					t.Errorf("flash-decode %d / split-KV %d launches on the attn_batched cell", r.faLaunches, r.skLaunches)
				}
			}},
		{"decode split-KV", map[string]string{"GOINFER_CUDA_FLASH_DECODE": "0", "GOINFER_SPLITKV_MIN_KEYS": "0"},
			func(t *testing.T, r *cudaResident) {
				if r.skLaunches == 0 {
					t.Errorf("no split-KV launches: the cell is vacuous")
				}
			}},
		{"decode flash-decode lane", map[string]string{"GOINFER_CUDA_FLASH_DECODE": "16", "GOINFER_CUDA_FLASH_DECODE_MIN_KEYS": "0"},
			func(t *testing.T, r *cudaResident) {
				if r.faLaunches == 0 {
					t.Errorf("no flash-decode launches: the cell is vacuous")
				}
			}},
		{"prefill attn_fused", map[string]string{"GOINFER_CUDA_FLASH_DECODE": "0", "GOINFER_CUDA_FAST_PREFILL_FLOOR": "0"},
			func(t *testing.T, r *cudaResident) {
				if r.fastAttnLaunches == 0 {
					t.Errorf("no fused prefill attention launches: the cell is vacuous")
				}
			}},
		{"prefill attn_batched", map[string]string{"GOINFER_CUDA_FLASH_DECODE": "0", "GOINFER_CUDA_FAST_PREFILL": "0"},
			func(t *testing.T, r *cudaResident) {
				if r.fastAttnLaunches != 0 {
					t.Errorf("%d fused prefill launches with GOINFER_CUDA_FAST_PREFILL=0", r.fastAttnLaunches)
				}
			}},
	}
	for _, model := range []string{"qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"} {
		for _, c := range cells {
			t.Run(model[:len("qwen2.5-coder-0.5b")]+"/"+c.name, func(t *testing.T) {
				const K, steps = 300, 8
				for k, v := range c.env {
					t.Setenv(k, v)
				}
				m32, _, r32, r16 := kv16PairOf(t, model, K+64)
				tk, err := tokenizer.LoadGGUF(modelPath(model))
				if err != nil {
					t.Skip(err)
				}
				_, files := decoder.PrefillGatePromptSet()
				ids := decoder.PrefillGateProseIDsForTest(t, tk, files[0], K+steps)
				embs := make([][]float32, K)
				for i := range embs {
					embs[i] = m32.EmbedResidentForTest(ids[i])
				}
				ctx := context.Background()
				a, err := r32.PrefillLast(ctx, embs, 0)
				if err != nil {
					t.Fatal(err)
				}
				b, err := r16.PrefillLast(ctx, embs, 0)
				if err != nil {
					t.Fatal(err)
				}
				cos := []float64{cosine64(a, b)}
				agree := 0
				if argmaxF32(a) == argmaxF32(b) {
					agree++
				}
				for i := 0; i < steps; i++ {
					e := m32.EmbedResidentForTest(ids[K+i])
					if a, err = r32.Forward(e, K+i); err != nil {
						t.Fatal(err)
					}
					if b, err = r16.Forward(e, K+i); err != nil {
						t.Fatal(err)
					}
					cos = append(cos, cosine64(a, b))
					if argmaxF32(a) == argmaxF32(b) {
						agree++
					}
				}
				minCos := cos[0]
				for _, v := range cos {
					minCos = math.Min(minCos, v)
				}
				t.Logf("prefill cosine %.6f, min over %d rows %.5f, argmax agreement %d/%d (f16 arm: flash %d, split-KV %d, fused prefill %d)",
					cos[0], len(cos), minCos, agree, len(cos), r16.faLaunches, r16.skLaunches, r16.fastAttnLaunches)
				// A smoke bar, not the fidelity gate: the readers' correctness is TestKV16_exactInputIdentity_*, bit for bit. What is left here is
				// f16's own rounding of K and V, which a 0.5B model's unfused attention shows at about 0.99 over a few hundred keys.
				if minCos < 0.98 {
					t.Errorf("min cosine %.5f against the f32 arm: the f16 path is wrong beyond rounding noise", minCos)
				}
				if agree < len(cos)*3/4 {
					t.Errorf("argmax agrees on %d of %d rows", agree, len(cos))
				}
				c.check(t, r16)
			})
		}
	}
}
