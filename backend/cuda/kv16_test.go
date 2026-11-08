//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"sync"
	"syscall"
	"testing"

	"github.com/townsendmerino/aikit/linalg"
	gpu "github.com/townsendmerino/goinfer/backend/cuda/internal/gpudev"
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

// TestKV16_cacheIsHalfAndTheKnobIsHonoured: -kv f16 on CUDA now does what it says. The f16 arm's per-layer pools are half the f32 arm's,
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
		// Each layer's K and V pool holds its group's blocks for every slot plus the null block, in elements of the layer's own type.
		if Ly := &r16.layers[l]; !Ly.noKV() && !Ly.isMLA() {
			if want := r16.kvPoolBlocks(r16.kvGroupOf(l), r16.KVSlots()) * kvBlock * Ly.kvDim; r16.kc[l].Len() != want || r16.vc[l].Len() != want {
				t.Fatalf("layer %d: f16 pool holds %d/%d elements, want %d", l, r16.kc[l].Len(), r16.vc[l].Len(), want)
			}
		}
		if !r16.layers[l].kvHalf || r32.layers[l].kvHalf {
			t.Fatalf("layer %d: kvHalf f16 arm %v, f32 arm %v", l, r16.layers[l].kvHalf, r32.layers[l].kvHalf)
		}
	}
	// A slot's bytes are its K/V blocks plus its block tables; the tables are int32 at either precision, so only the blocks halve. The null
	// blocks are per build and halve with the rows they stand in for.
	b32, b16 := kvBytesForCap(r32.ctxCap, r32.layers), kvBytesForCap(r16.ctxCap, r16.layers)
	tabs := int64(kvGroups * 4 * kvBlocksFor(r16.ctxCap))
	if r32.ctxCap != r16.ctxCap || (b16-tabs)*2 != b32-tabs {
		t.Errorf("KV block bytes at the same capacity %d: f32 %d, f16 %d (want exactly half)", r32.ctxCap, b32-tabs, b16-tabs)
	}
	n32, n16 := kvPoolNullBytes(r32.layers), kvPoolNullBytes(r16.layers)
	if n16*2 != n32 {
		t.Errorf("null blocks: f32 %d bytes, f16 %d (want exactly half)", n32, n16)
	}
	// ResidentKVBytes prices a one-slot load: one slot's blocks and tables, and the null blocks.
	if want := m16.ResidentKVBytes("cuda", r16.ctxCap, true, false); want != b16+n16 {
		t.Errorf("decoder prices the f16 cache at %d bytes, a one-slot build allocates %d", want, b16+n16)
	}
	if want := m32.ResidentKVBytes("cuda", r32.ctxCap, false, false); want != b32+n32 {
		t.Errorf("decoder prices the f32 cache at %d bytes, a one-slot build allocates %d", want, b32+n32)
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
	nPos := nPrompt + nDecode
	n := nPos * r32.layers[0].kvDim
	rows := func(r *cudaResident) (k, v []float32) {
		if err := r.do(func() error {
			if e := r.stream.Sync(); e != nil {
				return e
			}
			var e error
			k, v, e = r.kvRowsForTest(r.kvSlot, 0, 0, nPos)
			return e
		}); err != nil {
			t.Fatal(err)
		}
		return k, v
	}
	k32, v32 := rows(r32)
	k16, v16 := rows(r16)
	for _, c := range []struct {
		name string
		f    []float32
		h    []uint16
	}{{"K", k32, kvWordsToHalves(k16)}, {"V", v32, kvWordsToHalves(v16)}} {
		f, h := c.f, c.h
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
	// Positions 0 and 1 first hold a sentinel (device memory is not cleared at allocation, so "untouched" has to be a known value).
	const sentinel = 1.5
	below := make([]float32, 2*kvDim)
	for i := range below {
		below[i] = sentinel
	}
	if e := r16.UploadKV(3, 0, below, below); e != nil {
		t.Fatal(e)
	}
	if e := r16.UploadKV(3, 2, keys, vals); e != nil {
		t.Fatal(e)
	}
	var words []float32
	if e := r16.do(func() error {
		var err error
		words, _, err = r16.kvRowsForTest(r16.kvSlot, 3, 0, 2+n)
		return err
	}); e != nil {
		t.Fatal(e)
	}
	got := kvWordsToHalves(words)
	for i := range keys {
		if want := f32ToF16RNE(keys[i]); got[2*kvDim+i] != want {
			t.Fatalf("K[%d]: cache %#04x, want %#04x", i, got[2*kvDim+i], want)
		}
	}
	for _, i := range []int{0, 1, kvDim - 1, 2*kvDim - 1} { // positions 0 and 1 hold the sentinel the upload at base 2 must not touch
		if want := f32ToF16RNE(sentinel); got[i] != want {
			t.Errorf("element %d below the upload base was disturbed: %#04x, want %#04x", i, got[i], want)
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

// The CUDA int8 KV cache delivered (S2 of docs/measurements/cuda-kv-i8-2026-11/scope.md): the dense families on one KV slot. The readers'
// own bar is S1's G1 exact-input identity (kernels/kv8_identity_test.go); these tests hold what S2 adds: the pools and scale pools the
// build allocates (G3), the store wired after the rope kernels, and every attention path an int8 cache takes reading them.

// kv8Fixtures are the dense layouts the int8 cache is delivered for: a tiny llama (head dim 16), phi3 (fused qkv), a head dim 128 llama,
// and a real 0.5B Qwen2 (head dim 64, GQA 7:1).
var kv8Fixtures = []string{"llama-tiny", "phi3-tiny", "llama-attnfa-tiny", "~qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"}

// kv8Path resolves a kv8Fixtures entry: a name under testdata/, or ~name under the models directory.
func kv8Path(fx string) string {
	if name, ok := strings.CutPrefix(fx, "~"); ok {
		return modelPath(name)
	}
	return filepath.Join("../../testdata", fx)
}

// kv8Allocated is the K/V cache bytes (pools and scale pools) each resident kv8Load built allocated, read from the allocations (kvAllocSeenForTest).
var kv8Allocated = map[*cudaResident]int64{}

// kv8Load loads fx at -kv kv on CUDA with opts and returns its resident, recording the K/V bytes it allocated.
func kv8Load(t *testing.T, fx, kv string, opts decoder.Options) (*decoder.Model, *cudaResident) {
	t.Helper()
	path := kv8Path(fx)
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture: %v", err)
	}
	o := opts
	o.Backend, o.KVPrecision = "cuda", kv
	if o.Quant == "" {
		o.Quant, o.ActQuantGroup = kv16Quant(fx), kv16ActGroup(fx) // Phi-3 loads int8int8 with per-32 activation scales, as the served bench does
	}
	if o.ResidentKVSlots == 0 {
		o.ResidentKVSlots = 1
	}
	var mu sync.Mutex
	seen := map[*cudaResident]int64{}
	kvAllocSeenForTest = func(r *cudaResident, b int) { mu.Lock(); seen[r] += int64(b); mu.Unlock() }
	m, err := decoder.Load(path, o)
	kvAllocSeenForTest = nil
	if err != nil {
		t.Fatalf("load %s (-kv %s): %v", fx, kv, err)
	}
	t.Cleanup(func() { m.Close() })
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("%s (-kv %s) not CUDA-resident: %s", fx, kv, m.ResidentDecline())
	}
	kv8Allocated[r] = seen[r]
	t.Cleanup(func() { delete(kv8Allocated, r) })
	return m, r
}

// TestKV8_flashLaneCompiledFromItsPrecisionsTwin: the flash-decode lane of an f32, f16 and int8 build is compiled from decode_fa, its _kv16
// twin and its _kv8 twin respectively, and the int8 build allocates exactly f32/4 + f32*nKV/kvDim: the two facts TestKV8GateVsReference's
// precondition 1 checks on each arm, asserted here on real builds so the gate's arithmetic is not taken on trust.
func TestKV8_flashLaneCompiledFromItsPrecisionsTwin(t *testing.T) {
	requireCUDADevice(t)
	t.Setenv("GOINFER_CUDA_FLASH_DECODE", "16") // the lane on at this short context, as the gate and the every-path test turn it on
	t.Setenv("GOINFER_CUDA_FLASH_DECODE_MIN_KEYS", "0")
	const fx = "llama-attnfa-tiny"
	alloc := map[string]int64{}
	var kvDim, nKV int
	for _, c := range []struct{ kv, module string }{{"f32", "decode_fa"}, {"f16", "decode_fa_kv16"}, {"i8", "decode_fa_kv8"}} {
		m, r := kv8Load(t, fx, c.kv, decoder.Options{})
		if r.faSplit < 1 || r.faCombine == (Pipeline{}) {
			t.Fatalf("-kv %s: the flash-decode lane is not loaded", c.kv)
		}
		if r.faModule != c.module {
			t.Errorf("-kv %s: the flash-decode lane was compiled from %q, not %q", c.kv, r.faModule, c.module)
		}
		for l := range r.layers {
			if Ly := &r.layers[l]; !Ly.noKV() && !Ly.kvShared {
				if kvDim == 0 {
					kvDim, nKV = Ly.kvDim, Ly.nKV
				} else if Ly.kvDim != kvDim || Ly.nKV != nKV {
					t.Fatalf("%s's KV layers differ in geometry; the exact int8 figure needs one", fx)
				}
			}
		}
		alloc[c.kv] = kv8Allocated[r]
		var pools int64 // TestKV8GateVsReference's precondition 1, on this build: each KV-owning layer's pool rows times its row bytes
		for l := range r.layers {
			if Ly := &r.layers[l]; !Ly.noKV() && !Ly.kvShared && Ly.kvDim > 0 {
				pools += 2 * int64(r.kc[l].Len()/Ly.kvDim) * kvRowBytes(Ly)
			}
		}
		if pools != alloc[c.kv] {
			t.Errorf("-kv %s: allocated %d B of K/V; its layers' pools at its precision are %d B", c.kv, alloc[c.kv], pools)
		}
		m.Close()
	}
	e, h, q := alloc["f32"], alloc["f16"], alloc["i8"]
	if e <= 0 || 2*h != e || 4*q*int64(kvDim) != e*int64(kvDim)+4*e*int64(nKV) {
		t.Errorf("allocations f32 %d B, f16 %d B, i8 %d B; i8 should be %d B", e, h, q, e/4+e*int64(nKV)/int64(kvDim))
	}
}

// TestKV8_deliveredBytes is G3 for the CUDA int8 cache (docs/tasks/task-kv-precision-everywhere-2026-10.md §4): at -kv i8 every layer that
// keeps K/V holds int8 pools with as many elements as the f32 build's and one f32 scale per (row, KV head) for each of K and V; the build
// ALLOCATES exactly 1 B an element plus 8 B of scales per position, head and layer (the f32 build's bytes / 4, plus the scales); and
// decoder.ResidentKVBytes prices exactly what it allocated; a second slot costs exactly CUDAKVSlotBytes more. The pool says i8.
func TestKV8_deliveredBytes(t *testing.T) {
	requireCUDADevice(t)
	for _, fx := range kv8Fixtures {
		t.Run(filepath.Base(fx), func(t *testing.T) {
			// The f32 arm first, read, then closed, so the int8 arm plans its context against the same free VRAM.
			m32, r32 := kv8Load(t, fx, "f32", decoder.Options{})
			ctx32, a32, kv32 := r32.ctxCap, kv8Allocated[r32], r32.kvI8
			lens32 := make([][2]int, len(r32.layers))
			for l := range r32.layers {
				lens32[l] = [2]int{r32.kc[l].Len(), r32.vc[l].Len()}
			}
			m32.Close()
			m8, r8 := kv8Load(t, fx, "i8", decoder.Options{})
			if got := m8.ResidentKVPrecision(); got != "i8" {
				t.Fatalf("-kv i8 reports %q", got)
			}
			if !r8.kvI8 || kv32 || r8.kvF16 {
				t.Fatalf("resident flags: i8 arm kvI8=%v kvF16=%v, f32 arm kvI8=%v", r8.kvI8, r8.kvF16, kv32)
			}
			if p := r8.KVPool().Precision; p != "i8" {
				t.Errorf("KVPool precision %q", p)
			}
			if r8.ctxCap != ctx32 {
				t.Fatalf("contexts differ (f32 %d, i8 %d): the arms are not comparable", ctx32, r8.ctxCap)
			}
			owners, scales := 0, int64(0)
			var formula int64
			for l := range r8.layers {
				Ly := &r8.layers[l]
				if Ly.noKV() || Ly.kvShared {
					continue
				}
				owners++
				if !Ly.kvI8 {
					t.Fatalf("layer %d: not int8", l)
				}
				rows := r8.kc[l].Len() / Ly.kvDim
				if r8.kc[l].Len() != r8.vc[l].Len() || r8.ksc[l].Len() != rows*Ly.nKV || r8.vsc[l].Len() != rows*Ly.nKV {
					t.Fatalf("layer %d: K %d, V %d int8; K scales %d, V scales %d; want %d rows of %d heads", l, r8.kc[l].Len(), r8.vc[l].Len(),
						r8.ksc[l].Len(), r8.vsc[l].Len(), rows, Ly.nKV)
				}
				if r8.kc[l].Len() != lens32[l][0] || r8.vc[l].Len() != lens32[l][1] {
					t.Fatalf("layer %d: element counts differ from the f32 build (%d/%d vs %d/%d)", l, r8.kc[l].Len(), r8.vc[l].Len(), lens32[l][0], lens32[l][1])
				}
				scales += int64(r8.ksc[l].Len()+r8.vsc[l].Len()) * 4
				formula += 2 * int64(rows) * (int64(Ly.kvDim) + 4*int64(Ly.nKV)) // K and V: 1 B an element and a 4 B scale per head, each
			}
			if owners == 0 {
				t.Fatal("no layer keeps K/V: the check is vacuous")
			}
			a8 := kv8Allocated[r8]
			if a8 != formula {
				t.Errorf("the int8 build allocated %d B of K/V; 1 B an element plus 8 B of scales per position, head and layer is %d B", a8, formula)
			}
			if a8 != a32/4+scales {
				t.Errorf("the int8 build allocated %d B, the f32 build %d B: not a quarter plus %d B of scales", a8, a32, scales)
			}
			tabs := int64(kvGroups * 4 * kvBlocksFor(r8.ctxCap))
			if got, want := kvBytesForCap(r8.ctxCap, r8.layers)+kvPoolNullBytes(r8.layers), a8+tabs; got != want {
				t.Errorf("the resident's own fit figure is %d B; it allocated %d B and its tables are %d B", got, a8, tabs)
			}
			if want := m8.ResidentKVBytes("cuda", r8.ctxCap, false, true); want != a8+tabs {
				t.Errorf("decoder prices the int8 cache at %d B; a one-slot build allocates %d B and %d B of tables", want, a8, tabs)
			}
			// Two slots, loaded with the one-slot build closed so both plan against the same free VRAM: each slot keeps the one-slot context as its
			// guaranteed share; the pools hold one more slot's blocks; the tables are one per slot per group, as long as one conversation's cap
			// (which may be raised past the share, kvRaise).
			ctx8 := r8.ctxCap
			m8.Close()
			m2, r2 := kv8Load(t, fx, "i8", decoder.Options{ResidentKVSlots: 2})
			if r2.KVSlots() != 2 || !r2.kvI8 {
				t.Fatalf("two slots asked at -kv i8: built %d (int8 %v)", r2.KVSlots(), r2.kvI8)
			}
			share, tabsS := r2.kvShareCtx, int64(kvGroups*4*kvBlocksFor(r2.kvShareCtx))
			if share != ctx8 {
				t.Errorf("each of two int8 slots is guaranteed %d positions; one slot took %d", share, ctx8)
			}
			if got, want := kv8Allocated[r2], m2.ResidentKVBytes("cuda", share, false, true)+m2.CUDAKVSlotBytes(share, false, true)-2*tabsS; got != want {
				t.Errorf("two int8 slots of %d positions allocate %d B of pools; ResidentKVBytes plus one CUDAKVSlotBytes, less their tables, price %d B", share, got, want)
			}
			if got, want := int64(r2.kvTabDev.Len())*4, int64(2*kvGroups*4*kvBlocksFor(r2.ctxCap)); got != want {
				t.Errorf("two slots' block tables hold %d B; two slots of %d groups at %d positions (the cap) is %d B", got, kvGroups, r2.ctxCap, want)
			}
			t.Logf("%d int8 layers, one slot at %d positions: allocated %d B (f32 build %d B), scales %d B; two slots %d B", owners, r8.ctxCap, a8, a32, scales, kv8Allocated[r2])
		})
	}
}

// kv8RowsForTest reads slot s's layer l rows [lo, hi) of an int8 cache: the int8 K and V, and their scales, [row][nKV].
func (r *cudaResident) kv8RowsForTest(s, l, lo, hi int) (k, v []int8, ks, vs []float32, err error) {
	Ly := &r.layers[l]
	k, v = make([]int8, (hi-lo)*Ly.kvDim), make([]int8, (hi-lo)*Ly.kvDim)
	ks, vs = make([]float32, (hi-lo)*Ly.nKV), make([]float32, (hi-lo)*Ly.nKV)
	err = r.kvRowRuns(s, r.kvGroupOf(l), lo, hi, func(p, n, row int) error {
		for _, c := range []struct {
			src Buffer
			dst []int8
		}{{r.kc[l], k}, {r.vc[l], v}} {
			if e := gpu.Download(c.src.At(row*Ly.kvDim), c.dst[(p-lo)*Ly.kvDim:(p-lo+n)*Ly.kvDim]); e != nil {
				return e
			}
		}
		for _, c := range []struct {
			src Buffer
			dst []float32
		}{{r.ksc[l], ks}, {r.vsc[l], vs}} {
			if e := gpu.Download(c.src.At(row*Ly.nKV*4), c.dst[(p-lo)*Ly.nKV:(p-lo+n)*Ly.nKV]); e != nil {
				return e
			}
		}
		return nil
	})
	return k, v, ks, vs, err
}

// TestKV8_layer0StoreIsTheQuantizedF32Store is G1's store half wired into the build: the first cache layer's K and V come from the same
// embedding by the same f32 arithmetic at either precision, so the int8 build's layer-0 cache is the f32 build's quantized per (position, head)
// by aikit's QuantizeRowInt8, the CPU int8 cache's own quantizer: the same int8 codes and the same f32 scale bits, at every position the
// batched prefill wrote and every position decode wrote.
func TestKV8_layer0StoreIsTheQuantizedF32Store(t *testing.T) {
	requireCUDADevice(t)
	for _, fx := range kv8Fixtures {
		t.Run(filepath.Base(fx), func(t *testing.T) {
			m8, r8 := kv8Load(t, fx, "i8", decoder.Options{})
			_, r32 := kv8Load(t, fx, "f32", decoder.Options{})
			const kl = 0
			_, _, _, _, _, _, vocab := m8.Dims()
			nPrompt, nDecode := min(70, r8.ctxCap/2), 6
			embs := make([][]float32, nPrompt)
			for i := range embs {
				embs[i] = m8.EmbedResidentForTest(3 + (37*i)%(vocab-3))
			}
			ctx := context.Background()
			for _, r := range []*cudaResident{r32, r8} {
				if _, err := r.PrefillLast(ctx, embs, 0); errors.Is(err, errPrefillDeclined) {
					for i, e := range embs { // a load the batched prefill declines (Phi-3's per-32 activation scales) writes its rows one at a time
						if _, err := r.Forward(e, i); err != nil {
							t.Fatalf("prefill by decode at %d: %v", i, err)
						}
					}
				} else if err != nil {
					t.Fatalf("prefill (int8 %v): %v", r.kvI8, err)
				}
				for i := range nDecode {
					if _, err := r.Forward(m8.EmbedResidentForTest(3+(11*i)%(vocab-3)), nPrompt+i); err != nil {
						t.Fatalf("decode %d: %v", i, err)
					}
				}
			}
			nPos := nPrompt + nDecode
			Ly := &r8.layers[kl]
			var k32, v32 []float32
			var k8, v8 []int8
			var ks, vs []float32
			if err := r32.do(func() error {
				var e error
				k32, v32, e = r32.kvRowsForTest(r32.kvSlot, kl, 0, nPos)
				return e
			}); err != nil {
				t.Fatal(err)
			}
			if err := r8.do(func() error {
				if e := r8.stream.Sync(); e != nil {
					return e
				}
				var e error
				k8, v8, ks, vs, e = r8.kv8RowsForTest(r8.kvSlot, kl, 0, nPos)
				return e
			}); err != nil {
				t.Fatal(err)
			}
			q := make([]int8, Ly.hd)
			for _, c := range []struct {
				name string
				f    []float32
				q    []int8
				sc   []float32
			}{{"K", k32, k8, ks}, {"V", v32, v8, vs}} {
				bad, nonzero := 0, 0
				for p := range nPos {
					for h := range Ly.nKV {
						off := p*Ly.kvDim + h*Ly.hd
						sc := linalg.QuantizeRowInt8(c.f[off:off+Ly.hd], q)
						got := c.sc[p*Ly.nKV+h]
						same := math.Float32bits(sc) == math.Float32bits(got)
						for d := range Ly.hd {
							same = same && q[d] == c.q[off+d]
							if q[d] != 0 {
								nonzero++
							}
						}
						if !same {
							if bad < 3 {
								t.Errorf("layer %d %s position %d head %d: cache scale %g codes %v, the f32 row quantizes to %g %v", kl, c.name, p, h, got,
									c.q[off:off+min(8, Ly.hd)], sc, q[:min(8, Ly.hd)])
							}
							bad++
						}
					}
				}
				if bad > 0 {
					t.Errorf("layer %d %s: %d of %d (position, head) rows differ from the quantized f32 store", kl, c.name, bad, nPos*Ly.nKV)
				}
				if nonzero < nPos*Ly.kvDim/2 {
					t.Errorf("layer %d %s: only %d of %d codes are nonzero: the comparison is vacuous", kl, c.name, nonzero, nPos*Ly.kvDim)
				}
			}
		})
	}
}

// TestKV8_everyAttentionPathStaysCloseToF32 drives an int8 and an f32 build over real prose and compares last-position logits step by step on
// each attention path an int8 cache takes: decode through attn_batched (M=1), split-KV and the flash-decode lane; prefill through attn_batched,
// through the row-batched split-KV kernels, and with fast prefill on, which takes the tensor-core tiles' _kv8 twins (S6's route gate: a fast-prefill
// cell with no fused launch is a failure). The bar is a smoke bar, registered with S2 before its first run (cosine >= 0.97 on every row,
// argmax agreement on 3 of 4); the readers' correctness is S1's bit-exact G1, and the fidelity read is S4's. What this catches is a reader bound
// without its scales, a store that never ran, or a path that never ran (a launch counter at 0). The 1.5B only: what int8 KV costs the 0.5B
// is under the bar on the CPU's own int8 cache too, which makes it a quality reading, not a correctness one (scope.md, "S2").
func TestKV8_everyAttentionPathStaysCloseToF32(t *testing.T) {
	requireCUDADevice(t)
	type cell struct {
		name  string
		env   map[string]string
		force bool // prefillSplitKVForceForTest
		check func(t *testing.T, r *cudaResident)
	}
	noFused := func(t *testing.T, r *cudaResident) {
		if r.fastAttnLaunches != 0 {
			t.Errorf("%d fused prefill launches on an int8 cache", r.fastAttnLaunches)
		}
	}
	cells := []cell{
		{"decode attn_batched M=1", map[string]string{"GOINFER_CUDA_FLASH_DECODE": "0", "GOINFER_SPLITKV_ATTN": "0"}, false,
			func(t *testing.T, r *cudaResident) {
				if r.faLaunches != 0 || r.skLaunches != 0 {
					t.Errorf("flash-decode %d / split-KV %d launches on the attn_batched cell", r.faLaunches, r.skLaunches)
				}
			}},
		{"decode split-KV", map[string]string{"GOINFER_CUDA_FLASH_DECODE": "0", "GOINFER_SPLITKV_MIN_KEYS": "0"}, false,
			func(t *testing.T, r *cudaResident) {
				if r.skLaunches == 0 {
					t.Errorf("no split-KV launches: the cell is vacuous")
				}
			}},
		{"decode flash-decode lane", map[string]string{"GOINFER_CUDA_FLASH_DECODE": "16", "GOINFER_CUDA_FLASH_DECODE_MIN_KEYS": "0"}, false,
			func(t *testing.T, r *cudaResident) {
				if r.faLaunches == 0 {
					t.Errorf("no flash-decode launches: the cell is vacuous")
				}
			}},
		{"prefill with fast prefill on", map[string]string{"GOINFER_CUDA_FLASH_DECODE": "0", "GOINFER_CUDA_FAST_PREFILL_FLOOR": "0"}, false,
			func(t *testing.T, r *cudaResident) {
				if r.fastAttnLaunches == 0 {
					t.Errorf("the int8 cache took no tensor-core tile launch with fast prefill on (S6's route gate)")
				}
			}},
		// (the f32 control of this cell must have taken the fused tiles; asserted below, after both arms ran)
		{"prefill attn_batched", map[string]string{"GOINFER_CUDA_FLASH_DECODE": "0", "GOINFER_CUDA_FAST_PREFILL": "0"}, false, noFused},
		{"prefill row-batched split-KV", map[string]string{"GOINFER_CUDA_FLASH_DECODE": "0", "GOINFER_CUDA_FAST_PREFILL": "0"}, true,
			func(t *testing.T, r *cudaResident) {
				noFused(t, r)
				if r.psLaunches == 0 {
					t.Errorf("no row-batched split-KV launches: the cell is vacuous")
				}
			}},
	}
	// The 1.5B only. The 0.5B misses the 0.97 bar on every cell (min cosine 0.86 to 0.87) because int8 KV costs that model as much on the
	// CPU's own int8 cache (min cosine 0.82, top-1 35/65 against f32): a quality reading (G4), not a reader fault; the exact-input G2 holds
	// the 0.5B's readers (scope.md, "S2", "Readings so far").
	for _, model := range []string{"qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"} {
		for _, c := range cells {
			t.Run(model[:len("qwen2.5-coder-0.5b")]+"/"+c.name, func(t *testing.T) {
				const K, steps = 300, 8
				for k, v := range c.env {
					t.Setenv(k, v)
				}
				prefillSplitKVForceForTest = c.force
				defer func() { prefillSplitKVForceForTest = false }()
				m32, r32 := kv8Load(t, "~"+model, "f32", decoder.Options{ResidentContext: K + 64})
				_, r8 := kv8Load(t, "~"+model, "i8", decoder.Options{ResidentContext: K + 64})
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
				b, err := r8.PrefillLast(ctx, embs, 0)
				if errors.Is(err, errPrefillDeclined) || err != nil {
					t.Fatalf("int8 prefill: %v", err)
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
					if b, err = r8.Forward(e, K+i); err != nil {
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
				t.Logf("prefill cosine %.6f, min over %d rows %.5f, argmax agreement %d/%d (int8 arm: flash %d, split-KV %d, fused prefill %d)",
					cos[0], len(cos), minCos, agree, len(cos), r8.faLaunches, r8.skLaunches, r8.fastAttnLaunches)
				if minCos < 0.97 {
					t.Errorf("min cosine %.5f against the f32 arm: the int8 path is wrong beyond quantization noise", minCos)
				}
				if agree < len(cos)*3/4 {
					t.Errorf("argmax agrees on %d of %d rows", agree, len(cos))
				}
				c.check(t, r8)
				if c.name == "prefill with fast prefill on" && r32.fastAttnLaunches == 0 {
					t.Errorf("the f32 control took no fused prefill launch: the cell does not compare the two builds' tensor-core tiles")
				}
			})
		}
	}
}

// TestKV8_G2_matchesTheCPUInt8Cache is G2 for the CUDA int8 cache (docs/tasks/task-kv-precision-everywhere-2026-10.md §4): the same model and
// prose at -kv i8 on the CUDA resident against the CPU path's int8 cache (the same quantizer, decoder/kvcache.go), with the same pair at f32 as
// the reference for what two backends' arithmetic alone moves. Prefill 300 tokens, then 64 teacher-forced decode steps; 65 rows.
//
// Bars, registered before the first run (docs/measurements/cuda-kv-i8-2026-11/scope.md, "S2"):
//   - top-1 agreement of CUDA i8 with CPU i8 is at least the CUDA f32 / CPU f32 rate less 2 points (the doc's bar);
//   - the minimum logit cosine of CUDA i8 with CPU i8 is at least 0.99 (a different cache, or the wrong scales, reads far below it).
//
// The CPU int8 cache's own cosine to the CPU f32 cache is logged beside it: it is what int8 KV costs the model, which is G4's read, not a
// correctness bar.
//
// It read FAIL on its first run and its bars are kept as registered. The owner's decision (c) on #755 put the correctness question to the
// exact-input G2 (TestKV8_G2exact_readersMatchTheCPUOnOneCache); this one is kept as a reading and runs with the diagnostics
// (GOINFER_KV8_DIAG=1), so a run of the suite stays green on the gate that decides.
func TestKV8_G2_matchesTheCPUInt8Cache(t *testing.T) {
	kv8Diag(t)
	for _, model := range []string{"qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"} {
		t.Run(model[:len("qwen2.5-coder-0.5b")], func(t *testing.T) {
			_, files := decoder.PrefillGatePromptSet()
			g := kv8G2(t, model, files[0])
			if g.ag8 < g.ag32-0.02 {
				t.Errorf("G2 top-1: CUDA i8 agrees with CPU i8 on %.4f of the rows, under the f32 pair's %.4f less 2 points", g.ag8, g.ag32)
			}
			if g.min8 < 0.99 {
				t.Errorf("G2 logits: CUDA i8 against CPU i8 min cosine %.5f, under 0.99", g.min8)
			}
		})
	}
}

// kv8G2Read is one G2 run: per pair, the minimum logit cosine and the top-1 agreement rate over the rows.
type kv8G2Read struct{ min32, ag32, min8, ag8, minQ, agQ, minG, agG float64 }

// kv8G2 runs G2's four arms (CUDA and CPU, f32 and int8) of model over prose file: prefill 300 tokens, then 64 teacher-forced steps.
func kv8G2(t *testing.T, model, file string) kv8G2Read {
	t.Helper()
	const K, steps = 300, 64
	path := modelPath(model)
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture: %v", err)
	}
	tk, err := tokenizer.LoadGGUF(path)
	if err != nil {
		t.Skip(err)
	}
	ids := decoder.PrefillGateProseIDsForTest(t, tk, file, K+steps)
	ctx := context.Background()
	// rows runs one arm over the prose: the CUDA resident (r != nil) or the CPU path of m, and returns its 65 logit rows.
	rows := func(m *decoder.Model, r *cudaResident) [][]float32 {
		var out [][]float32
		if r != nil {
			embs := make([][]float32, K)
			for i := range embs {
				embs[i] = m.EmbedResidentForTest(ids[i])
			}
			lg, err := r.PrefillLast(ctx, embs, 0)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), lg...))
			for i := range steps {
				if lg, err = r.Forward(m.EmbedResidentForTest(ids[K+i]), K+i); err != nil {
					t.Fatal(err)
				}
				out = append(out, append([]float32(nil), lg...))
			}
			return out
		}
		cache := m.NewCache(K + steps + 1)
		lg, err := m.PrefillLogitsForTest(ctx, ids[:K], cache)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg...))
		for i := range steps {
			if lg, err = m.ForwardForTest(ids[K+i], cache); err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), lg...))
		}
		return out
	}
	cpu := func(kv string) *decoder.Model {
		m, err := decoder.Load(path, decoder.Options{Quant: "int4", KVQuant: kv})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { m.Close() })
		if got := m.CPUKVPrecision(); got != kv {
			t.Fatalf("the CPU cache is %q, asked %q", got, kv)
		}
		return m
	}
	mg32, rg32 := kv8Load(t, "~"+model, "f32", decoder.Options{ResidentContext: K + steps + 64})
	g32 := rows(mg32, rg32)
	mg8, rg8 := kv8Load(t, "~"+model, "i8", decoder.Options{ResidentContext: K + steps + 64})
	g8 := rows(mg8, rg8)
	c32, c8 := rows(cpu("f32"), nil), rows(cpu("i8"), nil)
	pair := func(a, b [][]float32) (minCos, agree float64) {
		minCos = 1
		for i := range a {
			minCos = math.Min(minCos, cosine64(a[i], b[i]))
			if argmaxF32(a[i]) == argmaxF32(b[i]) {
				agree++
			}
		}
		return minCos, agree / float64(len(a))
	}
	var g kv8G2Read
	g.min32, g.ag32 = pair(g32, c32)
	g.min8, g.ag8 = pair(g8, c8)
	g.minQ, g.agQ = pair(c8, c32)
	g.minG, g.agG = pair(g8, g32)
	n := float64(len(g32))
	t.Logf("%s, %s: CUDA f32 vs CPU f32: min cos %.5f top-1 %.0f/%.0f", model, filepath.Base(file), g.min32, g.ag32*n, n)
	t.Logf("%s, %s: CUDA i8  vs CPU i8:  min cos %.5f top-1 %.0f/%.0f", model, filepath.Base(file), g.min8, g.ag8*n, n)
	t.Logf("%s, %s: CPU i8 vs CPU f32 (what int8 KV costs on the CPU): min cos %.5f top-1 %.0f/%.0f", model, filepath.Base(file), g.minQ, g.agQ*n, n)
	t.Logf("%s, %s: CUDA i8 vs CUDA f32 (what it costs on CUDA): min cos %.5f top-1 %.0f/%.0f", model, filepath.Base(file), g.minG, g.agG*n, n)
	return g
}

// TestKV8_verifyIsSequentialDecode holds the speculative-verify path on an int8 cache: ForwardN's batched pass (rope_kv_batched, kv8_store over
// its rows, the batched or multi-row-lane attention) gives the same logits, bit for bit, as decoding the same rows one at a time from the same
// state, with the flash-decode lane on (multi-row verify lane against single-row lane) and off (attn_batched rows against M=1).
func TestKV8_verifyIsSequentialDecode(t *testing.T) {
	requireCUDADevice(t)
	for _, lane := range []string{"0", "16"} {
		for _, fx := range []string{"llama-attnfa-tiny", "~qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"} {
			t.Run(filepath.Base(fx)+"/lane="+lane, func(t *testing.T) {
				t.Setenv("GOINFER_CUDA_FLASH_DECODE", lane)
				t.Setenv("GOINFER_CUDA_FLASH_DECODE_MIN_KEYS", "0")
				const K, n = 90, 5
				run := func(batched bool) [][]float32 {
					m, r := kv8Load(t, fx, "i8", decoder.Options{ResidentContext: 256})
					_, _, _, _, _, _, vocab := m.Dims()
					embs := make([][]float32, K+n)
					for i := range embs {
						embs[i] = m.EmbedResidentForTest(3 + (29*i)%(vocab-3))
					}
					if _, err := r.PrefillLast(context.Background(), embs[:K], 0); err != nil {
						t.Fatal(err)
					}
					if batched {
						out, err := r.ForwardN(embs[K:], K)
						if err != nil {
							t.Fatal(err)
						}
						if ok, why := r.VerifyPath(); !ok {
							t.Fatalf("ForwardN is not the batched pass here: %s", why)
						}
						if lane != "0" && r.faRowLaunches == 0 {
							t.Fatalf("the multi-row verify lane did not run: the lane cell is vacuous")
						}
						return out
					}
					var out [][]float32
					for i := range n {
						lg, err := r.Forward(embs[K+i], K+i)
						if err != nil {
							t.Fatal(err)
						}
						out = append(out, append([]float32(nil), lg...))
					}
					return out
				}
				seq, bat := run(false), run(true)
				for i := range seq {
					for j := range seq[i] {
						if math.Float32bits(seq[i][j]) != math.Float32bits(bat[i][j]) {
							t.Fatalf("row %d logit %d: sequential %g, ForwardN %g", i, j, seq[i][j], bat[i][j])
						}
					}
				}
			})
		}
	}
}

// TestKV8_uploadRefused: the CPU-prefill UploadKV bridge (the VL families' path) is refused by name on an int8 cache, never written as floats.
// A host snapshot carries the scales (TestKV7_restoreFull/i8) and is priced with them: kvDim bytes and 4 B per head for each of K and V a row.
func TestKV8_uploadRefused(t *testing.T) {
	requireCUDADevice(t)
	_, r := kv8Load(t, "llama-tiny", "i8", decoder.Options{})
	Ly := &r.layers[0]
	if b, want := r.KVSnapshotBytes(decoder.KVBlockRows), int64(decoder.KVBlockRows)*int64(len(r.layers))*2*(int64(Ly.kvDim)+4*int64(Ly.nKV)); b != want {
		t.Errorf("KVSnapshotBytes of 64 int8 rows: %d, want %d", b, want)
	}
	if err := r.UploadKV(0, 0, make([]float32, r.layers[0].kvDim), make([]float32, r.layers[0].kvDim)); err == nil || !strings.Contains(err.Error(), "int8") {
		t.Errorf("UploadKV into an int8 cache: %v, want a refusal naming int8", err)
	}
}

// The S2 diagnostics behind G2's reading (docs/measurements/cuda-kv-i8-2026-11/scope.md, "S2"): not gates, so they run only with
// GOINFER_KV8_DIAG=1 (about 5 minutes each on the card). kv8Diag skips otherwise.
func kv8Diag(t *testing.T) {
	t.Helper()
	if os.Getenv("GOINFER_KV8_DIAG") != "1" {
		t.Skip("a diagnostic: GOINFER_KV8_DIAG=1 runs it")
	}
	requireCUDADevice(t)
}

// TestKV8_G2Diag_allPrompts is G2's four arms over the first eight prose files of the set, to size the one-prompt read's noise.
func TestKV8_G2Diag_allPrompts(t *testing.T) {
	kv8Diag(t)
	for _, model := range []string{"qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"} {
		_, files := decoder.PrefillGatePromptSet()
		files = files[:8] // about 5 minutes on the card
		var s kv8G2Read
		for _, f := range files {
			var g kv8G2Read
			t.Run(model[:18]+"/"+f[len(f)-12:], func(t *testing.T) { g = kv8G2(t, model, f) }) // its own subtest: its loads are released at its end
			s.ag32 += g.ag32 / float64(len(files))
			s.ag8 += g.ag8 / float64(len(files))
			s.agQ += g.agQ / float64(len(files))
			s.agG += g.agG / float64(len(files))
		}
		t.Logf("%s over %d prompts: top-1 CUDA f32/CPU f32 %.4f, CUDA i8/CPU i8 %.4f, CPU i8/CPU f32 %.4f, CUDA i8/CUDA f32 %.4f", model, len(files), s.ag32, s.ag8, s.agQ, s.agG)
	}
}

// TestKV8_G2Diag_amplification: within CUDA, two arms that differ only by the flash-decode lane (non-bit-identical decode attention), at f32
// and at i8. If int8 KV turns that arithmetic noise into more top-1 disagreement than f32 does, it does so whatever the two backends are.
func TestKV8_G2Diag_amplification(t *testing.T) {
	kv8Diag(t)
	const K, steps = 300, 64
	for _, model := range []string{"qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"} {
		_, files := decoder.PrefillGatePromptSet()
		var ag [2]float64 // f32, i8
		n := 0
		for _, f := range files[:8] {
			t.Run(model[:18]+"/"+f[len(f)-12:], func(t *testing.T) {
				t.Setenv("GOINFER_CUDA_FAST_PREFILL", "0")
				tk, err := tokenizer.LoadGGUF(modelPath(model))
				if err != nil {
					t.Skip(err)
				}
				ids := decoder.PrefillGateProseIDsForTest(t, tk, f, K+steps)
				for i, kv := range []string{"f32", "i8"} {
					var arms [2][][]float32
					for a, lane := range []string{"0", "16"} {
						t.Setenv("GOINFER_CUDA_FLASH_DECODE", lane)
						t.Setenv("GOINFER_CUDA_FLASH_DECODE_MIN_KEYS", "0")
						func() {
							m, r := kv8Load(t, "~"+model, kv, decoder.Options{ResidentContext: K + steps + 64})
							embs := make([][]float32, K)
							for j := range embs {
								embs[j] = m.EmbedResidentForTest(ids[j])
							}
							if _, err := r.PrefillLast(context.Background(), embs, 0); err != nil {
								t.Fatal(err)
							}
							for j := range steps {
								lg, err := r.Forward(m.EmbedResidentForTest(ids[K+j]), K+j)
								if err != nil {
									t.Fatal(err)
								}
								arms[a] = append(arms[a], append([]float32(nil), lg...))
							}
							m.Close()
						}()
					}
					for j := range arms[0] {
						if argmaxF32(arms[0][j]) == argmaxF32(arms[1][j]) {
							ag[i]++
						}
					}
				}
				n += steps
			})
		}
		t.Logf("%s over %d decode rows: top-1 lane-off vs lane-on, f32 %.4f, i8 %.4f", model, n, ag[0]/float64(n), ag[1]/float64(n))
	}
}

// TestKV8_G2exact_readersMatchTheCPUOnOneCache is the exact-input G2 (Francis's decision (c) on #755, docs/measurements/cuda-kv-i8-2026-11/scope.md,
// "S2, exact-input G2"): the CUDA int8 readers and the CPU's production int8 reader (decoder.Model.AttendInt8ForTest: the layer dequantized,
// then the float64-accumulated attention) fed ONE int8 cache, the rows and scales the CUDA resident stored, with the model's own queries,
// give the same attention output row, to float-reduction noise. Tier 1: llama-attnfa-tiny (head dim 128, in the repository, random tokens)
// and Qwen2.5-Coder 0.5B and 1.5B (prose); layers 0, the middle one and the last; seconds.
//
// Readers: every decode reader (glue attention, attn_batched M=1, the flash-decode lane, split-KV with and without its split V-sum) at four
// positions, and every prefill reader (attn_batched M=8, the row-batched split-KV kernels, the multi-row verify lane) over rows 200 to 207,
// each row with its OWN query (the one the model attended with at that position, captured while decoding it), against the CPU at that row's
// position. The output buffer is filled with NaN before every launch, so a reader that writes nothing, or only part of a row, reads as an
// infinite difference. The store is held on every layer: at the decode step at 300, each layer's int8 row and scales are aikit
// QuantizeRowInt8 of the rotated K and the V that layer handed kv8_store (layer 0's prefill rows are TestKV8_layer0StoreIsTheQuantizedF32Store's).
//
// The bar, registered before the first run: on every row, max |CUDA - CPU| <= 1e-4 x max |CPU row|. Two plants in the CPU copy only must break
// it on every row, decode and prefill: the most-attended key's V scale x1.05, and one V code of that key moved by 16. (They were x1.01 and +-1
// until #755's last review: the weakest then cleared the bar by 1%, close enough to drift green; scope.md, "S3, plants made decisive".)
func TestKV8_G2exact_readersMatchTheCPUOnOneCache(t *testing.T) {
	requireCUDADevice(t)
	const K, bar, M, start = 300, 1e-4, 8, 200
	t.Setenv("GOINFER_CUDA_FLASH_DECODE", "16") // the lane on at this short context, as the every-path test turns it on
	t.Setenv("GOINFER_CUDA_FLASH_DECODE_MIN_KEYS", "0")
	t.Setenv("GOINFER_SPLITKV_VSUM_SPLIT", "4") // loads the split V-sum; kv8Readers runs split-KV with it and without
	for _, fx := range []string{"llama-attnfa-tiny", "~qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", "~qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"} {
		t.Run(strings.TrimPrefix(fx, "~"), func(t *testing.T) {
			path := kv8Path(fx)
			if _, err := os.Stat(path); err != nil {
				t.Skipf("no fixture: %v", err)
			}
			mg, r := kv8Load(t, fx, "i8", decoder.Options{ResidentContext: K + 64})
			mc, err := decoder.Load(path, decoder.Options{Quant: "int4", KVQuant: "i8"})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { mc.Close() })
			_, _, _, _, _, _, vocab := mg.Dims()
			ids := make([]int, K+1)
			for i := range ids {
				ids[i] = 3 + (i*2654435761+11)%(vocab-3) // a repository fixture has no tokenizer: fixed random tokens
			}
			if strings.HasPrefix(fx, "~") {
				tk, err := tokenizer.LoadGGUF(path)
				if err != nil {
					t.Skip(err)
				}
				_, files := decoder.PrefillGatePromptSet()
				ids = decoder.PrefillGateProseIDsForTest(t, tk, files[0], K+1)
			}
			emb := func(lo, hi int) [][]float32 {
				e := make([][]float32, hi-lo)
				for i := range e {
					e[i] = mg.EmbedResidentForTest(ids[lo+i])
				}
				return e
			}
			// Rows [0, start) by prefill; start..start+M-1 decoded one at a time, each layer's rotated query captured; the rest by prefill;
			// then the decode step at K, which also captures each layer's rotated K and V as handed to kv8_store. Graphs off, so each layer's
			// launches have run when its hook returns.
			nL, nH := len(r.layers), r.nH
			var qs, kIn, vIn [][]float32
			graphs := r.graphs
			r.graphs = false
			legacyLayerForTest = func(r *cudaResident, Ly *cudaLayer, l, pos, ropePos int, qt float32, gA, gB, gC bool) error {
				if e := r.walkLayer(Ly, l, pos, ropePos, qt, gA, gB, gC); e != nil {
					return e
				}
				if e := r.stream.Sync(); e != nil {
					return e
				}
				qs[l], kIn[l], vIn[l] = make([]float32, r.nH*Ly.hd), make([]float32, Ly.kvDim), make([]float32, Ly.kvDim)
				for _, c := range []struct {
					b Buffer
					h []float32
				}{{r.qB, qs[l]}, {r.kB, kIn[l]}, {r.vB, vIn[l]}} {
					if e := gpu.Download(c.b, c.h); e != nil {
						return e
					}
				}
				return nil
			}
			step := func(pos int) {
				qs, kIn, vIn = make([][]float32, nL), make([][]float32, nL), make([][]float32, nL)
				if _, err := r.Forward(mg.EmbedResidentForTest(ids[pos]), pos); err != nil {
					t.Fatal(err)
				}
			}
			rowQs := make([][][]float32, M) // [row][layer]
			_, err = r.PrefillLast(context.Background(), emb(0, start), 0)
			for m := 0; err == nil && m < M; m++ {
				step(start + m)
				rowQs[m] = qs
			}
			if err == nil {
				_, err = r.PrefillLast(context.Background(), emb(start+M, K), start+M)
			}
			if err == nil {
				step(K)
			}
			legacyLayerForTest, r.graphs = nil, graphs
			if err != nil {
				t.Fatal(err)
			}
			// The store, every layer, row K.
			stored := 0
			for l := range nL {
				Ly := &r.layers[l]
				var kq, vq []int8
				var ks, vs []float32
				if err := r.do(func() error {
					var e error
					kq, vq, ks, vs, e = r.kv8RowsForTest(r.kvSlot, l, K, K+1)
					return e
				}); err != nil {
					t.Fatal(err)
				}
				q := make([]int8, Ly.hd)
				for _, c := range []struct {
					name string
					in   []float32
					code []int8
					sc   []float32
				}{{"K", kIn[l], kq, ks}, {"V", vIn[l], vq, vs}} {
					for h := range Ly.nKV {
						sc := linalg.QuantizeRowInt8(c.in[h*Ly.hd:(h+1)*Ly.hd], q)
						if math.Float32bits(sc) != math.Float32bits(c.sc[h]) || !slices.Equal(q, c.code[h*Ly.hd:(h+1)*Ly.hd]) {
							t.Errorf("layer %d %s head %d at position %d: the cache is not kv8_store's quantization of the row it was handed", l, c.name, h, K)
						}
						stored++
					}
				}
			}
			rows, plantRows := 0, 0
			ran := map[string]int{}
			worst, plantMin := 0.0, math.Inf(1) // the largest row's diff / scale, and the smallest a plant made
			for _, l := range []int{0, nL / 2, nL - 1} {
				Ly := &r.layers[l]
				hd, nKV, G := Ly.hd, Ly.nKV, nH/Ly.nKV
				var kq, vq []int8
				var ks, vs []float32
				if err := r.do(func() error {
					var e error
					kq, vq, ks, vs, e = r.kv8RowsForTest(r.kvSlot, l, 0, K+1)
					return e
				}); err != nil {
					t.Fatal(err)
				}
				cpuAt := func(pos int, q []float32, vq []int8, vs []float32) []float32 {
					n := pos + 1
					c, err := mc.AttendInt8ForTest(l, pos, q, kq[:n*Ly.kvDim], vq[:n*Ly.kvDim], ks[:n*nKV], vs[:n*nKV])
					if err != nil {
						t.Fatal(err)
					}
					return c
				}
				// check holds a CUDA row against the CPU's on the one cache, and against the two plants of the CPU copy, which must exceed the bar.
				check := func(name string, pos int, q, got []float32) {
					d, scale := kv8RowDiff(got, cpuAt(pos, q, vq, vs))
					worst = math.Max(worst, d/scale)
					if d > bar*scale {
						t.Errorf("layer %d position %d, %s: max |CUDA - CPU| %.3g, over %.0e x the row's %.3g", l, pos, name, d, bar, scale)
					}
					rows++
					n := pos + 1
					top := kv8TopKey(q, kq[:n*Ly.kvDim], ks[:n*nKV], 0, hd, nKV, G, r.attnScale)
					vs2 := append([]float32(nil), vs...)
					vs2[top*nKV] *= 1.05
					vq2 := append([]int8(nil), vq...)
					if vq2[top*Ly.kvDim] <= 127-16 {
						vq2[top*Ly.kvDim] += 16
					} else {
						vq2[top*Ly.kvDim] -= 16
					}
					for _, p := range []struct {
						name string
						vq   []int8
						vs   []float32
					}{{"V scale x1.05", vq, vs2}, {"V code +-16", vq2, vs}} {
						d1, s1 := kv8RowDiff(got, cpuAt(pos, q, p.vq, p.vs))
						plantMin = math.Min(plantMin, d1/s1)
						if d1 <= bar*s1 {
							t.Errorf("plant %q at layer %d position %d, %s (key %d): max diff %.3g is within the bar %.3g: the gate cannot see it", p.name, l, pos, name, top, d1, bar*s1)
						}
						plantRows++
					}
				}
				for _, pos := range []int{63, 64, K / 2, K} {
					for _, rd := range kv8Readers(r, l, pos) {
						ran[rd.name]++
						check(rd.name, pos, qs[l], kv8RunReader(t, r, qs[l], rd))
					}
				}
				qRows := make([]float32, 0, M*nH*hd)
				for m := range M {
					qRows = append(qRows, rowQs[m][l]...)
				}
				for _, rd := range kv8RowReaders(r, l, start, M) {
					ran[rd.name]++
					got := kv8RunRowReader(t, r, qRows, M, rd)
					for m := range M {
						check(rd.name, start+m, rowQs[m][l], got[m*nH*hd:(m+1)*nH*hd])
					}
				}
			}
			t.Logf("%d reader rows compared, %d plant rows; launches per reader %v; largest |CUDA - CPU| / max|row| %.2e (bar %.0e); smallest a plant made %.2e; "+
				"%d stored (layer, head, K/V) rows held to the quantizer", rows, plantRows, ran, worst, bar, plantMin, stored)
			for _, name := range []string{"glue attention", "attn_batched M=1", "flash-decode lane", "split-KV", "split-KV, V-sum split",
				"attn_batched M>1", "row-batched split-KV", "multi-row verify lane"} {
				if ran[name] == 0 {
					t.Errorf("%s never ran: its rows are not compared", name)
				}
			}
		})
	}
}

// kv8Reader is one CUDA attention reader at (layer, pos), or over a block of rows (kv8RowReaders).
type kv8Reader struct {
	name   string
	launch func() error
	rows   func(q, ctx Buffer) error // a prefill reader: reads q's rows, writes ctx's
}

// kv8Readers lists the decode readers that serve layer l at pos on r, each reading r.qB and writing r.cctx through the bound slot's table.
func kv8Readers(r *cudaResident, l, pos int) []kv8Reader {
	Ly := &r.layers[l]
	nKeys := pos + 1
	shm := uint32((nKeys + 128) * 4)
	out := []kv8Reader{
		{name: "glue attention", launch: func() error {
			return r.launch(r.fAttn, LaunchConfig{GridX: uint32(r.nH), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: shm},
				r.kvScaled(l, r.withCap(Arg(r.qB), Arg(r.kvK(l)), Arg(r.kvV(l)), Arg(r.kvT(l)), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)), gpu.ArgValue(int32(nKeys)), gpu.ArgValue(r.attnScale), gpu.ArgValue(Ly.window), Arg(r.cctx))...)...)
		}},
		{name: "attn_batched M=1", launch: func() error {
			return r.launch(r.bAttn, LaunchConfig{GridX: uint32(r.nH), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: shm},
				r.kvScaled(l, r.withCap(Arg(r.qB), Arg(r.kvK(l)), Arg(r.kvV(l)), Arg(r.kvT(l)), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)), gpu.ArgValue(int32(pos)), gpu.ArgValue(r.attnScale), gpu.ArgValue(Ly.window), gpu.ArgValue(int32(1)), Arg(r.cctx), r.sinkArg(l))...)...)
		}},
	}
	if r.faEligible(l) {
		out = append(out, kv8Reader{name: "flash-decode lane", launch: func() error { return r.flashDecodeAttn(l, pos) }})
	}
	if r.skScores != (Pipeline{}) {
		// split-KV's V-sum both ways: the single fold, and the split one GOINFER_SPLITKV_VSUM_SPLIT loads (skVsumSplit chunks).
		split := r.skVsumSplit
		out = append(out, kv8Reader{name: "split-KV", launch: func() error {
			defer func() { r.skVsumSplit = split }()
			r.skVsumSplit = 0
			return r.splitKVAttnDecode(l, pos)
		}})
		if split >= 1 && r.skVsumPartial != (Pipeline{}) {
			out = append(out, kv8Reader{name: "split-KV, V-sum split", launch: func() error { return r.splitKVAttnDecode(l, pos) }})
		}
	}
	return out
}

// kv8RowReaders lists the prefill readers that serve layer l over M rows at positions start.. on r.
func kv8RowReaders(r *cudaResident, l, start, M int) []kv8Reader {
	Ly := &r.layers[l]
	shm := uint32((start + M + 128) * 4)
	out := []kv8Reader{{name: "attn_batched M>1", rows: func(q, ctx Buffer) error {
		return r.launch(r.bAttn, LaunchConfig{GridX: uint32(r.nH), GridY: uint32(M), GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: shm},
			r.kvScaled(l, r.withCap(Arg(q), Arg(r.kvK(l)), Arg(r.kvV(l)), Arg(r.kvT(l)), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)), gpu.ArgValue(int32(start)), gpu.ArgValue(r.attnScale), gpu.ArgValue(Ly.window), gpu.ArgValue(int32(M)), Arg(ctx), r.sinkArg(l))...)...)
	}}}
	if r.prefillSplitKVReady() {
		out = append(out, kv8Reader{name: "row-batched split-KV", rows: func(q, ctx Buffer) error {
			stride := start + M // the widest row's keys
			sc, inv := r.tf(M*r.nH*stride), r.tf(M*r.nH)
			defer r.dev.ReleaseBuf(sc)
			defer r.dev.ReleaseBuf(inv)
			return r.prefillSplitKVAttn(l, start, M, q, ctx, sc, inv, stride, M, Buffer{}, 0)
		}})
	}
	if r.faEligible(l) {
		out = append(out, kv8Reader{name: "multi-row verify lane", rows: func(q, ctx Buffer) error { return r.flashVerifyAttn(l, start, 0, M, q, ctx) }})
	}
	return out
}

// kv8RunReader uploads q into r.qB, fills r.cctx with NaN, runs rd, and returns r.cctx: a reader that leaves any of the row unwritten
// returns NaN there.
func kv8RunReader(t *testing.T, r *cudaResident, q []float32, rd kv8Reader) []float32 {
	t.Helper()
	out := make([]float32, len(q))
	poison := make([]float32, len(q))
	for i := range poison {
		poison[i] = float32(math.NaN())
	}
	if err := r.do(func() error {
		if e := gpu.Upload(r.qB, q); e != nil {
			return e
		}
		if e := gpu.Upload(r.cctx, poison); e != nil {
			return e
		}
		if e := r.stream.Sync(); e != nil {
			return e
		}
		if e := rd.launch(); e != nil {
			return e
		}
		if e := r.stream.Sync(); e != nil {
			return e
		}
		return gpu.Download(r.cctx, out)
	}); err != nil {
		t.Fatalf("%s: %v", rd.name, err)
	}
	return out
}

// kv8RunRowReader runs a prefill reader over M rows of q in buffers of the test's own, the output filled with NaN first, and returns it.
func kv8RunRowReader(t *testing.T, r *cudaResident, q []float32, M int, rd kv8Reader) []float32 {
	t.Helper()
	out := make([]float32, len(q))
	poison := make([]float32, len(q))
	for i := range poison {
		poison[i] = float32(math.NaN())
	}
	if err := r.do(func() error {
		qb, cb := r.tup32(q), r.tup32(poison)
		defer r.dev.ReleaseBuf(qb)
		defer r.dev.ReleaseBuf(cb)
		if e := r.stream.Sync(); e != nil {
			return e
		}
		if e := rd.rows(qb, cb); e != nil {
			return e
		}
		if e := r.stream.Sync(); e != nil {
			return e
		}
		return gpu.Download(cb, out)
	}); err != nil {
		t.Fatalf("%s: %v", rd.name, err)
	}
	return out
}

// kv8RowDiff is max |a - b| and max |b| over a row. A NaN or infinite element of a (a value no reader produced) is an infinite difference.
func kv8RowDiff(a, b []float32) (d, scale float64) {
	for i := range b {
		x := float64(a[i])
		if math.IsNaN(x) || math.IsInf(x, 0) {
			d = math.Inf(1)
		} else {
			d = math.Max(d, math.Abs(x-float64(b[i])))
		}
		scale = math.Max(scale, math.Abs(float64(b[i])))
	}
	return d, scale
}

// kv8TopKey is the key query head h attends most to over an int8 cache's dequantized K (float64 scores).
func kv8TopKey(q []float32, kq []int8, ks []float32, h, hd, nKV, G int, scale float32) int {
	kvh, kvDim := h/G, nKV*hd
	best, top := math.Inf(-1), 0
	for s := range len(kq) / kvDim {
		var dot float64
		for d := range hd {
			dot += float64(q[h*hd+d]) * float64(kq[s*kvDim+kvh*hd+d]) * float64(ks[s*nKV+kvh])
		}
		if dot*float64(scale) > best {
			best, top = dot*float64(scale), s
		}
	}
	return top
}

// TestKV8_extraSlotPricedAtTheDeliveredPrecision: ctxForSlots prices each further slot (cudaExtraSlotKVBytes) at the precision the build
// allocates, which is what ResolveKV delivers, never the raw --kv request: a load asking int8 that the CUDA policy holds at f32 (a drafter)
// builds f32 pools, and its extra slots must be priced f32, or the context search sizes slots a quarter of their real cost. No device: the
// models load on the CPU, and only the CUDA pricing is read.
func TestKV8_extraSlotPricedAtTheDeliveredPrecision(t *testing.T) {
	dir := "../../testdata/llama-tiny"
	for _, c := range []struct {
		name string
		o    decoder.Options
		want string // the precision the extra slot must be priced at
	}{
		{"int8 delivered", decoder.Options{KVPrecision: "i8"}, "i8"},
		{"int8 asked with a drafter (held at f32)", decoder.Options{KVPrecision: "i8", Drafter: true}, "f32"},
		{"f16", decoder.Options{KVPrecision: "f16"}, "f16"},
	} {
		m, err := decoder.Load(dir, c.o)
		if err != nil {
			t.Fatal(err)
		}
		want := map[string]int64{"f32": m.CUDAKVSlotBytes(128, false, false), "f16": m.CUDAKVSlotBytes(128, true, false), "i8": m.CUDAKVSlotBytes(128, false, true)}[c.want]
		if got := cudaExtraSlotKVBytes(m, 128); got != want {
			t.Errorf("%s: an extra slot is priced %d B; the build allocates %s, %d B (delivered %q)", c.name, got, c.want, want, m.KVDeliveredFor("cuda"))
		}
		m.Close()
	}
}

// TestKV8_exactInputIdentity_fusedPrefill is S6's G1 (docs/measurements/cuda-kv-i8-2026-11/s6-registration.md): every tensor-core prefill tile's
// _kv8 twin, on an int8 cache, equals its _kv16 twin bit for bit on an f16 cache holding f16(code * scale) of the same rows: the int8 twin stages each
// element as that f16 value and runs the rest of the tile as the f16 build does. The int8 rows come from a real prefill of the int8 resident (its
// store's codes and scales, read back), the f16 cache from those rows dequantized and rounded to nearest even on the host. The cases put a ragged
// last key tile (the scalar staging) beside full tiles (the vector staging).
func TestKV8_exactInputIdentity_fusedPrefill(t *testing.T) {
	requireCUDADevice(t)
	for _, fx := range []string{"llama-attnfa-tiny", "~qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"} {
		t.Run(filepath.Base(fx), func(t *testing.T) {
			// The 0.5B is the only hd64 case, so on a box with a device its absence fails the test instead of skipping hd64's tiles.
			if _, err := os.Stat(kv8Path(fx)); err != nil {
				t.Fatalf("no %s (the hd64 tiles' only fixture): %v", fx, err)
			}
			const nTot = 300
			m8, r8 := kv8Load(t, fx, "i8", decoder.Options{ResidentContext: 1024})
			_, r16 := kv8Load(t, fx, "f16", decoder.Options{ResidentContext: 1024})
			Ly := &r8.layers[0]
			nH, hd, kvDim, nKV := r8.nH, Ly.hd, Ly.kvDim, Ly.nKV
			type kernel struct {
				name    string
				p8, p16 Pipeline
				bm, bn  int // query-tile and key-tile rows; 0 for the 64x64 kernel
			}
			var ks []kernel
			switch hd {
			case 64:
				ks = []kernel{{"attn_fused hd64", r8.bAttnFused64, r16.bAttnFused64, 0, 0}, {"attn_fused_bm32x64 hd64", r8.bAttnBM32x64hd64, r16.bAttnBM32x64hd64, 32, 64},
					{"attn_fused_bm32x32 hd64", r8.bAttnBM32x32hd64, r16.bAttnBM32x32hd64, 32, 32}, {"attn_fused_bm128 hd64", r8.bAttnBM128hd64, r16.bAttnBM128hd64, 128, 64}}
			case 128:
				ks = []kernel{{"attn_fused hd128", r8.bAttnFused128, r16.bAttnFused128, 0, 0}, {"attn_fused_bm32x64 hd128", r8.bAttnBM32x64hd128, r16.bAttnBM32x64hd128, 32, 64},
					{"attn_fused_bm32x32 hd128", r8.bAttnBM32x32hd128, r16.bAttnBM32x32hd128, 32, 32}, {"attn_fused_bm128 hd128", r8.bAttnBM128hd128, r16.bAttnBM128hd128, 128, 64}}
			default:
				t.Fatalf("no fused tile for hd=%d", hd)
			}
			for _, k := range ks {
				if k.p8 == (Pipeline{}) || k.p16 == (Pipeline{}) {
					t.Fatalf("%s did not load on both arms (i8 %v, f16 %v): a missing kernel is not a pass", k.name, k.p8 != (Pipeline{}), k.p16 != (Pipeline{}))
				}
			}
			// The int8 rows: a real prefill of the int8 resident, read back as codes and scales.
			_, _, _, _, _, _, vocab := m8.Dims()
			embs := make([][]float32, nTot)
			for i := range embs {
				embs[i] = m8.EmbedResidentForTest(3 + (37*i)%(vocab-3))
			}
			if _, err := r8.PrefillLast(context.Background(), embs, 0); err != nil {
				t.Fatalf("int8 prefill: %v", err)
			}
			var k8, v8 []int8
			var ksc, vsc []float32
			if err := r8.do(func() error {
				if e := r8.stream.Sync(); e != nil {
					return e
				}
				var e error
				k8, v8, ksc, vsc, e = r8.kv8RowsForTest(r8.kvSlot, 0, 0, nTot)
				return e
			}); err != nil {
				t.Fatal(err)
			}
			deq := func(c []int8, sc []float32) []float32 {
				out := make([]float32, len(c))
				for i := range c {
					p, h := i/kvDim, (i%kvDim)/hd
					out[i] = linalg.F16ToF32(f32ToF16RNE(float32(c[i]) * sc[p*nKV+h]))
				}
				return out
			}
			if err := r16.UploadKV(0, 0, deq(k8, ksc), deq(v8, vsc)); err != nil {
				t.Fatal(err)
			}
			for _, tc := range []struct{ startPos, M int }{{0, 70}, {0, 300}, {37, 200}, {100, 129}} {
				q := kv16Representable(tc.M*nH*hd, uint32(13+tc.M))
				for _, k := range ks {
					var out [2][]float32
					for i, r := range []*cudaResident{r8, r16} {
						p := k.p8
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
							args := r.kvScaled(0, Arg(qB), Arg(r.kvK(0)), Arg(r.kvV(0)), Arg(r.kvT(0)), gpu.ArgValue(int32(nH)), gpu.ArgValue(int32(nKV)),
								gpu.ArgValue(int32(hd)), gpu.ArgValue(int32(tc.startPos)), gpu.ArgValue(r.attnScale), gpu.ArgValue(Ly.window),
								gpu.ArgValue(int32(tc.M)), Arg(cB), r.sinkArg(0))
							cfg := LaunchConfig{GridX: uint32(nH), GridY: uint32((tc.M + attnFusedBM - 1) / attnFusedBM), GridZ: 1,
								BlockX: attnFusedThreads, BlockY: 1, BlockZ: 1, SharedMemBytes: attnFusedShmem(hd)}
							if k.bm > 0 {
								cfg.GridY = uint32((tc.M + k.bm - 1) / k.bm)
								cfg.BlockX = uint32(k.bm / 16 * 32)
								cfg.SharedMemBytes = uint32(2 * (k.bn*(hd+attnFusedKPAD) + hd*(k.bn+attnFusedKPAD)))
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
						t.Errorf("%s startPos=%d M=%d: the int8 arm's output is all zero, so the comparison is vacuous", k.name, tc.startPos, tc.M)
					}
					if diff != 0 {
						t.Errorf("%s startPos=%d M=%d hd=%d: %d/%d elements differ between the int8 tile and the f16 tile on f16(code*scale)", k.name, tc.startPos, tc.M, hd, diff, len(out[0]))
					}
				}
			}
		})
	}
}

// TestMoEExpertsUnpinned_identityAndPricing is MEM10's opt-in at tier 0 (docs/measurements/mem10-2026-11/ab-registration.md): on qwen3moe-tiny with
// paged experts, a load with MoEExpertsUnpinned builds every expert stack from a pageable file mapping and none pinned, its teacher-forced logits
// equal the pinned load's bit for bit (the slots hold the same bytes), and the CUDA build demand drops the pinned copy from its price. The flag
// off builds no pageable stack.
func TestMoEExpertsUnpinned_identityAndPricing(t *testing.T) {
	requireCUDADevice(t)
	const fx, steps = "qwen3moe-tiny", 24
	mA, rA := kv8Load(t, fx, "f32", decoder.Options{MoECacheExperts: true})
	_, rB := kv8Load(t, fx, "f32", decoder.Options{MoECacheExperts: true, MoEExpertsUnpinned: true})
	stacks := 0
	for l := range rA.layers {
		if rA.layers[l].expGU.srcW != nil {
			stacks += 2
		}
	}
	if stacks == 0 {
		t.Fatalf("%s built no expert cache: the comparison is vacuous", fx)
	}
	if rA.pageableStacks != 0 || rB.pageableStacks != stacks {
		t.Fatalf("pageable stacks: pinned load %d (want 0), unpinned load %d (want %d)", rA.pageableStacks, rB.pageableStacks, stacks)
	}
	_, _, _, _, _, _, vocab := mA.Dims()
	for i := range steps {
		e := mA.EmbedResidentForTest(3 + (37*i)%(vocab-3))
		a, err := rA.Forward(e, i)
		if err != nil {
			t.Fatal(err)
		}
		b, err := rB.Forward(e, i)
		if err != nil {
			t.Fatal(err)
		}
		for j := range a {
			if math.Float32bits(a[j]) != math.Float32bits(b[j]) {
				t.Fatalf("position %d: logit %d differs (pinned %g, unpinned %g)", i, j, a[j], b[j])
			}
		}
	}
	// The batched prefill and the speculative verify step (ForwardN) upload slots the same way: both arms, the same rows, the same bits.
	const pre, ver = 32, 4
	embs := make([][]float32, pre+ver)
	for i := range embs {
		embs[i] = mA.EmbedResidentForTest(5 + (41*i)%(vocab-5))
	}
	same := func(what string, a, b []float32) {
		t.Helper()
		if len(a) == 0 || len(a) != len(b) {
			t.Fatalf("%s: %d and %d logits", what, len(a), len(b))
		}
		for j := range a {
			if math.Float32bits(a[j]) != math.Float32bits(b[j]) {
				t.Fatalf("%s: logit %d differs (pinned %g, unpinned %g)", what, j, a[j], b[j])
			}
		}
	}
	pa, err := rA.PrefillLast(context.Background(), embs[:pre], 0)
	if err != nil {
		t.Fatalf("prefill: %v", err)
	}
	pb, err := rB.PrefillLast(context.Background(), embs[:pre], 0)
	if err != nil {
		t.Fatalf("prefill: %v", err)
	}
	same("batched prefill", pa, pb)
	if ok, why := rA.VerifyPath(); !ok {
		t.Fatalf("no speculative verify path on %s: %s", fx, why)
	}
	va, err := rA.ForwardN(embs[pre:], pre)
	if err != nil {
		t.Fatalf("verify: %v", err)
	}
	vb, err := rB.ForwardN(embs[pre:], pre)
	if err != nil {
		t.Fatalf("verify: %v", err)
	}
	if len(va) != ver || len(vb) != ver {
		t.Fatalf("verify returned %d and %d rows, want %d", len(va), len(vb), ver)
	}
	for i := range va {
		same(fmt.Sprintf("verify row %d", i), va[i], vb[i])
	}
	// Pricing, both terms: the pinned copy of a resident cache, and a .gguf build's peak.
	q := decoder.BuildQuery{Options: decoder.Options{Backend: "cuda", MoECacheExperts: true}, ExpertCache: true, FeaturesCovered: true,
		MoELayers: 2, NumExperts: 8, ExpertIntermediate: 64, Hidden: 32, GGUF: true, WeightBytes: 10 << 30, ExpertBytes: 7 << 30}
	pinned := buildDemand(q)
	if pinned.PinnedExpertBytes <= 0 || pinned.HostBuildPeak != 27<<30 {
		t.Fatalf("the pinned load prices a %d-byte copy and a %d-byte build peak: the comparison below is vacuous", pinned.PinnedExpertBytes, pinned.HostBuildPeak)
	}
	q.Options.MoEExpertsUnpinned = true
	if got := buildDemand(q); got.PinnedExpertBytes != 0 || got.HostBuildPeak != 20<<30 {
		t.Errorf("the unpinned load prices a %d-byte pinned copy and a %d-byte build peak (want 0 and %d)", got.PinnedExpertBytes, got.HostBuildPeak, int64(20<<30))
	}
}

// TestKV8_everyFusedPrefillLaunchPassesScales reads prefill.go: every launch of a tensor-core prefill tile (attn_fused, attn_fused_bm) passes
// the layer's scale pools through kvScaled, since the identity test launches each tile itself and would not see a dispatch site that forgot
// them (S6, Gemini's review of #785). Eight sites: two attn_fused head dims and three attn_fused_bm shapes at two head dims.
func TestKV8_everyFusedPrefillLaunchPassesScales(t *testing.T) {
	b, err := os.ReadFile("prefill.go")
	if err != nil {
		t.Fatal(err)
	}
	launch := regexp.MustCompile(`r\.launch\(r\.(bAttnFused(?:64|128)|bAttnBM\w+), ([^\n]*)`)
	sites := launch.FindAllStringSubmatch(string(b), -1)
	if len(sites) != 8 {
		t.Fatalf("found %d tensor-core tile launches in prefill.go, expected 8: the scan no longer sees them all", len(sites))
	}
	for _, m := range sites {
		if !strings.Contains(m[2], "r.kvScaled(l, ") {
			t.Errorf("prefill.go launches %s without r.kvScaled(l, ...): an int8 cache's tile would read no scales (%s)", m[1], m[0])
		}
	}
}

// TestKV8_prefillTakesTheTileOnTheTinyFixture is S6's route gate at tier 0 (s6-registration.md, "Route"): on llama-attnfa-tiny at -kv i8, a
// prefill above the fast-prefill floor launches the tensor-core tile and no attn_batched (every layer is tile-served: no window, hd 128).
func TestKV8_prefillTakesTheTileOnTheTinyFixture(t *testing.T) {
	requireCUDADevice(t)
	t.Setenv("GOINFER_CUDA_FAST_PREFILL_FLOOR", "0")
	m, r := kv8Load(t, "llama-attnfa-tiny", "i8", decoder.Options{ResidentContext: 1024})
	_, _, _, _, _, _, vocab := m.Dims()
	embs := make([][]float32, 300)
	for i := range embs {
		embs[i] = m.EmbedResidentForTest(3 + (37*i)%(vocab-3))
	}
	if _, err := r.PrefillLast(context.Background(), embs, 0); err != nil {
		t.Fatalf("int8 prefill: %v", err)
	}
	if r.fastAttnLaunches == 0 || r.exactAttnLaunches != 0 {
		t.Errorf("int8 prefill launched the tile %d times and attn_batched %d times: every layer should take the tile", r.fastAttnLaunches, r.exactAttnLaunches)
	}
}

// TestMoEExpertsUnpinned_refusesMemoryBackedOrMissingDirs: the packed-experts file is refused on tmpfs (its pages would still be RAM) and with no
// model path (never the working directory), and a load with the flag whose model lives on tmpfs declines by name rather than holding the copy.
func TestMoEExpertsUnpinned_refusesMemoryBackedOrMissingDirs(t *testing.T) {
	if _, _, err := pageableStackSources("", 4096, 4096); err == nil || !strings.Contains(err.Error(), "no file path") {
		t.Errorf("no model path: %v", err)
	}
	var fs syscall.Statfs_t
	if err := syscall.Statfs("/dev/shm", &fs); err != nil || fs.Type != tmpfsMagic {
		t.Skipf("/dev/shm is not tmpfs here (%v)", err)
	}
	if _, _, err := pageableStackSources("/dev/shm", 4096, 4096); err == nil || !strings.Contains(err.Error(), "tmpfs") {
		t.Errorf("tmpfs: %v", err)
	}
	requireCUDADevice(t)
	dir, err := os.MkdirTemp("/dev/shm", "unpinned-")
	if err != nil {
		t.Skip(err)
	}
	defer os.RemoveAll(dir)
	src := "../../testdata/qwen3moe-tiny"
	ents, err := os.ReadDir(src)
	if err != nil {
		t.Skipf("no fixture: %v", err)
	}
	for _, e := range ents {
		b, err := os.ReadFile(filepath.Join(src, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, e.Name()), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	m, err := decoder.Load(dir, decoder.Options{Backend: "cuda", Quant: "int4", MoECacheExperts: true, MoEExpertsUnpinned: true, ResidentKVSlots: 1})
	if err != nil {
		if !strings.Contains(err.Error(), "tmpfs") {
			t.Fatalf("the load failed without naming tmpfs: %v", err)
		}
		return
	}
	defer m.Close()
	if r, ok := m.ResidentForwardForTest().(*cudaResident); ok && r != nil {
		t.Fatalf("a model on tmpfs built a resident with --moe-experts-unpinned (%d pageable stacks)", r.pageableStacks)
	}
	if why := m.ResidentDecline(); !strings.Contains(why, "tmpfs") {
		t.Errorf("the decline does not name tmpfs: %s", why)
	}
}
