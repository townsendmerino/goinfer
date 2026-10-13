//go:build cuda && goinfer_testhooks

package cuda

import (
	"context"
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"math"
	"os"
	"slices"
	"strconv"
	"testing"
	"time"

	gpu "github.com/townsendmerino/aikit/gpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestKV16_wholeTokenDecodeAB is the in-process whole-token A/B of docs/tasks/task-cuda-f16-kv-2026-10.md's speed question: one process, both caches
// resident side by side, the same prompt, decode steps timed ABBA-interleaved. It gives a direction and a size per cell; the served check is the
// instrument for any claim. Exploratory by day: KV16_AB=1, KV16_AB_MODELS and KV16_AB_DEPTHS narrow it.
//
//	KV16_AB=1 go test -tags 'cuda goinfer_testhooks' ./cuda/ -run TestKV16_wholeTokenDecodeAB -v -timeout 20m
func TestKV16_wholeTokenDecodeAB(t *testing.T) {
	if os.Getenv("KV16_AB") == "" {
		t.Skip("set KV16_AB=1")
	}
	models := []string{"qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", "phi3-mini-4k-gguf/Phi-3-mini-4k-instruct-q4.gguf"}
	if v := os.Getenv("KV16_AB_MODELS"); v != "" {
		models = []string{v}
	}
	for _, model := range models {
		t.Run(model, func(t *testing.T) {
			m32, _, r32, r16 := kv16PairOf(t, model, 4200)
			const steps, reps = 24, 6
			for _, depth := range []int{128, 512, 2048, 3900} {
				embs := kv16Prompt(m32, depth)
				// One prefill per arm and depth: the timed steps rewrite positions depth..depth+steps-1 with the same values each time (a sequential-prefill
				// model like Phi-3 would otherwise spend its minutes in the untimed part).
				for _, r := range []*cudaResident{r32, r16} {
					if _, err := r.PrefillLast(context.Background(), embs, 0); err != nil {
						t.Fatal(err)
					}
				}
				run := func(r *cudaResident) float64 {
					e := m32.EmbedResidentForTest(7)
					t0 := time.Now()
					for i := range steps {
						if _, err := r.Forward(e, depth+i); err != nil {
							t.Fatal(err)
						}
					}
					return float64(time.Since(t0).Microseconds()) / steps
				}
				run(r32)
				run(r16) // warm both
				var a, b []float64
				for i := range reps {
					if i%2 == 0 {
						a, b = append(a, run(r32)), append(b, run(r16))
					} else {
						b, a = append(b, run(r16)), append(a, run(r32))
					}
				}
				slices.Sort(a)
				slices.Sort(b)
				ma, mb := a[len(a)/2], b[len(b)/2]
				fmt.Printf("[kv16-ab] %s depth %4d: f32 %.0f us/token, f16 %.0f us/token, f16/f32 rate %.3f  (n=%d, ABBA)\n", model, depth, ma, mb, ma/mb, reps)
			}
		})
	}
}

// TestKV16_singleArmDecode times one arm alone (KV16_ARM=f32|f16), for a model whose two copies do not fit the card together (Phi-3 at int8int8 is 3.7 GB of
// weights). The caller alternates the two processes, ABBA; it is a direction, as the paired test above is.
func TestKV16_singleArmDecode(t *testing.T) {
	arm := os.Getenv("KV16_ARM")
	if arm == "" {
		t.Skip("set KV16_ARM=f32|f16 and KV16_AB_MODELS")
	}
	model := os.Getenv("KV16_AB_MODELS")
	requireCUDADevice(t)
	path := modelPath(model)
	m, err := decoder.Load(path, decoder.Options{Backend: "cuda", Quant: kv16Quant(model), ActQuantGroup: kv16ActGroup(model), KVPrecision: arm, ResidentContext: 4200})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("no resident: %s", m.ResidentDecline())
	}
	embs := kv16Prompt(m, 3900)
	pos := 0
	for _, depth := range []int{512, 2048, 3900} {
		for ; pos < depth; pos++ { // sequential: the per-32 activation quantizer has no batched prefill
			if _, err := r.Forward(embs[pos], pos); err != nil {
				t.Fatal(err)
			}
		}
		e := m.EmbedResidentForTest(7)
		var us []float64
		for rep := range 8 {
			t0 := time.Now()
			for i := range 24 {
				if _, err := r.Forward(e, depth+i); err != nil {
					t.Fatal(err)
				}
			}
			if rep > 0 {
				us = append(us, float64(time.Since(t0).Microseconds())/24)
			}
		}
		slices.Sort(us)
		fmt.Printf("[kv16-arm] %s -kv %s depth %4d: %.0f us/token (median of 7)\n", model, arm, depth, us[len(us)/2])
	}
}

// TestKV16_profileSteps runs a few decode steps on synthetic K/V uploaded at KV16_DEPTH (no prefill), for `ncu` to capture the attention kernels of one arm
// (KV16_ARM=f32|f16, KV16_AB_MODELS). It asserts nothing.
func TestKV16_profileSteps(t *testing.T) {
	arm := os.Getenv("KV16_ARM")
	depth, _ := strconv.Atoi(os.Getenv("KV16_DEPTH"))
	if arm == "" || depth == 0 {
		t.Skip("set KV16_ARM, KV16_DEPTH, KV16_AB_MODELS")
	}
	model := os.Getenv("KV16_AB_MODELS")
	requireCUDADevice(t)
	m, err := decoder.Load(modelPath(model), decoder.Options{Backend: "cuda", Quant: kv16Quant(model), ActQuantGroup: kv16ActGroup(model), KVPrecision: arm, ResidentContext: 4200})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("no resident: %s", m.ResidentDecline())
	}
	for l := range r.layers {
		k := kv16Representable(depth*r.layers[l].kvDim, uint32(31+l))
		v := kv16Representable(depth*r.layers[l].kvDim, uint32(77+l))
		if err := r.UploadKV(l, 0, k, v); err != nil {
			t.Fatal(err)
		}
	}
	e := m.EmbedResidentForTest(7)
	for i := range 3 {
		if _, err := r.Forward(e, depth+i); err != nil {
			t.Fatal(err)
		}
	}
	fmt.Printf("[kv16-profile] %s -kv %s depth %d: 3 steps done (split-KV launches %d, flash %d)\n", model, arm, depth, r.skLaunches, r.faLaunches)
}

// TestKV16_readerKernelTime times each decode attention reader on synthetic K/V at KV16_DEPTH for one arm (KV16_ARM, KV16_AB_MODELS): the whole-token gap's
// location. One call is timed by 200 back-to-back launches between two stream syncs. It asserts nothing.
func TestKV16_readerKernelTime(t *testing.T) {
	arm := os.Getenv("KV16_ARM")
	depth, _ := strconv.Atoi(os.Getenv("KV16_DEPTH"))
	if arm == "" || depth == 0 {
		t.Skip("set KV16_ARM, KV16_DEPTH, KV16_AB_MODELS")
	}
	model := os.Getenv("KV16_AB_MODELS")
	requireCUDADevice(t)
	m, err := decoder.Load(modelPath(model), decoder.Options{Backend: "cuda", Quant: kv16Quant(model), ActQuantGroup: kv16ActGroup(model), KVPrecision: arm, ResidentContext: 4200})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	r, ok := m.ResidentForwardForTest().(*cudaResident)
	if !ok {
		t.Fatalf("no resident: %s", m.ResidentDecline())
	}
	Ly := &r.layers[0]
	if err := r.UploadKV(0, 0, kv16Representable(depth*Ly.kvDim, 5), kv16Representable(depth*Ly.kvDim, 9)); err != nil {
		t.Fatal(err)
	}
	if err := r.do(func() error { return gpu.Upload(r.qB, kv16Representable(r.nH*Ly.hd, 3)) }); err != nil {
		t.Fatal(err)
	}
	pos, nWin := depth-1, depth
	readers := []struct {
		name string
		run  func() error
	}{
		{"glue attention", func() error {
			return r.launch(r.fAttn, LaunchConfig{GridX: uint32(r.nH), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((nWin + 128) * 4)},
				Arg(r.qB), Arg(r.kvK(0)), Arg(r.kvV(0)), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)),
				gpu.ArgValue(int32(nWin)), gpu.ArgValue(r.attnScale), gpu.ArgValue(Ly.window), Arg(r.cctx))
		}},
		{"attn_batched M=1", func() error {
			return r.launch(r.bAttn, LaunchConfig{GridX: uint32(r.nH), GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((nWin + 128) * 4)},
				Arg(r.qB), Arg(r.kvK(0)), Arg(r.kvV(0)), gpu.ArgValue(int32(r.nH)), gpu.ArgValue(int32(Ly.nKV)), gpu.ArgValue(int32(Ly.hd)),
				gpu.ArgValue(int32(pos)), gpu.ArgValue(r.attnScale), gpu.ArgValue(Ly.window), gpu.ArgValue(int32(1)), Arg(r.cctx), r.sinkArg(0))
		}},
		{"split-KV", func() error { return r.splitKVAttnDecode(0, pos) }},
	}
	if r.faSplit > 0 && r.faEligible(0) {
		readers = append(readers, struct {
			name string
			run  func() error
		}{"flash-decode lane", func() error { return r.flashDecodeAttn(0, pos) }})
	}
	fmt.Printf("[kv16-reader] %s -kv %s depth %d: nH=%d nKV=%d hd=%d prefillReady=%v splitkvAttn=%v lane=%v\n", model, arm, depth, r.nH, Ly.nKV, Ly.hd, r.prefillReady, r.splitkvAttn, r.faSplit > 0 && r.faEligible(0))
	for _, rd := range readers {
		if rd.name == "attn_batched M=1" && r.bAttn == (Pipeline{}) {
			continue
		}
		var us float64
		for rep := range 4 {
			if err := r.do(func() error {
				if e := r.stream.Sync(); e != nil {
					return e
				}
				t0 := time.Now()
				for range 200 {
					if e := rd.run(); e != nil {
						return e
					}
				}
				if e := r.stream.Sync(); e != nil {
					return e
				}
				if rep > 0 {
					us += float64(time.Since(t0).Microseconds()) / 200 / 3
				}
				return nil
			}); err != nil {
				t.Fatalf("%s: %v", rd.name, err)
			}
		}
		// The context the reader writes, hashed: the same inputs on the other arm (another process, for a model whose two copies do not fit the card
		// together) must print the same hash, which is the exact-input identity for that geometry.
		ctx := make([]float32, r.nH*Ly.hd)
		if err := r.do(func() error { return gpu.Download(r.cctx, ctx) }); err != nil {
			t.Fatal(err)
		}
		h := fnv.New64a()
		for _, v := range ctx {
			_ = binary.Write(h, binary.LittleEndian, math.Float32bits(v))
		}
		fmt.Printf("[kv16-reader]   %-18s %7.1f us/call  ctx hash %016x\n", rd.name, us, h.Sum64())
	}
}
