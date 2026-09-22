//go:build gpu && goinfer_testhooks

package gpu

import (
	"os"
	"sort"
	"testing"
	"time"

	"github.com/townsendmerino/aikit/linalg"
	"github.com/townsendmerino/goinfer/decoder"
)

// TestGEMVLMHeadIsolation is G38's decisive step: TestGEMVNSweep found the GEMV KERNEL clean and
// on-roofline (874us at N=151936, ~267 GB/s) when isolated to its own process with a synthetic
// buffer -- yet the SAME shape, SAME N, inside a loaded real model (TestDecodeArgmaxHeadroom) costs
// 8512us, a ~10x gap the kernel itself does not explain. This isolates the remaining variable: is
// it the REAL model's OWN resident lm-head buffer specifically (co-residency/allocator state from
// the rest of the model's weights and KV cache), or is it something about running the GEMV as the
// LAST dispatch of a big multi-layer pass (Run()'s own recording shape) rather than standalone?
//
// Calls the model's OWN r.lmHead buffer directly via ctx.MatmulW8A8GEMV, in its own
// encoder/pass/submit, with a synthetic activation -- BEFORE any Forward() has run (so nothing else
// in the process has touched the allocator except the load itself), and again AFTER 200 real decode
// steps (so the KV cache and everything else is now also resident, matching TestDecodeArgmaxHeadroom's
// own state) -- same buffer, same call shape, two points in the model's own lifetime.
//
//	GOINFER_DECODE_GGUF=~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
//	  go test -tags 'gpu goinfer_testhooks' ./gpu/ -run TestGEMVLMHeadIsolation -v
func TestGEMVLMHeadIsolation(t *testing.T) {
	requireHeavyModel(t)
	path := os.Getenv("GOINFER_DECODE_GGUF")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("model not found: %s", path)
	}
	m, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int8int8"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	if !m.ResidentActive() {
		t.Skipf("model not GPU-resident")
	}
	rd, ok := m.ResidentForwardForTest().(*residentDecoder)
	if !ok {
		t.Skipf("resident forward is not the gpu DecodeRunner (%T)", m.ResidentForwardForTest())
	}
	r := rd.runner
	c := rd.c
	rm, ok := r.LMHeadForTest().(*ResidentW8A8)
	if !ok {
		t.Skipf("lmHead is not *ResidentW8A8 (%T) -- this model's quant path is not the one G38 measured", r.LMHeadForTest())
	}
	_, _, _, _, _, _, vocab := m.Dims()
	hidden, _, _, _, _, _, _ := m.Dims()

	act := make([]float32, hidden)
	for i := range act {
		act[i] = float32(i%7-3) * 0.1
	}
	aq, aScales := linalg.QuantizeRowsInt8(act, 1, hidden)

	median := func(xs []float64) float64 {
		s := append([]float64(nil), xs...)
		sort.Float64s(s)
		return s[len(s)/2]
	}
	measure := func(label string) float64 {
		for range 5 {
			if _, e := c.MatmulW8A8GEMV(aq, aScales[0], rm); e != nil {
				t.Fatalf("%s warmup: %v", label, e)
			}
		}
		const reps = 30
		times := make([]float64, reps)
		for i := range reps {
			t0 := time.Now()
			if _, e := c.MatmulW8A8GEMV(aq, aScales[0], rm); e != nil {
				t.Fatalf("%s: %v", label, e)
			}
			times[i] = float64(time.Since(t0).Microseconds())
		}
		med := median(times)
		t.Logf("%s: median %.1f us (%.1f GB/s)", label, med, float64(vocab*hidden)/(med*1e-6)/1e9)
		return med
	}

	fresh := measure("standalone GEMV on r.lmHead, RIGHT AFTER LOAD (nothing else resident yet)")

	rd.Reset()
	rng := make([]float32, hidden)
	for i := range rng {
		rng[i] = float32((i*37)%23-11) * 0.07
	}
	for i := range 200 {
		if _, e := rd.Forward(rng, i); e != nil {
			t.Fatalf("decode step %d: %v", i, e)
		}
	}
	warm := measure("standalone GEMV on r.lmHead, AFTER 200 real decode steps")

	t.Logf("ratio (after/fresh): %.2fx", warm/fresh)
}
