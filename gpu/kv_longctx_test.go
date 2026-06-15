//go:build gpu

package gpu

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestKVLongCtx sweeps decode tok/s vs context depth for int8 / f16 / f32 KV on the
// residency path — the residual the GPU int8-KV + f16-KV tasks left open: the
// attention kernel is KV-read-bound, so at long context fewer KV bytes (f32 > f16 >
// i8) should make decode FASTER, but the shipped fit test only measured ~1k (0.96×,
// weight-stream-bound). This maps the curve at 1k/4k/8k to locate any crossover —
// far cheaper than a single 16k point (prefill is O(L²), no batched prefill). One
// model load per precision, reused across depths.
//
//	go test -tags gpu -run TestKVLongCtx -v -timeout 5400s
func TestKVLongCtx(t *testing.T) {
	path := os.Getenv("GOINFER_GPU_7B")
	if path == "" {
		home, _ := os.UserHomeDir()
		path = filepath.Join(home, "models", "qwen2.5-7b-instruct-q4_k_m.gguf")
	}
	if os.Getenv("GOINFER_KV_LONGCTX") == "" {
		t.Skip("long-context KV measurement (~70 min) — set GOINFER_KV_LONGCTX=1 to run")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no 7B GGUF at %s (set GOINFER_GPU_7B)", path)
	}
	depths := []int{1024, 4096, 8192}
	precs := []string{"i8", "f16", "f32"}
	const decodeN = 32

	// tps[prec][depth] = steady-state decode tok/s.
	tps := map[string]map[int]float64{}
	for _, p := range precs {
		tps[p] = map[int]float64{}
	}

	for _, prec := range precs {
		m, err := decoder.Load(path, decoder.Options{Backend: "webgpu", Quant: "int4", KVPrecision: prec})
		if err != nil {
			t.Fatalf("Load(%s): %v", prec, err)
		}
		if !m.ResidentActive() {
			m.Close()
			t.Skipf("7B not GPU-resident in int4 (kv=%s)", prec)
		}
		_, _, _, _, _, _, vocab := m.Dims()
		for _, d := range depths {
			prompt := make([]int, d)
			for i := range prompt {
				prompt[i] = (i*7 + 13) % vocab
			}
			ch, _ := m.Generate(context.Background(), prompt, decodeN, decoder.SamplingParams{Temperature: 0})
			var first time.Time
			n := 0
			for range ch {
				if n == 0 {
					first = time.Now() // prefill done; steady-state decode begins
				}
				n++
			}
			if n > 1 {
				tps[prec][d] = float64(n-1) / time.Since(first).Seconds()
			}
			t.Logf("  %-3s @ depth %5d: %.2f tok/s", prec, d, tps[prec][d])
		}
		m.Close()
	}

	t.Logf("=== decode tok/s vs context depth (7B int4, RTX 2070 SUPER) ===")
	t.Logf("  depth |   i8   |  f16   |  f32   | i8/f32 | f16/f32")
	for _, d := range depths {
		i8, f16, f32 := tps["i8"][d], tps["f16"][d], tps["f32"][d]
		r8, r16 := 0.0, 0.0
		if f32 > 0 {
			r8, r16 = i8/f32, f16/f32
		}
		t.Logf("  %5d | %6.2f | %6.2f | %6.2f | %5.2fx | %5.2fx", d, i8, f16, f32, r8, r16)
	}
	t.Logf("(>1 i8/f32 at depth D ⇒ int8 KV-read-bound win realized by D; trend toward 1 ⇒ crossover above 8k)")
}
