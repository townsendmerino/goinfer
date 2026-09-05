//go:build darwin && goinfer_testhooks

package metal

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
)

// TestSpecVerifyCurveMetal is the Metal side of cuda/spec_verify_ceiling_test.go (D1/P10): the
// batched-verify curve T(M) = W + C*M a resident Metal target needs so P10's speedup formula
// (decode_ms*(1+accepted)/(draft_ms+verify_ms(k))) can be projected on this backend instead of
// only CUDA's. See docs/prompts/metal-verify-curve.md for the full task.
//
// CRITICAL, and the reason this needs stating before any number below: `PrefillLast` — the batched
// primitive this test times — DECLINES BY DEFAULT on Metal (metal/backend.go), because its f16-MMA
// activation path is NOT bit-identical to decode's int8 path (54% stream divergence measured,
// §A2-Metal, when that gate last ran). GOINFER_METAL_BATCHED_PREFILL=1 forces it on for exactly the
// "measurement/TTFT-at-the-cost-of-exactness" use this test is. So every number this test produces
// characterizes a kernel that is NOT currently usable as P10's verify oracle on Metal — P10's own
// design requires the verify step to reproduce sequential greedy exactly (00-core's lossless
// contract). A real Metal P10 leg needs that bit-identity gap closed FIRST; this test answers "is
// the timing shape even worth it", not "is this safe to ship".
//
//	GOINFER_HEAVY_TESTS=1 go test ./metal -run TestSpecVerifyCurveMetal -v
func TestSpecVerifyCurveMetal(t *testing.T) {
	if os.Getenv("GOINFER_HEAVY_TESTS") == "" {
		t.Skip("set GOINFER_HEAVY_TESTS=1 (loads a 1.5B model)")
	}
	path := os.Getenv("GOINFER_METAL_MODEL")
	if path == "" {
		path = os.ExpandEnv("$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no fixture at %s (set GOINFER_METAL_MODEL)", path)
	}
	t.Setenv("GOINFER_METAL_BATCHED_PREFILL", "1") // see the caveat above — measurement-only

	m, err := decoder.Load(path, decoder.Options{Backend: "metal", Quant: "int4"})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()
	rf, ok := m.ResidentForwardForTest().(*metalResident)
	if !ok {
		t.Skip("metal resident not built for this model (declined — see stderr)")
	}
	_, _, _, _, _, _, vocab := m.Dims()
	emb := func(i int) []float32 { return m.EmbedResidentForTest((i*2654435761 + 1) % (vocab - 1)) }

	const depth = 1024 // matches cuda/spec_verify_ceiling_test.go's depth for comparability
	warm := make([][]float32, depth)
	for i := range warm {
		warm[i] = emb(i)
	}
	if _, e := rf.PrefillLast(context.Background(), warm, 0); e != nil {
		t.Fatalf("warm prefill (forced batched): %v", e)
	}

	timeIt := func(f func()) time.Duration {
		best := time.Hour
		for range 5 {
			t0 := time.Now()
			f()
			if d := time.Since(t0); d < best {
				best = d
			}
		}
		return best
	}

	// REAL decode_ms — via Forward (the shipped int8 pipelined-executor path), matching
	// cuda/spec_verify_ceiling_test.go's OWN reference method (it measures this as "one", a
	// separate quantity from the batched curve's T(1) — not derived from the fit). On Metal
	// this is expected to differ substantially from the batched curve's T(1): PrefillLast is
	// an entirely different kernel family (f16-MMA) from Forward's shipped int8 decode path,
	// unlike CUDA where the two are close enough that T(1)≈decode_ms is a fair simplification.
	realDecode := timeIt(func() {
		if _, e := rf.Forward(emb(5), depth); e != nil {
			t.Fatal(e)
		}
	})
	t.Logf("REAL decode_ms (Forward, shipped int8 path) @depth %d: %.3f ms", depth, float64(realDecode.Microseconds())/1000)

	type point struct {
		m int
		t time.Duration
	}
	var points []point
	for _, k := range []int{1, 2, 4, 6, 8, 10, 12, 16} {
		ek := make([][]float32, k)
		for i := range ek {
			ek[i] = emb(100 + i)
		}
		d := timeIt(func() {
			if _, e := rf.PrefillLast(context.Background(), ek, depth); e != nil {
				t.Fatal(e)
			}
		})
		points = append(points, point{k, d})
		t.Logf("M=%-3d T(M)=%.3f ms", k, float64(d.Microseconds())/1000)
	}

	// Least-squares fit T(M) = W + C*M over the measured points.
	var n, sumM, sumT, sumMM, sumMT float64
	for _, p := range points {
		mF, tF := float64(p.m), float64(p.t.Microseconds())/1000
		n++
		sumM += mF
		sumT += tF
		sumMM += mF * mF
		sumMT += mF * tF
	}
	c := (n*sumMT - sumM*sumT) / (n*sumMM - sumM*sumM)
	w := (sumT - c*sumM) / n
	t.Logf("fit: T(M) = %.3f + %.3f*M  (ms)", w, c)
	t.Logf("T(1) via the fit (batched-path M=1, NOT the same kernel as real decode) = %.3f ms", w+c)
	realDecodeMs := float64(realDecode.Microseconds()) / 1000
	t.Logf("ratio T(1)_fit / real_decode_ms = %.2fx — the gap IS the finding: PrefillLast's f16-MMA "+
		"path is never tuned for small M (nobody ships it there; it's declined by default), so its "+
		"per-call floor is far above one real int8 decode step", (w+c)/realDecodeMs)

	fmt.Printf("\nMETAL VERIFY CURVE — %s, int4, depth %d\nW=%.4f C=%.4f  real_decode_ms=%.4f\n",
		path, depth, w, c, realDecodeMs)
}
