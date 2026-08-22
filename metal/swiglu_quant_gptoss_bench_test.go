//go:build darwin && goinfer_testhooks

package metal

import (
	"testing"
	"time"
)

// BenchmarkSwigluQuantGptOss isolates swiglu_quant_gptoss's (metal/moe.go) own throughput at
// gpt-oss-20b's real expert intermediate size (2880, per docs/task-mxfp4-gptoss.md) and its real
// production dispatch shape (moe.go: mo.pActGptOss, (256, 256), hasBias=1, once per selected
// expert per layer).
func BenchmarkSwigluQuantGptOss(b *testing.B) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		b.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		b.Fatalf("compile: %v", err)
	}
	pipe, err := d.NewComputePipeline(lib, "swiglu_quant_gptoss")
	if err != nil {
		b.Fatalf("pipeline: %v", err)
	}

	const goI = 2880
	const nExp, slot = 1, 0
	rndf := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(i%23-11) * 0.7
		}
		return s
	}
	dG := NewBufferFloats(d, rndf(goI))
	dU := NewBufferFloats(d, rndf(goI))
	dDq := d.NewBufferBytes(goI)
	dDs := d.NewBufferLen(1)
	uI := NewBufferU32(d, uint32(goI))
	dBias := NewBufferFloats(d, rndf(nExp*2*goI))
	dIdx := NewBufferUint32s(d, []uint32{0})
	uSlot := NewBufferU32(d, uint32(slot))
	uHasBias := NewBufferU32(d, 1)
	uAlpha := NewBufferFloats(d, []float32{1.702})
	uLimit := NewBufferFloats(d, []float32{7.0})

	q_ := d.NewCommandQueue()
	const reps = 8
	run := func() {
		e := q_.Begin()
		for range reps {
			e.Dispatch(pipe, 256, 256, dG, dU, dDq, dDs, uI, dBias, dIdx, uSlot, uHasBias, uAlpha, uLimit)
		}
		e.End()
	}

	for range 4 {
		run()
	}
	b.ResetTimer()
	best := time.Hour
	for range b.N {
		t0 := time.Now()
		run()
		if dt := time.Since(t0); dt < best {
			best = dt
		}
	}
	b.ReportMetric(float64(best/time.Duration(reps))/float64(time.Nanosecond), "ns/dispatch(best)")
}
