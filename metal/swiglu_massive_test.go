//go:build darwin

package metal

import (
	"math"
	"testing"
)

// TestSwigluQuant_MassiveChannel isolates the BOS geglu bug (b04d799) to KERNEL vs DATA. The L0
// BOS trace found swiglu_quant emitting ~0 for the dominant channel (gate=12.14, up=-18.49, so
// geglu should be -224.4). This drives the SHIPPED swiglu_quant (allKernels) with exactly that
// value at one channel and small values elsewhere. If the dequantized output there is ~-224 the
// kernel is fine and the r.gu data was corrupted upstream; if it's ~0 the kernel itself drops the
// massive geglu channel.
func TestSwigluQuant_MassiveChannel(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pSw, err := d.NewComputePipeline(lib, "swiglu_quant")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	const I = 10240
	const ch = 7027
	gu := make([]float32, 2*I) // [gate(I) | up(I)]
	for i := 0; i < I; i++ {
		gu[i] = 0.3    // gate: small
		gu[I+i] = -0.2 // up: small
	}
	gu[ch] = 12.14    // the BOS massive gate
	gu[I+ch] = -18.49 // the BOS massive up

	guB := d.NewBufferFloats(gu)
	dq := d.NewBufferInt8(make([]int8, I))
	dSc := d.NewBufferLen(1)
	uI := d.NewBufferU32(I)
	for _, act := range []struct {
		name string
		v    uint32
	}{{"GELU_TANH", 0}, {"SILU", 1}} {
		uAct := d.NewBufferU32(act.v)
		q := d.NewCommandQueue()
		e := q.begin()
		e.dispatch(pSw, 256, 256, guB, guB.At(I*4), dq, dSc, uI, uAct)
		e.end()
		sc := dSc.Floats()[0]
		got := float32(dq.Int8s()[ch]) * sc
		// reference geglu at ch, using the same activation
		var ref float32
		g, u := gu[ch], gu[I+ch]
		if act.v == 1 {
			ref = float32(float64(g)/(1+math.Exp(-float64(g)))) * u
		} else {
			gf := float64(g)
			ref = float32(0.5*gf*(1+math.Tanh(0.7978845608028654*(gf+0.044715*gf*gf*gf)))) * u
		}
		t.Logf("act=%s: scale=%.4g (absmax=%.3f) | ch%d int8-rt=%.3f  ref=%.3f", act.name, sc, sc*127, ch, got, ref)
		if math.Abs(float64(got-ref)) > 2 {
			t.Errorf("act=%s: swiglu_quant DROPPED the massive channel — got %.3f, ref %.3f (KERNEL bug)", act.name, got, ref)
		}
	}
}
