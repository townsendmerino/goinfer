//go:build cuda && goinfer_testhooks

package cuda

import (
	"math"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	gpu "github.com/townsendmerino/aikit/gpu"
)

// requireBareCUDADevice needs only the driver — no checkpoint, no GOINFER_HEAVY_TESTS — since a
// pure kernel test compiles its own tiny PTX modules and launches with synthetic data.
func requireBareCUDADevice(t *testing.T) *Device {
	t.Helper()
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit: %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no device: %v", err)
	}
	dev, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("CreateSystemDefaultDevice: %v", err)
	}
	t.Cleanup(dev.ReleaseAll)
	return dev
}

// cpuLayerNorm mirrors aikit/vision/encoder.go's layerNormInto exactly (mean/variance in
// float64, then (x-mean)*inv*w+b in mixed f64/f32 — transcribed from that function, not
// re-derived, since it is the CPU reference this kernel is held to).
func cpuLayerNorm(x, w, b []float32, rows, dim int, eps float64) []float32 {
	out := make([]float32, rows*dim)
	for r := 0; r < rows; r++ {
		xr := x[r*dim : r*dim+dim]
		var mean float64
		for _, v := range xr {
			mean += float64(v)
		}
		mean /= float64(dim)
		var variance float64
		for _, v := range xr {
			d := float64(v) - mean
			variance += d * d
		}
		variance /= float64(dim)
		inv := 1.0 / math.Sqrt(variance+eps)
		dst := out[r*dim : r*dim+dim]
		for d := 0; d < dim; d++ {
			dst[d] = float32((float64(xr[d])-mean)*inv)*w[d] + b[d]
		}
	}
	return out
}

func cosineF32(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// dequantRows undoes the symmetric per-row int8 quantize+pack this file's kernels use (4 values
// packed per int32 word, scale = rowmax/127), for comparing a kernel's quantized output against
// an f32 CPU reference.
func dequantRows(packed []int32, scale []float32, rows, dim int) []float32 {
	out := make([]float32, rows*dim)
	for r := 0; r < rows; r++ {
		s := scale[r]
		for i4 := 0; i4 < dim/4; i4++ {
			word := uint32(packed[r*(dim/4)+i4])
			for b := 0; b < 4; b++ {
				v := int8(byte(word >> (8 * b)))
				out[r*dim+i4*4+b] = float32(v) * s
			}
		}
	}
	return out
}

// TestLayerNormQuantBatched_matchesCPUReference pins layernorm_quant_batched against
// cpuLayerNorm on synthetic data covering a real head-dim-shaped case (SigLIP so400m: hidden
// 1152) at a small M, so the test stays cheap while exercising the kernel's real shared-memory
// sizing path.
func TestLayerNormQuantBatched_matchesCPUReference(t *testing.T) {
	dev := requireBareCUDADevice(t)
	mod, err := dev.CompileLibrary(layernormQuantPTX)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := dev.NewComputePipeline(mod, "layernorm_quant_batched")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const M, N = 6, 1152
	const eps = 1e-6
	x := make([]float32, M*N)
	w := make([]float32, N)
	b := make([]float32, N)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*0.037)) * 12.0 // real activations span a wide range
	}
	for i := range w {
		w[i] = 1.0 + float32(math.Cos(float64(i)*0.011))*0.3
		b[i] = float32(math.Sin(float64(i)*0.019)) * 0.2
	}
	ref := cpuLayerNorm(x, w, b, M, N, eps)

	q := dev.NewCommandQueue()
	xb := gpu.NewBufferLenOf[float32](dev, len(x))
	wb := gpu.NewBufferLenOf[float32](dev, len(w))
	bb := gpu.NewBufferLenOf[float32](dev, len(b))
	if err := gpu.Upload(xb, x); err != nil {
		t.Fatalf("upload x: %v", err)
	}
	if err := gpu.Upload(wb, w); err != nil {
		t.Fatalf("upload w: %v", err)
	}
	if err := gpu.Upload(bb, b); err != nil {
		t.Fatalf("upload b: %v", err)
	}
	qOut := gpu.NewBufferLenOf[int32](dev, M*N/4)
	sOut := gpu.NewBufferLenOf[float32](dev, M)
	if err := q.Launch(pipe, LaunchConfig{GridX: uint32(M), GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: uint32((256 + N) * 4)},
		Arg(xb), Arg(wb), Arg(bb), gpu.ArgValue(int32(N)), gpu.ArgValue(float32(eps)), Arg(qOut), Arg(sOut)); err != nil {
		t.Fatalf("launch: %v", err)
	}
	if err := q.Sync(); err != nil {
		t.Fatalf("sync: %v", err)
	}
	packed := make([]int32, M*N/4)
	scale := make([]float32, M)
	if err := gpu.Download(qOut, packed); err != nil {
		t.Fatalf("download q: %v", err)
	}
	if err := gpu.Download(sOut, scale); err != nil {
		t.Fatalf("download s: %v", err)
	}
	got := dequantRows(packed, scale, M, N)

	cos := cosineF32(ref, got)
	t.Logf("layernorm_quant_batched vs CPU reference: cosine=%.6f", cos)
	if cos < 0.999 {
		t.Errorf("cosine %.6f < 0.999 — layernorm_quant_batched diverges from the CPU reference", cos)
	}
}

// TestLayerNormF32Batched_matchesCPUReference pins the tower's final (non-quantized) post-LN
// kernel, same reference, no dequantize step.
func TestLayerNormF32Batched_matchesCPUReference(t *testing.T) {
	dev := requireBareCUDADevice(t)
	mod, err := dev.CompileLibrary(layernormQuantPTX)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := dev.NewComputePipeline(mod, "layernorm_f32_batched")
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}

	const M, N = 6, 1152
	const eps = 1e-6
	x := make([]float32, M*N)
	w := make([]float32, N)
	b := make([]float32, N)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*0.041)) * 9.0
	}
	for i := range w {
		w[i] = 1.0 + float32(math.Cos(float64(i)*0.013))*0.3
		b[i] = float32(math.Sin(float64(i)*0.017)) * 0.2
	}
	ref := cpuLayerNorm(x, w, b, M, N, eps)

	q := dev.NewCommandQueue()
	xb := gpu.NewBufferLenOf[float32](dev, len(x))
	wb := gpu.NewBufferLenOf[float32](dev, len(w))
	bb := gpu.NewBufferLenOf[float32](dev, len(b))
	if err := gpu.Upload(xb, x); err != nil {
		t.Fatalf("upload x: %v", err)
	}
	if err := gpu.Upload(wb, w); err != nil {
		t.Fatalf("upload w: %v", err)
	}
	if err := gpu.Upload(bb, b); err != nil {
		t.Fatalf("upload b: %v", err)
	}
	if err := q.Launch(pipe, LaunchConfig{GridX: 1, GridY: uint32(M), GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1, SharedMemBytes: 256 * 4},
		Arg(xb), Arg(wb), Arg(bb), gpu.ArgValue(int32(N)), gpu.ArgValue(float32(eps)), gpu.ArgValue(int32(M))); err != nil {
		t.Fatalf("launch: %v", err)
	}
	if err := q.Sync(); err != nil {
		t.Fatalf("sync: %v", err)
	}
	got := make([]float32, M*N)
	if err := gpu.Download(xb, got); err != nil {
		t.Fatalf("download: %v", err)
	}

	cos := cosineF32(ref, got)
	t.Logf("layernorm_f32_batched vs CPU reference: cosine=%.6f", cos)
	if cos < 0.9999 {
		t.Errorf("cosine %.6f < 0.9999 — layernorm_f32_batched diverges from the CPU reference", cos)
	}
}
