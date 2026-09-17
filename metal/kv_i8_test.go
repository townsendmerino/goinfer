//go:build darwin

package metal

import (
	"math"
	"math/rand"
	"testing"

	"github.com/townsendmerino/goinfer/decoder"
)

func cosineV(a, b []float32) float64 {
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

// TestMetalKVI8_PipelineBuild validates that kv_store_i8 and attention_i8 compile and build pipelines.
func TestMetalKVI8_PipelineBuild(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	if _, err := d.NewComputePipeline(lib, "kv_store_i8"); err != nil {
		t.Fatalf("pipeline kv_store_i8: %v", err)
	}
	if _, err := d.NewComputePipeline(lib, "attention_i8"); err != nil {
		t.Fatalf("pipeline attention_i8: %v", err)
	}
}

// TestMetalKVI8_KVStoreAndAttentionParity validates INT8 KV store and attention against CPU reference.
func TestMetalKVI8_KVStoreAndAttentionParity(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pKvI8, err := d.NewComputePipeline(lib, "kv_store_i8")
	if err != nil {
		t.Fatalf("pipeline kv_store_i8: %v", err)
	}
	pAttnI8, err := d.NewComputePipeline(lib, "attention_i8")
	if err != nil {
		t.Fatalf("pipeline attention_i8: %v", err)
	}

	const nH, nKV, hd, nKeys = 12, 2, 128, 96 // GQA 6:1
	kvDim := nKV * hd
	scale := float32(1 / math.Sqrt(float64(hd)))
	rng := rand.New(rand.NewSource(42))

	q := make([]float32, nH*hd)
	for i := range q {
		q[i] = rng.Float32()*2 - 1
	}

	// Buffers for Metal:
	kcBuf := byteBuf(d, nKeys*kvDim)
	vcBuf := byteBuf(d, nKeys*kvDim)
	ksBuf := d.NewBufferLen(nKeys * nKV)
	vsBuf := d.NewBufferLen(nKeys * nKV)

	cq := d.NewCommandQueue()

	// Store keys one position at a time via kv_store_i8:
	kCpu := make([][]float32, nKeys)
	vCpu := make([][]float32, nKeys)
	kScalesCpu := make([]float32, nKeys*nKV)
	vScalesCpu := make([]float32, nKeys*nKV)
	kInt8Cpu := make([]int8, nKeys*kvDim)
	vInt8Cpu := make([]int8, nKeys*kvDim)

	for pos := 0; pos < nKeys; pos++ {
		kRow := make([]float32, kvDim)
		vRow := make([]float32, kvDim)
		for i := range kRow {
			kRow[i] = rng.Float32()*2 - 1
			vRow[i] = rng.Float32()*2 - 1
		}
		kCpu[pos] = kRow
		vCpu[pos] = vRow

		kBuf := NewBufferFloats(d, kRow)
		vBuf := NewBufferFloats(d, vRow)

		cq.Run1D(pKvI8, nKV, 1,
			kBuf, vBuf, kcBuf, vcBuf, ksBuf, vsBuf,
			NewBufferU32(d, nKV), NewBufferU32(d, hd), NewBufferU32(d, uint32(pos)),
		)

		// Mirror CPU quantization:
		for h := 0; h < nKV; h++ {
			base := h * hd
			var amaxK, amaxV float32
			for dd := 0; dd < hd; dd++ {
				if a := float32(math.Abs(float64(kRow[base+dd]))); a > amaxK {
					amaxK = a
				}
				if a := float32(math.Abs(float64(vRow[base+dd]))); a > amaxV {
					amaxV = a
				}
			}
			scK := amaxK / 127.0
			if scK == 0 {
				scK = 1.0
			}
			scV := amaxV / 127.0
			if scV == 0 {
				scV = 1.0
			}
			kScalesCpu[pos*nKV+h] = scK
			vScalesCpu[pos*nKV+h] = scV
			invK := 1.0 / scK
			invV := 1.0 / scV
			for dd := 0; dd < hd; dd++ {
				qK := int8(math.Round(float64(kRow[base+dd] * invK)))
				qV := int8(math.Round(float64(vRow[base+dd] * invV)))
				kInt8Cpu[pos*kvDim+base+dd] = qK
				vInt8Cpu[pos*kvDim+base+dd] = qV
			}
		}
	}

	// Run attention_i8:
	outBuf := d.NewBufferLen(nH * hd)
	cq.Run1D(pAttnI8, nH*tgReduceAttn, tgReduceAttn,
		NewBufferFloats(d, q), kcBuf, vcBuf, ksBuf, vsBuf, outBuf,
		NewBufferU32(d, nH), NewBufferU32(d, nKV), NewBufferU32(d, hd),
		NewBufferU32(d, uint32(nKeys)), NewBufferFloats(d, []float32{scale}),
		NewBufferU32(d, 0), NewBufferFloats(d, make([]float32, nH)), NewBufferU32(d, 0),
	)

	got := outBuf.Floats()

	// CPU reference:
	ref := make([]float32, nH*hd)
	for qh := 0; qh < nH; qh++ {
		kvh := qh / (nH / nKV)
		sc := make([]float64, nKeys)
		mx := math.Inf(-1)
		for s := 0; s < nKeys; s++ {
			kScale := float64(kScalesCpu[s*nKV+kvh])
			var dot float64
			for dd := 0; dd < hd; dd++ {
				kVal := float64(kInt8Cpu[s*kvDim+kvh*hd+dd]) * kScale
				dot += float64(q[qh*hd+dd]) * kVal
			}
			sc[s] = dot * float64(scale)
			if sc[s] > mx {
				mx = sc[s]
			}
		}
		var sum float64
		for s := 0; s < nKeys; s++ {
			sc[s] = math.Exp(sc[s] - mx)
			sum += sc[s]
		}
		for dd := 0; dd < hd; dd++ {
			var acc float64
			for s := 0; s < nKeys; s++ {
				vScale := float64(vScalesCpu[s*nKV+kvh])
				vVal := float64(vInt8Cpu[s*kvDim+kvh*hd+dd]) * vScale
				acc += sc[s] * vVal
			}
			ref[qh*hd+dd] = float32(acc / sum)
		}
	}

	// Verify got vs ref:
	var dot, na, nb, maxabs float64
	for i := range got {
		dot += float64(got[i]) * float64(ref[i])
		na += float64(got[i]) * float64(got[i])
		nb += float64(ref[i]) * float64(ref[i])
		if dd := math.Abs(float64(got[i] - ref[i])); dd > maxabs {
			maxabs = dd
		}
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	t.Logf("attention_i8 vs CPU ref: cosine=%.7f maxAbs=%.2e", cos, maxabs)
	if cos < 0.9999 || maxabs > 1e-3 {
		t.Fatalf("attention_i8 FAIL: cosine=%.7f maxAbs=%.2e", cos, maxabs)
	}
}

// TestMetalKVI8_DeepContext tests attention_i8 with 8192 keys (tiled online softmax).
func TestMetalKVI8_DeepContext(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no metal device: %v", err)
	}
	lib, err := d.CompileLibrary(allKernels, MSL3_1)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pAttnI8, err := d.NewComputePipeline(lib, "attention_i8")
	if err != nil {
		t.Fatalf("pipeline attention_i8: %v", err)
	}

	const nH, nKV, hd, nKeys = 8, 2, 128, 8192
	kvDim := nKV * hd
	scale := float32(1 / math.Sqrt(float64(hd)))
	rng := rand.New(rand.NewSource(123))

	q := make([]float32, nH*hd)
	for i := range q {
		q[i] = rng.Float32()*2 - 1
	}

	kInt8 := make([]int8, nKeys*kvDim)
	vInt8 := make([]int8, nKeys*kvDim)
	for i := range kInt8 {
		kInt8[i] = int8(rng.Intn(255) - 127)
		vInt8[i] = int8(rng.Intn(255) - 127)
	}
	kScales := make([]float32, nKeys*nKV)
	vScales := make([]float32, nKeys*nKV)
	for i := range kScales {
		kScales[i] = rng.Float32()*0.02 + 0.001
		vScales[i] = rng.Float32()*0.02 + 0.001
	}

	kcBuf := NewBufferInt8(d, kInt8)
	vcBuf := NewBufferInt8(d, vInt8)
	ksBuf := NewBufferFloats(d, kScales)
	vsBuf := NewBufferFloats(d, vScales)
	outBuf := d.NewBufferLen(nH * hd)

	cq := d.NewCommandQueue()
	cq.Run1D(pAttnI8, nH*tgReduceAttn, tgReduceAttn,
		NewBufferFloats(d, q), kcBuf, vcBuf, ksBuf, vsBuf, outBuf,
		NewBufferU32(d, nH), NewBufferU32(d, nKV), NewBufferU32(d, hd),
		NewBufferU32(d, uint32(nKeys)), NewBufferFloats(d, []float32{scale}),
		NewBufferU32(d, 0), NewBufferFloats(d, make([]float32, nH)), NewBufferU32(d, 0),
	)

	got := outBuf.Floats()
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("got[%d] = %v (NaN or Inf)", i, v)
		}
	}
	t.Logf("attention_i8 deep context (nKeys=%d) completed with finite values (got[0]=%.4f)", nKeys, got[0])
}

// TestMetalBuildResident_KVI8 verifies end-to-end decode with INT8 KV cache on a real model.
func TestMetalBuildResident_KVI8(t *testing.T) {
	if _, err := CreateSystemDefaultDevice(); err != nil {
		t.Skipf("no metal device: %v", err)
	}

	// 1. Load with KVPrecision: "i8"
	mI8, err := decoder.Load("../testdata/llama-tiny", decoder.Options{Quant: "int4", KVPrecision: "i8", ResidentContext: 64})
	if err != nil {
		t.Fatalf("Load with KVPrecision i8: %v", err)
	}
	defer mI8.Close()

	bI8 := &metalBackend{}
	rfI8, ok, err := bI8.BuildResident(mI8)
	if err != nil || !ok {
		t.Fatalf("BuildResident (i8): ok=%v err=%v", ok, err)
	}
	defer bI8.Close()

	mResI8, ok := rfI8.(*metalResident)
	if !ok || !mResI8.r.kvI8 {
		t.Fatalf("expected metalResident with kvI8=true")
	}

	// 2. Load with FP16 default KV cache as baseline
	mF16, err := decoder.Load("../testdata/llama-tiny", decoder.Options{Quant: "int4", ResidentContext: 64})
	if err != nil {
		t.Fatalf("Load with FP16 KV: %v", err)
	}
	defer mF16.Close()

	bF16 := &metalBackend{}
	rfF16, ok, err := bF16.BuildResident(mF16)
	if err != nil || !ok {
		t.Fatalf("BuildResident (f16): ok=%v err=%v", ok, err)
	}
	defer bF16.Close()

	// 3. Step through 5 tokens and compare logits
	emb := make([]float32, mI8.Config().HiddenDim)
	rng := rand.New(rand.NewSource(99))

	for pos := 0; pos < 5; pos++ {
		for i := range emb {
			emb[i] = rng.Float32()*2 - 1
		}
		logitsI8, err := rfI8.Forward(emb, pos)
		if err != nil {
			t.Fatalf("Forward (i8) at pos %d: %v", pos, err)
		}
		logitsF16, err := rfF16.Forward(emb, pos)
		if err != nil {
			t.Fatalf("Forward (f16) at pos %d: %v", pos, err)
		}

		for i, v := range logitsI8 {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("logitsI8[%d] = %v at pos %d", i, v, pos)
			}
		}

		cos := cosineV(logitsI8, logitsF16)
		t.Logf("pos %d: INT8 KV vs FP16 KV logits cosine = %.7f", pos, cos)
		if cos < 0.99 {
			t.Errorf("pos %d: cosine %.7f < 0.99", pos, cos)
		}
	}
}
