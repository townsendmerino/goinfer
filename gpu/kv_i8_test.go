//go:build gpu

package gpu

import (
	"math"
	"math/rand"
	"testing"

	"github.com/cogentcore/webgpu/wgpu"
)

// int8 KV kernel correctness (task-gpu-kv-i8.md Inc 1). Real-HW; validates the
// three kernels compile and that the WRITE kernels quantize per-head identically
// to the CPU reference (maxabs/127), and the READ kernel matches f32 attention
// over the dequantized cache.

func i8rng() *rand.Rand { return rand.New(rand.NewSource(17)) }

func randF(rng *rand.Rand, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(rng.NormFloat64())
	}
	return s
}

// dispatch binds storage buffers (binding 0..k-1) + a uniform (binding k) and runs.
func (c *Context) dispatchI8(pl *wgpu.ComputePipeline, layout *wgpu.BindGroupLayout, groups int, storage []*wgpu.Buffer, uni *wgpu.Buffer) error {
	entries := make([]wgpu.BindGroupEntry, 0, len(storage)+1)
	for i, b := range storage {
		entries = append(entries, wgpu.BindGroupEntry{Binding: uint32(i), Buffer: b, Size: b.GetSize()})
	}
	entries = append(entries, wgpu.BindGroupEntry{Binding: uint32(len(storage)), Buffer: uni, Size: uni.GetSize()})
	bg, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: layout, Entries: entries})
	if err != nil {
		return err
	}
	defer bg.Release()
	enc, _ := c.device.CreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pl)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(uint32(groups), 1, 1)
	pass.End()
	cb, _ := enc.Finish(nil)
	c.queue.Submit(cb)
	c.device.Poll(true, nil)
	return nil
}

func (c *Context) sbuf(data []float32) *wgpu.Buffer {
	b, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "i8t-s", Contents: wgpu.ToBytes(data), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	return b
}
func (c *Context) ubuf(data []uint32) *wgpu.Buffer {
	b, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "i8t-u", Contents: wgpu.ToBytes(data), Usage: wgpu.BufferUsageUniform})
	return b
}
func (c *Context) zbuf(n int) *wgpu.Buffer { // zeroed storage, readable
	b, _ := c.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "i8t-z", Size: uint64(n * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	return b
}
func (c *Context) readWords(b *wgpu.Buffer, n int) []uint32 {
	f, _ := c.Readback(&DeviceBuffer{buf: b, n: n})
	out := make([]uint32, n)
	for i, v := range f {
		out[i] = math.Float32bits(v)
	}
	return out
}
func (c *Context) readF(b *wgpu.Buffer, n int) []float32 {
	f, _ := c.Readback(&DeviceBuffer{buf: b, n: n})
	return f
}

func i8byte(words []uint32, e int) int { return int(int8(words[e>>2] >> ((e & 3) * 8))) }

func cosF(a, b []float32) float64 {
	var d, na, nb float64
	for i := range a {
		d += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return d / (math.Sqrt(na) * math.Sqrt(nb))
}

func TestKVI8Kernels(t *testing.T) {
	c := newOrSkipHW(t)
	defer c.Close()
	if err := c.ensureAttn(); err != nil {
		t.Fatalf("ensureAttn (compile int8 kernels): %v", err)
	}
	const nKV, hd, pos = 4, 128, 9
	half := hd / 2
	kvDim := nKV * hd
	maxLen := pos + 1
	rng := i8rng()
	invFreq := make([]float32, half)
	for i := range invFreq {
		invFreq[i] = float32(math.Pow(10000, -2*float64(i)/float64(hd)))
	}

	// per-head quantize-dequant reference (maxabs/127), matching the kernels.
	dq := func(head []float32) []float32 {
		var amax float32
		for _, v := range head {
			if a := float32(math.Abs(float64(v))); a > amax {
				amax = a
			}
		}
		sc := amax / 127
		if sc == 0 {
			sc = 1
		}
		out := make([]float32, len(head))
		for i, v := range head {
			q := math.Round(float64(v) / float64(sc))
			out[i] = float32(math.Max(-127, math.Min(127, q))) * sc
		}
		return out
	}
	rope := func(headK []float32) []float32 {
		out := make([]float32, hd)
		for d := range half {
			th := float64(pos) * float64(invFreq[d])
			cs, sn := float32(math.Cos(th)), float32(math.Sin(th))
			out[d] = headK[d]*cs - headK[half+d]*sn
			out[half+d] = headK[half+d]*cs + headK[d]*sn
		}
		return out
	}

	// --- ropeStoreI8: rotate K then per-head int8. ---
	kSrc := randF(rng, kvDim)
	kCache := c.zbuf((maxLen*kvDim + 3) / 4)
	kScale := c.zbuf(maxLen * nKV)
	uniRope := c.ubuf([]uint32{nKV, uint32(hd), uint32(half), pos, math.Float32bits(1.0), uint32(pos * kvDim), nKV, 0})
	if err := c.dispatchI8(c.ropeStoreI8Pipeline, c.ropeStoreI8Layout, 1, []*wgpu.Buffer{c.sbuf(kSrc), c.sbuf(invFreq), kCache, kScale}, uniRope); err != nil {
		t.Fatal(err)
	}
	kWords := c.readWords(kCache, (maxLen*kvDim+3)/4)
	kSc := c.readF(kScale, maxLen*nKV)
	var gotK, wantK []float32
	for h := range nKV {
		want := dq(rope(kSrc[h*hd : h*hd+hd]))
		wantK = append(wantK, want...)
		for d := range hd {
			gotK = append(gotK, float32(i8byte(kWords, pos*kvDim+h*hd+d))*kSc[pos*nKV+h])
		}
	}
	if cos := cosF(gotK, wantK); cos < 0.99999 {
		t.Errorf("ropeStoreI8 vs CPU rope+quant: cosine %.6f", cos)
	} else {
		t.Logf("ropeStoreI8 vs CPU: cosine %.6f", cos)
	}

	// --- kvStoreI8: V int8 (no rope). ---
	vSrc := randF(rng, kvDim)
	vCache := c.zbuf((maxLen*kvDim + 3) / 4)
	vScale := c.zbuf(maxLen * nKV)
	uniV := c.ubuf([]uint32{nKV, uint32(hd), uint32(pos * kvDim), pos, nKV, 0, 0, 0})
	if err := c.dispatchI8(c.kvStoreI8Pipeline, c.kvStoreI8Layout, 1, []*wgpu.Buffer{c.sbuf(vSrc), vCache, vScale}, uniV); err != nil {
		t.Fatal(err)
	}
	vWords := c.readWords(vCache, (maxLen*kvDim+3)/4)
	vSc := c.readF(vScale, maxLen*nKV)
	var gotV, wantV []float32
	for h := range nKV {
		wantV = append(wantV, dq(vSrc[h*hd:h*hd+hd])...)
		for d := range hd {
			gotV = append(gotV, float32(i8byte(vWords, pos*kvDim+h*hd+d))*vSc[pos*nKV+h])
		}
	}
	if cos := cosF(gotV, wantV); cos < 0.99999 {
		t.Errorf("kvStoreI8 vs CPU quant: cosine %.6f", cos)
	} else {
		t.Logf("kvStoreI8 vs CPU: cosine %.6f", cos)
	}
}

func (c *Context) wbuf(data []uint32) *wgpu.Buffer {
	b, _ := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "i8t-w", Contents: wgpu.ToBytes(data), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	return b
}

// TestKVI8Attn: attnI8 (READ arm) must match f32 attention over the dequantized
// int8 cache it reads.
func TestKVI8Attn(t *testing.T) {
	c := newOrSkipHW(t)
	defer c.Close()
	if err := c.ensureAttn(); err != nil {
		t.Fatal(err)
	}
	const nH, nKV, hd, nKeys = 8, 4, 128, 40
	group := nH / nKV
	kvDim := nKV * hd
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	rng := i8rng()
	q := randF(rng, nH*hd)

	// Build the int8 K/V cache + scales on CPU (per-head maxabs/127); the f32
	// reference attends over the dequantized values (= what attnI8 reads).
	kWords := make([]uint32, (nKeys*kvDim+3)/4)
	vWords := make([]uint32, (nKeys*kvDim+3)/4)
	kSc := make([]float32, nKeys*nKV)
	vSc := make([]float32, nKeys*nKV)
	deK := make([]float32, nKeys*kvDim)
	deV := make([]float32, nKeys*kvDim)
	quant := func(head []float32, words []uint32, sc []float32, de []float32, base, sci int) {
		var amax float32
		for _, v := range head {
			if a := float32(math.Abs(float64(v))); a > amax {
				amax = a
			}
		}
		s := amax / 127
		if s == 0 {
			s = 1
		}
		sc[sci] = s
		for d, v := range head {
			qv := int8(math.Max(-127, math.Min(127, math.Round(float64(v)/float64(s)))))
			e := base + d
			words[e>>2] |= uint32(uint8(qv)) << ((e & 3) * 8)
			de[e] = float32(qv) * s
		}
	}
	for s := range nKeys {
		for h := range nKV {
			quant(randF(rng, hd), kWords, kSc, deK, s*kvDim+h*hd, s*nKV+h)
			quant(randF(rng, hd), vWords, vSc, deV, s*kvDim+h*hd, s*nKV+h)
		}
	}

	// f32 reference attention over deK/deV.
	want := make([]float32, nH*hd)
	for qh := range nH {
		kvh := qh / group
		sc := make([]float64, nKeys)
		mx := math.Inf(-1)
		for s := range nKeys {
			var dot float64
			for d := range hd {
				dot += float64(q[qh*hd+d]) * float64(deK[s*kvDim+kvh*hd+d])
			}
			sc[s] = dot * float64(scale)
			if sc[s] > mx {
				mx = sc[s]
			}
		}
		var sum float64
		for s := range sc {
			sc[s] = math.Exp(sc[s] - mx)
			sum += sc[s]
		}
		for d := range hd {
			var acc float64
			for s := range nKeys {
				acc += sc[s] / sum * float64(deV[s*kvDim+kvh*hd+d])
			}
			want[qh*hd+d] = float32(acc)
		}
	}

	ctx := c.zbuf(nH * hd)
	uni := c.ubuf([]uint32{nH, nKV, uint32(hd), nKeys, 0, uint32(group), math.Float32bits(scale), 0})
	if err := c.dispatchI8(c.attnI8Pipeline, c.attnI8Layout, nH, []*wgpu.Buffer{c.sbuf(q), c.wbuf(kWords), c.wbuf(vWords), c.sbuf(kSc), c.sbuf(vSc), ctx}, uni); err != nil {
		t.Fatal(err)
	}
	got := c.readF(ctx, nH*hd)
	if cos := cosF(got, want); cos < 0.9999 {
		t.Errorf("attnI8 vs f32-over-dequant: cosine %.6f", cos)
	} else {
		t.Logf("attnI8 vs f32-over-dequant (%d keys): cosine %.6f", nKeys, cos)
	}
}
