//go:build gpu

package kernels

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// w4Case is one random int4 projection with M quantized activation rows.
type w4Case struct {
	N, K, M int
	rm      *ResidentW4A8
	act     []int8 // [M, kp], zero past K
	aScales []float32
}

func newW4Case(t *testing.T, c *Context, rng *rand.Rand, N, K, M int) *w4Case {
	t.Helper()
	kp := PadK32(K)
	ng := kp / W4a8GroupSize
	nib := make([]uint8, N*K)
	for i := range nib {
		nib[i] = uint8(rng.Intn(16))
	}
	scales := make([]float32, N*ng)
	for i := range scales {
		scales[i] = 0.01 + rng.Float32()*0.05
	}
	rm, err := c.UploadW4A8(nib, scales, N, K)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(rm.Release)
	w := &w4Case{N: N, K: K, M: M, rm: rm, act: make([]int8, M*kp), aScales: make([]float32, M)}
	for m := range M {
		for k := range K {
			w.act[m*kp+k] = int8(rng.Intn(255) - 127)
		}
		w.aScales[m] = 0.005 + rng.Float32()*0.02
	}
	return w
}

// w4Encode runs enc's dispatches on a fresh encoder and returns dst's n floats.
func w4Encode(t *testing.T, c *Context, n int, enc func(dst *wgpu.Buffer, uni func([]uint32) *wgpu.Buffer, bind func(*wgpu.BindGroupLayout, ...*wgpu.Buffer) *wgpu.BindGroup, disp func(*wgpu.ComputePipeline, *wgpu.BindGroup, uint32, uint32))) []float32 {
	t.Helper()
	var keep []func()
	defer func() {
		for _, f := range keep {
			f()
		}
	}()
	dev := c.device
	dst, err := dev.TryCreateBuffer(&wgpu.BufferDescriptor{Size: uint64(n * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	if err != nil {
		t.Fatal(err)
	}
	keep = append(keep, dst.Release)
	stag, err := dev.TryCreateBuffer(&wgpu.BufferDescriptor{Size: uint64(n * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	if err != nil {
		t.Fatal(err)
	}
	keep = append(keep, stag.Release)
	e, err := dev.TryCreateCommandEncoder(nil)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Release()
	uni := func(v []uint32) *wgpu.Buffer {
		b, err := dev.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(v), Usage: wgpu.BufferUsageUniform})
		if err != nil {
			t.Fatal(err)
		}
		keep = append(keep, b.Release)
		return b
	}
	bind := func(layout *wgpu.BindGroupLayout, bufs ...*wgpu.Buffer) *wgpu.BindGroup {
		es := make([]wgpu.BindGroupEntry, len(bufs))
		for i, b := range bufs {
			es[i] = wgpu.BindGroupEntry{Binding: uint32(i), Buffer: b, Size: b.GetSize()}
		}
		bg, err := dev.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: layout, Entries: es})
		if err != nil {
			t.Fatal(err)
		}
		keep = append(keep, bg.Release)
		return bg
	}
	disp := func(pl *wgpu.ComputePipeline, bg *wgpu.BindGroup, gx, gy uint32) {
		pass := e.BeginComputePass(nil)
		pass.SetPipeline(pl)
		pass.SetBindGroup(0, bg, nil)
		pass.DispatchWorkgroups(gx, gy, 1)
		if err := pass.TryEnd(); err != nil {
			t.Fatal(err)
		}
		pass.Release()
	}
	enc(dst, uni, bind, disp)
	if err := e.TryCopyBufferToBuffer(dst, 0, stag, 0, uint64(n*4)); err != nil {
		t.Fatal(err)
	}
	cmd, err := e.TryFinish(nil)
	if err != nil {
		t.Fatal(err)
	}
	c.queue.Submit(cmd)
	cmd.Release()
	st := wgpu.MapAsyncStatus(0)
	if err := stag.TryMapAsync(wgpu.MapModeRead, 0, uint64(n*4), func(s wgpu.MapAsyncStatus) { st = s }); err != nil {
		t.Fatal(err)
	}
	c.device.Poll(true, nil)
	if st != wgpu.MapAsyncStatusSuccess {
		t.Fatalf("map failed: %v", st)
	}
	out := make([]float32, n)
	copy(out, wgpu.FromBytes[float32](stag.GetMappedRange(0, uint(n*4))))
	if err := stag.TryUnmap(); err != nil {
		t.Fatal(err)
	}
	return out
}

func (w *w4Case) storage(t *testing.T, c *Context, contents []byte) *wgpu.Buffer {
	t.Helper()
	b, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: contents, Usage: wgpu.BufferUsageStorage})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(b.Release)
	return b
}

// single is the reference: each row through gemvW4A8ShaderWGSL on its own.
func (w *w4Case) single(t *testing.T, c *Context) []float32 {
	t.Helper()
	kp := w.rm.kp
	out := make([]float32, 0, w.M*w.N)
	for m := range w.M {
		aq := w.storage(t, c, wgpu.ToBytes(packInt8(w.act[m*kp:(m+1)*kp], 1, kp)))
		as := w.storage(t, c, wgpu.ToBytes(w.aScales[m:m+1]))
		out = append(out, w4Encode(t, c, w.N, func(dst *wgpu.Buffer, uni func([]uint32) *wgpu.Buffer, bind func(*wgpu.BindGroupLayout, ...*wgpu.Buffer) *wgpu.BindGroup, disp func(*wgpu.ComputePipeline, *wgpu.BindGroup, uint32, uint32)) {
			gx, gy := GemvGrid(w.N)
			disp(c.gemvW4Pipeline, bind(c.gemvW4Layout, aq, w.rm.bq, as, w.rm.bScales, dst, uni([]uint32{1, uint32(kp), uint32(w.N), 0})), gx, gy)
		})...)
	}
	return out
}

// multi is all M rows through kind's kernel as built in p.
func (w *w4Case) multi(t *testing.T, c *Context, kind W4MultiKind, p tiledPipes) []float32 {
	t.Helper()
	aq := w.storage(t, c, wgpu.ToBytes(packInt8(w.act, w.M, w.rm.kp)))
	as := w.storage(t, c, wgpu.ToBytes(w.aScales))
	return w4Encode(t, c, w.M*w.N, func(dst *wgpu.Buffer, uni func([]uint32) *wgpu.Buffer, bind func(*wgpu.BindGroupLayout, ...*wgpu.Buffer) *wgpu.BindGroup, disp func(*wgpu.ComputePipeline, *wgpu.BindGroup, uint32, uint32)) {
		encodeW4Multi(kind, p, w.rm, w.M, aq, as, dst, uni, bind, disp)
	})
}

// w4Grade grades got against the single-row reference: every output of every row equal bit for bit. It returns "" on
// a pass.
func w4Grade(ref, got []float32, M, N int) string {
	for m := range M {
		var worst float64
		bits := 0
		for n := range N {
			r, g := ref[m*N+n], got[m*N+n]
			if math.Float32bits(r) != math.Float32bits(g) {
				bits++
				if d := math.Abs(float64(r) - float64(g)); !(d <= worst) { // a NaN difference counts as the worst
					worst = d
				}
			}
		}
		if bits != 0 {
			return fmt.Sprintf("row %d: %d of %d outputs differ in their bits (max |diff| %.3g)", m, bits, N, worst)
		}
	}
	return ""
}

// G-S12.4b-1a (docs/tasks/task-multimodal-support-2026-10.md): both multi-row int4 kernels against the single-row
// kernel on the same quantized rows, over aligned and ragged K and N and row counts on both sides of a 16-row block and
// a 64-row tile.
func TestW4Multi_matchesSingleRow(t *testing.T) {
	c := newOrSkipHW(t)
	t.Cleanup(func() { c.Close() }) // after the cases' own cleanups, which release their weights
	if err := c.ensureGEMVW4(); err != nil {
		t.Fatal(err)
	}
	if err := c.ensureW4Multi(); err != nil {
		t.Fatal(err)
	}
	shapes := []struct{ N, K int }{
		{256, 128},  // aligned: 4 groups
		{131, 100},  // ragged N and K: 4 groups, the last padded
		{64, 96},    // 3 groups: most lanes of the sum's tree are empty
		{200, 4160}, // 130 groups: lanes with one, two and three groups
		{70, 6144},  // 192 groups: three groups a lane, a real feed-forward width
	}
	for _, sh := range shapes {
		for _, M := range []int{1, 5, 16, 17, 300} {
			rng := rand.New(rand.NewSource(int64(sh.N*100003 + sh.K*101 + M)))
			w := newW4Case(t, c, rng, sh.N, sh.K, M)
			ref := w.single(t, c)
			for _, kind := range []W4MultiKind{W4MultiRows, W4MultiTiled} {
				got := w.multi(t, c, kind, c.w4Multi[kind])
				if why := w4Grade(ref, got, M, sh.N); why != "" {
					t.Errorf("%s N=%d K=%d M=%d: %s", kind, sh.N, sh.K, M, why)
				}
			}
		}
	}
}

// w4Defects are the planted defects of G-S12.4b-1a, as edits of each kernel's source.
var w4Defects = []struct {
	name     string
	kinds    []W4MultiKind
	old, new [2]string // indexed by w4MultiKind
	count    [2]int
}{
	{
		name:  "a group's scale taken from the next group",
		kinds: []W4MultiKind{W4MultiRows, W4MultiTiled},
		old:   [2]string{"let s = wBase + v;", "let si = bc * ng + g;"},
		new:   [2]string{"let s = wBase + v + 1u;", "let si = bc * ng + g + 1u;"},
		count: [2]int{1, 1},
	},
	{
		name:  "the nibbles of a byte swapped",
		kinds: []W4MultiKind{W4MultiRows, W4MultiTiled},
		old:   [2]string{"(ww >>  0u) & 0xFu) - 8) * al.x;\n    s = s + (i32((ww >>  4u)", "lo = nib4(ww & 0xFFFFu);"},
		new:   [2]string{"(ww >>  4u) & 0xFu) - 8) * al.x;\n    s = s + (i32((ww >>  0u)", "lo = nib4(((ww & 0xF0F0u) >> 4u) | ((ww & 0x0F0Fu) << 4u));"},
		count: [2]int{1, 1},
	},
	{
		name:  "a row's activation scale taken from another row",
		kinds: []W4MultiKind{W4MultiRows, W4MultiTiled},
		old:   [2]string{"partial[0] * aScales[row];", "let sa = aScales[row];"},
		new:   [2]string{"partial[0] * aScales[dims.row0 + (m + 1u) % dims.m];", "let sa = aScales[(row + 1u) % dims.m];"},
		count: [2]int{1, 1},
	},
	{
		name:  "the last partial column block of a tile skipped",
		kinds: []W4MultiKind{W4MultiTiled},
		old:   [2]string{"", "< dims.n) { dst["},
		new:   [2]string{"", "< (dims.n / 16u) * 16u) { dst["},
		count: [2]int{0, 4},
	},
}

// The planted defects of G-S12.4b-1a: each must fail its kernel's own bar on a ragged shape.
func TestW4Multi_plantedDefects(t *testing.T) {
	c := newOrSkipHW(t)
	t.Cleanup(func() { c.Close() }) // after the cases' own cleanups, which release their weights
	if err := c.ensureGEMVW4(); err != nil {
		t.Fatal(err)
	}
	if err := c.ensureW4Multi(); err != nil {
		t.Fatal(err)
	}
	const N, K, M = 131, 100, 17
	w := newW4Case(t, c, rand.New(rand.NewSource(7)), N, K, M)
	ref := w.single(t, c)
	for _, kind := range []W4MultiKind{W4MultiRows, W4MultiTiled} {
		if why := w4Grade(ref, w.multi(t, c, kind, c.w4Multi[kind]), M, N); why != "" {
			t.Fatalf("%s is red before any defect is planted: %s", kind, why)
		}
	}
	for _, d := range w4Defects {
		for _, kind := range d.kinds {
			code, label := c.w4MultiSource(kind)
			if got := strings.Count(code, d.old[kind]); got != d.count[kind] {
				t.Fatalf("%s, %s: the text to edit appears %d times, want %d", kind, d.name, got, d.count[kind])
			}
			sh, pl, layout, err := c.mkPipeline(label+"-defect", strings.ReplaceAll(code, d.old[kind], d.new[kind]))
			if err != nil {
				t.Fatalf("%s, %s: %v", kind, d.name, err)
			}
			why := w4Grade(ref, w.multi(t, c, kind, tiledPipes{sh: sh, Pl: pl, Layout: layout}), M, N)
			if why == "" {
				t.Errorf("%s, %s: the gate stayed green", kind, d.name)
				continue
			}
			t.Logf("%s, %s: red (%s)", kind, d.name, why)
		}
	}
}
