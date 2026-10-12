//go:build gpu

package gpu

import (
	"math"
	"strings"
	"testing"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// attnRowsOn runs a rows-form attention pipeline over M query rows: row r attends keys up to firstKeys+r, from its
// window's start when window is not 0. It returns the packed [M, nH*hd] contexts.
func attnRowsOn(t *testing.T, c *Context, p tiledPipes, q, keys, vals []float32, nH, nKV, hd, M, firstKeys, window int, scale float32) []float32 {
	t.Helper()
	var keep []func()
	defer func() {
		for _, f := range keep {
			f()
		}
	}()
	mk := func(contents []byte, usage wgpu.BufferUsage) *wgpu.Buffer {
		b, err := c.device.TryCreateBufferInit(&wgpu.BufferInitDescriptor{Contents: contents, Usage: usage})
		if err != nil {
			t.Fatal(err)
		}
		keep = append(keep, b.Release)
		return b
	}
	n := M * nH * hd
	bufs := []*wgpu.Buffer{
		mk(wgpu.ToBytes(q), wgpu.BufferUsageStorage), mk(wgpu.ToBytes(keys), wgpu.BufferUsageStorage), mk(wgpu.ToBytes(vals), wgpu.BufferUsageStorage),
		mk(wgpu.ToBytes(make([]float32, n)), wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc), mk(wgpu.ToBytes([]float32{0}), wgpu.BufferUsageStorage),
		mk(wgpu.ToBytes([]uint32{uint32(nH), uint32(nKV), uint32(hd), uint32(firstKeys), 0, uint32(nH / nKV), f32bits(scale), uint32(window)}), wgpu.BufferUsageUniform),
		mk(wgpu.ToBytes([]uint32{0, 0, 0, 0}), wgpu.BufferUsageUniform),
	}
	es := make([]wgpu.BindGroupEntry, len(bufs))
	for i, b := range bufs {
		es[i] = wgpu.BindGroupEntry{Binding: uint32(i), Buffer: b, Size: b.GetSize()}
	}
	bg, err := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: p.layout, Entries: es})
	if err != nil {
		t.Fatal(err)
	}
	keep = append(keep, bg.Release)
	enc, err := c.device.TryCreateCommandEncoder(nil)
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(p.pl)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(uint32(nH), uint32(M), 1)
	if err := pass.TryEnd(); err != nil {
		t.Fatal(err)
	}
	pass.Release()
	cmd, err := enc.TryFinish(nil)
	if err != nil {
		t.Fatal(err)
	}
	c.queue.Submit(cmd)
	cmd.Release()
	out, err := c.readbackRaw(bufs[3], n)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// G-S12.4c-b1 (docs/tasks/task-multimodal-support-2026-10.md): each key-split kernel's rows form against the single-row
// kernel run once a row, bit for bit, with and without a window, at key counts that cross a tile. Then the planted
// defects, each of which must differ.
func TestAttnKeysRows_matchSingleRow(t *testing.T) {
	c, err := New()
	if err != nil {
		t.Skipf("no GPU adapter: %v", err)
	}
	defer c.Close()
	for _, f := range []func() error{c.ensureAttn, c.ensureAttnWide, c.ensureAttnKeysRows} {
		if err := f(); err != nil {
			t.Fatal(err)
		}
	}
	const M = 5
	type shape struct {
		nH, nKV, hd int
		wide        bool
	}
	// differing counts the outputs of a rows pipeline that are not the single-row kernel's bits.
	differing := func(p tiledPipes, g shape, firstKeys, window int) int {
		kvDim := g.nKV * g.hd
		scale := float32(1.0 / math.Sqrt(float64(g.hd)))
		row := g.nH * g.hd
		last := firstKeys + M - 1
		q, keys, vals := randMat(M*row, 1), randMat(last*kvDim, 2), randMat(last*kvDim, 3)
		got := attnRowsOn(t, c, p, q, keys, vals, g.nH, g.nKV, g.hd, M, firstKeys, window, scale)
		pl, ly := c.attnKeysPipeline, c.attnKeysLayout
		if g.wide {
			pl, ly = c.attnKeysWide.pl, c.attnKeysWide.layout
		}
		diff := 0
		for r := range M {
			nKeys := firstKeys + r
			start := 0
			if window > 0 {
				start = max(0, nKeys-window)
			}
			want, err := c.attentionOn(pl, ly, q[r*row:(r+1)*row], keys[:nKeys*kvDim], vals[:nKeys*kvDim], g.nH, g.nKV, g.hd, nKeys, start, scale)
			if err != nil {
				t.Fatal(err)
			}
			for i := range want {
				if math.Float32bits(want[i]) != math.Float32bits(got[r*row+i]) {
					diff++
				}
			}
		}
		return diff
	}
	shapes := []shape{{12, 2, 128, false}, {8, 4, 256, true}}
	type cut struct{ firstKeys, window int }
	cuts := []cut{{1, 0}, {40, 0}, {126, 0}, {attnKeysTile - 2, 0}, {attnKeysTile + 30, 0}, {300, 128}, {1500, 1024}, {attnKeysTile + 600, 1024}, {3, 1024}}
	for _, g := range shapes {
		for _, k := range cuts {
			if d := differing(c.attnKeysRows[b2i(g.wide)], g, k.firstKeys, k.window); d != 0 {
				t.Errorf("%+v first keys %d window %d: %d outputs differ in their bits from the single-row kernel's", g, k.firstKeys, k.window, d)
			}
		}
	}
	for _, g := range shapes {
		src, err := attnKeysRowsSource(g.wide)
		if err != nil {
			t.Fatal(err)
		}
		for name, d := range map[string]struct {
			old, new string
			cut      cut
		}{
			"every row given the first row's key count": {"let rowKeys = p.nKeys + row;", "let rowKeys = p.nKeys;", cut{40, 0}},
			"the window ignored":                        {"if (p.window != 0u && ", "if (false && ", cut{300, 128}},
			"a row reading the next row's query":        {"let qb4 = (row * p.nH * hd + qh * hd) / 4u;", "let qb4 = ((row + 1u) * p.nH * hd + qh * hd) / 4u;", cut{40, 0}},
		} {
			if strings.Count(src, d.old) != 1 {
				t.Fatalf("%s: the text to edit appears %d times", name, strings.Count(src, d.old))
			}
			sh, pl, ly, err := c.mkPipeline("attn-keys-rows-defect", strings.Replace(src, d.old, d.new, 1))
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			if n := differing(tiledPipes{sh: sh, pl: pl, layout: ly}, g, d.cut.firstKeys, d.cut.window); n == 0 {
				t.Errorf("%+v, %s: the gate stayed green", g, name)
			} else {
				t.Logf("%+v, %s: red (%d outputs differ)", g, name, n)
			}
		}
	}
}
