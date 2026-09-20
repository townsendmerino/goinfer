//go:build gpu

package gpu

import (
	"fmt"
	"math"

	"github.com/oliverbestmann/webgpu/wgpu"
)

// gumbelState is a DecodeRunner's device-sampling resources (R7b), built lazily on the first RunSample so a
// runner that only ever returns logits pays nothing.
type gumbelState struct {
	pl1, pl2   *wgpu.ComputePipeline
	bg1, bg2   *wgpu.BindGroup
	prm        *wgpu.Buffer // [vocab, invT bits, key lo, key hi, draw lo, draw hi, nBlocks] as u32
	bkey, bidx *wgpu.Buffer // per-workgroup winners
	out        *wgpu.Buffer // the drawn id (i32), -1 if nothing was comparable
	outStag    *wgpu.Buffer // MapRead copy of out: 4 bytes back instead of vocab*4
	nb         int
	scratch    [7]uint32
}

// gumbelBlocks is the stage-1 dispatch size for a vocabulary of v entries (256 threads x 4 tokens each).
func gumbelBlocks(v int) int { return (v + 1023) / 1024 }

// newGumbelState builds the sampler's pipelines, buffers and bind groups over an existing logits buffer of `vocab`
// f32 values. Every object it creates is handed to `keep` for release. It is independent of DecodeRunner so the
// kernel can be driven directly (gumbel_test.go) against arbitrary logits.
func newGumbelState(c *Context, logits *wgpu.Buffer, vocab int, keep func(release func())) (*gumbelState, error) {
	nb := gumbelBlocks(vocab)
	_, pl1, lay1, err := c.mkPipeline("gumbelStage1", gumbelStage1WGSL)
	if err != nil {
		return nil, err
	}
	_, pl2, lay2, err := c.mkPipeline("gumbelStage2", gumbelStage2WGSL)
	if err != nil {
		return nil, err
	}
	s := &gumbelState{pl1: pl1, pl2: pl2, nb: nb}
	mk := func(size uint64, usage wgpu.BufferUsage) (*wgpu.Buffer, error) {
		b, e := c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Size: size, Usage: usage})
		if e != nil {
			return nil, fmt.Errorf("gpu: gumbel sampler buffer: %w", e)
		}
		keep(b.Release)
		return b, nil
	}
	if s.prm, err = mk(uint64(len(s.scratch)*4), wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst); err != nil {
		return nil, err
	}
	if s.bkey, err = mk(uint64(nb*4), wgpu.BufferUsageStorage); err != nil {
		return nil, err
	}
	if s.bidx, err = mk(uint64(nb*4), wgpu.BufferUsageStorage); err != nil {
		return nil, err
	}
	if s.out, err = mk(4, wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc); err != nil {
		return nil, err
	}
	if s.outStag, err = mk(4, wgpu.BufferUsageMapRead|wgpu.BufferUsageCopyDst); err != nil {
		return nil, err
	}
	bind := func(lay *wgpu.BindGroupLayout, bufs ...*wgpu.Buffer) (*wgpu.BindGroup, error) {
		es := make([]wgpu.BindGroupEntry, len(bufs))
		for i, b := range bufs {
			es[i] = wgpu.BindGroupEntry{Binding: uint32(i), Buffer: b, Size: b.GetSize()}
		}
		bg, e := c.device.TryCreateBindGroup(&wgpu.BindGroupDescriptor{Layout: lay, Entries: es})
		if e != nil {
			return nil, fmt.Errorf("gpu: gumbel sampler bind group: %w", e)
		}
		keep(bg.Release)
		return bg, nil
	}
	if s.bg1, err = bind(lay1, logits, s.prm, s.bkey, s.bidx); err != nil {
		return nil, err
	}
	if s.bg2, err = bind(lay2, s.prm, s.bkey, s.bidx, s.out); err != nil {
		return nil, err
	}
	return s, nil
}

// sample writes the draw parameters, runs `pre` (whatever must precede the draw in the pass: the runner's forward
// plan, or nothing), the two Gumbel dispatches, and reads back the id — 4 bytes. It returns -1 if nothing in the
// row was comparable; the caller decides the fallback.
func (s *gumbelState) sample(c *Context, pre func(*wgpu.ComputePassEncoder), vocab int, invT float32, seed, draw uint64) (int, error) {
	s.scratch = [7]uint32{uint32(vocab), math.Float32bits(invT), uint32(seed), uint32(seed >> 32), uint32(draw), uint32(draw >> 32), uint32(s.nb)}
	if err := c.queue.TryWriteBuffer(s.prm, 0, wgpu.ToBytes(s.scratch[:])); err != nil {
		return 0, err
	}
	enc, err := c.device.TryCreateCommandEncoder(nil)
	if err != nil {
		return 0, err
	}
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	if pre != nil {
		pre(pass)
	}
	pass.SetPipeline(s.pl1)
	pass.SetBindGroup(0, s.bg1, nil)
	pass.DispatchWorkgroups(uint32(s.nb), 1, 1)
	pass.SetPipeline(s.pl2)
	pass.SetBindGroup(0, s.bg2, nil)
	pass.DispatchWorkgroups(1, 1, 1)
	if err := pass.TryEnd(); err != nil {
		pass.Release()
		return 0, fmt.Errorf("gpu: end compute pass: %w", err)
	}
	pass.Release()
	if err := enc.TryCopyBufferToBuffer(s.out, 0, s.outStag, 0, 4); err != nil {
		return 0, fmt.Errorf("gpu: copy sampled id→stage: %w", err)
	}
	cmd, err := enc.TryFinish(nil)
	if err != nil {
		return 0, err
	}
	defer cmd.Release()
	c.queue.Submit(cmd)
	st := wgpu.MapAsyncStatus(0)
	if err := s.outStag.TryMapAsync(wgpu.MapModeRead, 0, 4, func(m wgpu.MapAsyncStatus) { st = m }); err != nil {
		return 0, err
	}
	c.device.Poll(true, nil)
	if st != wgpu.MapAsyncStatusSuccess {
		return 0, fmt.Errorf("gpu: gumbel sample map failed: %v", st)
	}
	id := int(wgpu.FromBytes[int32](s.outStag.GetMappedRange(0, 4))[0])
	s.outStag.TryUnmap()
	return id, nil
}

func (r *DecodeRunner) ensureSample() error {
	if r.smp != nil {
		return nil
	}
	s, err := newGumbelState(r.c, r.lastLogits, r.vocab, func(f func()) { r.keep = append(r.keep, f) })
	if err != nil {
		return err
	}
	r.smp = s
	return nil
}

// RunSample runs one token's forward and draws the NEXT token on-device by Gumbel-max, returning only the id:
// two extra dispatches on the logits the LM head just wrote, and a 4-byte readback instead of vocab*4. (seed,
// draw) are the sampler's own (decoder.Sampler.NextDraw), so this is the draw the host would have made.
func (r *DecodeRunner) RunSample(x []float32, pos, ropePos int, invT float32, seed, draw uint64) (int, error) {
	if err := r.ensureSample(); err != nil {
		return 0, err
	}
	if err := r.writeInputs(x, pos, ropePos); err != nil {
		return 0, err
	}
	id, err := r.smp.sample(r.c, r.record, r.vocab, invT, seed, draw)
	if err != nil {
		return 0, err
	}
	if id >= 0 {
		return id, nil
	}
	// Nothing comparable (an all-masked row): the host draws the argmax; read the row and do the same.
	return r.argmaxOfLastLogits()
}

// argmaxOfLastLogits reads the logits the last dispatch left in r.lastLogits and returns their (first) argmax.
func (r *DecodeRunner) argmaxOfLastLogits() (int, error) {
	c := r.c
	enc, err := c.device.TryCreateCommandEncoder(nil)
	if err != nil {
		return 0, err
	}
	defer enc.Release()
	if err := enc.TryCopyBufferToBuffer(r.lastLogits, 0, r.stag, 0, uint64(r.vocab*4)); err != nil {
		return 0, err
	}
	cmd, err := enc.TryFinish(nil)
	if err != nil {
		return 0, err
	}
	defer cmd.Release()
	c.queue.Submit(cmd)
	st := wgpu.MapAsyncStatus(0)
	if err := r.stag.TryMapAsync(wgpu.MapModeRead, 0, uint64(r.vocab*4), func(m wgpu.MapAsyncStatus) { st = m }); err != nil {
		return 0, err
	}
	c.device.Poll(true, nil)
	if st != wgpu.MapAsyncStatusSuccess {
		return 0, fmt.Errorf("gpu: argmax fallback map failed: %v", st)
	}
	row := wgpu.FromBytes[float32](r.stag.GetMappedRange(0, uint(r.vocab*4)))
	best, bi := float32(math.Inf(-1)), 0
	for i, v := range row {
		if v > best {
			best, bi = v, i
		}
	}
	r.stag.TryUnmap()
	return bi, nil
}
