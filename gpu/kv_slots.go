//go:build gpu

package gpu

import (
	"fmt"
	"os"
	"runtime"
	"slices"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/decoder"
)

// MC1 on WebGPU (docs/tasks/task-concurrency-2026-09.md): several resident KV caches ("slots"), one bound at a time.
//
// A slot here is a whole DecodeRunner, not a pointer swap as on CUDA and Metal: newDecodeRunner bakes every KV buffer
// into bind groups it builds once (ropeStore, vStore, qkvFinalize, the attention binds, the MLA latent store and
// attention), so a runner can only ever read and write the caches it was built over. Each slot therefore holds a
// runModel whose layers share every weight buffer with slot 0's and own only their KV (K, V, the int8 scales, the MLA
// latent), plus a runner built over it and its own lazily grown ForwardN pool. Binding a slot swaps rd.rm, rd.runner
// and rd.batch; everything that writes or reads KV (Forward, ForwardN, UploadKV, PrefillLast's runModelToModelW) reads
// those three at call time, so none of it changes.

// webgpuKVSlot is one resident KV slot: its runModel (slot 0's weights, its own KV), the runner built over it, and its
// ForwardN verify pool (batch[0] aliases runner once grown). The bound slot's copies live in residentDecoder's rm /
// runner / batch; its entry here is refreshed on a switch.
type webgpuKVSlot struct {
	rm     runModel
	runner *DecodeRunner
	batch  []*DecodeRunner
}

// webgpuSlotHeadroom is the device memory a further slot must leave free on a discrete GPU: the per-call prefill
// scratch (PrefillLastW8A8 allocates its activations per call) and the lazily grown ForwardN runners need room after
// the slots are built. The same margin cuda/resident.go's kvSlotsFit keeps.
const webgpuSlotHeadroom = 384 << 20

// hostRAMBytes / hostRAMAvailable are indirected so a test can price slots against a machine's worth of RAM it does
// not have.
var (
	hostRAMBytes     = decoder.HostRAMBytes
	hostRAMAvailable = decoder.HostRAMAvailableBytes
)

// slotAllocHook, when set (tests only), runs before slot `slot` allocates layer `layer`'s KV; an error it returns is
// treated as that allocation failing, which is how the clamp — and the release of a half-built slot — is exercised
// without exhausting a device.
var slotAllocHook func(slot, layer int) error

// kvSlotsWithin is the unified-memory clamp arithmetic (metal/backend.go's, restated because the gpu module cannot
// import metal): the largest slot count up to want whose build — base bytes for the first slot plus perSlot for each
// further one — fits budget, and never below 1 (nothing on this backend prices the first slot; that is the load-time
// fit guard's job).
func kvSlotsWithin(want int, budget, base, perSlot int64) int {
	n := max(1, want)
	for n > 1 && base+int64(n-1)*perSlot > budget {
		n--
	}
	return n
}

// darwinKVSlots prices slots where the device's buffers ARE host RAM (darwin), against the same two ceilings Metal's guard
// takes the smaller of (metalMemoryCeiling):
//   - the fit guard's share of physical RAM, against everything the build holds: the device weights, the host copy a
//     unified-memory backend keeps beside them (decoder.Model.ResidentHostCopyBytes), and one slot of KV each;
//   - the memory available when the build STARTED (avail0, read before its first upload, 0 when unknown), against only what
//     the build then allocated: the device weights and the slots. The host copy was already resident when avail0 was read, so
//     it is not counted again, and a live figure read after the upload would count the weights twice
//     (docs/tasks/task-concurrency-2026-09.md, "priced after the build").
//
// An unreadable RAM size grants one, as metalKVSlots does.
func darwinKVSlots(want int, ram, avail0, weights, hostCopy, perSlot int64) int {
	if ram <= 0 {
		return 1
	}
	n := kvSlotsWithin(want, int64(decoder.WeightsMemFraction*float64(ram)), weights+hostCopy+perSlot, perSlot)
	if avail0 > 0 {
		n = min(n, kvSlotsWithin(want, avail0, weights+perSlot, perSlot))
	}
	return n
}

// slotKVBuffers returns the addresses of every per-sequence buffer on a layer — what a new slot allocates afresh.
// Everything else on runLayer is a weight or a model constant and is shared. The Mamba / DeltaNet state is also
// per-sequence, but a model that has it keeps one slot (kvSlotsRequest), so it is never cloned.
func slotKVBuffers(l *runLayer) []**wgpu.Buffer {
	return []**wgpu.Buffer{&l.kCache, &l.vCache, &l.kScale, &l.vScale, &l.latCache}
}

// kvBytesPerSlot is the device bytes one slot's KV takes: slot 0's per-sequence buffers, each counted once.
func kvBytesPerSlot(rm *runModel) int64 {
	seen := map[*wgpu.Buffer]bool{}
	var n int64
	for i := range rm.layers {
		for _, p := range slotKVBuffers(&rm.layers[i]) {
			if b := *p; b != nil && !seen[b] {
				seen[b] = true
				n += int64(b.GetSize())
			}
		}
	}
	return n
}

// kvSlotsRequest is how many slots a build asks for: the model's request (decoder.Model.ResidentKVSlotsRequest,
// already 1 for a family with recurrent state), and 1 whenever the runModel carries Mamba or DeltaNet state, which is
// mutated in place and is not part of a slot — the same refusal cudaKVSlotsRequest makes, held here as well so the
// runner never depends on the decoder's check alone.
func kvSlotsRequest(m *decoder.Model, rm *runModel) int {
	if rm.mamba != nil || rm.dnet != nil {
		return 1
	}
	return m.ResidentKVSlotsRequest()
}

// newKVSlot allocates slot `slot`: a runModel sharing rd.rm's weights with fresh KV buffers of the same sizes, and a
// runner over it. On any error everything it allocated is released and the error returned; on success the returned
// frees release the slot's KV buffers (the runner is released with the slot).
func (rd *residentDecoder) newKVSlot(slot int) (webgpuKVSlot, []func(), error) {
	var frees []func()
	undo := func() {
		for _, f := range slices.Backward(frees) {
			f()
		}
	}
	rm := rd.rm
	rm.layers = slices.Clone(rd.rm.layers)
	fresh := map[*wgpu.Buffer]*wgpu.Buffer{} // a buffer two layers share stays shared in the new slot
	for i := range rm.layers {
		if slotAllocHook != nil {
			if err := slotAllocHook(slot, i); err != nil {
				undo()
				return webgpuKVSlot{}, nil, fmt.Errorf("layer %d KV: %w", i, err)
			}
		}
		for _, p := range slotKVBuffers(&rm.layers[i]) {
			old := *p
			if old == nil {
				continue
			}
			if b, ok := fresh[old]; ok {
				*p = b
				continue
			}
			b, err := rd.c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Label: "kvcache-slot", Size: old.GetSize(),
				Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst | wgpu.BufferUsageCopySrc})
			if err != nil {
				undo()
				return webgpuKVSlot{}, nil, fmt.Errorf("layer %d KV: %w", i, err)
			}
			d := newDeviceBuffer(b, 0)
			frees = append(frees, func() { _ = d.Close() })
			fresh[old], *p = b, b
		}
	}
	r, err := rd.runnerFor(rm)
	if err != nil {
		undo()
		return webgpuKVSlot{}, nil, fmt.Errorf("runner: %w", err)
	}
	if runtime.GOOS != "darwin" {
		// A discrete GPU has no free-memory query on this backend, so the room left after this slot is measured by
		// asking for it. (On darwin the device's memory is host RAM, which webgpuKVSlotsWithin priced up front; an
		// allocation there proves nothing.)
		probe, err := rd.c.device.TryCreateBuffer(&wgpu.BufferDescriptor{Label: "kvslot-headroom", Size: webgpuSlotHeadroom,
			Usage: wgpu.BufferUsageStorage})
		if err != nil {
			r.Release()
			undo()
			return webgpuKVSlot{}, nil, fmt.Errorf("less than %d MiB left after it", webgpuSlotHeadroom>>20)
		}
		probe.Release()
	}
	return webgpuKVSlot{rm: rm, runner: r}, frees, nil
}

// webgpuSlotCtxFloor is the shortest context slotsBeforeContext gives up to: CUDA's floor (cudaCtxCapDefault), so the
// two backends make the same trade.
const webgpuSlotCtxFloor = 4096

// kvBytesPerPosition is what one slot's KV costs per position, from the same geometry BuildResident allocates: K and V
// per layer at the resident precision — f32 4 B/elem, f16 2 B/elem, int8 1 B/elem plus one f32 scale per KV head —
// or one latent row of kvLoRA+qkRope f32 on an MLA model. It assumes every layer holds attention KV, which is true of
// every family that may have more than one slot (a recurrent one keeps one, so it is never priced here);
// TestWebGPUKVSlots_perPositionMatchesTheBuild pins it against kvBytesPerSlot on each KV layout.
func kvBytesPerPosition(m *decoder.Model, kvF16, kvI8 bool) int64 {
	_, nLayers, _, nKV, hd, _, _ := m.Dims()
	if _, kvLoRA, _, qkRope, _, _, _, _, mlaOK := m.MLAResidentParams(); mlaOK {
		return int64(nLayers) * int64(kvLoRA+qkRope) * 4
	}
	kvDim := int64(nKV * hd)
	per := kvDim * 4
	switch {
	case kvI8:
		per = kvDim + int64(nKV)*4
	case kvF16:
		per = kvDim * 2
	}
	return 2 * int64(nLayers) * per
}

// slotsBeforeContext is MC1's "slots before context" on WebGPU (the rule CUDA's ctxForSlots applies): when more than one KV
// slot is requested and the caller did not choose the context (decoder.Model.ResidentContextPinned: a fit-guard auto-pin is a
// one-slot ceiling, not a choice), give up context, down to webgpuSlotCtxFloor, until every requested slot fits; below the
// floor the slot count is clamped instead, by buildKVSlots as always. An explicit -ctx is never shrunk.
//
// darwin only: there the device's memory is host RAM and darwinKVSlots can price a context before anything is allocated. A
// discrete GPU has no free-memory query on this backend, so what fits is learned only by allocating, and no context
// shrinking is done there. KV is linear in the context, so "fits" is monotone and a binary search finds the edge.
// History and the open Vulkan question: docs/code-notes/gpu.md#slotsBeforeContext.
func slotsBeforeContext(m *decoder.Model, ctxCap int, kvF16, kvI8 bool, avail0 int64) int {
	want := m.ResidentKVSlotsRequest()
	if runtime.GOOS != "darwin" || want <= 1 || m.ResidentContextPinned() || ctxCap <= webgpuSlotCtxFloor {
		return ctxCap
	}
	perPos := kvBytesPerPosition(m, kvF16, kvI8)
	ram := hostRAMBytes()
	if perPos <= 0 || ram <= 0 {
		return ctxCap
	}
	weights, hostCopy := m.ResidentWeightBytes(), m.ResidentHostCopyBytes(0)
	fits := func(ctx int) bool {
		return darwinKVSlots(want, ram, avail0, weights, hostCopy, perPos*int64(ctx)) >= want
	}
	return ctxWhereSlotsFit(ctxCap, want, fits)
}

// ctxWhereSlotsFit is slotsBeforeContext's search, separated so its edges are testable without a model: the largest
// context in [webgpuSlotCtxFloor, oneSlot] at which fits holds, oneSlot when it already does, and the floor when not
// even the floor does (the slot clamp then takes over). It logs any shrink.
func ctxWhereSlotsFit(oneSlot, want int, fits func(ctx int) bool) int {
	if fits(oneSlot) {
		return oneSlot
	}
	if !fits(webgpuSlotCtxFloor) {
		fmt.Fprintf(os.Stderr, "webgpu: resident context %d (the floor), not %d, for the %d requested KV slots (--kv-sessions); "+
			"the build grants as many as fit there, and an explicit --ctx keeps a longer context with fewer slots\n",
			webgpuSlotCtxFloor, oneSlot, want)
		return webgpuSlotCtxFloor
	}
	lo, hi := webgpuSlotCtxFloor, oneSlot // fits(lo), !fits(hi)
	for hi-lo > 1 {
		if mid := lo + (hi-lo)/2; fits(mid) {
			lo = mid
		} else {
			hi = mid
		}
	}
	fmt.Fprintf(os.Stderr, "webgpu: resident context %d, not %d, so the %d requested KV slots fit (--kv-sessions); an "+
		"explicit --ctx keeps a longer context with fewer slots\n", lo, oneSlot, want)
	return lo
}

// availBeforeBuild is the memory available before BuildResident's first upload, for darwinKVSlots — read only on
// darwin and only when more than one slot is requested (it execs vm_stat), else 0.
func availBeforeBuild(m *decoder.Model) int64 {
	if runtime.GOOS != "darwin" || m.ResidentKVSlotsRequest() <= 1 {
		return 0
	}
	return hostRAMAvailable()
}

// buildKVSlots adds slots 1..want-1 after BuildResident has built slot 0 (rd.rm / rd.runner). On darwin the count is
// first priced against memory (darwinKVSlots, from the decoder's weight figures, slot 0's exact KV bytes, and avail0
// from availBeforeBuild); everywhere, a slot whose allocation fails stops the build at the slots already made. Either
// clamp logs the count it granted. A clamp never fails the build: slot 0 is the resident the build already had.
func (rd *residentDecoder) buildKVSlots(m *decoder.Model, want int, avail0 int64) {
	if want <= 1 {
		return
	}
	perSlot := kvBytesPerSlot(&rd.rm)
	n := want
	if runtime.GOOS == "darwin" {
		weights, hostCopy := m.ResidentWeightBytes(), m.ResidentHostCopyBytes(0)
		if n = darwinKVSlots(want, hostRAMBytes(), avail0, weights, hostCopy, perSlot); n < want {
			fmt.Fprintf(os.Stderr, "webgpu: %d resident KV slots of %d requested — each costs %.0f MB of KV at the resident context (%d positions); "+
				"with %.0f MB of weights (+%.0f MB host copy), %.0f%% of %.0f MB RAM and %.0f MB available before the build allow %d\n",
				n, want, float64(perSlot)/(1<<20), rd.ctxCap, float64(weights)/(1<<20), float64(hostCopy)/(1<<20),
				100*decoder.WeightsMemFraction, float64(hostRAMBytes())/(1<<20), float64(avail0)/(1<<20), n)
		}
	}
	rd.slots = []webgpuKVSlot{{rm: rd.rm, runner: rd.runner}}
	for s := 1; s < n; s++ {
		sl, frees, err := rd.newKVSlot(s)
		if err != nil {
			fmt.Fprintf(os.Stderr, "webgpu: %d resident KV slots of %d requested — slot %d (%.0f MB of KV at %d positions) did not fit: %v\n",
				s, want, s, float64(perSlot)/(1<<20), rd.ctxCap, err)
			break
		}
		rd.keep = append(rd.keep, frees...)
		rd.slots = append(rd.slots, sl)
	}
	if len(rd.slots) == 1 {
		rd.slots = nil
	}
}

var _ decoder.ResidentKVSlotter = (*residentDecoder)(nil)

// KVSlots / UseKVSlot implement decoder.ResidentKVSlotter (MC1): the slots buildKVSlots allocated, and binding one.
func (rd *residentDecoder) KVSlots() int { return max(1, len(rd.slots)) }

// UseKVSlot binds slot i: every later Forward / ForwardN / UploadKV / PrefillLast reads and writes its KV. Nothing is
// encoded across calls on this backend (each Run records and submits its own command buffer), so nothing is left
// against the previous slot. A bound adapter moves with the switch: generateInto binds the adapter (SetAdapter) BEFORE
// it acquires the slot, so the new slot's runner is given the adapter first — and on an error the binding is unchanged
// — then the old runner's copy is cleared.
func (rd *residentDecoder) UseKVSlot(i int) error {
	if i < 0 || i >= rd.KVSlots() {
		return fmt.Errorf("gpu: KV slot %d out of range (%d allocated)", i, rd.KVSlots())
	}
	if i == rd.kvSlot {
		return nil
	}
	next := &rd.slots[i]
	if rd.adapter != nil {
		if err := next.runner.SetAdapter(rd.adapter); err != nil {
			return err
		}
		_ = rd.runner.SetAdapter(nil) // clearing cannot fail
	}
	rd.slots[rd.kvSlot].batch = rd.batch
	rd.rm, rd.runner, rd.batch = next.rm, next.runner, next.batch
	rd.kvSlot = i
	return nil
}
