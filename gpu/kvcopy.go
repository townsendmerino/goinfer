//go:build gpu

package gpu

import (
	"fmt"

	"github.com/oliverbestmann/webgpu/wgpu"
	"github.com/townsendmerino/goinfer/decoder"
)

var _ decoder.ResidentKVCopier = (*residentDecoder)(nil)

// CopyKV (decoder.ResidentKVCopier) copies rows [start, end) of every per-sequence buffer (slotKVBuffers) from slot src
// to slot dst. Each buffer is position-major over the resident context, so a row is its size over ctxCap bytes and a
// range is one buffer-to-buffer copy. The copies run in one command buffer on the queue, after everything submitted
// before them, and the call waits for them.
func (rd *residentDecoder) CopyKV(src, dst, start, end int) error {
	n := rd.KVSlots()
	switch {
	case src < 0 || src >= n || dst < 0 || dst >= n || src == dst:
		return fmt.Errorf("gpu: CopyKV slots %d -> %d of %d", src, dst, n)
	case start < 0 || start > end || start%64 != 0 || end%64 != 0:
		return fmt.Errorf("gpu: CopyKV rows [%d, %d) are not whole 64-row blocks", start, end)
	case end > rd.ctxCap:
		return &decoder.KVCopyDecline{Reason: fmt.Sprintf("rows up to %d are past the %d-position cache", end, rd.ctxCap)}
	case len(rd.slots) < 2:
		return &decoder.KVCopyDecline{Reason: "one KV slot"}
	case rd.rm.mamba != nil || rd.rm.dnet != nil:
		return &decoder.KVCopyDecline{Reason: "recurrent state is not part of a KV slot"}
	}
	if start == end {
		return nil
	}
	type cp struct {
		s, d     *wgpu.Buffer
		off, len uint64
	}
	var copies []cp
	seen := map[*wgpu.Buffer]bool{} // a buffer two layers share is copied once
	sl, dl := rd.slots[src].rm.layers, rd.slots[dst].rm.layers
	for l := range sl {
		sp, dp := slotKVBuffers(&sl[l]), slotKVBuffers(&dl[l])
		for k := range sp {
			s, d := *sp[k], *dp[k]
			if s == nil || seen[s] {
				continue
			}
			seen[s] = true
			size := s.GetSize()
			if d == nil || d.GetSize() != size || size%uint64(rd.ctxCap) != 0 {
				return &decoder.KVCopyDecline{Reason: fmt.Sprintf("layer %d: a per-sequence buffer is not whole rows of the cache", l)}
			}
			row := size / uint64(rd.ctxCap)
			c := cp{s, d, uint64(start) * row, uint64(end-start) * row}
			if c.off%4 != 0 || c.len%4 != 0 {
				return &decoder.KVCopyDecline{Reason: fmt.Sprintf("layer %d: a %d-byte row range is not 4-byte aligned", l, row)}
			}
			copies = append(copies, c)
		}
	}
	enc, err := rd.c.device.TryCreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	defer enc.Release()
	for _, c := range copies {
		if err := enc.TryCopyBufferToBuffer(c.s, c.off, c.d, c.off, c.len); err != nil {
			return fmt.Errorf("gpu: CopyKV: %w", err)
		}
	}
	cmd, err := enc.TryFinish(nil)
	if err != nil {
		return err
	}
	defer cmd.Release()
	rd.c.queue.Submit(cmd)
	rd.c.device.Poll(true, nil)
	return nil
}
