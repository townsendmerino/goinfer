//go:build linux

package gpudev

import (
	"fmt"

	gc "github.com/eitamring/gocudrv/cuda"
)

// HostCopy is one host→device pair for UploadBatch: Src's bytes go into Dst at
// Dst's bind offset (Buffer.At).
//
// Src is []byte rather than a generic []T because a batch is heterogeneous by
// nature — the motivating caller loads an expert's f32 weights and its f16 scales
// in the same pass — and a generic batch would force one element type across all
// of them. Both of that caller's sites already hold byte slices for the same
// reason, so this costs nothing at the call site. Upload[T] stays generic; this
// is the byte-level sibling, not a replacement.
type HostCopy struct {
	Dst Buffer
	Src []byte
}

// UploadBatch uploads every pair and then synchronizes ONCE.
//
// WHY THIS EXISTS: THE SYNC COUNT IS THE COST, NOT THE COPY. Upload ends in a full
// device Synchronize — correct, and cheap when uploads are per-request (a corpus, a
// tower, a batch of patches), which is what it was written for. goinfer's C′ expert
// cache is the case it was not: a MoE decode token loads ~120 expert slots, two
// uploads each, so ~240 synchronizes land on a single token. Measured on an RTX
// 2070 SUPER that is ~3.6 ms of a 64 ms token — 5.6%, paid for nothing, since the
// bytes are already in flight and one sync at the end would have covered them all.
//
// This does exactly that: issue all the copies, wait once. It is the H2D twin of
// CopyDeviceBatch, and the same reasoning produced both.
//
// WHAT IT DOES NOT DO — and this was checked against the caller rather than assumed.
// It does not coalesce adjacent pairs the way CopyDeviceBatch does. On that path
// merging pays because a snapshot walks contiguous ranges; here it would collapse
// nothing. goinfer's loadExpertSlot derives its source offset from the ROUTED
// EXPERT index and its destination offset from the LRU VICTIM slot:
//
//	src: e    * perExpert   // which expert routing picked
//	dst: slot * perExpert   // which slot the LRU evicted
//
// Those two are independent, so consecutive loads are essentially never adjacent in
// BOTH — which is what merging requires — and the two uploads within one expert
// target different buffers (weights and scales) so they cannot merge either. A
// coalescing pass would be a linear scan that always returns its input. If a caller
// with genuinely contiguous ranges turns up, revisit; the code is in copy_coalesce.go.
//
// THE CORRECTNESS PROPERTY IS PRESERVED, NOT WEAKENED. Upload's synchronize exists
// for a real race — gocudrv streams are CU_STREAM_NON_BLOCKING and unordered against
// the legacy null stream, and cuMemcpyHtoD from PAGEABLE memory returns once the
// source is staged with the transfer still in flight. It surfaced once as an
// intermittently wrong GEMM. This batch still synchronizes before it returns, so the
// guarantee a caller gets is identical; only its cost is amortized. That is the whole
// difference from the async proposal, which buys the same 3.6 ms but moves ordering
// into a contract on the caller (see goinfer's aikit-subrange-async-upload.md, and
// its decision).
//
// Every pair is validated before any copy is issued, so a bad pair fails without
// leaving half the batch applied.
func UploadBatch(copies []HostCopy) error {
	if len(copies) == 0 {
		return nil
	}
	// Validate every pair against the FIRST destination's context before issuing
	// anything. The single trailing Synchronize is per-context, so a batch spanning
	// two contexts would return with copies on the second still in flight — the exact
	// class of bug Upload's synchronize was added to prevent. Refused, not sorted.
	cx := copies[0].Dst.cx
	for i, c := range copies {
		if err := checkHostCopy(c, cx); err != nil {
			return fmt.Errorf("cuda: upload %d of %d: %w", i, len(copies), err)
		}
	}
	// Pre-sync once for the whole batch — the write-after-read hazard
	// Buffer.upload documents (audit C-01): queued launches may still be reading
	// these destinations.
	if cx != nil {
		if err := cx.Synchronize(bg); err != nil {
			return err
		}
	}
	var issued bool
	for i, c := range copies {
		if len(c.Src) == 0 {
			continue
		}
		if err := c.Dst.b.CopyFromAt(bg, int(c.Dst.off), c.Src); err != nil {
			return fmt.Errorf("cuda: upload %d of %d: %w", i, len(copies), err)
		}
		issued = true
	}
	if !issued || cx == nil {
		return nil
	}
	return cx.Synchronize(bg)
}

// checkHostCopy validates one pair in aikit's error vocabulary. The messages name
// the size and offset because a caller meeting one of these is looking at a batch of
// 240 and needs to know which pair was wrong — gocudrv's own errors name neither.
func checkHostCopy(c HostCopy, cx *gc.Context) error {
	// A mapped-host Buffer is a device-usable pointer into pinned HOST memory and
	// has no gocudrv buffer behind it; uploading "into" one is a host memcpy wearing
	// a DMA's clothes. Refused by name rather than served silently.
	if c.Dst.raw != 0 {
		return fmt.Errorf("cuda: upload into a mapped-host buffer — those bytes live in host RAM; " +
			"write them directly instead")
	}
	if c.Dst.b == nil {
		return fmt.Errorf("cuda: upload into a nil buffer")
	}
	if got, want := c.Dst.b.Bytes(), uint64(c.Dst.off)+uint64(len(c.Src)); got < want {
		return fmt.Errorf("cuda: upload of %d bytes at offset %d overruns a %d-byte buffer",
			len(c.Src), c.Dst.off, got)
	}
	if c.Dst.cx != cx {
		return fmt.Errorf("cuda: upload into a buffer on a different context than the batch's first — " +
			"one Synchronize cannot cover both; split the batch per context")
	}
	return nil
}
