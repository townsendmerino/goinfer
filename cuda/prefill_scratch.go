//go:build cuda

package cuda

import (
	"fmt"
	"os"
	"strconv"
)

// prefillWidths are the default pass widths the build chooses among (prefillWidthFor), widest first; each a multiple of the block
// size 64. The last is prefillDefaultChunk.
var prefillWidths = []int{2048, 1024, prefillDefaultChunk}

// prefillWidthFor is the build's default pass width: the widest of prefillWidths whose pass (pass(rows), its device scratch) fits in
// free less margin less attach, the VRAM left after the build beyond what must stay free beside a pass (marginBesidePass) and what
// is still to attach. When none of the wider ones fits it is prefillDefaultChunk, whose pass the context margin holds. Measurement and gates:
// docs/tasks/task-kernel-survey-2026-10.md, "KS-B8 build".
func prefillWidthFor(free, margin, attach int64, pass func(rows int) int64) int {
	room := free - margin - attach
	for _, w := range prefillWidths[:len(prefillWidths)-1] {
		if pass(w) <= room {
			return w
		}
	}
	return prefillDefaultChunk
}

// choosePrefillWidth sets prefillWidth from the free VRAM after the build (the executor thread, the build's last step): a pass is
// priced at its scratch plus the span bound at the full context (PrefillSpanScratchBytes), which is at least admission's price for
// every prompt up to the context at that width. An unreadable figure keeps prefillDefaultChunk, and so does a windowed model: its slack
// bounds a pass (prefillPassRows), and a wider default would move its passes off the 512-row width that slack was sized for.
func (r *cudaResident) choosePrefillWidth() {
	r.prefillWidth = prefillDefaultChunk
	if r.kvWin {
		return
	}
	free, err := cudaFreeVRAM(r)
	if memReadSeenForTest != nil {
		memReadSeenForTest("prefill", int64(free), err == nil)
	}
	if err != nil {
		return
	}
	var line string
	r.prefillWidth, line = r.prefillWidthChoice(free)
	fmt.Fprint(os.Stderr, line)
}

// prefillWidthChoice is choosePrefillWidth's decision over the free VRAM it read: the width (prefillWidthFor over a pass's scratch plus
// its span bound at the full context, beside the margin kept beside a pass and the companion reserve) and the line the build logs.
func (r *cudaResident) prefillWidthChoice(free uint64) (int, string) {
	pass := func(rows int) int64 { return r.PrefillScratchBytes(rows) + r.PrefillSpanScratchBytes(rows, r.ctxCap) }
	keep := r.marginBesidePass()
	w := prefillWidthFor(int64(free), keep, r.extraBytes, pass)
	return w, fmt.Sprintf("cuda: prefill width %d rows (%.0f MiB of scratch; %.0f MiB free beyond the %.0f MiB kept beside a pass); passes take up to %d rows\n",
		w, float64(pass(w))/(1<<20), float64(int64(free)-keep-r.extraBytes)/(1<<20), float64(keep)/(1<<20),
		min(r.prefillPassRowsAt(w), r.ctxCap))
}

// prefillBaseRows is a pass's width before the per-prompt bounds: GOINFER_PREFILL_CHUNK when it parses to a positive integer, else
// the build's choice, lowered to the width a previous prompt learned the card can hold. An unusable knob value is ignored, not an
// error: a typo must not take a model off the fast path.
func (r *cudaResident) prefillBaseRows() int { return r.prefillBaseRowsAt(r.prefillWidth) }

// prefillBaseRowsAt is prefillBaseRows for a build whose chosen width is width.
func (r *cudaResident) prefillBaseRowsAt(width int) int {
	chunk := prefillDefaultChunk
	if width > 0 {
		chunk = width
	}
	if n, err := strconv.Atoi(r.knobValue("GOINFER_PREFILL_CHUNK")); err == nil && n > 0 {
		chunk = n
	}
	if learned := int(r.prefillChunkCap.Load()); learned > 0 && learned < chunk {
		chunk = learned // a previous prompt already found the default too wide for this card
	}
	return chunk
}

// prefillPassRows is the most rows one batched pass takes now: prefillBaseRows, lowered to a windowed model's slack.
// prefillChunked cuts by it; PrefillChunkRows reports it to admission.
func (r *cudaResident) prefillPassRows() int { return r.prefillPassRowsAt(r.prefillWidth) }

// prefillPassRowsAt is prefillPassRows for a build whose chosen width is width.
func (r *cudaResident) prefillPassRowsAt(width int) int {
	chunk := r.prefillBaseRowsAt(width)
	if r.kvWin {
		chunk = min(chunk, kvWindowSlack+1) // a windowed layer's slack bounds the rows one pass can add (kvRoomFor)
	}
	return chunk
}

// PrefillChunkRows (decoder.ResidentPrefillChunks, matched structurally: TestKV3_residentReportsItsChunks holds the wiring) is the most rows one pass of a long prompt's prefill takes.
func (r *cudaResident) PrefillChunkRows() int { return r.prefillPassRows() }

// PrefillTailMinRows (decoder.ResidentPrefillTailMin) is the fewest rows a prefill's last pass may have: a shorter one runs the
// exact kernels while the pass before it ran the fused ones (attnFusedMinRows).
func (r *cudaResident) PrefillTailMinRows() int { return attnFusedMinRows }

// prefillScratchSeenForTest, when set (tests only), is told the byte size of each scratch buffer prefillCore asks for, before the
// allocator's rounding, so a test can hold PrefillScratchBytes to the pass.
var prefillScratchSeenForTest func(bytes int64)

// prefillPassDoneForTest, when set (tests only), is called after each pass of a chunked prefill completes, so a test can split
// prefillScratchSeenForTest's figures by pass.
var prefillPassDoneForTest func()

// scratchBufBytes is what the device charges for one scratch buffer of n 4-byte elements: the request, rounded up to the
// allocator's quantum from a quantum upward (allocRoundSlack). A request of no elements is one (prefillCore's af and ai).
func scratchBufBytes(n int) int64 {
	b := int64(max(n, 1)) * 4
	return b + allocRoundSlack(b)
}

// PrefillScratchBytes (decoder.ResidentPrefillChunks) is what one ordinary prefill pass of rows rows allocates on the device beyond
// the standing buffers: the M-row buffers prefillCore creates, each after the allocator's rounding. It prices a pass whose tail is
// the last row only (PrefillLast), text only; the all-logits tails of speculative verify add an M x vocab logits block, an image pass
// its block table, and neither is a prompt's prefill. It mirrors prefillCore's allocations by the same arithmetic, and
// TestKV3_prefillScratchBytes_matchesThePass holds it to them.
func (r *cudaResident) PrefillScratchBytes(rows int) int64 {
	M := rows
	if M <= 0 {
		return 0
	}
	maxQDim, maxKvDim := r.prefillMaxGeom()
	hidden, inter := r.hidden, r.inter
	if r.eModel {
		for l := range r.layers {
			inter = max(inter, r.layers[l].ffnI)
		}
	}
	var sum int64
	for _, n := range []int{
		M * hidden,        // xB
		M * hidden / 4, M, // aqB, aScB
		M * maxQDim, M * maxKvDim, M * maxKvDim, // qBb, kBb, vBb
		M * maxQDim,        // cctxB
		M * maxQDim / 4, M, // cqB, cScB
		M * hidden / 4, M, // mqB, mScB
		M * inter, M * inter, // gOb, uOb
		M * inter / 4, M, M * inter, // dqB, dScB, dScrB
	} {
		sum += scratchBufBytes(n)
	}
	if dp := r.dnet; dp != nil {
		for _, n := range []int{M * dp.convDim, M * dp.nv, M * dp.nv, M * dp.valueDim, M * dp.convDim, M * dp.keyDim, M * dp.keyDim,
			M * dp.nv * 2, M * dp.valueDim, M * dp.valueDim, M * dp.valueDim / 4, M, M * 2 * maxQDim, M * maxQDim} {
			sum += scratchBufBytes(n)
		}
	}
	if r.sandwich || r.postOnly || r.parallelBlock {
		sum += scratchBufBytes(M * hidden)
	}
	if r.pleP > 0 {
		sum += scratchBufBytes(r.nLayers*M*r.pleP) + scratchBufBytes(M*r.pleP)
	}
	return sum
}
