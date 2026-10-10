package multimodal

import (
	"fmt"
	"slices"
	"strings"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// This is the one image-block splice every vision caller shares (goinfer-serve's vision routes, goinfer-chat --image, the examples);
// serveapp's spliceImageBlock is a one-line delegate to it.

// SpliceImageBlock re-tags the image block as its own Special segment, splitting the content segment that contains it.
//
// The block does NOT start its segment: the template renders the role prefix and the message body into one non-special span, so a
// Gemma-4 user turn arrives as "user\n<start_of_image>…<end_of_image>\nhello", and the block must be found inside the span, not as
// its prefix.
func SpliceImageBlock(segs []tokenizer.Segment, block string) ([]tokenizer.Segment, error) {
	// Search from the END and splice the LAST occurrence, not the first. The block is always appended to the CURRENT (last) user turn,
	// which renders last; an EARLIER turn that happens to contain the same literal text (a user asking what the sentinel means, say) must
	// not be mistaken for it. Splicing the wrong occurrence reopens the special-token-forging class: the real image tokens stay unspliced
	// plain text (and fail downstream with a misleading "template mismatch"), while unrelated earlier text gets tagged Special and parsed as
	// sentinels it was never meant to be.
	segIdx := -1
	for i, seg := range slices.Backward(segs) {
		if !seg.Special && strings.Contains(seg.Text, block) {
			segIdx = i
			break
		}
	}
	if segIdx < 0 {
		return nil, fmt.Errorf("vision: the image block was not found in the rendered prompt " +
			"(template changed?); refusing to encode its sentinels as ordinary text")
	}
	sg := segs[segIdx]
	i := strings.LastIndex(sg.Text, block) // last occurrence within the segment too, same reason
	out := make([]tokenizer.Segment, 0, len(segs)+2)
	out = append(out, segs[:segIdx]...)
	if before := sg.Text[:i]; before != "" {
		out = append(out, tokenizer.Segment{Text: before}) // the template's role prefix
	}
	out = append(out, tokenizer.Segment{Text: block, Special: true})
	if after := sg.Text[i+len(block):]; after != "" {
		out = append(out, tokenizer.Segment{Text: after}) // the user's own words
	}
	out = append(out, segs[segIdx+1:]...)
	return out, nil
}

// SpliceImageBlocks is SpliceImageBlock for several images: blocks in prompt order, spliced from the END, so the last block is the
// last occurrence of its text and each earlier block is the last occurrence before the one after it. That keeps the last-occurrence
// rule (an earlier turn quoting the sentinel text is never taken for an image) for every block; a block the user's own words forge
// after the real ones takes a real block's place, and the run-length check downstream then refuses the request.
func SpliceImageBlocks(segs []tokenizer.Segment, blocks []string) ([]tokenizer.Segment, error) {
	if len(blocks) == 0 {
		return nil, fmt.Errorf("vision: no image block to splice")
	}
	head := segs
	var tail []tokenizer.Segment // already spliced: from the latest block to the end
	for k := len(blocks) - 1; k >= 0; k-- {
		out, err := SpliceImageBlock(head, blocks[k])
		if err != nil {
			return nil, fmt.Errorf("vision: image %d of %d: %w", k+1, len(blocks), err)
		}
		// The block just spliced is the last Special segment whose text is blocks[k] (SpliceImageBlock searches from the end
		// and the template's own Special segments never carry a block's text): everything from it on is settled.
		cut := -1
		for i, sg := range slices.Backward(out) {
			if sg.Special && sg.Text == blocks[k] {
				cut = i
				break
			}
		}
		tail = append(slices.Clone(out[cut:]), tail...)
		head = out[:cut]
	}
	return append(slices.Clone(head), tail...), nil
}
