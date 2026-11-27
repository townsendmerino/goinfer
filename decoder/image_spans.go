package decoder

import (
	"github.com/townsendmerino/goinfer/internal/runtime/cpuengine"
	"github.com/townsendmerino/goinfer/internal/runtime/vlprep"
)

// ImageSpan is one image's placeholder run in a prompt: ids[Pos, Pos+Len)
// are replaced by that image's projected features. Hash identifies the image for resident prefix reuse; 0 makes no claim
// and never matches.
type ImageSpan = cpuengine.ImageSpan

var (
	imageSpansTotal       = vlprep.TotalTokens
	checkImageSpans       = vlprep.CheckSpans
	spliceImageSpans      = vlprep.Splice
	imageSpanBlocks       = vlprep.Blocks
	residentImageBlocksOf = vlprep.ResidentBlocks
	residentImageClaimsOf = vlprep.ResidentClaims
)
