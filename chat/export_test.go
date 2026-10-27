package chat

import (
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// CodecCensusThrough runs the codec census in check mode with its render, splice and encode steps replaced, for an
// external test that cannot be in this package (one importing a package that imports chat).
func CodecCensusThrough(t *testing.T,
	segments func(tm *Template, system string, turns []Turn, tools []Tool) []Segment,
	image func(tm *Template, system string, turns []Turn, block string) ([]Segment, error),
	encode func(tk *tokenizer.Tokenizer, segs []Segment) ([]int, error),
) {
	codecCensus(t, censusPath{segments: segments, image: image, encode: encode}, false)
}
