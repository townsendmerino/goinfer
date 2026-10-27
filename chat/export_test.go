package chat

import (
	"testing"

	"github.com/townsendmerino/goinfer/tokenizer"
)

// CensusPath is the census's seam, for an external test that cannot be in this package (one importing a package that
// imports chat): the render, splice and encode steps, the reply splitter, and the tool-call settle.
type CensusPath struct {
	Segments func(tm *Template, system string, turns []Turn, tools []Tool) []Segment
	Image    func(tm *Template, system string, turns []Turn, block string) ([]Segment, error)
	Encode   func(tk *tokenizer.Tokenizer, segs []Segment) ([]int, error)
	Reply    func(tm *Template, turns []Turn) *ReplySplitter
	ParseFor func(tm *Template, out string, tools []Tool) ([]ToolCall, string)
}

func (p CensusPath) internal() censusPath {
	return censusPath{segments: p.Segments, image: p.Image, encode: p.Encode, reply: p.Reply, parseFor: p.ParseFor}
}

// CodecCensusThrough runs the codec census through p, in check mode.
func CodecCensusThrough(t *testing.T, p CensusPath) { codecCensus(t, p.internal(), false) }

// ToolCallParseCensusThrough runs the tool-call parse census through p, in check mode.
func ToolCallParseCensusThrough(t *testing.T, p CensusPath) {
	toolCallParseCensus(t, p.internal(), false)
}
