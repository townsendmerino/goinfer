package chat_test

import (
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/codec"
)

// codecPath is the census's seam filled by internal/codec. Without a reply router, codec.SettleCalls is the template's
// ParseToolCallsFor, so the parse table's settle column must read the same through it.
var codecPath = chat.CensusPath{
	Segments: codec.Segments,
	Image:    codec.ImageSegments,
	Encode:   codec.Encode,
	Reply:    codec.NewReplySplitter,
	ParseFor: func(tm *chat.Template, out string, tools []chat.Tool) ([]chat.ToolCall, string) {
		calls, lead, _ := codec.SettleCalls(tm, out, tools, nil, "")
		return calls, lead
	},
}

// TestCodecCensus_throughCodec is the codec census run through internal/codec: the routes' path from a conversation to
// token IDs, and from a reply to its reasoning and answer, must match the census goldens byte for byte
// (task-subsystems-2026-10.md §8 SB8, gate G-CX-arm).
func TestCodecCensus_throughCodec(t *testing.T) { chat.CodecCensusThrough(t, codecPath) }

// TestToolCallParseCensus_throughCodec is the tool-call parse census with the settle step through internal/codec.
func TestToolCallParseCensus_throughCodec(t *testing.T) {
	chat.ToolCallParseCensusThrough(t, codecPath)
}
