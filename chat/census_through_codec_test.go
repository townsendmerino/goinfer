package chat_test

import (
	"testing"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/codec"
)

// TestCodecCensus_throughCodec is the codec census run through internal/codec: the routes' path from a conversation to
// token IDs must match the census goldens byte for byte (task-subsystems-2026-10.md §8 SB8, gate G-CX-arm).
func TestCodecCensus_throughCodec(t *testing.T) {
	chat.CodecCensusThrough(t, codec.Segments, codec.ImageSegments, codec.Encode)
}
