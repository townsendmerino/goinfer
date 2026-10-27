// Package codec is the conversation codec: the one path from a conversation (system prompt, turns, tool declarations,
// image and audio blocks) to the token IDs a model reads. Its parts are the public packages chat (templates and tool-call
// dialects) and tokenizer; the serving routes and goinfer-chat call this package instead of composing them, so they
// cannot disagree about what a model sees.
//
// Every prompt is encoded with tokenizer.EncodeSegments and no added BOS: the template writes its own BOS marker, and the
// template's control tokens are Special segments while everything a user, a tool or the model wrote is plain text, so a
// marker typed into a message stays text instead of forging a turn.
//
// chat's codec census (chat/codec_census_test.go, TestCodecCensus_throughCodec) checks this package against the census
// goldens byte for byte; docs/tasks/task-subsystems-2026-10.md §8 SB8 registers that gate.
package codec

import (
	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// Template is the chat template tk's checkpoint declares: chat.Detect over its chat template, with its vocabulary as the
// probe for a bare checkpoint. chat.ErrUnknownTemplate means the caller falls back to a raw completion.
func Template(tk *tokenizer.Tokenizer) (*chat.Template, error) {
	return chat.Detect(chat.Meta{ChatTemplate: tk.ChatTemplate(), HasToken: tk.Has})
}

// Segments renders system and turns through tm, with the tool declarations when tools is non-empty.
func Segments(tm *chat.Template, system string, turns []chat.Turn, tools []chat.Tool) []chat.Segment {
	if len(tools) > 0 {
		return tm.RenderToolsSegments(system, turns, tools)
	}
	return tm.RenderSegments(system, turns)
}

// ImageSegments renders a prompt whose last user turn already holds block (an image or audio placeholder) and re-tags the
// block as its own Special segment. It fails, rather than encode the block's sentinels as text, when the rendered prompt
// does not contain it.
func ImageSegments(tm *chat.Template, system string, turns []chat.Turn, block string) ([]chat.Segment, error) {
	return multimodal.SpliceImageBlock(tm.RenderSegments(system, turns), block)
}

// ImagesSegments is ImageSegments for several blocks, in prompt order.
func ImagesSegments(tm *chat.Template, system string, turns []chat.Turn, blocks []string) ([]chat.Segment, error) {
	return multimodal.SpliceImageBlocks(tm.RenderSegments(system, turns), blocks)
}

// Encode is the token IDs of a rendered prompt.
func Encode(tk *tokenizer.Tokenizer, segs []chat.Segment) ([]int, error) {
	return tk.EncodeSegments(segs, false)
}

// EncodePrompt is Encode of Segments.
func EncodePrompt(tk *tokenizer.Tokenizer, tm *chat.Template, system string, turns []chat.Turn, tools []chat.Tool) ([]int, error) {
	return Encode(tk, Segments(tm, system, turns, tools))
}
