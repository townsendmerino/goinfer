package serveapp

import (
	"encoding/json"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/media"
)

const (
	imagesOmittedHeader = media.ImagesOmittedHeader
	olderImageNote      = media.OlderImageNote
	olderAudioNote      = media.OlderAudioNote
)

var noteOmittedImages = media.NoteOmittedImages

// omitChatHistoryImages rewrites earlier OpenAI chat messages' media parts into omission notes.
func omitChatHistoryImages(msgs []chatMessage) int {
	cs := make([]*json.RawMessage, len(msgs))
	for i := range msgs {
		cs[i] = &msgs[i].Content
	}
	return media.OmitHistoryMedia(cs, map[string]string{"image_url": olderImageNote, "input_audio": olderAudioNote})
}

// omitAnthropicHistoryImages rewrites earlier Anthropic messages' image parts into omission notes.
func omitAnthropicHistoryImages(msgs []anthropicMessage) int {
	cs := make([]*json.RawMessage, len(msgs))
	for i := range msgs {
		cs[i] = &msgs[i].Content
	}
	return media.OmitHistoryImages(cs, "image")
}
