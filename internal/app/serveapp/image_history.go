package serveapp

import (
	"encoding/json"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/media"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/wirecore"
)

const (
	imagesOmittedHeader = media.ImagesOmittedHeader
	olderImageNote      = media.OlderImageNote
	olderAudioNote      = media.OlderAudioNote
)

var (
	noteOmittedImages     = media.NoteOmittedImages
	omitChatHistoryImages = wirecore.OmitChatHistoryImages
)

func omitAnthropicHistoryImages(msgs []anthropicMessage) int {
	cs := make([]*json.RawMessage, len(msgs))
	for i := range msgs {
		cs[i] = &msgs[i].Content
	}
	return media.OmitHistoryImages(cs, "image")
}
