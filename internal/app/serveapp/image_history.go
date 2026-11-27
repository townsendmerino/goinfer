package serveapp

import (
	"github.com/townsendmerino/goinfer/internal/app/serveapp/anthropicapi"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/media"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/wirecore"
)

const (
	imagesOmittedHeader = media.ImagesOmittedHeader
	olderImageNote      = media.OlderImageNote
	olderAudioNote      = media.OlderAudioNote
)

var (
	noteOmittedImages          = media.NoteOmittedImages
	omitChatHistoryImages      = wirecore.OmitChatHistoryImages
	omitAnthropicHistoryImages = anthropicapi.OmitHistoryImages
)
