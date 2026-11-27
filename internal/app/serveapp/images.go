package serveapp

import (
	"github.com/townsendmerino/goinfer/internal/app/serveapp/media"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/wirecore"
)

type (
	imageRef    = media.ImageRef
	contentPart = media.ContentPart
)

var (
	contentPartsText   = media.ContentPartsText
	contentPartsImages = media.ContentPartsImages
	decodeDataURI      = media.DecodeDataURI
	chatImages         = wirecore.ChatImages
)
