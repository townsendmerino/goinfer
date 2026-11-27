package serveapp

import "github.com/townsendmerino/goinfer/internal/app/serveapp/media"

type (
	imageRef    = media.ImageRef
	contentPart = media.ContentPart
)

var (
	contentPartsText   = media.ContentPartsText
	contentPartsImages = media.ContentPartsImages
	decodeDataURI      = media.DecodeDataURI
	decodeBase64Image  = media.DecodeBase64Image
)

// chatImages collects inline images and audio clips across an OpenAI chat message list in order.
func chatImages(msgs []chatMessage) ([]imageRef, error) {
	var out []imageRef
	for _, m := range msgs {
		imgs, err := m.imageData()
		if err != nil {
			return nil, err
		}
		out = append(out, imgs...)
	}
	return out, nil
}
