package media

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
)

// ImageRef is one decoded inline media item from a request content part: an
// image, or (when Audio is true) a WAV clip. Only inline base64 inputs are
// accepted; remote URLs are refused as an SSRF guard.
type ImageRef struct {
	MediaType string
	Data      []byte
	Audio     bool
	At        int // byte offset in the message's joined text parts where this item appeared
}

// ContentPart is one element of an OpenAI chat message's content array.
type ContentPart struct {
	Type     string `json:"type"`
	Text     string `json:"text"`
	ImageURL *struct {
		URL string `json:"url"`
	} `json:"image_url"`
	InputAudio *struct {
		Data   string `json:"data"`
		Format string `json:"format"`
	} `json:"input_audio"`
}

// ContentPartsText returns a chat message's text: the plain-string content, or
// the concatenated text parts of an OpenAI content array. Empty for null or
// absent content.
func ContentPartsText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var parts []ContentPart
	if json.Unmarshal(raw, &parts) != nil {
		return ""
	}
	var b strings.Builder
	for _, p := range parts {
		if p.Type == "text" {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

// ContentPartsImages extracts inline images and audio clips from an OpenAI
// content array. A plain-string content yields nil. Any non-data image URL
// returns an error.
func ContentPartsImages(raw json.RawMessage) ([]ImageRef, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return nil, nil
	}
	var parts []ContentPart
	if json.Unmarshal(raw, &parts) != nil {
		return nil, nil
	}
	var out []ImageRef
	at := 0
	for _, p := range parts {
		if p.Type == "text" {
			at += len(p.Text)
			continue
		}
		if p.Type == "input_audio" {
			ref, err := decodeInputAudio(p.InputAudio)
			if err != nil {
				return nil, err
			}
			ref.At = at
			out = append(out, ref)
			continue
		}
		if p.Type != "image_url" || p.ImageURL == nil {
			continue
		}
		ref, err := DecodeDataURI(p.ImageURL.URL)
		if err != nil {
			return nil, err
		}
		ref.At = at
		out = append(out, ref)
	}
	return out, nil
}

func decodeInputAudio(ia *struct {
	Data   string `json:"data"`
	Format string `json:"format"`
}) (ImageRef, error) {
	if ia == nil || ia.Data == "" {
		return ImageRef{}, fmt.Errorf("input_audio needs data")
	}
	if ia.Format != "" && ia.Format != "wav" {
		return ImageRef{}, fmt.Errorf("input_audio format %q: only wav (16-bit PCM) is taken", ia.Format)
	}
	var data []byte
	if strings.HasPrefix(ia.Data, "data:") {
		ref, err := DecodeDataURI(ia.Data)
		if err != nil {
			return ImageRef{}, fmt.Errorf("input_audio: %w", err)
		}
		data = ref.Data
	} else {
		d, err := base64.StdEncoding.DecodeString(strings.TrimSpace(ia.Data))
		if err != nil {
			return ImageRef{}, fmt.Errorf("input_audio data is neither a data: URI nor valid base64: %w", err)
		}
		data = d
	}
	return ImageRef{MediaType: "audio/wav", Data: data, Audio: true}, nil
}

// DecodeDataURI parses a base64 data: URI ("data:<media>;base64,<payload>") into
// an ImageRef. Any non-data URL is rejected.
func DecodeDataURI(uri string) (ImageRef, error) {
	rest, ok := strings.CutPrefix(uri, "data:")
	if !ok {
		return ImageRef{}, fmt.Errorf("image url must be a base64 data: URI (server-side URL fetch is not supported)")
	}
	header, payload, ok := strings.Cut(rest, ",")
	if !ok {
		return ImageRef{}, fmt.Errorf("malformed data: URI (no comma)")
	}
	if !strings.Contains(header, "base64") {
		return ImageRef{}, fmt.Errorf("data: URI must be base64-encoded")
	}
	media := header
	if before, _, ok := strings.Cut(header, ";"); ok {
		media = before
	}
	data, err := base64.StdEncoding.DecodeString(strings.TrimSpace(payload))
	if err != nil {
		return ImageRef{}, fmt.Errorf("decode base64 image: %w", err)
	}
	return ImageRef{MediaType: media, Data: data}, nil
}

// DecodeBase64Image decodes an Anthropic image block's base64 source payload.
func DecodeBase64Image(mediaType, b64 string) (ImageRef, error) {
	data, err := base64.StdEncoding.DecodeString(strings.TrimSpace(b64))
	if err != nil {
		return ImageRef{}, fmt.Errorf("decode base64 image: %w", err)
	}
	return ImageRef{MediaType: mediaType, Data: data}, nil
}
