package serveapp

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
)

// Multimodal image input. v1 accepts inline base64 images only — a data: URI in
// an OpenAI image_url part, or an Anthropic image block's base64 source. A
// remote URL is never fetched: a server that GETs an attacker-chosen URL is an
// SSRF primitive, and the project just spent a release hardening attacker-
// supplied bytes (a `--allow-image-urls` opt-in can come later). Decoded bytes
// flow to the same vision pipeline (preprocess → encoder → projector).

// imageRef is one decoded inline media item from a request content part: an image, or (audio set) an audio clip.
type imageRef struct {
	mediaType string // e.g. "image/png" (informational; preprocess sniffs the real format), "audio/wav"
	data      []byte // raw bytes (base64 already decoded)
	audio     bool   // an OpenAI input_audio part (S5 of docs/tasks/task-multimodal-support-2026-10.md): a WAV for Gemma 4's audio tower
	at        int    // S11: the byte offset in its message's joined text parts where this part sat (placeImageBlocks)
}

// contentPart is one element of an OpenAI chat message's content array.
type contentPart struct {
	Type     string `json:"type"`
	Text     string `json:"text"`
	ImageURL *struct {
		URL string `json:"url"`
	} `json:"image_url"`
	InputAudio *struct {
		Data   string `json:"data"`   // base64 (bare, or a data: URI)
		Format string `json:"format"` // "wav" (the only one taken)
	} `json:"input_audio"`
}

// contentPartsText returns a chat message's text: the plain-string content, or
// the concatenated text parts of an OpenAI content array (non-text parts
// ignored). Empty for null/absent content.
func contentPartsText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var parts []contentPart
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

// contentPartsImages extracts inline images from an OpenAI content array's
// image_url parts (data: URIs only). A plain-string content has none. A non-data
// URL is an error (SSRF guard).
func contentPartsImages(raw json.RawMessage) ([]imageRef, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return nil, nil
	}
	var parts []contentPart
	if json.Unmarshal(raw, &parts) != nil {
		return nil, nil
	}
	var out []imageRef
	at := 0 // the joined text so far (contentPartsText's), where the next media part sits
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
			ref.at = at
			out = append(out, ref)
			continue
		}
		if p.Type != "image_url" || p.ImageURL == nil {
			continue
		}
		ref, err := decodeDataURI(p.ImageURL.URL)
		if err != nil {
			return nil, err
		}
		ref.at = at
		out = append(out, ref)
	}
	return out, nil
}

// decodeInputAudio decodes an OpenAI input_audio part: base64 data (bare, or a data: URI) in format "wav" (or unset).
// Its content (16-bit PCM; any rate and up to 8 channels, brought to 16 kHz mono; the length cap) is checked where the clip is used.
func decodeInputAudio(ia *struct {
	Data   string `json:"data"`
	Format string `json:"format"`
}) (imageRef, error) {
	if ia == nil || ia.Data == "" {
		return imageRef{}, fmt.Errorf("input_audio needs data")
	}
	if ia.Format != "" && ia.Format != "wav" {
		return imageRef{}, fmt.Errorf("input_audio format %q: only wav (16-bit PCM) is taken", ia.Format)
	}
	var data []byte
	if strings.HasPrefix(ia.Data, "data:") {
		ref, err := decodeDataURI(ia.Data)
		if err != nil {
			return imageRef{}, fmt.Errorf("input_audio: %w", err)
		}
		data = ref.data
	} else {
		d, err := base64.StdEncoding.DecodeString(strings.TrimSpace(ia.Data))
		if err != nil {
			return imageRef{}, fmt.Errorf("input_audio data is neither a data: URI nor valid base64: %w", err)
		}
		data = d
	}
	return imageRef{mediaType: "audio/wav", data: data, audio: true}, nil
}

// decodeDataURI parses a base64 data: URI ("data:<media>;base64,<payload>") into
// an imageRef. Any non-data: URL is rejected — v1 never fetches a server-side URL.
func decodeDataURI(uri string) (imageRef, error) {
	if !strings.HasPrefix(uri, "data:") {
		return imageRef{}, fmt.Errorf("image url must be a base64 data: URI (server-side URL fetch is not supported)")
	}
	comma := strings.IndexByte(uri, ',')
	if comma < 0 {
		return imageRef{}, fmt.Errorf("malformed data: URI (no comma)")
	}
	header, payload := uri[len("data:"):comma], uri[comma+1:]
	if !strings.Contains(header, "base64") {
		return imageRef{}, fmt.Errorf("data: URI must be base64-encoded")
	}
	media := header
	if before, _, ok := strings.Cut(header, ";"); ok {
		media = before
	}
	data, err := base64.StdEncoding.DecodeString(strings.TrimSpace(payload))
	if err != nil {
		return imageRef{}, fmt.Errorf("decode base64 image: %w", err)
	}
	return imageRef{mediaType: media, data: data}, nil
}

// chatImages collects the inline images across an OpenAI chat message list (in
// order). A non-data URL anywhere is an error (SSRF guard). No images → nil.
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

// decodeBase64Image decodes an Anthropic image block's base64 source data.
func decodeBase64Image(mediaType, b64 string) (imageRef, error) {
	data, err := base64.StdEncoding.DecodeString(strings.TrimSpace(b64))
	if err != nil {
		return imageRef{}, fmt.Errorf("decode base64 image: %w", err)
	}
	return imageRef{mediaType: mediaType, data: data}, nil
}
