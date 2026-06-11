package main

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

// imageRef is one decoded inline image from a request content part.
type imageRef struct {
	mediaType string // e.g. "image/png" (informational; preprocess sniffs the real format)
	data      []byte // raw image bytes (base64 already decoded)
}

// contentPart is one element of an OpenAI chat message's content array.
type contentPart struct {
	Type     string `json:"type"`
	Text     string `json:"text"`
	ImageURL *struct {
		URL string `json:"url"`
	} `json:"image_url"`
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
	for _, p := range parts {
		if p.Type != "image_url" || p.ImageURL == nil {
			continue
		}
		ref, err := decodeDataURI(p.ImageURL.URL)
		if err != nil {
			return nil, err
		}
		out = append(out, ref)
	}
	return out, nil
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
	if i := strings.IndexByte(header, ';'); i >= 0 {
		media = header[:i]
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
