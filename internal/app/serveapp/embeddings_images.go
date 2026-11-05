package serveapp

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/townsendmerino/goinfer/embeddinggemma2"
)

// Image inputs to /v1/embeddings (docs/tasks/task-embeddinggemma2.md). `input` is a string, an object, or an array
// whose every element is one input; an element is a string, an {"image": ..., "text": ...} object (the dict
// sentence-transformers takes), or an OpenAI content-part array ([{"type": "image_url", "image_url": {"url":
// "data:..."}}, {"type": "text", "text": "..."}]). A lone part object with a "type" is a one-part input. Images are
// inline base64 only (a data: URI, or bare base64 in the object form), never a fetched URL (decodeDataURI's SSRF
// guard). One image per input, and text only after it, the layout EmbeddingGemma 2's reference builds for an image
// followed by text.
//
// Audio takes the same two shapes: {"audio": <a data: URI or bare base64 of a WAV>, "text": ...}, and the OpenAI part
// {"type": "input_audio", "input_audio": {"data": <base64>, "format": "wav"}}. The WAV is 16-bit PCM, mono, at 16
// kHz, at most 30 s (embeddinggemma2.DecodeWAV and MaxAudioSeconds; nothing is resampled or cut). An input carries
// one image or one audio clip, not both.

const (
	maxEmbedImages     = 16
	maxEmbedImageBytes = 16 << 20 // decoded
	maxEmbedAudio      = 16       // clips per request
)

// embedItem is one /v1/embeddings input: text, or an image or an audio clip with optional text after it.
type embedItem struct {
	text     string
	image    []byte
	hasImage bool
	audio    []float32 // 16 kHz samples
	hasAudio bool
}

// audioEmbedder is the optional capability of an embedder that embeds audio (EmbeddingGemma 2 with its audio tower).
type audioEmbedder interface {
	EmbedAudioTask(samples []float32, text, prompt string) ([]float32, int, error)
}

// imageEmbedder is the optional capability of an embedder that embeds images (EmbeddingGemma 2 with its vision tower).
type imageEmbedder interface {
	EmbedImageTask(img []byte, text, prompt string) ([]float32, int, error)
}

// parseEmbedItems reads `input` in every accepted shape. A request with no image or audio yields only text items,
// which the handler serves as plain text inputs.
func parseEmbedItems(raw json.RawMessage) ([]embedItem, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var one string
	if json.Unmarshal(raw, &one) == nil {
		return []embedItem{{text: one}}, nil
	}
	var obj map[string]json.RawMessage
	if json.Unmarshal(raw, &obj) == nil {
		it, err := parseEmbedObject(obj)
		if err != nil {
			return nil, err
		}
		return []embedItem{it}, nil
	}
	var many []json.RawMessage
	if json.Unmarshal(raw, &many) != nil {
		return nil, fmt.Errorf("input must be a string, an image object, or an array of them")
	}
	items := make([]embedItem, 0, len(many))
	for i, el := range many {
		it, err := parseEmbedElement(el)
		if err != nil {
			return nil, fmt.Errorf("input %d: %w", i, err)
		}
		items = append(items, it)
	}
	return items, nil
}

func parseEmbedElement(el json.RawMessage) (embedItem, error) {
	var s string
	if json.Unmarshal(el, &s) == nil {
		return embedItem{text: s}, nil
	}
	var obj map[string]json.RawMessage
	if json.Unmarshal(el, &obj) == nil {
		return parseEmbedObject(obj)
	}
	var parts []map[string]json.RawMessage
	if json.Unmarshal(el, &parts) == nil {
		return parseEmbedParts(parts)
	}
	return embedItem{}, fmt.Errorf("must be a string, an {\"image\", \"text\"} object or a content-part array")
}

// parseEmbedObject reads {"image": ..., "text": ...}, or a single content part.
func parseEmbedObject(obj map[string]json.RawMessage) (embedItem, error) {
	if _, ok := obj["type"]; ok {
		return parseEmbedParts([]map[string]json.RawMessage{obj})
	}
	var it embedItem
	for k, v := range obj {
		switch k {
		case "text":
			if err := json.Unmarshal(v, &it.text); err != nil {
				return embedItem{}, fmt.Errorf("text must be a string")
			}
		case "image":
			var s string
			if err := json.Unmarshal(v, &s); err != nil {
				return embedItem{}, fmt.Errorf("image must be a base64 string or a data: URI")
			}
			img, err := decodeEmbedImage(s, true)
			if err != nil {
				return embedItem{}, err
			}
			it.image, it.hasImage = img, true
		case "audio":
			var s string
			if err := json.Unmarshal(v, &s); err != nil {
				return embedItem{}, fmt.Errorf("audio must be a base64 string or a data: URI of a WAV")
			}
			raw, err := decodeEmbedImage(s, true)
			if err != nil {
				return embedItem{}, fmt.Errorf("audio: %w", err)
			}
			if it.audio, err = decodeEmbedAudio(raw); err != nil {
				return embedItem{}, err
			}
			it.hasAudio = true
		default:
			return embedItem{}, fmt.Errorf("unknown field %q (an input object takes image, audio and text)", k)
		}
	}
	if it.hasImage && it.hasAudio {
		return embedItem{}, fmt.Errorf("an input takes one image or one audio clip, not both")
	}
	if !it.hasImage && !it.hasAudio && it.text == "" {
		return embedItem{}, fmt.Errorf("an input object needs an image, audio or text")
	}
	return it, nil
}

// parseEmbedParts reads an OpenAI content-part array: at most one image_url part, and text parts only after it.
func parseEmbedParts(parts []map[string]json.RawMessage) (embedItem, error) {
	var it embedItem
	var text []string
	for i, p := range parts {
		var typ string
		if err := json.Unmarshal(p["type"], &typ); err != nil {
			return embedItem{}, fmt.Errorf("part %d has no type", i)
		}
		switch typ {
		case "text":
			var s string
			if err := json.Unmarshal(p["text"], &s); err != nil {
				return embedItem{}, fmt.Errorf("part %d: text must be a string", i)
			}
			text = append(text, s)
		case "image_url":
			if it.hasImage || it.hasAudio {
				return embedItem{}, fmt.Errorf("part %d: one image or audio clip per input", i)
			}
			if len(text) > 0 {
				return embedItem{}, fmt.Errorf("part %d: text before the image is not supported yet; put the image part first", i)
			}
			var u struct {
				URL string `json:"url"`
			}
			if err := json.Unmarshal(p["image_url"], &u); err != nil || u.URL == "" {
				return embedItem{}, fmt.Errorf("part %d: image_url needs a url", i)
			}
			img, err := decodeEmbedImage(u.URL, false)
			if err != nil {
				return embedItem{}, fmt.Errorf("part %d: %w", i, err)
			}
			it.image, it.hasImage = img, true
		case "input_audio":
			if it.hasImage || it.hasAudio {
				return embedItem{}, fmt.Errorf("part %d: one image or audio clip per input", i)
			}
			if len(text) > 0 {
				return embedItem{}, fmt.Errorf("part %d: text before the audio is not supported yet; put the audio part first", i)
			}
			var ia struct {
				Data   string `json:"data"`
				Format string `json:"format"`
			}
			if err := json.Unmarshal(p["input_audio"], &ia); err != nil || ia.Data == "" {
				return embedItem{}, fmt.Errorf("part %d: input_audio needs data", i)
			}
			if ia.Format != "" && ia.Format != "wav" {
				return embedItem{}, fmt.Errorf("part %d: input_audio format %q (only wav)", i, ia.Format)
			}
			raw, err := decodeEmbedImage(ia.Data, true)
			if err != nil {
				return embedItem{}, fmt.Errorf("part %d: audio: %w", i, err)
			}
			if it.audio, err = decodeEmbedAudio(raw); err != nil {
				return embedItem{}, fmt.Errorf("part %d: %w", i, err)
			}
			it.hasAudio = true
		default:
			return embedItem{}, fmt.Errorf("part %d: type %q (want image_url, input_audio or text)", i, typ)
		}
	}
	it.text = strings.Join(text, "")
	return it, nil
}

// decodeEmbedImage decodes a data: URI, or (allowBare) bare base64, refusing anything over maxEmbedImageBytes.
func decodeEmbedImage(s string, allowBare bool) ([]byte, error) {
	var data []byte
	if strings.HasPrefix(s, "data:") || !allowBare {
		ref, err := decodeDataURI(s)
		if err != nil {
			return nil, err
		}
		data = ref.data
	} else {
		d, err := base64.StdEncoding.DecodeString(strings.TrimSpace(s))
		if err != nil {
			return nil, fmt.Errorf("image is neither a data: URI nor valid base64: %w", err)
		}
		data = d
	}
	if len(data) > maxEmbedImageBytes {
		return nil, fmt.Errorf("image is %d bytes decoded, over the %d-byte limit", len(data), maxEmbedImageBytes)
	}
	return data, nil
}

// decodeEmbedAudio reads a WAV into 16 kHz samples, refusing a clip over embeddinggemma2.MaxAudioSeconds.
func decodeEmbedAudio(raw []byte) ([]float32, error) {
	s, err := embeddinggemma2.DecodeWAV(raw)
	if err != nil {
		return nil, err
	}
	if len(s) > embeddinggemma2.MaxAudioSeconds*16000 {
		return nil, fmt.Errorf("audio is %.1f s, over the %d s the model reads", float64(len(s))/16000, embeddinggemma2.MaxAudioSeconds)
	}
	return s, nil
}

// embedMedia counts the items that carry an image and an audio clip.
func embedMedia(items []embedItem) (images, audio int) {
	for _, it := range items {
		if it.hasImage {
			images++
		}
		if it.hasAudio {
			audio++
		}
	}
	return images, audio
}

// encodeMediaItems embeds a request that carries images or audio: each image item through ie, each audio item through
// ae, each text item through te, in order, all under one prompt.
func encodeMediaItems(ie imageEmbedder, ae audioEmbedder, te taskEmbedder, items []embedItem, task string) ([][]float32, int, error) {
	vecs := make([][]float32, len(items))
	total := 0
	for i, it := range items {
		if it.hasImage || it.hasAudio {
			var v []float32
			var n int
			var err error
			if it.hasImage {
				v, n, err = ie.EmbedImageTask(it.image, it.text, task)
			} else {
				v, n, err = ae.EmbedAudioTask(it.audio, it.text, task)
			}
			if err != nil {
				return nil, 0, fmt.Errorf("input %d: %w", i, err)
			}
			vecs[i], total = v, total+n
			continue
		}
		vs, ns, err := te.EncodeTasks([]string{it.text}, task)
		if err != nil {
			return nil, 0, fmt.Errorf("input %d: %w", i, err)
		}
		vecs[i], total = vs[0], total+ns[0]
	}
	return vecs, total, nil
}
