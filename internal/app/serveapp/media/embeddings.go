package media

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/townsendmerino/goinfer/embeddinggemma2"
)

// Per-request input bounds for /v1/embeddings.
const (
	MaxEmbedInputs     = 2048
	MaxEmbedInputBytes = 1 << 20
	MaxEmbedImages     = 16
	MaxEmbedImageBytes = 16 << 20
	MaxEmbedAudio      = 16
)

// EmbedItem is one /v1/embeddings input: text, or an image or audio clip with
// optional trailing text.
type EmbedItem struct {
	Text     string
	Image    []byte
	HasImage bool
	Audio    []float32
	HasAudio bool
}

// AudioEmbedder is the capability of an embedder that embeds audio clips.
type AudioEmbedder interface {
	EmbedAudioTask(samples []float32, text, prompt string) ([]float32, int, error)
}

// ImageEmbedder is the capability of an embedder that embeds images.
type ImageEmbedder interface {
	EmbedImageTask(img []byte, text, prompt string) ([]float32, int, error)
}

// TaskEmbedder is the capability of an embedder parameterized by named task prompts.
type TaskEmbedder interface {
	PromptNames() []string
	PromptText(name string) (string, error)
	EncodeTasks(texts []string, prompt string) ([][]float32, []int, error)
}

// CheckEmbedInputBounds rejects text batches exceeding MaxEmbedInputs or
// MaxEmbedInputBytes per input.
func CheckEmbedInputBounds(inputs []string) error {
	if len(inputs) > MaxEmbedInputs {
		return fmt.Errorf("too many inputs: %d (max %d per request)", len(inputs), MaxEmbedInputs)
	}
	for i, in := range inputs {
		if len(in) > MaxEmbedInputBytes {
			return fmt.Errorf("input %d is %d bytes, exceeds the %d-byte per-input limit", i, len(in), MaxEmbedInputBytes)
		}
	}
	return nil
}

// ParseEmbedItems decodes a /v1/embeddings `input` field across string,
// {"image","audio","text"} object, and OpenAI content-part array forms.
func ParseEmbedItems(raw json.RawMessage) ([]EmbedItem, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var one string
	if json.Unmarshal(raw, &one) == nil {
		return []EmbedItem{{Text: one}}, nil
	}
	var obj map[string]json.RawMessage
	if json.Unmarshal(raw, &obj) == nil {
		it, err := parseEmbedObject(obj)
		if err != nil {
			return nil, err
		}
		return []EmbedItem{it}, nil
	}
	var many []json.RawMessage
	if json.Unmarshal(raw, &many) != nil {
		return nil, fmt.Errorf("input must be a string, an image object, or an array of them")
	}
	if len(many) > MaxEmbedInputs {
		return nil, fmt.Errorf("too many inputs: %d (max %d per request)", len(many), MaxEmbedInputs)
	}
	items := make([]EmbedItem, 0, len(many))
	var ni, na int
	for i, el := range many {
		it, err := parseEmbedElement(el)
		if err != nil {
			return nil, fmt.Errorf("input %d: %w", i, err)
		}
		if len(it.Image) > 0 {
			ni++
			if ni > MaxEmbedImages {
				return nil, fmt.Errorf("too many images: %d (max %d per request)", ni, MaxEmbedImages)
			}
		}
		if len(it.Audio) > 0 {
			na++
			if na > MaxEmbedAudio {
				return nil, fmt.Errorf("too many audio clips: %d (max %d per request)", na, MaxEmbedAudio)
			}
		}
		items = append(items, it)
	}
	return items, nil
}

func parseEmbedElement(el json.RawMessage) (EmbedItem, error) {
	var s string
	if json.Unmarshal(el, &s) == nil {
		return EmbedItem{Text: s}, nil
	}
	var obj map[string]json.RawMessage
	if json.Unmarshal(el, &obj) == nil {
		return parseEmbedObject(obj)
	}
	var parts []map[string]json.RawMessage
	if json.Unmarshal(el, &parts) == nil {
		return parseEmbedParts(parts)
	}
	return EmbedItem{}, fmt.Errorf("must be a string, an {\"image\", \"text\"} object or a content-part array")
}

func parseEmbedObject(obj map[string]json.RawMessage) (EmbedItem, error) {
	if _, ok := obj["type"]; ok {
		return parseEmbedParts([]map[string]json.RawMessage{obj})
	}
	var it EmbedItem
	for k, v := range obj {
		switch k {
		case "text":
			if err := json.Unmarshal(v, &it.Text); err != nil {
				return EmbedItem{}, fmt.Errorf("text must be a string")
			}
		case "image":
			var s string
			if err := json.Unmarshal(v, &s); err != nil {
				return EmbedItem{}, fmt.Errorf("image must be a base64 string or a data: URI")
			}
			img, err := DecodeEmbedImage(s, true)
			if err != nil {
				return EmbedItem{}, err
			}
			it.Image, it.HasImage = img, true
		case "audio":
			var s string
			if err := json.Unmarshal(v, &s); err != nil {
				return EmbedItem{}, fmt.Errorf("audio must be a base64 string or a data: URI of a WAV")
			}
			raw, err := DecodeEmbedImage(s, true)
			if err != nil {
				return EmbedItem{}, fmt.Errorf("audio: %w", err)
			}
			if it.Audio, err = DecodeEmbedAudio(raw); err != nil {
				return EmbedItem{}, err
			}
			it.HasAudio = true
		default:
			return EmbedItem{}, fmt.Errorf("unknown field %q (an input object takes image, audio and text)", k)
		}
	}
	if it.HasImage && it.HasAudio {
		return EmbedItem{}, fmt.Errorf("an input takes one image or one audio clip, not both")
	}
	if !it.HasImage && !it.HasAudio && it.Text == "" {
		return EmbedItem{}, fmt.Errorf("an input object needs an image, audio or text")
	}
	return it, nil
}

func parseEmbedParts(parts []map[string]json.RawMessage) (EmbedItem, error) {
	var it EmbedItem
	var text []string
	for i, p := range parts {
		var typ string
		if err := json.Unmarshal(p["type"], &typ); err != nil {
			return EmbedItem{}, fmt.Errorf("part %d has no type", i)
		}
		switch typ {
		case "text":
			var s string
			if err := json.Unmarshal(p["text"], &s); err != nil {
				return EmbedItem{}, fmt.Errorf("part %d: text must be a string", i)
			}
			text = append(text, s)
		case "image_url":
			if it.HasImage || it.HasAudio {
				return EmbedItem{}, fmt.Errorf("part %d: one image or audio clip per input", i)
			}
			if len(text) > 0 {
				return EmbedItem{}, fmt.Errorf("part %d: text before the image is not supported yet; put the image part first", i)
			}
			var u struct {
				URL string `json:"url"`
			}
			if err := json.Unmarshal(p["image_url"], &u); err != nil || u.URL == "" {
				return EmbedItem{}, fmt.Errorf("part %d: image_url needs a url", i)
			}
			img, err := DecodeEmbedImage(u.URL, false)
			if err != nil {
				return EmbedItem{}, fmt.Errorf("part %d: %w", i, err)
			}
			it.Image, it.HasImage = img, true
		case "input_audio":
			if it.HasImage || it.HasAudio {
				return EmbedItem{}, fmt.Errorf("part %d: one image or audio clip per input", i)
			}
			if len(text) > 0 {
				return EmbedItem{}, fmt.Errorf("part %d: text before the audio is not supported yet; put the audio part first", i)
			}
			var ia struct {
				Data   string `json:"data"`
				Format string `json:"format"`
			}
			if err := json.Unmarshal(p["input_audio"], &ia); err != nil || ia.Data == "" {
				return EmbedItem{}, fmt.Errorf("part %d: input_audio needs data", i)
			}
			if ia.Format != "" && ia.Format != "wav" {
				return EmbedItem{}, fmt.Errorf("part %d: input_audio format %q (only wav)", i, ia.Format)
			}
			raw, err := DecodeEmbedImage(ia.Data, true)
			if err != nil {
				return EmbedItem{}, fmt.Errorf("part %d: audio: %w", i, err)
			}
			if it.Audio, err = DecodeEmbedAudio(raw); err != nil {
				return EmbedItem{}, fmt.Errorf("part %d: %w", i, err)
			}
			it.HasAudio = true
		default:
			return EmbedItem{}, fmt.Errorf("part %d: type %q (want image_url, input_audio or text)", i, typ)
		}
	}
	it.Text = strings.Join(text, "")
	return it, nil
}

// DecodeEmbedImage decodes a data: URI or (when allowBare is true) bare base64,
// refusing payloads larger than MaxEmbedImageBytes.
func DecodeEmbedImage(s string, allowBare bool) ([]byte, error) {
	var data []byte
	if strings.HasPrefix(s, "data:") || !allowBare {
		ref, err := DecodeDataURI(s)
		if err != nil {
			return nil, err
		}
		data = ref.Data
	} else {
		d, err := base64.StdEncoding.DecodeString(strings.TrimSpace(s))
		if err != nil {
			return nil, fmt.Errorf("image is neither a data: URI nor valid base64: %w", err)
		}
		data = d
	}
	if len(data) > MaxEmbedImageBytes {
		return nil, fmt.Errorf("image is %d bytes decoded, over the %d-byte limit", len(data), MaxEmbedImageBytes)
	}
	return data, nil
}

// DecodeEmbedAudio decodes a 16 kHz mono 16-bit PCM WAV bounded by embeddinggemma2.MaxAudioSeconds.
func DecodeEmbedAudio(raw []byte) ([]float32, error) {
	s, err := embeddinggemma2.DecodeWAV(raw)
	if err != nil {
		return nil, err
	}
	if len(s) > embeddinggemma2.MaxAudioSeconds*16000 {
		return nil, fmt.Errorf("audio is %.1f s, over the %d s the model reads", float64(len(s))/16000, embeddinggemma2.MaxAudioSeconds)
	}
	return s, nil
}

// EmbedMedia counts the items carrying an image and an audio clip.
func EmbedMedia(items []EmbedItem) (images, audio int) {
	for _, it := range items {
		if it.HasImage {
			images++
		}
		if it.HasAudio {
			audio++
		}
	}
	return images, audio
}

// EncodeMediaItems embeds a batch containing images, audio clips, or plain text
// in input order under one task prompt.
func EncodeMediaItems(ie ImageEmbedder, ae AudioEmbedder, te TaskEmbedder, items []EmbedItem, task string) ([][]float32, int, error) {
	vecs := make([][]float32, len(items))
	total := 0
	for i, it := range items {
		if it.HasImage || it.HasAudio {
			var v []float32
			var n int
			var err error
			if it.HasImage {
				v, n, err = ie.EmbedImageTask(it.Image, it.Text, task)
			} else {
				v, n, err = ae.EmbedAudioTask(it.Audio, it.Text, task)
			}
			if err != nil {
				return nil, 0, fmt.Errorf("input %d: %w", i, err)
			}
			vecs[i], total = v, total+n
			continue
		}
		vs, ns, err := te.EncodeTasks([]string{it.Text}, task)
		if err != nil {
			return nil, 0, fmt.Errorf("input %d: %w", i, err)
		}
		vecs[i], total = vs[0], total+ns[0]
	}
	return vecs, total, nil
}
