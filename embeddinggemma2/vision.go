package embeddinggemma2

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// Images (Phase V, docs/tasks/task-embeddinggemma2.md). EmbeddingGemma 2's vision tower is gemma4_vision, the same
// tower and projector aikit already runs for Gemma 4 (vision.Gemma4Encoder, its tensors under the same names), so an
// image's soft tokens come from aikit and replace their placeholder rows in the encoder's input, unscaled (the
// composite model's masked_scatter runs after the token embedding's sqrt(hidden) scale). The text encoder, its PLE
// block and the mean pool then run over the whole sequence as for text.

// MaxImageSoftTokens is the soft-token budget the checkpoint's processor uses (vision_soft_tokens_per_image, 280).
const MaxImageSoftTokens = 280

type imageTokens struct {
	Image int `json:"image_token_id"`
	BOI   int `json:"boi_token_id"`
	EOI   int `json:"eoi_token_id"`
}

// visionTower is the loaded image path: aikit's tower and projector, the three image token ids, and the tower's
// accelerator when it runs on another device (vision_accel.go).
type visionTower struct {
	enc    *vision.Gemma4Encoder
	tok    imageTokens
	accel  VisionAccelerator // nil: the CPU tower
	device string            // VisionDevice's account when not plain CPU
}

// EnableVision loads the checkpoint's vision tower (from the same directory as the encoder) so EmbedImage works. The
// tower runs on the encoder's accelerator when one of that name is registered for it (RegisterVisionAccelerator), and
// on the CPU otherwise (VisionDevice says which); the text encoder after it runs where the encoder does.
func (e *Encoder) EnableVision() error {
	if e.dir == "" {
		return fmt.Errorf("embeddinggemma2: the encoder was not loaded from a directory")
	}
	raw, err := os.ReadFile(filepath.Join(e.dir, "config.json"))
	if err != nil {
		return fmt.Errorf("embeddinggemma2: %w", err)
	}
	var cfg struct {
		imageTokens
		Vision *json.RawMessage `json:"vision_config"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return fmt.Errorf("embeddinggemma2: config.json: %w", err)
	}
	if cfg.Vision == nil || cfg.Image <= 0 || cfg.BOI <= 0 || cfg.EOI <= 0 {
		return fmt.Errorf("embeddinggemma2: this checkpoint has no vision_config or image token ids")
	}
	enc, err := vision.LoadGemma4Encoder(e.dir, false)
	if err != nil {
		return fmt.Errorf("embeddinggemma2: vision tower: %w", err)
	}
	if enc.TextHiddenSize != e.m.cfg.Hidden {
		return fmt.Errorf("embeddinggemma2: the vision projector emits %d wide, the encoder takes %d", enc.TextHiddenSize, e.m.cfg.Hidden)
	}
	e.vis = &visionTower{enc: enc, tok: cfg.imageTokens}
	e.bindVisionAccel()
	return nil
}

// VisionEnabled reports whether EnableVision has loaded the tower.
func (e *Encoder) VisionEnabled() bool { return e.vis != nil }

// ImageInput is one image-bearing input: the image's encoded bytes (PNG, JPEG, ...), optional text after it, and the
// named prompt before it ("" for none).
type ImageInput struct {
	Image  []byte
	Text   string
	Prompt string
}

// TokenizeImage returns the ids for an image input with n soft tokens: <bos>, the prompt's text, <|image>, n image
// tokens, <image|>, the text, <eos>, the layout sentence-transformers builds (probed, Phase V Gate 0), and the index
// of the first soft token.
func (e *Encoder) TokenizeImage(in ImageInput, n int) (ids []int, imgPos int, err error) {
	if e.vis == nil {
		return nil, 0, fmt.Errorf("embeddinggemma2: vision is not enabled (EnableVision)")
	}
	p, err := e.PromptText(in.Prompt)
	if err != nil {
		return nil, 0, err
	}
	ids = []int{e.bos}
	if p != "" {
		pi, err := e.tok.Encode(p, false)
		if err != nil {
			return nil, 0, err
		}
		ids = append(ids, pi...)
	}
	ids = append(ids, e.vis.tok.BOI)
	imgPos = len(ids)
	for range n {
		ids = append(ids, e.vis.tok.Image)
	}
	ids = append(ids, e.vis.tok.EOI)
	if in.Text != "" {
		ti, err := e.tok.Encode(in.Text, false)
		if err != nil {
			return nil, 0, err
		}
		ids = append(ids, ti...)
	}
	ids = append(ids, e.eos)
	if len(ids) > MaxTokens {
		return nil, 0, fmt.Errorf("input is %d tokens, over the model's %d-token context", len(ids), MaxTokens)
	}
	return ids, imgPos, nil
}

// ImageFeatures preprocesses encoded image bytes (the encoder's resize, SetImageResize) and runs aikit's tower: the
// pooled, projected soft tokens [n, hidden] and n.
func (e *Encoder) ImageFeatures(img []byte) ([]float32, int, error) {
	if e.vis == nil {
		return nil, 0, fmt.Errorf("embeddinggemma2: vision is not enabled (EnableVision)")
	}
	patches, pos, err := e.preprocessImage(img)
	if err != nil {
		return nil, 0, fmt.Errorf("embeddinggemma2: image: %w", err)
	}
	return e.featuresFrom(patches, pos)
}

// ImageFeaturesFrom is ImageFeatures from already preprocessed patches and their (x, y) positions (a test feeds the
// reference processor's own, to judge the tower alone).
func (e *Encoder) ImageFeaturesFrom(patches []float32, pos [][2]int) ([]float32, int, error) {
	if e.vis == nil {
		return nil, 0, fmt.Errorf("embeddinggemma2: vision is not enabled (EnableVision)")
	}
	return e.featuresFrom(patches, pos)
}

func (e *Encoder) featuresFrom(patches []float32, pos [][2]int) ([]float32, int, error) {
	return e.featuresOn(e.vis.accel, patches, pos)
}

// featuresOn runs the tower on a (nil: the CPU), then aikit's tail.
func (e *Encoder) featuresOn(a VisionAccelerator, patches []float32, pos [][2]int) ([]float32, int, error) {
	n := multimodal.Gemma4PooledTokens(pos, e.vis.enc.Cfg.PoolingKernelSize)
	var feats []float32
	var err error
	if a != nil {
		var h []float32
		if h, err = a.Hidden(patches, pos); err == nil {
			feats, err = e.vis.enc.FinishHidden(h, pos)
		}
	} else {
		feats, err = e.vis.enc.Forward(patches, pos)
	}
	if err != nil {
		return nil, 0, fmt.Errorf("embeddinggemma2: vision tower: %w", err)
	}
	if H := e.m.cfg.Hidden; len(feats) != n*H {
		return nil, 0, fmt.Errorf("embeddinggemma2: the tower emitted %d values, want %d soft tokens of %d", len(feats), n, H)
	}
	return feats, n, nil
}

// EmbedImage is the sentence embedding of an image input, unit length, and its token count.
func (e *Encoder) EmbedImage(in ImageInput) ([]float32, int, error) {
	feats, n, err := e.ImageFeatures(in.Image)
	if err != nil {
		return nil, 0, err
	}
	return e.EmbedImageFeatures(in, feats, n)
}

// EmbedImageFeatures is EmbedImage from soft tokens already computed (ImageFeatures or ImageFeaturesFrom).
func (e *Encoder) EmbedImageFeatures(in ImageInput, feats []float32, n int) ([]float32, int, error) {
	ids, imgPos, err := e.TokenizeImage(in, n)
	if err != nil {
		return nil, 0, err
	}
	x, err := e.m.EmbedTokens(ids)
	if err != nil {
		return nil, 0, err
	}
	H := e.m.cfg.Hidden
	copy(x[imgPos*H:(imgPos+n)*H], feats) // unscaled, as the reference's masked_scatter
	var h []float32
	if e.accel != nil {
		h, _, err = e.accel.ForwardEmbeds(x, len(ids), false)
	} else {
		h, _, err = e.m.forwardEmbeds(x, len(ids), false)
	}
	if err != nil {
		return nil, 0, err
	}
	return poolNormalize(h, len(ids), e.m.cfg.EmbeddingDim), len(ids), nil
}

// visMu serialises the lazy EnableVision of EmbedImageTask.
var visMu sync.Mutex

// OnVisionLoad sets a function EmbedImageTask calls once, when its first image loads the tower, with where the tower
// runs (VisionDevice) and how long the load took: a server reports it, since that happens after its startup line.
func (e *Encoder) OnVisionLoad(f func(device string, took time.Duration)) { e.onVis = f }

// EmbedImageTask embeds one image input for a server: it loads the vision tower on first use (a text-only server never
// pays for it), then EmbedImage. prompt names a task prompt or is "" for none.
func (e *Encoder) EmbedImageTask(img []byte, text, prompt string) ([]float32, int, error) {
	visMu.Lock()
	if e.vis == nil {
		t0 := time.Now()
		if err := e.EnableVision(); err != nil {
			visMu.Unlock()
			return nil, 0, err
		}
		if e.onVis != nil {
			e.onVis(e.VisionDevice(), time.Since(t0))
		}
	}
	visMu.Unlock()
	return e.EmbedImage(ImageInput{Image: img, Text: text, Prompt: prompt})
}
