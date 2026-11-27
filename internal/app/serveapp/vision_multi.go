package serveapp

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/towers"
)

func (lm *loadedModel) imagesPrompt(tm *chat.Template, system string, turns []chat.Turn, idx int, imgs []imageRef, ordered bool, msgText string, cache bool) (visionInput, error) {
	var wrapCache func(func() ([]float32, error), []byte) func() ([]float32, error)
	if cache {
		family := lm.towerFamily(false)
		wrapCache = func(tower func() ([]float32, error), raw []byte) func() ([]float32, error) {
			return lm.cachedFeatures(tower, raw, family)
		}
	}
	return towers.AssembleImagesPrompt(towers.ImagesPromptArgs{
		Vision:    lm.vision,
		TK:        lm.tk,
		HiddenDim: lm.hiddenDim(),
		TM:        tm,
		System:    system,
		Turns:     turns,
		Idx:       idx,
		Imgs:      imgs,
		Ordered:   ordered,
		MsgText:   msgText,
		WrapCache: wrapCache,
	})
}

func decoderSpans(spans []imageSpan) []decoder.ImageSpan {
	if spans == nil {
		return nil
	}
	out := make([]decoder.ImageSpan, len(spans))
	for i, s := range spans {
		out[i] = decoder.ImageSpan(s)
	}
	return out
}

func (lm *loadedModel) cachedFeatures(tower func() ([]float32, error), raw []byte, family string) func() ([]float32, error) {
	return lm.visionFeatureCache().Wrap(raw, func() ([]float32, error) { return recoverDeviceTower(family, tower) })
}

func (lm *loadedModel) visionPromptN(tm *chat.Template, system string, turns []chat.Turn, imgs []imageRef, ordered bool, msgText string) (visionInput, error) {
	for _, img := range imgs {
		if img.Audio && len(imgs) > 1 {
			return visionInput{}, fmt.Errorf("an audio clip must be the only media in a request (got %d media parts); send several images, or one clip", len(imgs))
		}
	}
	if len(imgs) == 1 && (imgs[0].Audio || lm.glmOcr() != nil || lm.tmpl == nil) {
		return lm.visionPrompt(tm, system, turns, imgs[0])
	}
	if lm.glmOcr() != nil {
		return visionInput{}, fmt.Errorf("GLM-OCR takes one image per request (its task prompts are written for one page); got %d", len(imgs))
	}
	if lm.tmpl == nil {
		return visionInput{}, fmt.Errorf("this model has no chat template for vision")
	}
	idx := lastUserTurn(turns)
	if idx < 0 {
		return visionInput{}, fmt.Errorf("no user turn to attach the image to")
	}
	return lm.imagesPrompt(tm, system, turns, idx, imgs, ordered, msgText, true)
}

func tooManyImages(n int) string {
	return fmt.Sprintf("at most %d images per request (each runs the vision tower), got %d", maxImagesPerTurn, n)
}

func chatMediaText(msgs []chatMessage) (string, bool) {
	media, lastUser := -1, -1
	for i, m := range msgs {
		switch m.Role {
		case "system", "developer", "tool", "assistant":
		default:
			lastUser = i
		}
		if contentHasMedia(m.Content) {
			media = i
		}
	}
	if media < 0 {
		return "", false
	}
	return msgs[media].Text(), media == lastUser
}

func contentHasMedia(raw json.RawMessage) bool {
	var parts []contentPart
	if len(raw) == 0 || json.Unmarshal(raw, &parts) != nil {
		return false
	}
	for _, p := range parts {
		if (p.Type == "image_url" && p.ImageURL != nil) || p.Type == "input_audio" {
			return true
		}
	}
	return false
}

func anthropicMediaText(req *anthropicReq) (string, bool) {
	media, lastUser := -1, -1
	var text string
	for i, m := range req.Messages {
		if m.Role == "user" {
			lastUser = i
		}
		var blocks []anthropicBlock
		if json.Unmarshal(m.Content, &blocks) != nil {
			continue
		}
		var b strings.Builder
		has := false
		for _, bl := range blocks {
			switch bl.Type {
			case "text":
				b.WriteString(bl.Text)
			case "image":
				has = true
			}
		}
		if has {
			media, text = i, b.String()
		}
	}
	if media < 0 {
		return "", false
	}
	return text, media == lastUser
}
