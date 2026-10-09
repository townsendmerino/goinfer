package serveapp

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/multimodal"
)

// S11 (docs/tasks/task-multimodal-support-2026-10.md, "S11, plan and gates"): several images in one message. Each image
// is prepared on its own (its block, its placeholder count, its lazy tower), the blocks go where the image parts sat
// among the text parts, and every block becomes its own decoder.ImageSpan: never two images in one block, even adjacent
// and the same size (docs/multimodal.md, "Do not pair images"). One image takes the same path, so the single-image
// prompt is this code with N = 1.

// imagePrep is one image of an image prompt: its block text, its placeholder count, the lazy tower that fills it, and
// what decoding needs (the Qwen grid, the DeepStack set count).
type imagePrep struct {
	n        int // the image's feature rows
	runs     int // placeholder runs it fills (0 = 1: one block; Pixtral: one per merged row, the breaks between them text)
	runLen   int // each run's length (when runs > 1)
	block    string
	features func() ([]float32, error)
	hash     uint64
	grid     [3]int
	deepSets int
}

// imageKind is the image path a model's family takes: "gemma3", "qwen" or "gemma4". The others (GLM-OCR, audio,
// Qwen3-ASR) keep their own one-media builders.
func (lm *loadedModel) imageKind() string {
	switch {
	case lm.pixtral != nil:
		return "pixtral"
	case lm.qwenEnc != nil || lm.qwen3 != nil:
		return "qwen"
	case lm.gemma4Enc != nil:
		return "gemma4"
	}
	return "gemma3"
}

// imageTok is the placeholder token id of kind's image block.
func (lm *loadedModel) imageTok(kind string) int {
	switch kind {
	case "pixtral":
		return lm.pixtral.imgTok
	case "qwen":
		return lm.qwenImgTok
	case "gemma4":
		return lm.gemma4ImgTok
	}
	return lm.vimgTok
}

// prepImage preprocesses one image for kind's family: nothing runs the tower until features is called.
func (lm *loadedModel) prepImage(kind string, img imageRef) (imagePrep, error) {
	if kind == "pixtral" {
		return lm.pixtralPrep(img)
	}
	hiddenDim := lm.model.Config().HiddenDim
	p := imagePrep{hash: multimodal.HashImageBytes(img.data)}
	switch kind {
	case "qwen":
		pv, grid, err := multimodal.QwenPreprocess(img.data, lm.qwenPP)
		if err != nil {
			return p, err
		}
		n := multimodal.QwenMergedTokens(grid, lm.qwenMerge)
		deepSets := lm.qwenDeepstackSets() // Qwen3-VL (S10): the features carry this many DeepStack sets after the merged rows
		p.n, p.grid, p.deepSets = n, grid, deepSets
		p.features = func() ([]float32, error) {
			feats, err := lm.qwenForward(pv, grid)
			if err != nil {
				return nil, fmt.Errorf("qwen vision encoder: %w", err)
			}
			if len(feats) != n*hiddenDim*(1+deepSets) {
				return nil, fmt.Errorf("qwen encoder emitted %d features, want %d (%d rows x %d, %d DeepStack sets)", len(feats), n*hiddenDim*(1+deepSets), n, hiddenDim, deepSets)
			}
			return feats, nil
		}
		// M-38 (audit-2026-09-10): Qwen2.5-VL's real chat_template.json (verified live against
		// Qwen/Qwen2.5-VL-7B-Instruct) splices <|vision_start|><|image_pad|><|vision_end|> inline
		// with NO adjacent newline on either side — the trailing "\n" this used to append doesn't
		// exist in the real template.
		p.block = multimodal.QwenImageBlock(n)
	case "gemma4":
		patches, positionIDs, err := vision.Gemma4Preprocess(img.data, lm.gemma4MaxSoft)
		if err != nil {
			return p, err
		}
		n := multimodal.Gemma4PooledTokens(positionIDs, lm.gemma4Enc.Cfg.PoolingKernelSize)
		p.n = n
		p.features = func() ([]float32, error) {
			feats, err := multimodal.Gemma4TowerFeatures(lm.gemma4Enc, lm.gemma4Tower, patches, positionIDs)
			if err != nil {
				return nil, fmt.Errorf("gemma4 vision encoder: %w", err)
			}
			if len(feats) != n*hiddenDim {
				return nil, fmt.Errorf("gemma4 encoder emitted %d features, want %d", len(feats), n*hiddenDim)
			}
			return feats, nil
		}
		// M-38 (audit-2026-09-10): Gemma 4's own processor (processing_gemma4.py, verified against the
		// real transformers source) does f"{boi_token}{image_tokens}{eoi_token}" — no adjacent
		// newline at all, unlike the trailing "\n" this used to append.
		p.block = multimodal.Gemma4ImageBlock(n)
	default:
		pv, err := vision.Preprocess(img.data, lm.vcfg)
		if err != nil {
			return p, err
		}
		n := lm.vproj.MMTokens()
		p.n = n
		p.features = func() ([]float32, error) {
			hidden, err := lm.venc.Forward(pv.Data)
			if err != nil {
				return nil, fmt.Errorf("vision encoder: %w", err)
			}
			feats, err := lm.vproj.Forward(hidden)
			if err != nil {
				return nil, fmt.Errorf("vision projector: %w", err)
			}
			if len(feats) != n*hiddenDim {
				return nil, fmt.Errorf("projector emitted %d features, want %d", len(feats), n*hiddenDim)
			}
			return feats, nil
		}
		p.block = multimodal.Gemma3PromptBlock(n)
	}
	return p, nil
}

// placeImageBlocks puts blocks[k] where image k sat in its message (S11). ordered says the images' offsets (imageRef.at,
// byte offsets into msgText, the message's joined text parts) apply to content: true when content ends with msgText,
// the message's own turn (or a turn it was merged onto). Otherwise every block leads the content in order, which is
// where one image always went before S11.
func placeImageBlocks(content, msgText string, ordered bool, imgs []imageRef, blocks []string) string {
	if !ordered || msgText == "" || !strings.HasSuffix(content, msgText) {
		return strings.Join(blocks, "") + content
	}
	base, prev := len(content)-len(msgText), 0
	for _, im := range imgs {
		if im.at < prev || im.at > len(msgText) {
			return strings.Join(blocks, "") + content
		}
		prev = im.at
	}
	var b strings.Builder
	cut := 0
	for k, im := range imgs {
		b.WriteString(content[cut : base+im.at])
		b.WriteString(blocks[k])
		cut = base + im.at
	}
	b.WriteString(content[cut:])
	return b.String()
}

// imagesPrompt assembles an image prompt for kind's family from one prep per image: the blocks placed in the last user
// turn (placeImageBlocks), each spliced back as its own Special segment, one placeholder run per image found, and the
// features concatenated in span order. With cache, each image's tower output comes from this model's feature cache.
func (lm *loadedModel) imagesPrompt(tm *chat.Template, system string, turns []chat.Turn, idx int, kind string, imgs []imageRef, ordered bool, msgText string, cache bool) (visionInput, error) {
	preps := make([]imagePrep, len(imgs))
	blocks := make([]string, len(imgs))
	for k, img := range imgs {
		p, err := lm.prepImage(kind, img)
		if err != nil {
			if len(imgs) > 1 {
				return visionInput{}, fmt.Errorf("image %d of %d: %w", k+1, len(imgs), err)
			}
			return visionInput{}, err
		}
		if cache {
			p.features = lm.cachedFeatures(p.features, img.data, lm.towerFamily(false))
		}
		preps[k], blocks[k] = p, p.block
	}
	if kind == "pixtral" && len(imgs) == 1 {
		ordered = false // the checkpoint's template moves a [text, image] message's one image first
	}
	turns[idx].Content = placeImageBlocks(turns[idx].Content, msgText, ordered, imgs, blocks)
	segs, err := multimodal.SpliceImageBlocks(tm.RenderSegments(system, turns), blocks)
	if err != nil {
		return visionInput{}, fmt.Errorf("encode: %w", err)
	}
	ids, err := lm.tk.EncodeSegments(segs, false)
	if err != nil {
		return visionInput{}, fmt.Errorf("encode: %w", err)
	}
	runs := multimodal.FindImageRuns(ids, lm.imageTok(kind))
	spans, grids, err := imageSpans(runs, preps, kind)
	if err != nil {
		return visionInput{}, err
	}
	vi := visionInput{ids: ids, qwen: kind == "qwen", gemma4: kind == "gemma4", pixtral: kind == "pixtral", deepSets: preps[0].deepSets, spans: spans, grids: grids}
	vi.imgPos, vi.imgLen, vi.imgHash, vi.grid = vi.spans[0].Pos, vi.spans[0].Len, vi.spans[0].Hash, preps[0].grid
	if kind == "pixtral" {
		vi.grids = nil // no m-RoPE
	}
	vi.features = concatImageFeatures(preps, lm.model.Config().HiddenDim)
	return vi, nil
}

// imageSpans pairs the prompt's placeholder runs (FindImageRuns, in order) with the prepared images: one span per image,
// or, for an image that fills several runs (Pixtral's merged rows, the breaks between them text), one span per run, each
// with the image's hash. The Qwen grids come back one per image.
func imageSpans(runs [][2]int, preps []imagePrep, kind string) ([]decoder.ImageSpan, [][3]int, error) {
	want := 0
	for _, p := range preps {
		want += max(p.runs, 1)
	}
	if len(runs) != want {
		return nil, nil, fmt.Errorf("found %d image placeholder runs, want %d (tokenizer/template mismatch)", len(runs), want)
	}
	var spans []decoder.ImageSpan
	var grids [][3]int
	r := 0
	for _, p := range preps {
		if p.runs > 1 {
			for range p.runs {
				if runs[r][1] != p.runLen {
					return nil, nil, fmt.Errorf("image placeholder run = %d tokens, want %d (tokenizer/template mismatch)", runs[r][1], p.runLen)
				}
				spans = append(spans, decoder.ImageSpan{Pos: runs[r][0], Len: p.runLen, Hash: p.hash})
				r++
			}
			continue
		}
		if runs[r][1] != p.n {
			unit := map[string]string{"qwen": "pads", "gemma4": "soft tokens", "gemma3": "soft tokens", "pixtral": "tokens"}[kind]
			return nil, nil, fmt.Errorf("image placeholder run = %d %s, want %d (tokenizer/template mismatch)", runs[r][1], unit, p.n)
		}
		spans = append(spans, decoder.ImageSpan{Pos: runs[r][0], Len: p.n, Hash: p.hash})
		grids = append(grids, p.grid)
		r++
	}
	return spans, grids, nil
}

// concatImageFeatures runs every image's tower and lays the rows out as the decoder's span entries take them: every
// image's merged rows in order, then (Qwen3-VL) each DeepStack set across every image in the same order.
func concatImageFeatures(preps []imagePrep, hidden int) func() ([]float32, error) {
	if len(preps) == 1 {
		return preps[0].features
	}
	return func() ([]float32, error) {
		flats := make([][]float32, len(preps))
		total := 0
		for k, p := range preps {
			f, err := p.features()
			if err != nil {
				return nil, fmt.Errorf("image %d of %d: %w", k+1, len(preps), err)
			}
			flats[k] = f
			total += p.n
		}
		sets := preps[0].deepSets
		out := make([]float32, 0, total*hidden*(1+sets))
		for s := range 1 + sets {
			for k, p := range preps {
				rows := p.n * hidden
				out = append(out, flats[k][s*rows:(s+1)*rows]...)
			}
		}
		return out, nil
	}
}

// cachedFeatures answers tower from this model's per-image cache when it holds raw's encode (withFeatureCache, per image).
func (lm *loadedModel) cachedFeatures(tower func() ([]float32, error), raw []byte, family string) func() ([]float32, error) {
	return lm.visionFeatureCache().wrap(raw, func() ([]float32, error) { return recoverDeviceTower(family, tower) })
}

// visionPromptN is visionPrompt for every image of a message (S11). One medium that is not an image of a multi-image
// family (an audio clip, a GLM-OCR page, Qwen3-ASR) takes visionPrompt; two or more must all be images, on a family that
// takes several.
func (lm *loadedModel) visionPromptN(tm *chat.Template, system string, turns []chat.Turn, imgs []imageRef, ordered bool, msgText string) (visionInput, error) {
	for _, img := range imgs {
		if img.audio && len(imgs) > 1 {
			return visionInput{}, fmt.Errorf("an audio clip must be the only media in a request (got %d media parts); send several images, or one clip", len(imgs))
		}
	}
	if len(imgs) == 1 && (imgs[0].audio || lm.glm != nil || lm.tmpl == nil) {
		return lm.visionPrompt(tm, system, turns, imgs[0])
	}
	if lm.glm != nil {
		return visionInput{}, fmt.Errorf("GLM-OCR takes one image per request (its task prompts are written for one page); got %d", len(imgs))
	}
	if lm.tmpl == nil {
		return visionInput{}, fmt.Errorf("this model has no chat template for vision")
	}
	idx := lastUserTurn(turns)
	if idx < 0 {
		return visionInput{}, fmt.Errorf("no user turn to attach the image to")
	}
	return lm.imagesPrompt(tm, system, turns, idx, lm.imageKind(), imgs, ordered, msgText, true)
}

// tooManyImages is the 400 for more images than maxImagesPerTurn.
func tooManyImages(n int) string {
	return fmt.Sprintf("at most %d images per request (each runs the vision tower), got %d", maxImagesPerTurn, n)
}

// chatMediaText returns the joined text of the message that carries the media (after the history rule, the one message
// that still does) and whether it is the last user message: only then do the images' offsets apply to the last user
// turn (placeImageBlocks).
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
	return msgs[media].text(), media == lastUser
}

// contentHasMedia reports whether an OpenAI content array carries an image_url or input_audio part.
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

// anthropicMediaText is chatMediaText for an Anthropic request: the joined text blocks of the message carrying the
// image blocks, and whether it is the last user message.
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
