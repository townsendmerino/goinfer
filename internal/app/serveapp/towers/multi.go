package towers

import (
	"fmt"
	"strings"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/media"
	"github.com/townsendmerino/goinfer/internal/codec"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// PlaceImageBlocks inserts blocks[k] where image k sat in msgText when ordered holds, or prepends all blocks in order.
func PlaceImageBlocks(content, msgText string, ordered bool, imgs []media.ImageRef, blocks []string) string {
	if !ordered || msgText == "" || !strings.HasSuffix(content, msgText) {
		return strings.Join(blocks, "") + content
	}
	base, prev := len(content)-len(msgText), 0
	for _, im := range imgs {
		if im.At < prev || im.At > len(msgText) {
			return strings.Join(blocks, "") + content
		}
		prev = im.At
	}
	var b strings.Builder
	cut := 0
	for k, im := range imgs {
		b.WriteString(content[cut : base+im.At])
		b.WriteString(blocks[k])
		cut = base + im.At
	}
	b.WriteString(content[cut:])
	return b.String()
}

// ImageSpans pairs placeholder runs in encoded prompt IDs with prepared images.
func ImageSpans(runs [][2]int, preps []ImagePrep, kind string) ([]ImageSpan, [][3]int, error) {
	want := 0
	for _, p := range preps {
		want += max(len(p.RunLens), 1)
	}
	if len(runs) != want {
		return nil, nil, fmt.Errorf("found %d image placeholder runs, want %d (tokenizer/template mismatch)", len(runs), want)
	}
	var spans []ImageSpan
	var grids [][3]int
	r := 0
	for _, p := range preps {
		if len(p.RunLens) > 0 {
			for _, ln := range p.RunLens {
				if runs[r][1] != ln {
					return nil, nil, fmt.Errorf("image placeholder run = %d tokens, want %d (tokenizer/template mismatch)", runs[r][1], ln)
				}
				spans = append(spans, ImageSpan{Pos: runs[r][0], Len: ln, Hash: p.Hash})
				r++
			}
			continue
		}
		if runs[r][1] != p.N {
			unit := map[string]string{"qwen": "pads", "gemma4": "soft tokens", "gemma3": "soft tokens", "pixtral": "tokens", "lfm2vl": "tokens"}[kind]
			return nil, nil, fmt.Errorf("image placeholder run = %d %s, want %d (tokenizer/template mismatch)", runs[r][1], unit, p.N)
		}
		spans = append(spans, ImageSpan{Pos: runs[r][0], Len: p.N, Hash: p.Hash})
		grids = append(grids, p.Grid)
		r++
	}
	return spans, grids, nil
}

// ConcatImageFeatures concatenates merged image rows followed by each DeepStack set across all images in order.
func ConcatImageFeatures(preps []ImagePrep, hidden int) func() ([]float32, error) {
	if len(preps) == 1 {
		return preps[0].Features
	}
	return func() ([]float32, error) {
		flats := make([][]float32, len(preps))
		total := 0
		for k, p := range preps {
			f, err := p.Features()
			if err != nil {
				return nil, fmt.Errorf("image %d of %d: %w", k+1, len(preps), err)
			}
			flats[k] = f
			total += p.N
		}
		sets := preps[0].DeepSets
		out := make([]float32, 0, total*hidden*(1+sets))
		for s := range 1 + sets {
			for k, p := range preps {
				rows := p.N * hidden
				out = append(out, flats[k][s*rows:(s+1)*rows]...)
			}
		}
		return out, nil
	}
}

// ImagesPromptArgs bundles the inputs to AssembleImagesPrompt.
type ImagesPromptArgs struct {
	Vision    VisionTower
	TK        *tokenizer.Tokenizer
	HiddenDim int
	TM        *chat.Template
	System    string
	Turns     []chat.Turn
	Idx       int
	Imgs      []media.ImageRef
	Ordered   bool
	MsgText   string
	WrapCache func(tower func() ([]float32, error), raw []byte) func() ([]float32, error)
}

// AssembleImagesPrompt prepares imgs on a.Vision, splices their blocks into a.Turns[a.Idx], encodes segments, and locates spans.
func AssembleImagesPrompt(a ImagesPromptArgs) (VisionInput, error) {
	kind := a.Vision.Kind()
	preps := make([]ImagePrep, len(a.Imgs))
	blocks := make([]string, len(a.Imgs))
	for k, img := range a.Imgs {
		p, err := a.Vision.Prep(a.HiddenDim, img)
		if err != nil {
			if len(a.Imgs) > 1 {
				return VisionInput{}, fmt.Errorf("image %d of %d: %w", k+1, len(a.Imgs), err)
			}
			return VisionInput{}, err
		}
		if a.WrapCache != nil {
			p.Features = a.WrapCache(p.Features, img.Data)
		}
		preps[k], blocks[k] = p, p.Block
	}
	ordered := a.Ordered
	if kind == "pixtral" && len(a.Imgs) == 1 {
		ordered = false
	}
	a.Turns[a.Idx].Content = PlaceImageBlocks(a.Turns[a.Idx].Content, a.MsgText, ordered, a.Imgs, blocks)
	segs, err := codec.ImagesSegments(a.TM, a.System, a.Turns, blocks)
	if err != nil {
		return VisionInput{}, fmt.Errorf("encode: %w", err)
	}
	ids, err := codec.Encode(a.TK, segs)
	if err != nil {
		return VisionInput{}, fmt.Errorf("encode: %w", err)
	}
	runs := multimodal.FindImageRuns(ids, a.Vision.ImageTok())
	spans, grids, err := ImageSpans(runs, preps, kind)
	if err != nil {
		return VisionInput{}, err
	}
	causal := kind == "pixtral" || kind == "lfm2vl"
	vi := VisionInput{
		IDs:      ids,
		Qwen:     kind == "qwen",
		Gemma4:   kind == "gemma4",
		Causal:   causal,
		DeepSets: preps[0].DeepSets,
		Spans:    spans,
		Grids:    grids,
		Images:   len(preps),
	}
	vi.ImgPos, vi.ImgLen, vi.ImgHash, vi.Grid = vi.Spans[0].Pos, vi.Spans[0].Len, vi.Spans[0].Hash, preps[0].Grid
	if causal {
		vi.Grids = nil
	}
	vi.Features = ConcatImageFeatures(preps, a.HiddenDim)
	return vi, nil
}
