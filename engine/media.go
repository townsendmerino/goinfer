package engine

import (
	"context"

	"github.com/townsendmerino/goinfer/decoder"
)

// MediaKind is the decoder generator a Media request takes, named after its entry point.
type MediaKind int

const (
	MediaImage       MediaKind = iota // bidirectional image tokens: decoder.Model.GenerateVLSpans
	MediaImageCausal                  // causal image tokens, one span per run: GenerateVLCausalSpans
	MediaGemma4                       // Gemma 4's image tokens: GenerateGemma4VLSpans
	MediaQwen                         // Qwen-VL's m-RoPE images: GenerateQwenVLDeepstackSpans
	MediaAudio                        // an audio clip's encoder rows: GenerateAudio, on the first span
)

// Media is a request's image or audio input: the placeholder runs of the prompt the features replace, in prompt order,
// and the features themselves. Features returns every span's rows concatenated; it runs at most once, lazily, and not
// at all when a resident backend reuses a resent image's KV. For MediaQwen, Grids is each span's m-RoPE grid (t, h,
// w in patch units), Merge the spatial merge and ImageToken the image placeholder's id, and when DeepSets is above 0
// Features returns the merged rows followed by that many DeepStack sets of the same size.
type Media struct {
	Kind       MediaKind
	Spans      []decoder.ImageSpan
	Features   func() ([]float32, error)
	Grids      [][3]int
	Merge      int
	ImageToken int
	DeepSets   int
}

// mediaModel is the part of *decoder.Model a Media request calls, so a test can fake the dispatch.
type mediaModel interface {
	Config() *decoder.Config
	GenerateAudio(context.Context, []int, int, int, func() ([]float32, error), int, decoder.SamplingParams) (<-chan int, *decoder.Generation)
	GenerateGemma4VLSpans(context.Context, []int, []decoder.ImageSpan, func() ([]float32, error), int, decoder.SamplingParams) (<-chan int, *decoder.Generation)
	GenerateQwenVLDeepstackSpans(context.Context, []int, []decoder.ImageSpan, func() ([]float32, [][]float32, error), [][3]int, int, int, int, decoder.SamplingParams) (<-chan int, *decoder.Generation)
	GenerateVLCausalSpans(context.Context, []int, []decoder.ImageSpan, func() ([]float32, error), int, decoder.SamplingParams) (<-chan int, *decoder.Generation)
	GenerateVLSpans(context.Context, []int, []decoder.ImageSpan, func() ([]float32, error), int, decoder.SamplingParams) (<-chan int, *decoder.Generation)
}

var _ mediaModel = (*decoder.Model)(nil)

// generateMedia starts req, whose Media is set, on the generator its Kind names. Media turns take no session (a
// multimodal prompt opts out of the CPU-side prefix reuse).
func generateMedia(ctx context.Context, m mediaModel, req Request) (<-chan int, *decoder.Generation) {
	md := req.Media
	switch md.Kind {
	case MediaAudio:
		return m.GenerateAudio(ctx, req.PromptIDs, md.Spans[0].Pos, md.Spans[0].Len, md.Features, req.MaxTokens, req.Sampling)
	case MediaQwen:
		features := func() ([]float32, [][]float32, error) {
			f, err := md.Features()
			return f, nil, err
		}
		if md.DeepSets > 0 { // split the flat features into the merged rows and the DeepStack sets
			total := 0
			for _, sp := range md.Spans {
				total += sp.Len
			}
			n := total * m.Config().HiddenDim
			features = func() ([]float32, [][]float32, error) {
				flat, err := md.Features()
				if err != nil {
					return nil, nil, err
				}
				deep := make([][]float32, md.DeepSets)
				for l := range deep {
					deep[l] = flat[(l+1)*n : (l+2)*n]
				}
				return flat[:n], deep, nil
			}
		}
		return m.GenerateQwenVLDeepstackSpans(ctx, req.PromptIDs, md.Spans, features, md.Grids, md.Merge, md.ImageToken, req.MaxTokens, req.Sampling)
	case MediaGemma4:
		return m.GenerateGemma4VLSpans(ctx, req.PromptIDs, md.Spans, md.Features, req.MaxTokens, req.Sampling)
	case MediaImageCausal:
		return m.GenerateVLCausalSpans(ctx, req.PromptIDs, md.Spans, md.Features, req.MaxTokens, req.Sampling)
	default:
		return m.GenerateVLSpans(ctx, req.PromptIDs, md.Spans, md.Features, req.MaxTokens, req.Sampling)
	}
}
