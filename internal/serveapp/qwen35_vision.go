package serveapp

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// Qwen3.5+ image serving (P8a, docs/multimodal.md P8 record). The route is the Qwen2.5-VL one —
// preprocess → tower → GenerateQwenVL with m-RoPE — with two differences that live here:
// the tower is loaded on first use, and an image is capped at qwen3MaxImageTokens.

// qwen3MaxImageTokens is the serve-side ceiling on merged image tokens per image. The checkpoint's own
// preprocessor_config allows up to 16384 (longest_edge 16777216 px), which is not a usable request on
// the CPU decode path an image turn takes today: a Gated-DeltaNet hybrid prefills one token at a time,
// and the resident GPU paths cannot yet carry an image turn (docs/multimodal.md P8 record, item 5).
// 1024 tokens is a 1024x1024 image. Larger images are smart-resized down to fit, exactly as the
// processor would for its own max_pixels; a build-time constant, not an environment read.
const qwen3MaxImageTokens = 1024

// qwen3Tower is a Qwen3.5+ vision tower that loads on first use. encoder() is safe for concurrent
// callers; a failed load is remembered rather than retried on every request.
type qwen3Tower struct {
	dir   string
	quant bool
	once  sync.Once
	enc   *vision.Qwen3VisionEncoder
	err   error
}

func (q *qwen3Tower) encoder() (*vision.Qwen3VisionEncoder, error) {
	q.once.Do(func() {
		q.enc, q.err = vision.LoadQwen3VisionEncoder(q.dir, q.quant)
		if q.err != nil {
			q.err = fmt.Errorf("load qwen3.5 vision tower (%s): %w", q.dir, q.err)
		}
	})
	return q.enc, q.err
}

// isQwen35VisionDir reports whether dir is a Qwen3.5+ checkpoint that carries a usable vision tower:
// model_type qwen3_5/qwen3_5_moe, a non-empty vision_config with no DeepStack, and a preprocessor
// config LoadQwen3PreprocessConfig accepts. Used for AUTO-discovery only: a stripped text-only copy
// (no vision_config or no preprocessor_config.json) is simply not a vision model, not an error.
func isQwen35VisionDir(dir string) bool {
	if mt := visionModelType(dir); mt != "qwen3_5" && mt != "qwen3_5_moe" {
		return false
	}
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return false
	}
	var c struct {
		Vision *struct {
			Depth     int   `json:"depth"`
			Deepstack []int `json:"deepstack_visual_indexes"`
		} `json:"vision_config"`
	}
	if json.Unmarshal(raw, &c) != nil || c.Vision == nil || c.Vision.Depth <= 0 || len(c.Vision.Deepstack) != 0 {
		return false
	}
	_, err = multimodal.LoadQwen3PreprocessConfig(dir)
	return err == nil
}

// setupQwen35Vision does everything about a Qwen3.5+ tower that can be decided without the weights: it
// validates the directory, reads the preprocessor config, lowers max_pixels to the serve cap, and
// returns the (not yet loaded) tower. Split from loadQwen35VisionTower so the cap and the laziness are
// testable without a tokenizer.
func setupQwen35Vision(dir string, int8Tower bool) (*qwen3Tower, multimodal.QwenPreprocessConfig, error) {
	if !isQwen35VisionDir(dir) {
		return nil, multimodal.QwenPreprocessConfig{}, fmt.Errorf("%s is not a Qwen3.5+ checkpoint with a usable vision tower (needs vision_config with deepstack_visual_indexes [], and a preprocessor_config.json with size.shortest_edge/longest_edge)", dir)
	}
	pp, err := multimodal.LoadQwen3PreprocessConfig(dir)
	if err != nil {
		return nil, pp, fmt.Errorf("qwen3.5 preprocessor config (%s): %w", dir, err)
	}
	if limit := qwen3MaxImageTokens * pp.MergeSize * pp.MergeSize * pp.PatchSize * pp.PatchSize; pp.MaxPixels > limit {
		pp.MaxPixels = limit
	}
	return &qwen3Tower{dir: dir, quant: int8Tower}, pp, nil
}

// loadQwen35VisionTower attaches a Qwen3.5+ tower to the single loaded model. Everything that can be
// checked without reading the tower's weights is checked here, at startup, so a bad -vision dir fails
// loudly now rather than at the first image; the weights themselves load on first use.
func (s *server) loadQwen35VisionTower(dir string, int8Tower bool) error {
	tower, pp, err := setupQwen35Vision(dir, int8Tower)
	if err != nil {
		return err
	}
	for _, lm := range s.models {
		lm.qwen3 = tower
		lm.qwenPP = pp
		lm.qwenMerge = pp.MergeSize
		lm.qwenImgTok = -1
		if id, ok := lm.tk.TokenID(multimodal.QwenImagePad); ok {
			lm.qwenImgTok = id
		}
		if lm.qwenImgTok < 0 {
			return fmt.Errorf("vision: tokenizer has no %q token (needed to place image embeddings)", multimodal.QwenImagePad)
		}
		fmt.Fprintf(os.Stderr, "Qwen3.5 vision for %q: tower loads on first image (merge %d, image-pad id %d, <= %d tokens/image) from %s\n", lm.name, lm.qwenMerge, lm.qwenImgTok, qwen3MaxImageTokens, dir)
	}
	return nil
}
