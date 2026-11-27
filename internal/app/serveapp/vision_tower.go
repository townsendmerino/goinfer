package serveapp

import (
	"fmt"
	"sync"

	"github.com/townsendmerino/goinfer/internal/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// visionTower is one family's image input: its preprocessing, its tower and the placeholder its block carries. An entry
// holds at most one (loadVisionTower attaches one family per server). The startup tower is shared by every startup
// entry, adapters included; each entry holds its own visionTower value over it, and only the last holder closes it
// (visionShare).
type visionTower interface {
	// family names the tower in errors and keys the per-image feature cache (withFeatureCache); it must stay exact.
	family() string
	// kind is imagesPrompt's family key: "gemma3", "qwen", "gemma4", "pixtral" or "lfm2vl".
	kind() string
	// imageTok is the placeholder token id of the family's image block.
	imageTok() int
	// prep preprocesses one image; nothing runs the tower until the result's features is called.
	prep(lm *loadedModel, img imageRef) (imagePrep, error)
	// close releases the tower: its device tower first (a device ledger only its Close frees), then the CPU encoder,
	// whose weights alias its checkpoint's mappings (aikit vision aliasQMat) until Close. Called once, by the tower's
	// last holder.
	close()
}

// qwenGrid is what the Qwen-shaped towers (Qwen2.5-VL, Qwen3.5+ and Qwen3-VL, GLM-OCR) share: the smart-resize
// preprocessing, the spatial merge and the image-pad id. Their turns go through GenerateQwenVL with m-RoPE.
type qwenGrid struct {
	pp     multimodal.QwenPreprocessConfig
	merge  int // spatial_merge_size
	imgTok int // <|image_pad|> (GLM-OCR: <|image|>)
}

// qwenGridTower is a Qwen-shaped tower: features returns the merged rows for one image's grid, then deepSets DeepStack
// sets of the same size (Qwen3-VL; 0 for the others).
type qwenGridTower interface {
	visionTower
	grid() *qwenGrid
	features(pv []float32, grid [3]int) ([]float32, error)
	deepSets() int
}

// qwenGrid is this entry's Qwen grid parameters; the zero grid for any other tower, or none.
func (lm *loadedModel) qwenGrid() qwenGrid {
	if t, ok := lm.vision.(qwenGridTower); ok {
		return *t.grid()
	}
	return qwenGrid{}
}

// gemma3Vision is Gemma 3's SigLIP tower and its separate projector.
type gemma3Vision struct {
	enc    *vision.Encoder
	proj   *multimodal.Projector
	cfg    vision.Config
	imgTok int // <image_soft_token>
}

func (g *gemma3Vision) family() string { return "Gemma 3 SigLIP" }
func (g *gemma3Vision) kind() string   { return "gemma3" }
func (g *gemma3Vision) imageTok() int  { return g.imgTok }
func (g *gemma3Vision) close()         { g.enc.Close() }

func (g *gemma3Vision) prep(lm *loadedModel, img imageRef) (imagePrep, error) {
	hiddenDim := lm.model.Config().HiddenDim
	p := imagePrep{hash: multimodal.HashImageBytes(img.Data)}
	pv, err := vision.Preprocess(img.Data, g.cfg)
	if err != nil {
		return p, err
	}
	n := g.proj.MMTokens()
	p.n = n
	p.features = func() ([]float32, error) {
		hidden, err := g.enc.Forward(pv.Data)
		if err != nil {
			return nil, fmt.Errorf("vision encoder: %w", err)
		}
		feats, err := g.proj.Forward(hidden)
		if err != nil {
			return nil, fmt.Errorf("vision projector: %w", err)
		}
		if len(feats) != n*hiddenDim {
			return nil, fmt.Errorf("projector emitted %d features, want %d", len(feats), n*hiddenDim)
		}
		return feats, nil
	}
	p.block = multimodal.Gemma3PromptBlock(n)
	return p, nil
}

// qwen25Vision is the Qwen2.5-VL ViT, loaded at startup. The merger is in the encoder (no separate projector).
type qwen25Vision struct {
	qwenGrid
	enc     *vision.QwenVisionEncoder
	devMu   sync.Mutex // serializes the tower and its CPU fallback (qwenDeviceForward)
	require bool       // -require-backend: a device-memory failure fails the request instead of falling back to the CPU
}

func (q *qwen25Vision) family() string  { return "Qwen2.5-VL" }
func (q *qwen25Vision) kind() string    { return "qwen" }
func (q *qwen25Vision) imageTok() int   { return q.imgTok }
func (q *qwen25Vision) grid() *qwenGrid { return &q.qwenGrid }
func (q *qwen25Vision) deepSets() int   { return 0 }
func (q *qwen25Vision) close()          { q.enc.Close() }

func (q *qwen25Vision) prep(lm *loadedModel, img imageRef) (imagePrep, error) {
	return qwenPrep(lm, q, img)
}

// features is the ViT with the device-memory fallback (qwenDeviceForward).
func (q *qwen25Vision) features(pv []float32, grid [3]int) ([]float32, error) {
	q.devMu.Lock()
	defer q.devMu.Unlock()
	return qwenDeviceForward(q.enc, q.require, func() ([]float32, error) { return q.enc.Forward(pv, [][3]int{grid}) })
}

// qwen3Vision is an entry's view of a Qwen3.5+ or Qwen3-VL tower, which loads on the first image (qwen3Tower): every
// Qwen3.5 checkpoint carries one, so loading it at startup would tax every text-only user's startup and memory.
type qwen3Vision struct {
	qwenGrid
	*qwen3Tower
}

func (q *qwen3Vision) family() string  { return "Qwen3.5+ vision" }
func (q *qwen3Vision) kind() string    { return "qwen" }
func (q *qwen3Vision) imageTok() int   { return q.imgTok }
func (q *qwen3Vision) grid() *qwenGrid { return &q.qwenGrid }
func (q *qwen3Vision) deepSets() int   { return q.deep }

func (q *qwen3Vision) prep(lm *loadedModel, img imageRef) (imagePrep, error) {
	return qwenPrep(lm, q, img)
}

func (q *qwen3Vision) close() {
	if q.acc != nil {
		_ = q.acc.Close()
	}
	if q.enc != nil {
		q.enc.Close()
	}
}

// glmVision is an entry's view of a GLM-OCR tower, which loads on the first image (glmOcrTower). Its requests take
// glmOcrVisionPrompt, one page each, not imagesPrompt.
type glmVision struct {
	qwenGrid
	*glmOcrTower
}

func (g *glmVision) family() string  { return "GLM-OCR" }
func (g *glmVision) kind() string    { return "qwen" }
func (g *glmVision) imageTok() int   { return g.imgTok }
func (g *glmVision) grid() *qwenGrid { return &g.qwenGrid }
func (g *glmVision) deepSets() int   { return 0 }

func (g *glmVision) prep(lm *loadedModel, img imageRef) (imagePrep, error) {
	return qwenPrep(lm, g, img)
}

func (g *glmVision) close() {
	if g.acc != nil {
		_ = g.acc.Close()
	}
	if g.enc != nil {
		g.enc.Close()
	}
}

// qwenPrep prepares one image for a Qwen-shaped tower: the smart-resized grid, the merged-row count, and the lazy tower.
func qwenPrep(lm *loadedModel, t qwenGridTower, img imageRef) (imagePrep, error) {
	hiddenDim := lm.model.Config().HiddenDim
	p := imagePrep{hash: multimodal.HashImageBytes(img.Data)}
	g := t.grid()
	pv, grid, err := multimodal.QwenPreprocess(img.Data, g.pp)
	if err != nil {
		return p, err
	}
	n := multimodal.QwenMergedTokens(grid, g.merge)
	deepSets := t.deepSets() // Qwen3-VL: the features carry this many DeepStack sets after the merged rows
	p.n, p.grid, p.deepSets = n, grid, deepSets
	p.features = func() ([]float32, error) {
		feats, err := t.features(pv, grid)
		if err != nil {
			return nil, fmt.Errorf("qwen vision encoder: %w", err)
		}
		if len(feats) != n*hiddenDim*(1+deepSets) {
			return nil, fmt.Errorf("qwen encoder emitted %d features, want %d (%d rows x %d, %d DeepStack sets)", len(feats), n*hiddenDim*(1+deepSets), n, hiddenDim, deepSets)
		}
		return feats, nil
	}
	// Qwen2.5-VL's real chat_template.json splices <|vision_start|><|image_pad|><|vision_end|> inline with no
	// adjacent newline on either side, so the block carries none.
	p.block = multimodal.QwenImageBlock(n)
	return p, nil
}

// gemma4Vision is Gemma 4's vision tower. There is no separate projector: Gemma4Encoder.Forward includes the
// embed_vision projection.
type gemma4Vision struct {
	enc   *vision.Gemma4Encoder
	tower multimodal.Gemma4TowerAccelerator // nil: the CPU tower (docs/multimodal.md)
	swap  *towerSwap                        // non-nil: the device tower loads for each encode (towerSwap)
	// maxSoft is the checkpoint's vision_soft_tokens_per_image budget (vision.Gemma4Preprocess's maxSoftTokens); an
	// image's token count is data-dependent (multimodal.Gemma4PooledTokens).
	maxSoft int
	imgTok  int // <|image|>
}

func (g *gemma4Vision) family() string { return "Gemma 4 vision" }
func (g *gemma4Vision) kind() string   { return "gemma4" }
func (g *gemma4Vision) imageTok() int  { return g.imgTok }

func (g *gemma4Vision) close() {
	if g.tower != nil {
		_ = g.tower.Close()
	}
	if g.enc != nil {
		g.enc.Close()
	}
}

func (g *gemma4Vision) prep(lm *loadedModel, img imageRef) (imagePrep, error) {
	hiddenDim := lm.model.Config().HiddenDim
	p := imagePrep{hash: multimodal.HashImageBytes(img.Data)}
	patches, positionIDs, err := vision.Gemma4Preprocess(img.Data, g.maxSoft)
	if err != nil {
		return p, err
	}
	n := multimodal.Gemma4PooledTokens(positionIDs, g.enc.Cfg.PoolingKernelSize)
	p.n = n
	p.features = func() ([]float32, error) {
		var feats []float32
		var err error
		if g.swap != nil {
			feats, err = g.swap.Features(patches, positionIDs)
		} else {
			feats, err = multimodal.Gemma4TowerFeatures(g.enc, g.tower, patches, positionIDs)
		}
		if err != nil {
			return nil, fmt.Errorf("gemma4 vision encoder: %w", err)
		}
		if len(feats) != n*hiddenDim {
			return nil, fmt.Errorf("gemma4 encoder emitted %d features, want %d", len(feats), n*hiddenDim)
		}
		return feats, nil
	}
	// Gemma 4's own processor (processing_gemma4.py) builds f"{boi_token}{image_tokens}{eoi_token}": no adjacent
	// newline at all, so the block carries none.
	p.block = multimodal.Gemma4ImageBlock(n)
	return p, nil
}
