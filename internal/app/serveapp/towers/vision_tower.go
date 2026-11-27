package towers

import (
	"fmt"
	"sync"

	"github.com/townsendmerino/goinfer/internal/app/serveapp/media"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/towerfit"
	"github.com/townsendmerino/goinfer/internal/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

// Gemma3Vision holds Gemma 3's SigLIP encoder and projector.
type Gemma3Vision struct {
	Enc    *vision.Encoder
	Proj   *multimodal.Projector
	Cfg    vision.Config
	ImgTok int
}

func (g *Gemma3Vision) Family() string { return "Gemma 3 SigLIP" }
func (g *Gemma3Vision) Kind() string   { return "gemma3" }
func (g *Gemma3Vision) ImageTok() int  { return g.ImgTok }
func (g *Gemma3Vision) Close()         { g.Enc.Close() }

func (g *Gemma3Vision) Prep(hiddenDim int, img media.ImageRef) (ImagePrep, error) {
	p := ImagePrep{Hash: multimodal.HashImageBytes(img.Data)}
	pv, err := vision.Preprocess(img.Data, g.Cfg)
	if err != nil {
		return p, err
	}
	n := g.Proj.MMTokens()
	p.N = n
	p.Features = func() ([]float32, error) {
		hidden, err := g.Enc.Forward(pv.Data)
		if err != nil {
			return nil, fmt.Errorf("vision encoder: %w", err)
		}
		feats, err := g.Proj.Forward(hidden)
		if err != nil {
			return nil, fmt.Errorf("vision projector: %w", err)
		}
		if len(feats) != n*hiddenDim {
			return nil, fmt.Errorf("projector emitted %d features, want %d", len(feats), n*hiddenDim)
		}
		return feats, nil
	}
	p.Block = multimodal.Gemma3PromptBlock(n)
	return p, nil
}

// Qwen25Vision holds Qwen2.5-VL's ViT encoder and preprocessing grid.
type Qwen25Vision struct {
	QwenGrid
	Enc     *vision.QwenVisionEncoder
	devMu   sync.Mutex
	Require bool
}

func (q *Qwen25Vision) Family() string  { return "Qwen2.5-VL" }
func (q *Qwen25Vision) Kind() string    { return "qwen" }
func (q *Qwen25Vision) ImageTok() int   { return q.ImgTok }
func (q *Qwen25Vision) Grid() *QwenGrid { return &q.QwenGrid }
func (q *Qwen25Vision) DeepSets() int   { return 0 }
func (q *Qwen25Vision) Close()          { q.Enc.Close() }

func (q *Qwen25Vision) Prep(hiddenDim int, img media.ImageRef) (ImagePrep, error) {
	return qwenPrep(hiddenDim, q, img)
}

func (q *Qwen25Vision) Features(pv []float32, grid [3]int) ([]float32, error) {
	q.devMu.Lock()
	defer q.devMu.Unlock()
	return QwenDeviceForward(q.Enc, q.Require, func() ([]float32, error) { return q.Enc.Forward(pv, [][3]int{grid}) })
}

// Qwen3Vision binds a lazy Qwen3Tower to one model entry's preprocessing grid and placeholder token.
type Qwen3Vision struct {
	QwenGrid
	*Qwen3Tower
}

func (q *Qwen3Vision) Family() string  { return "Qwen3.5+ vision" }
func (q *Qwen3Vision) Kind() string    { return "qwen" }
func (q *Qwen3Vision) ImageTok() int   { return q.ImgTok }
func (q *Qwen3Vision) Grid() *QwenGrid { return &q.QwenGrid }
func (q *Qwen3Vision) DeepSets() int   { return q.Deep }

func (q *Qwen3Vision) Prep(hiddenDim int, img media.ImageRef) (ImagePrep, error) {
	return qwenPrep(hiddenDim, q, img)
}

func (q *Qwen3Vision) Close() {
	if q.Acc != nil {
		_ = q.Acc.Close()
	}
	if q.Enc != nil {
		q.Enc.Close()
	}
}

// GlmVision binds a lazy GlmOcrTower to one model entry's preprocessing grid and placeholder token.
type GlmVision struct {
	QwenGrid
	*GlmOcrTower
}

func (g *GlmVision) Family() string  { return "GLM-OCR" }
func (g *GlmVision) Kind() string    { return "qwen" }
func (g *GlmVision) ImageTok() int   { return g.ImgTok }
func (g *GlmVision) Grid() *QwenGrid { return &g.QwenGrid }
func (g *GlmVision) DeepSets() int   { return 0 }

func (g *GlmVision) Prep(hiddenDim int, img media.ImageRef) (ImagePrep, error) {
	return qwenPrep(hiddenDim, g, img)
}

func (g *GlmVision) Close() {
	if g.Acc != nil {
		_ = g.Acc.Close()
	}
	if g.Enc != nil {
		g.Enc.Close()
	}
}

func qwenPrep(hiddenDim int, t QwenGridTower, img media.ImageRef) (ImagePrep, error) {
	p := ImagePrep{Hash: multimodal.HashImageBytes(img.Data)}
	g := t.Grid()
	pv, grid, err := multimodal.QwenPreprocess(img.Data, g.PP)
	if err != nil {
		return p, err
	}
	n := multimodal.QwenMergedTokens(grid, g.Merge)
	deepSets := t.DeepSets()
	p.N, p.Grid, p.DeepSets = n, grid, deepSets
	p.Features = func() ([]float32, error) {
		feats, err := t.Features(pv, grid)
		if err != nil {
			return nil, fmt.Errorf("qwen vision encoder: %w", err)
		}
		if len(feats) != n*hiddenDim*(1+deepSets) {
			return nil, fmt.Errorf("qwen encoder emitted %d features, want %d (%d rows x %d, %d DeepStack sets)", len(feats), n*hiddenDim*(1+deepSets), n, hiddenDim, deepSets)
		}
		return feats, nil
	}
	p.Block = multimodal.QwenImageBlock(n)
	return p, nil
}

// Gemma4Vision holds Gemma 4's vision encoder and optional device accelerator or swapper.
type Gemma4Vision struct {
	Enc     *vision.Gemma4Encoder
	Tower   multimodal.Gemma4TowerAccelerator
	Swap    *towerfit.TowerSwap
	MaxSoft int
	ImgTok  int
}

func (g *Gemma4Vision) Family() string { return "Gemma 4 vision" }
func (g *Gemma4Vision) Kind() string   { return "gemma4" }
func (g *Gemma4Vision) ImageTok() int  { return g.ImgTok }

func (g *Gemma4Vision) Close() {
	if g.Tower != nil {
		_ = g.Tower.Close()
	}
	if g.Enc != nil {
		g.Enc.Close()
	}
}

func (g *Gemma4Vision) Prep(hiddenDim int, img media.ImageRef) (ImagePrep, error) {
	p := ImagePrep{Hash: multimodal.HashImageBytes(img.Data)}
	patches, positionIDs, err := vision.Gemma4Preprocess(img.Data, g.MaxSoft)
	if err != nil {
		return p, err
	}
	n := multimodal.Gemma4PooledTokens(positionIDs, g.Enc.Cfg.PoolingKernelSize)
	p.N = n
	p.Features = func() ([]float32, error) {
		var feats []float32
		var err error
		if g.Swap != nil {
			feats, err = g.Swap.Features(patches, positionIDs)
		} else {
			feats, err = multimodal.Gemma4TowerFeatures(g.Enc, g.Tower, patches, positionIDs)
		}
		if err != nil {
			return nil, fmt.Errorf("gemma4 vision encoder: %w", err)
		}
		if len(feats) != n*hiddenDim {
			return nil, fmt.Errorf("gemma4 encoder emitted %d features, want %d", len(feats), n*hiddenDim)
		}
		return feats, nil
	}
	p.Block = multimodal.Gemma4ImageBlock(n)
	return p, nil
}
