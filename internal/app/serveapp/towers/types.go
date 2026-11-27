package towers

import (
	"slices"

	"github.com/townsendmerino/goinfer/chat"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/media"
	"github.com/townsendmerino/goinfer/internal/codec"
	"github.com/townsendmerino/goinfer/multimodal"
	"github.com/townsendmerino/goinfer/tokenizer"
)

// ImagePrep holds one preprocessed image's block text, placeholder count, and lazy tower runner.
type ImagePrep struct {
	N        int
	RunLens  []int
	Block    string
	Features func() ([]float32, error)
	Hash     uint64
	Grid     [3]int
	DeepSets int
}

// ImageSpan locates one image placeholder run in an encoded prompt.
type ImageSpan struct {
	Pos, Len int
	Hash     uint64
}

// VisionInput is an assembled multimodal prompt ready for generation.
type VisionInput struct {
	IDs            []int
	Features       func() ([]float32, error)
	ImgHash        uint64
	ImgPos, ImgLen int
	Grid           [3]int
	Qwen           bool
	DeepSets       int
	Gemma4         bool
	ASR            bool
	ReplyPrefix    string
	Causal         bool
	Images         int
	Spans          []ImageSpan
	Grids          [][3]int
}

// QwenGrid holds smart-resize preprocessing, spatial merge factor, and placeholder token ID for Qwen-shaped towers.
type QwenGrid struct {
	PP     multimodal.QwenPreprocessConfig
	Merge  int
	ImgTok int
}

// VisionTower is one family's image preprocessing and encoder handle.
type VisionTower interface {
	Family() string
	Kind() string
	ImageTok() int
	Prep(hiddenDim int, img media.ImageRef) (ImagePrep, error)
	Close()
}

// QwenGridTower extends VisionTower with m-RoPE grid parameters and optional DeepStack sets.
type QwenGridTower interface {
	VisionTower
	Grid() *QwenGrid
	Features(pv []float32, grid [3]int) ([]float32, error)
	DeepSets() int
}

// AudioModel carries the model metadata an AudioTower needs to build a prompt.
type AudioModel struct {
	Name        string
	HiddenDim   int
	HasTemplate bool
	TK          *tokenizer.Tokenizer
}

// AudioTower is one family's audio clip-to-prompt builder and lazy encoder handle.
type AudioTower interface {
	Family() string
	Prompt(m AudioModel, tm *chat.Template, system string, turns []chat.Turn, clip media.ImageRef) (VisionInput, error)
	Close()
}

// LastUserTurn returns the index of the last user turn in turns, or -1 if none exists.
func LastUserTurn(turns []chat.Turn) int {
	for i, turn := range slices.Backward(turns) {
		if turn.Role == "user" {
			return i
		}
	}
	return -1
}

// EncodeVisionSegments renders and encodes turns with block spliced as a Special segment.
func EncodeVisionSegments(tk *tokenizer.Tokenizer, tm *chat.Template, system string, turns []chat.Turn, block string) ([]int, error) {
	segs, err := codec.ImageSegments(tm, system, turns, block)
	if err != nil {
		return nil, err
	}
	return codec.Encode(tk, segs)
}
