package families

import (
	"github.com/townsendmerino/goinfer/decoder/config"
	"github.com/townsendmerino/goinfer/internal/runtime/archparts"
	"github.com/townsendmerino/goinfer/internal/runtime/ggufconfig"
	"github.com/townsendmerino/goinfer/internal/runtime/program"
)

type (
	Config    = config.Config
	ActKind   = program.ActKind
	MoEConfig = program.MoEConfig
)

const (
	kvRowPadAlignDefault = archparts.KVRowPadAlignDefault
	maxGGUFLayers        = ggufconfig.MaxLayers
	maxGGUFHidden        = ggufconfig.MaxHidden
	maxGGUFHeads         = ggufconfig.MaxHeads
	maxGGUFVocabSize     = ggufconfig.MaxVocabSize
	maxGGUFExperts       = ggufconfig.MaxExperts
	maxGGUFIntermediate  = ggufconfig.MaxIntermediate
)
