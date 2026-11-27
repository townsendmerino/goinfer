package serveapp

import "github.com/townsendmerino/goinfer/internal/app/serveapp/towers"

type (
	visionTower   = towers.VisionTower
	qwenGrid      = towers.QwenGrid
	qwenGridTower = towers.QwenGridTower
	audioTower    = towers.AudioTower
	visionInput   = towers.VisionInput
	imagePrep     = towers.ImagePrep
	imageSpan     = towers.ImageSpan
	gridTowerPlan = towers.GridTowerPlan

	gemma3Vision = towers.Gemma3Vision
	qwen25Vision = towers.Qwen25Vision
	qwen3Vision  = towers.Qwen3Vision
	qwen3Tower   = towers.Qwen3Tower
	glmVision    = towers.GlmVision
	glmOcrTower  = towers.GlmOcrTower
	gemma4Vision = towers.Gemma4Vision
	lfm2vlTower  = towers.Lfm2VLTower
	pixtralTower = towers.PixtralTower
	gemma4Audio  = towers.Gemma4Audio
	qwenASRAudio = towers.QwenASRAudio
	voxtralAudio = towers.VoxtralAudio
)

const (
	gemma4AudioMaxSoftTokens = towers.Gemma4AudioMaxSoftTokens
	gemma4AudioMaxSeconds    = towers.Gemma4AudioMaxSeconds
	qwenASRMaxSeconds        = towers.QwenASRMaxSeconds
	voxtralMaxSeconds        = towers.VoxtralMaxSeconds
)

var (
	placeImageBlocks   = towers.PlaceImageBlocks
	imageSpans         = towers.ImageSpans
	planGridTower      = towers.PlanGridTower
	chooseGemma4Tower  = towers.ChooseGemma4Tower
	qwenDeviceForward  = towers.QwenDeviceForward
	qwenTowerPlacement = towers.QwenTowerPlacement
	isQwen35VisionDir  = towers.IsQwen35VisionDir
	setupQwen35Vision  = towers.SetupQwen35Vision
	setupQwen35MMProj  = towers.SetupQwen35MMProj
	isGlmOcrVisionDir  = towers.IsGlmOcrVisionDir
	setupGlmOcrVision  = towers.SetupGlmOcrVision
)

func (lm *loadedModel) qwenGrid() qwenGrid {
	if t, ok := lm.vision.(qwenGridTower); ok {
		return *t.Grid()
	}
	return qwenGrid{}
}

func (lm *loadedModel) hiddenDim() int {
	if lm != nil && lm.model != nil {
		return lm.model.Config().HiddenDim
	}
	return 0
}

func (lm *loadedModel) audioModel() towers.AudioModel {
	if lm == nil {
		return towers.AudioModel{}
	}
	return towers.AudioModel{
		Name:        lm.name,
		HiddenDim:   lm.hiddenDim(),
		HasTemplate: lm.tmpl != nil,
		TK:          lm.tk,
	}
}
