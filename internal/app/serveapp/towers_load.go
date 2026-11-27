package serveapp

import (
	"fmt"
	"os"
	"slices"
	"strings"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/embeddinggemma2"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/towerfit"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/towers"
	"github.com/townsendmerino/goinfer/internal/vision"
	"github.com/townsendmerino/goinfer/multimodal"
)

type (
	visionRoute    = towerfit.VisionRoute
	deviceFallback = towerfit.DeviceFallback
	towerSwap      = towerfit.TowerSwap
)

const (
	discreteKeptNote = towerfit.DiscreteKeptNote
	mmprojKeptNote   = towerfit.MMProjKeptNote
)

var (
	towerInt8           = towerfit.TowerInt8
	enableResidentTower = towerfit.EnableResidentTower
	attachResidentTower = towerfit.AttachResidentTower
	attachWhisperDevice = towerfit.AttachWhisperDevice
	visionModelType     = towerfit.VisionModelType
	readTowerDims       = towerfit.ReadTowerDims
	towerVRAMEstimate   = towerfit.TowerVRAMEstimate
	metalTowerEstimate  = towerfit.MetalTowerEstimate
)

func (cfg *config) towerConfig() towerfit.Config {
	return towerfit.Config{
		Backend:         cfg.load.Backend,
		VisionPath:      cfg.visionPath,
		VisionDevice:    cfg.visionDevice,
		VisionQuant:     cfg.visionQuant,
		VisionMaxPixels: cfg.visionMaxPixels,
	}
}

func (cfg *config) towerBackend() string { return cfg.towerConfig().TowerBackend() }

func (cfg *config) routeVisionTower(family string, mmproj bool) visionRoute {
	return cfg.towerConfig().RouteVisionTower(family, mmproj)
}

func towerReserve(cfg config, modelPath string) int64 {
	return towerfit.Reserve(cfg.towerConfig(), modelPath)
}

func webgpuTowerFits(cfg config, dir string) error {
	return towerfit.WebGPUTowerFits(cfg.towerConfig(), dir)
}

func (s *server) loadVisionTower(cfg config) error {
	switch cfg.visionDevice {
	case "", "auto", "cpu", "gpu":
	default:
		return fmt.Errorf("-vision-device %q: want auto, cpu or gpu", cfg.visionDevice)
	}
	dir := cfg.visionPath
	if dir == "" {
		if len(cfg.models) == 1 {
			cand := s.soleModelSource(cfg)
			if fi, err := os.Stat(cand); err == nil && fi.IsDir() {
				if visionModelType(cand) == "qwen2_5_vl" {
					dir = cand
				} else if towers.IsQwen35VisionDir(cand) {
					dir = cand
				} else if towers.IsGlmOcrVisionDir(cand) {
					dir = cand
				} else if visionModelType(cand) == "gemma4" {
					dir = cand
				} else if visionModelType(cand) == "qwen3_asr" {
					dir = cand
				} else if visionModelType(cand) == "voxtral" {
					dir = cand
				} else if visionModelType(cand) == "mistral3" {
					dir = cand
				} else if visionModelType(cand) == "lfm2_vl" {
					dir = cand
				} else if _, err := multimodal.LoadProjector(cand); err == nil {
					dir = cand
				}
			}
		}
		if dir == "" {
			return nil
		}
	}
	if err := towers.VisionPathError(dir); err != nil {
		return err
	}
	if len(cfg.models) != 1 {
		return fmt.Errorf("-vision needs exactly one --model (got %d)", len(cfg.models))
	}
	if err := webgpuTowerFits(cfg, dir); err != nil {
		return err
	}
	if fi, err := os.Stat(dir); err == nil && !fi.IsDir() && strings.HasSuffix(strings.ToLower(dir), ".gguf") {
		route := cfg.routeVisionTower("qwen3_5", true)
		be := route.Backend
		if route.Note != "" {
			fmt.Fprintln(os.Stderr, route.Note)
		}
		return s.loadQwen35MMProj(dir, towerInt8("qwen3_5", cfg.visionQuant, be), be, cfg.requireBE)
	}
	mt := visionModelType(dir)
	route := cfg.routeVisionTower(mt, false)
	vbe := route.Backend
	if route.Note != "" {
		fmt.Fprintln(os.Stderr, route.Note)
	}
	int8Tower := towerInt8(mt, cfg.visionQuant, vbe)
	if mt == "qwen2_5_vl" {
		return s.loadQwenVisionTower(dir, int8Tower, vbe, cfg.requireBE)
	}
	if mt == "qwen3_5" || mt == "qwen3_5_moe" || mt == "qwen3_vl" || mt == "qwen3_vl_moe" {
		return s.loadQwen35VisionTower(dir, int8Tower, vbe, cfg.requireBE)
	}
	if mt == "qwen3_asr" {
		return s.loadQwenASR(dir)
	}
	if mt == "voxtral" {
		return s.loadVoxtral(dir, cfg.towerBackend())
	}
	if mt == "glm_ocr" {
		return s.loadGlmOcrVisionTower(dir, int8Tower, cfg.visionMaxPixels, vbe, cfg.requireBE)
	}
	if mt == "gemma4" {
		return s.loadGemma4VisionTower(dir, int8Tower, vbe, cfg.requireBE)
	}
	if mt == "mistral3" {
		return s.loadPixtralVisionTower(dir)
	}
	if mt == "lfm2_vl" {
		return s.loadLfm2VLVisionTower(dir)
	}
	towerBE := vbe
	enc, int8Tower, residentOK, err := towers.AttachGemma3Tower(towers.SiglipLoader(dir, towerBE == "metal"), int8Tower, gemma3FloatDefault(cfg, int8Tower), towerBE, cfg.requireBE, os.Stderr)
	if err != nil {
		return err
	}
	proj, err := multimodal.LoadProjector(dir)
	if err != nil {
		return fmt.Errorf("load vision projector (%s): %w", dir, err)
	}
	for _, lm := range s.models {
		g := &gemma3Vision{Enc: enc, Proj: proj, Cfg: vision.Gemma3(), ImgTok: -1}
		lm.vision = g
		if id, ok := lm.tk.TokenID(imageSoftToken); ok {
			g.ImgTok = id
		}
		if g.ImgTok < 0 {
			return fmt.Errorf("vision: tokenizer has no %q token (needed to place image embeddings)", imageSoftToken)
		}
		vq := "f32"
		if int8Tower {
			vq = "int8"
		}
		if residentOK {
			vq += "/" + towerBE + "-resident"
		}
		fmt.Fprintf(os.Stderr, "loaded vision tower for %q (%d image tokens/image, soft-token id %d, encoder %s) from %s\n", lm.name, proj.MMTokens(), g.ImgTok, vq, dir)
	}
	return nil
}

func (s *server) soleModelSource(cfg config) string {
	for _, lm := range s.models {
		if lm.source != "" {
			return lm.source
		}
	}
	return cfg.models[0].path
}

func (s *server) loadQwenVisionTower(dir string, int8Tower bool, backend string, require bool) error {
	enc, err := vision.LoadQwenVisionEncoder(dir, int8Tower)
	if err != nil {
		return fmt.Errorf("load qwen2.5-vl vision encoder (%s): %w", dir, err)
	}
	where := "CPU"
	if backend == "metal" {
		ok, err := attachResidentTower("Qwen2.5-VL", enc, backend, require, os.Stderr)
		if err != nil {
			return err
		}
		if ok {
			where = "Metal"
		}
	} else if where, err = towers.QwenTowerPlacement(backend, int8Tower, require, enc.EnableResident, os.Stderr); err != nil {
		return err
	}
	if !strings.HasPrefix(where, "CPU") {
		towers.ReleaseTowerBlocks("Qwen2.5-VL", enc)
	}
	pp, err := multimodal.LoadQwenPreprocessConfig(dir)
	if err != nil {
		return fmt.Errorf("qwen2.5-vl preprocessor config (%s): %w", dir, err)
	}
	for _, lm := range s.models {
		q := &qwen25Vision{QwenGrid: qwenGrid{PP: pp, Merge: enc.Cfg.SpatialMergeSize, ImgTok: -1}, Enc: enc, Require: require}
		lm.vision = q
		if id, ok := lm.tk.TokenID(multimodal.QwenImagePad); ok {
			q.ImgTok = id
		}
		if q.ImgTok < 0 {
			return fmt.Errorf("vision: tokenizer has no %q token (needed to place image embeddings)", multimodal.QwenImagePad)
		}
		fmt.Fprintf(os.Stderr, "loaded Qwen2.5-VL vision tower for %q (merge %d, image-pad id %d, tower on %s) from %s\n", lm.name, q.Merge, q.ImgTok, where, dir)
	}
	return nil
}

func (s *server) loadGemma4VisionTower(dir string, int8Tower bool, backend string, require bool) error {
	for _, lm := range s.models {
		if bd := lm.model.Config().UseBidirectionalAttention; bd != "" && bd != "vision" {
			return fmt.Errorf("gemma4 vision: %q sets use_bidirectional_attention=%q, not the supported %q value; GenerateGemma4VL only implements the E2B/E4B-class causal case (unset) and the 26B-A4B/31B-class %q blockwise case", lm.name, bd, "vision", "vision")
		}
	}
	enc, err := vision.LoadGemma4Encoder(dir, int8Tower)
	if err != nil {
		return fmt.Errorf("load gemma4 vision encoder (%s): %w", dir, err)
	}
	pp, err := multimodal.LoadGemma4PreprocessConfig(dir)
	if err != nil {
		return fmt.Errorf("gemma4 vision preprocessor config (%s): %w", dir, err)
	}
	tower, where, err := towers.ChooseGemma4Tower(enc, int8Tower, backend, require)
	if err != nil {
		return err
	}
	swap := gemma4TowerSwap(s.models, enc, tower, backend, dir)
	if swap != nil {
		tower = nil
		where += ", loaded for each encode and released after it"
	} else if tower != nil {
		towers.ReleaseTowerBlocks("Gemma 4", enc)
	}
	for _, lm := range s.models {
		g := &gemma4Vision{Enc: enc, Tower: tower, Swap: swap, MaxSoft: pp.MaxSoftTokens, ImgTok: -1}
		lm.vision = g
		if id, ok := lm.tk.TokenID(multimodal.Gemma4ImageSoftToken); ok {
			g.ImgTok = id
		}
		if g.ImgTok < 0 {
			return fmt.Errorf("vision: tokenizer has no %q token (needed to place image embeddings)", multimodal.Gemma4ImageSoftToken)
		}
		fmt.Fprintf(os.Stderr, "loaded Gemma 4 vision tower for %q (max %d soft tokens/image, soft-token id %d, tower on %s) from %s\n", lm.name, g.MaxSoft, g.ImgTok, where, dir)
		if towers.HasAudioConfig(dir) {
			id, ok := lm.tk.TokenID(multimodal.Gemma4AudioSoftToken)
			if !ok {
				return fmt.Errorf("audio: %s has an audio tower but its tokenizer has no %q token", dir, multimodal.Gemma4AudioSoftToken)
			}
			at := &gemma4Audio{Dir: dir, Tok: id}
			lm.audio = at
			where := "CPU"
			if backend == "metal" && slices.Contains(embeddinggemma2.AudioAccelerators(), "metal") {
				at.Device, where = "metal", "Metal (the conformer blocks; subsample and tail on the CPU)"
			}
			fmt.Fprintf(os.Stderr, "Gemma 4 audio input on for %q (audio-token id %d; the tower loads on the first clip, %s) from %s\n", lm.name, id, where, dir)
		}
	}
	return nil
}

func (s *server) loadQwenASR(dir string) error {
	for _, lm := range s.models {
		id, ok := lm.tk.TokenID("<|audio_pad|>")
		if !ok {
			return fmt.Errorf("audio: %s is a Qwen3-ASR checkpoint but its tokenizer has no <|audio_pad|> token", dir)
		}
		lm.audio = &qwenASRAudio{Dir: dir}
		fmt.Fprintf(os.Stderr, "Qwen3-ASR audio input on for %q (audio-pad id %d; the encoder loads on the first clip, CPU float32) from %s\n", lm.name, id, dir)
	}
	return nil
}

func (s *server) loadVoxtral(dir, backend string) error {
	for _, lm := range s.models {
		if _, ok := lm.tk.TokenID("[AUDIO]"); !ok {
			return fmt.Errorf("audio: %s is a Voxtral checkpoint but its tokenizer has no [AUDIO] token (a Tekken tokenizer is needed: tekken.json beside the weights)", dir)
		}
		lm.audio = &voxtralAudio{Dir: dir, Backend: backend}
		fmt.Fprintf(os.Stderr, "Voxtral audio input on for %q (the encoder loads on the first clip, float32, on %s when this binary has a device encoder for it, else the CPU; a clip over %d s is refused) from %s\n", lm.name, backend, voxtralMaxSeconds, dir)
	}
	return nil
}

func gemma4TowerSwap(models map[string]*loadedModel, enc *vision.Gemma4Encoder, tower multimodal.Gemma4TowerAccelerator, backend, dir string) *towerSwap {
	if tower == nil || backend != "cuda" || len(models) != 1 {
		return nil
	}
	var m *decoder.Model
	for _, lm := range models {
		m = lm.model
	}
	if m == nil {
		return nil
	}
	before, ok := m.ExpertCacheStats()
	if !ok {
		return nil
	}
	d, ok := towerfit.ReadTowerDims(dir)
	if !ok {
		return nil
	}
	_ = tower.Close()
	if after, err := m.ResizeExpertCache(0); err != nil {
		fmt.Fprintf(os.Stderr, "vision: the expert cache stays at %d slots/layer between images: %v\n", before.Slots, err)
	} else {
		fmt.Fprintf(os.Stderr, "vision: the expert cache grows from %d to %d slots/layer between images; the Gemma 4 tower loads for each encode\n", before.Slots, after)
	}
	return towerfit.NewTowerSwap(enc, backend, m, towerfit.TowerVRAMEstimate("gemma4", d, 0))
}
