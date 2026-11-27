package serveapp

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/embeddinggemma2"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/towerfit"
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
	refuseCPUFallback   = towerfit.RefuseCPUFallback
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

// loadVisionTower attaches a vision tower to the (single) loaded model, making it vision-capable (serve then accepts
// image content parts). The family is chosen by visionModelType. The dir is -vision if set, else the sole --model's own
// dir when it carries a vision tower (auto-discovery). A tower needs exactly one --model, so -vision with a model zoo
// is an error. Absent a tower it is a no-op: text-only serving is unchanged.
func (s *server) loadVisionTower(cfg config) error {
	switch cfg.visionDevice {
	case "", "auto", "cpu", "gpu": // "" (a config built without flags) is auto
	default:
		return fmt.Errorf("-vision-device %q: want auto, cpu or gpu", cfg.visionDevice)
	}
	dir := cfg.visionPath
	if dir == "" {
		// Auto-discover: a single --model dir that holds a vision tower — either the
		// Gemma 3 projector or a Qwen2.5-VL checkpoint (its ViT lives in the same dir).
		if len(cfg.models) == 1 {
			cand := s.soleModelSource(cfg)
			if fi, err := os.Stat(cand); err == nil && fi.IsDir() {
				if visionModelType(cand) == "qwen2_5_vl" {
					dir = cand
				} else if isQwen35VisionDir(cand) {
					dir = cand
				} else if isGlmOcrVisionDir(cand) {
					dir = cand
				} else if visionModelType(cand) == "gemma4" {
					dir = cand
				} else if visionModelType(cand) == "qwen3_asr" {
					dir = cand
				} else if visionModelType(cand) == "voxtral" { // Voxtral Mini's audio tower, in the model's own directory
					dir = cand
				} else if visionModelType(cand) == "mistral3" { // Ministral 3's Pixtral tower
					dir = cand
				} else if visionModelType(cand) == "lfm2_vl" { // LFM2-VL's SigLIP2 NaFlex tower
					dir = cand
				} else if _, err := multimodal.LoadProjector(cand); err == nil {
					dir = cand
				}
			}
		}
		if dir == "" {
			return nil // no vision tower — text-only model
		}
	}
	if err := visionPathError(dir); err != nil {
		return err
	}
	// cfg.models, not s.models: loadAdapters (called just above) already put each --adapter's own served name in s.models, so
	// `--model base --adapter ft=...` would count 2 and refuse a valid single-base-model vision setup. cfg.models is the raw
	// --model list, matching the auto-discovery branch's len(cfg.models) == 1 check.
	if len(cfg.models) != 1 {
		return fmt.Errorf("-vision needs exactly one --model (got %d)", len(cfg.models))
	}
	// A WebGPU tower shares the decoder's device memory, which towerReserve priced into the KV slots before the load: on a
	// unified-memory machine a tower that needs more than the memory available refuses the load (webgpu_tower_fit.go).
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
	// the tower backend, not the model's backend: -vision-device cpu keeps Gemma 3's SigLIP tower on the CPU, in f32, under
	// --backend cuda or webgpu too, and a discrete GPU under webgpu keeps a tower on the CPU until the device is asked for.
	route := cfg.routeVisionTower(mt, false)
	vbe := route.Backend
	if route.Note != "" {
		fmt.Fprintln(os.Stderr, route.Note)
	}
	int8Tower := towerInt8(mt, cfg.visionQuant, vbe)
	if mt == "qwen2_5_vl" {
		return s.loadQwenVisionTower(dir, int8Tower, vbe, cfg.requireBE)
	}
	if mt == "qwen3_5" || mt == "qwen3_5_moe" || mt == "qwen3_vl" || mt == "qwen3_vl_moe" { // qwen3_vl, qwen3_vl_moe: the same tower plus DeepStack; literals, which the support-table loader test reads
		return s.loadQwen35VisionTower(dir, int8Tower, vbe, cfg.requireBE)
	}
	if mt == "qwen3_asr" { // speech to text; the audio encoder is in the model's own directory
		return s.loadQwenASR(dir)
	}
	if mt == "voxtral" { // speech to text; the audio tower and projector are in the model's own directory
		return s.loadVoxtral(dir, cfg.towerBackend())
	}
	if mt == "glm_ocr" {
		return s.loadGlmOcrVisionTower(dir, int8Tower, cfg.visionMaxPixels, vbe, cfg.requireBE)
	}
	if mt == "gemma4" {
		return s.loadGemma4VisionTower(dir, int8Tower, vbe, cfg.requireBE)
	}
	if mt == "mistral3" { // Ministral 3's Pixtral tower, on the CPU in float32
		return s.loadPixtralVisionTower(dir)
	}
	if mt == "lfm2_vl" { // LFM2-VL's SigLIP2 NaFlex tower, on the CPU in float32
		return s.loadLfm2VLVisionTower(dir)
	}
	// On CUDA the unset default is the float32 tower (towerInt8); if it does not attach, attachGemma3Tower releases it and
	// attaches the int8 device tower instead of leaving the CPU one (an explicit -vision-quant never falls back). The unset
	// default was settled before the plan was made: resolveGemma3VisionQuant makes it int8 on a card too tight for the
	// decoder, one KV slot and the float32 tower. Both are in gemma3_tower.go. The CUDA default is float32 because the int8
	// tower changes the reply, at the cost of about 1.7 GiB of VRAM against 0.56. The CUDA resident tower registers its factory with vision.RegisterResident
	// (cuda/vision_register.go) through cuda/cmd/serve's blank import.
	towerBE := vbe
	enc, int8Tower, residentOK, err := attachGemma3Tower(siglipLoader(dir, towerBE == "metal"), int8Tower, gemma3FloatDefault(cfg, int8Tower), towerBE, cfg.requireBE, os.Stderr)
	if err != nil {
		return err
	}
	proj, err := multimodal.LoadProjector(dir)
	if err != nil {
		return fmt.Errorf("load vision projector (%s): %w", dir, err)
	}
	for _, lm := range s.models {
		g := &gemma3Vision{enc: enc, proj: proj, cfg: vision.Gemma3(), imgTok: -1}
		lm.vision = g
		if id, ok := lm.tk.TokenID(imageSoftToken); ok {
			g.imgTok = id
		}
		if g.imgTok < 0 {
			return fmt.Errorf("vision: tokenizer has no %q token (needed to place image embeddings)", imageSoftToken)
		}
		vq := "f32"
		if int8Tower {
			vq = "int8"
		}
		if residentOK {
			vq += "/" + towerBE + "-resident"
		}
		fmt.Fprintf(os.Stderr, "loaded vision tower for %q (%d image tokens/image, soft-token id %d, encoder %s) from %s\n", lm.name, proj.MMTokens(), g.imgTok, vq, dir)
	}
	return nil
}

// towerInt8 says whether a vision tower loads with int8 matmul weights. Gemma 3's SigLIP defaults to float32 on every
// backend, because the int8 tower is lossy at real size (docs/measurements/siglip-int8-fidelity-2026-10-07.md). Under
// WebGPU the default is the float32 device tower (its speed gate: docs/completed/task-multimodal-support-2026-10.md,
// the float32 SigLIP tower on WebGPU); -vision-quant int8 asks for the int8 one.
// Every other tower gets int8 only when asked for, whatever the backend: its device towers are float32 and its gates ran
// float32, and forcing int8 under cuda/webgpu bought no speed (the CPU int8 tower is not faster) and cost fidelity
// (docs/measurements/vision-tower-int8-fidelity-2026-10-02.md). A new vision model type must be added to the list below,
// or it falls into the backend default and gets an int8 CPU tower nobody validated under cuda/webgpu.
// blockReleaser is a vision encoder that can drop its encoder blocks from the host once a device tower holds them, and
// load them again on its CPU path's first use (aikit's ReleaseBlocks).
type blockReleaser interface{ ReleaseBlocks() error }

// releaseTowerBlocks drops a device-attached tower's host blocks: the
// device tower serves the images, and a CPU fallback reloads them. Called once per tower, after its device tower is built
// and before any request runs it. A variable so a test can count the calls.
var releaseTowerBlocks = func(family string, enc any) {
	r, ok := enc.(blockReleaser)
	if !ok {
		return
	}
	if err := r.ReleaseBlocks(); err != nil {
		fmt.Fprintf(os.Stderr, "vision: the %s tower keeps its host blocks: %v\n", family, err)
	}
}

// soleModelSource is the single --model as loadDecoder resolved it: for an `hf:<repo>:safetensors` reference, the
// checkpoint directory it fetched, which is where a VL repo's tower lives (the typed string names no directory, so
// stat-ing it would find no tower and serve the model text-only without a word). Falls back to the typed path for an
// entry built without a source.
func (s *server) soleModelSource(cfg config) string {
	for _, lm := range s.models {
		if lm.source != "" {
			return lm.source
		}
	}
	return cfg.models[0].path
}

// loadQwenVisionTower attaches the Qwen2.5-VL ViT (aikit) to the single loaded
// model. Unlike Gemma 3 there's no separate projector — the merger is in the
// encoder; preprocessing + m-RoPE are Qwen-specific (the image path branches on
// qwenEnc). Image placeholders use <|image_pad|>, expanded per image to the merged
// patch count.
func (s *server) loadQwenVisionTower(dir string, int8Tower bool, backend string, require bool) error {
	enc, err := vision.LoadQwenVisionEncoder(dir, int8Tower)
	if err != nil {
		return fmt.Errorf("load qwen2.5-vl vision encoder (%s): %w", dir, err)
	}
	// Metal's tower attaches through attachResidentTower; CUDA's (aikit's gpu/qwencuda) and WebGPU's through
	// qwenTowerPlacement, which names every CPU fallback; every other backend runs the CPU tower.
	where := "CPU"
	if backend == "metal" {
		ok, err := attachResidentTower("Qwen2.5-VL", enc, backend, require, os.Stderr)
		if err != nil {
			return err
		}
		if ok {
			where = "Metal"
		}
	} else if where, err = qwenTowerPlacement(backend, int8Tower, require, enc.EnableResident, os.Stderr); err != nil {
		return err
	}
	if !strings.HasPrefix(where, "CPU") {
		releaseTowerBlocks("Qwen2.5-VL", enc)
	}
	pp, err := multimodal.LoadQwenPreprocessConfig(dir)
	if err != nil {
		return fmt.Errorf("qwen2.5-vl preprocessor config (%s): %w", dir, err)
	}
	for _, lm := range s.models {
		q := &qwen25Vision{pp: pp, merge: enc.Cfg.SpatialMergeSize, imgTok: -1, enc: enc, require: require}
		lm.vision = q
		if id, ok := lm.tk.TokenID(multimodal.QwenImagePad); ok {
			q.imgTok = id
		}
		if q.imgTok < 0 {
			return fmt.Errorf("vision: tokenizer has no %q token (needed to place image embeddings)", multimodal.QwenImagePad)
		}
		fmt.Fprintf(os.Stderr, "loaded Qwen2.5-VL vision tower for %q (merge %d, image-pad id %d, tower on %s) from %s\n", lm.name, q.merge, q.imgTok, where, dir)
	}
	return nil
}

// loadGemma4VisionTower attaches the Gemma 4 vision tower (aikit) to the single loaded model. There is no separate
// projector: Gemma4Encoder.Forward bakes in the embed_vision projection. decoder.GenerateGemma4VL implements two
// forwards, chosen by the checkpoint's use_bidirectional_attention: unset (the E2B/E4B-class sequential/causal path)
// and "vision" (the 26B-A4B/31B-class batched path, decoder/forward_gemma4_batched.go). Any other value is refused at
// load rather than served with the wrong mask. Placement is chooseGemma4Tower's; aikit's
// Gemma4Encoder has no EnableResident method.
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
	tower, where, err := chooseGemma4Tower(enc, int8Tower, backend, require)
	if err != nil {
		return err
	}
	swap := gemma4TowerSwap(s.models, enc, tower, backend, dir)
	if swap != nil {
		tower = nil
		where += ", loaded for each encode and released after it"
	} else if tower != nil {
		releaseTowerBlocks("Gemma 4", enc)
	}
	for _, lm := range s.models {
		g := &gemma4Vision{enc: enc, tower: tower, swap: swap, maxSoft: pp.MaxSoftTokens, imgTok: -1}
		lm.vision = g
		if id, ok := lm.tk.TokenID(multimodal.Gemma4ImageSoftToken); ok {
			g.imgTok = id
		}
		if g.imgTok < 0 {
			return fmt.Errorf("vision: tokenizer has no %q token (needed to place image embeddings)", multimodal.Gemma4ImageSoftToken)
		}
		fmt.Fprintf(os.Stderr, "loaded Gemma 4 vision tower for %q (max %d soft tokens/image, soft-token id %d, tower on %s) from %s\n", lm.name, g.maxSoft, g.imgTok, where, dir)
		// A checkpoint with an audio_config also takes audio clips; its tower loads on the first one.
		if hasAudioConfig(dir) {
			id, ok := lm.tk.TokenID(multimodal.Gemma4AudioSoftToken)
			if !ok {
				return fmt.Errorf("audio: %s has an audio tower but its tokenizer has no %q token", dir, multimodal.Gemma4AudioSoftToken)
			}
			at := &gemma4Audio{dir: dir, tok: id}
			lm.audio = at
			where := "CPU"
			if backend == "metal" && slices.Contains(embeddinggemma2.AudioAccelerators(), "metal") { // the conformer blocks run on Metal
				at.device, where = "metal", "Metal (the conformer blocks; subsample and tail on the CPU)"
			}
			fmt.Fprintf(os.Stderr, "Gemma 4 audio input on for %q (audio-token id %d; the tower loads on the first clip, %s) from %s\n", lm.name, id, where, dir)
		}
	}
	return nil
}

// loadQwenASR turns on audio input for a Qwen3-ASR model: the audio encoder lives beside the decoder in the same
// safetensors, so there is no tower to attach, only the placeholder id to find and the directory to load the encoder from
// on the first clip.
func (s *server) loadQwenASR(dir string) error {
	for _, lm := range s.models {
		id, ok := lm.tk.TokenID("<|audio_pad|>")
		if !ok {
			return fmt.Errorf("audio: %s is a Qwen3-ASR checkpoint but its tokenizer has no <|audio_pad|> token", dir)
		}
		lm.audio = &qwenASRAudio{dir: dir}
		fmt.Fprintf(os.Stderr, "Qwen3-ASR audio input on for %q (audio-pad id %d; the encoder loads on the first clip, CPU float32) from %s\n", lm.name, id, dir)
	}
	return nil
}

// loadVoxtral turns on audio input for a Voxtral Mini model, as loadQwenASR does for Qwen3-ASR: the encoder lives beside the decoder in the same safetensors and loads on the first clip, so
// there is only the [AUDIO] placeholder to check for and the directory to remember.
func (s *server) loadVoxtral(dir, backend string) error {
	for _, lm := range s.models {
		if _, ok := lm.tk.TokenID("[AUDIO]"); !ok {
			return fmt.Errorf("audio: %s is a Voxtral checkpoint but its tokenizer has no [AUDIO] token (a Tekken tokenizer is needed: tekken.json beside the weights)", dir)
		}
		lm.audio = &voxtralAudio{dir: dir, backend: backend}
		fmt.Fprintf(os.Stderr, "Voxtral audio input on for %q (the encoder loads on the first clip, float32, on %s when this binary has a device encoder for it, else the CPU; a clip over %d s is refused) from %s\n", lm.name, backend, voxtralMaxSeconds, dir)
	}
	return nil
}

// hasAudioConfig reports whether dir/config.json carries an audio_config (a Gemma 4 checkpoint with an audio tower).
func hasAudioConfig(dir string) bool {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return false
	}
	var c struct {
		Audio json.RawMessage `json:"audio_config"`
	}
	return json.Unmarshal(raw, &c) == nil && len(c.Audio) > 0 && string(c.Audio) != "null"
}

// gemma4TowerSwap decides whether the device tower holds its memory only while it encodes: a CUDA tower serving one model
// whose expert cache can grow into that memory between images. Then it closes the tower, grows the cache, and returns the
// towerSwap that encodes from here on; otherwise it returns nil and the tower stays resident. The reserve is towerReserve's
// figure, which the model's load already held back, so for an encode the cache shrinks to the size it was built at.
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

// chooseGemma4Tower puts Gemma 4's vision tower on the GPU when the backend is Metal, CUDA or WebGPU and this binary
// registers a tower for it, and says where it runs. The device tower is float32, so an int8 tower (-vision-quant int8)
// stays on the CPU; a tower that fails to start falls back to the CPU with the reason, unless -require-backend asks for a
// refusal. WebGPU's device tower is the default like the others' (its speed gate:
// docs/completed/task-multimodal-support-2026-10.md, Gemma 4's vision tower on WebGPU); -vision-device cpu arrives here
// as backend "cpu". Other backends run it on the CPU.
func chooseGemma4Tower(enc *vision.Gemma4Encoder, int8Tower bool, backend string, require bool) (multimodal.Gemma4TowerAccelerator, string, error) {
	name, ok := deviceTowerName(backend)
	if backend == "webgpu" {
		name, ok = "WebGPU", true
	}
	if !ok {
		return nil, "CPU", nil
	}
	if int8Tower {
		if require {
			return nil, "", fmt.Errorf("-require-backend: the Gemma 4 %s tower is float32; -vision-quant int8 keeps it on the CPU", name)
		}
		return nil, "CPU (-vision-quant int8; the " + name + " tower is float32)", nil
	}
	if !slices.Contains(multimodal.Gemma4Towers(), backend) {
		if require {
			return nil, "", fmt.Errorf("-require-backend: this binary has no %s Gemma 4 tower", name)
		}
		return nil, "CPU (no " + name + " tower in this binary)", nil
	}
	t, err := multimodal.NewGemma4Tower(backend, enc)
	if err != nil {
		if require {
			return nil, "", fmt.Errorf("-require-backend: the Gemma 4 tower could not start on %s: %w", name, err)
		}
		return nil, "CPU (" + name + " declined: " + err.Error() + ")", nil
	}
	return t, name, nil
}

// visionPathError refuses a -vision path that is a file, other than a GGUF mmproj (which loadVisionTower routes to
// loadQwen35MMProj). A path that does not exist, or a directory, is the loaders' to judge, with their own messages.
func visionPathError(dir string) error {
	fi, err := os.Stat(dir)
	if err != nil || fi.IsDir() {
		return nil
	}
	if strings.HasSuffix(strings.ToLower(dir), ".gguf") {
		return nil // a GGUF mmproj: loadVisionTower routes it (Qwen3.5+ only), with its own refusals
	}
	return fmt.Errorf("-vision %s is a file; -vision takes a directory with a vision tower (config.json and safetensors)", dir)
}

// qwenTowerPlacement decides where Qwen2.5-VL's vision tower runs and attaches it: the device tower under --backend
// cuda or webgpu when the binary registers one (cuda/vision_towers.go imports aikit's gpu/qwencuda) and the tower is
// float32; the CPU everywhere else, with the reason named.
// -require-backend turns each CPU fallback on a device backend into a refusal. -vision-device cpu arrives here as backend
// "cpu". Metal's tower is attached by loadQwenVisionTower.
func qwenTowerPlacement(backend string, int8Tower, require bool, attach func() error, warn io.Writer) (string, error) {
	name := "CUDA"
	switch backend {
	case "cuda":
	case "webgpu":
		name = "WebGPU"
	default:
		return "CPU", nil
	}
	if int8Tower {
		if require {
			return "", fmt.Errorf("-require-backend: the Qwen2.5-VL %s tower is float32; -vision-quant int8 keeps it on the CPU", name)
		}
		return "CPU (-vision-quant int8; the " + name + " tower is float32)", nil
	}
	if err := attach(); err != nil {
		if require {
			return "", fmt.Errorf("-require-backend: the Qwen2.5-VL tower could not start on %s: %w", backend, err)
		}
		fmt.Fprintf(warn, "vision: the Qwen2.5-VL tower runs on the CPU: %s declined it: %v\n", backend, err)
		return "CPU (" + backend + " declined)", nil
	}
	return name, nil
}
