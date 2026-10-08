package serveapp

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"

	"github.com/townsendmerino/goinfer/multimodal"
)

// The VRAM a CUDA vision tower will claim, priced before the decoder builds (S4 of docs/tasks/task-multimodal-support-2026-10.md, Gate 0 step 4). A tower loads lazily on
// the first image, after the resident decoder, and nothing used to reserve room for it: serve's default KV plan took every free byte, the tower's upload failed
// (`CUDA_ERROR_OUT_OF_MEMORY`) and it ran on the CPU, so a served run on the default plan read as a pass for the wrong reason (G-S4q's first served run). The resident build
// already reserves for a drafter through Options.ExtraResidentBytes; a tower rides the same field. The figure is the tower's float32 weights plus the scratch its largest
// image needs (the tower releases its scratch after each call, so this is a peak, not a standing cost), from the checkpoint's own vision_config: an estimate, deliberately
// not exact, and a ceiling the elastic terms (the KV context, the expert cache) give way to.

type towerDims struct {
	hidden, inter, layers, patch, temporal, inChan, merge int
	posTable                                              int // Gemma 4's two position tables, rows each
	outputLength                                          int // Gemma 4's default soft-token count
	pool                                                  int // Gemma 4's pooling kernel
}

func readTowerDims(dir string) (towerDims, bool) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return towerDims{}, false
	}
	var c struct {
		Vision map[string]any `json:"vision_config"`
	}
	if json.Unmarshal(raw, &c) != nil || c.Vision == nil {
		return towerDims{}, false
	}
	num := func(keys ...string) int {
		for _, k := range keys {
			if v, ok := c.Vision[k].(float64); ok {
				return int(v)
			}
		}
		return 0
	}
	d := towerDims{hidden: num("hidden_size", "embed_dim"), inter: num("intermediate_size"), layers: num("num_hidden_layers", "depth"),
		patch: num("patch_size"), temporal: max(num("temporal_patch_size"), 1), inChan: max(num("in_channels", "num_channels"), 3), merge: max(num("spatial_merge_size"), 1),
		posTable: num("position_embedding_size"), outputLength: num("default_output_length"), pool: max(num("pooling_kernel_size"), 1)}
	return d, d.hidden > 0 && d.inter > 0 && d.layers > 0 && d.patch > 0
}

// towerVRAMEstimate is the float32 weights plus the peak scratch of one tower, in bytes, for the given model type and (GLM-OCR's) pixel ceiling (0 = the model's own).
func towerVRAMEstimate(mt string, d towerDims, maxPixels int) int64 {
	patchIn := d.inChan * d.temporal * d.patch * d.patch
	var mlpMats, np int
	switch mt {
	case "gemma4":
		patchIn = 3 * d.patch * d.patch
		mlpMats = 3
		np = max(d.outputLength, 1) * d.pool * d.pool
	case "qwen3_5", "qwen3_5_moe":
		mlpMats = 2
		np = qwen3MaxImageTokens * d.merge * d.merge // serve's cap on the Qwen3.5 tower
	case "glm_ocr":
		mlpMats = 3
		px := maxPixels
		if px <= 0 {
			px = 4816896 // the model's own ceiling, serve's default (-vision-max-pixels 0)
		}
		// The ceiling's scratch alone is ~2.5 GB and, reserved, took GLM-OCR's CUDA context from 16384 to 4931 positions on the 8 GB card (measured 2026-10-07). Reserve for
		// about 1.5 MP and let a larger image fall back to the CPU tower (deviceFallback) instead.
		np = min(px, 1_500_000) / (d.patch * d.patch)
	case "qwen2_5_vl":
		mlpMats = 3
		np = 8192 // about 1.6 MP; aikit's tower sizes its scratch per call and its default ceiling (65536 patches) cannot be reserved, so a larger image fails by name (recoverDeviceTower)
	default:
		return 0
	}
	h, i := int64(d.hidden), int64(d.inter)
	params := int64(d.layers)*(4*h*h+int64(mlpMats)*h*i) + int64(patchIn)*h
	if mt == "gemma4" {
		params += 2 * int64(d.posTable) * h
	}
	// scratch: about twelve hidden-wide buffers, two MLP-wide ones, the widest projection's input copy and the pixel rows, per patch
	scratch := int64(np) * 4 * (12*h + 2*i + max(h, i) + int64(patchIn))
	return params*4 + scratch
}

// towerReserve is what to add to Options.ExtraResidentBytes for this model's CUDA vision tower: zero unless the tower will run on CUDA (the backend is cuda, -vision-device is
// not cpu, the tower is float32 and this binary registers a CUDA tower for the family).
func towerReserve(cfg config, modelPath string) int64 {
	if cfg.towerBackend() != "cuda" {
		return 0
	}
	dir := cfg.visionPath
	if dir == "" {
		dir = modelPath
	}
	if fi, err := os.Stat(dir); err != nil || !fi.IsDir() {
		return 0 // a GGUF mmproj or an unresolved reference: nothing priced
	}
	mt := visionModelType(dir)
	if towerInt8(mt, cfg.visionQuant, "cuda") {
		return 0
	}
	registered := false
	switch mt {
	case "gemma4":
		registered = slices.Contains(multimodal.Gemma4Towers(), "cuda")
	case "qwen3_5", "qwen3_5_moe":
		registered = slices.Contains(multimodal.Qwen3Towers(), "cuda")
	case "glm_ocr":
		registered = slices.Contains(multimodal.GlmOcrTowers(), "cuda")
	case "qwen2_5_vl":
		registered = true // aikit's qwencuda registers through the vision package, not multimodal's registry; a cuda binary imports it
	}
	if !registered {
		return 0
	}
	d, ok := readTowerDims(dir)
	if !ok {
		return 0
	}
	return towerVRAMEstimate(mt, d, cfg.visionMaxPixels)
}
