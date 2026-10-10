package serveapp

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/multimodal"
)

// The VRAM a CUDA vision tower will claim, priced before the decoder builds (docs/tasks/task-multimodal-support-2026-10.md). A
// tower loads lazily on the first image, after the resident decoder; without a reserve, serve's default KV plan takes every
// free byte, the tower's upload fails (`CUDA_ERROR_OUT_OF_MEMORY`) and it runs on the CPU, so a served run on the default plan
// reads as a pass for the wrong reason. The resident build already reserves for a drafter through Options.ExtraResidentBytes;
// a tower rides the same field. The figure is the tower's float32 weights plus the scratch its largest image needs (the tower
// releases its scratch after each call, so this is a peak, not a standing cost), from the checkpoint's own vision_config: an
// estimate, deliberately not exact, and a ceiling the elastic terms (the KV context, the expert cache) give way to.

type towerDims struct {
	hidden, inter, layers, patch, temporal, inChan, merge int
	posTable                                              int // Gemma 4's two position tables, rows each
	outputLength                                          int // Gemma 4's default soft-token count
	pool                                                  int // Gemma 4's pooling kernel
	imageSize                                             int // Gemma 3's SigLIP image side in pixels
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
		posTable: num("position_embedding_size"), outputLength: num("default_output_length"), pool: max(num("pooling_kernel_size"), 1), imageSize: num("image_size")}
	return d, d.hidden > 0 && d.inter > 0 && d.layers > 0 && d.patch > 0
}

// towerVRAMEstimate is the float32 weights plus the peak scratch of one tower, in bytes, for the given model type and (GLM-OCR's) pixel ceiling (0 = the model's own).
func towerVRAMEstimate(mt string, d towerDims, maxPixels int) int64 {
	if mt == "qwen2_5_vl" {
		return qwen25VLTowerEstimate(d)
	}
	params, scratch := towerParts(mt, d, maxPixels)
	return params*4 + scratch
}

// towerParts is a tower's parameter count and its peak scratch in bytes (towerVRAMEstimate's two terms), 0, 0 for a family it does not know.
func towerParts(mt string, d towerDims, maxPixels int) (params, scratch int64) {
	patchIn := d.inChan * d.temporal * d.patch * d.patch
	var mlpMats, np int
	switch mt {
	case "gemma3":
		// SigLIP: a fixed (image_size / patch)^2 patches, a position table of the same rows, two MLP matrices, biased separate q/k/v/o (the 4*h*h below)
		patchIn = 3 * d.patch * d.patch
		mlpMats = 2
		side := d.imageSize / max(d.patch, 1)
		np = side * side
	case "gemma4":
		patchIn = 3 * d.patch * d.patch
		mlpMats = 3
		np = max(d.outputLength, 1) * d.pool * d.pool
	case "qwen3_5", "qwen3_5_moe", "qwen3_vl": // Qwen3-VL's is the same tower; its DeepStack mergers run on the host
		mlpMats = 2
		np = qwen3MaxImageTokens * d.merge * d.merge // serve's cap on the Qwen3.5 tower
	case "glm_ocr":
		mlpMats = 3
		px := maxPixels
		if px <= 0 {
			px = 4816896 // the model's own ceiling, serve's default (-vision-max-pixels 0)
		}
		// Reserve for about 1.5 MP and let a larger image fall back to the CPU tower (deviceFallback) instead: reserving the
		// ceiling's scratch (~2.5 GB) would starve the CUDA context of positions.
		np = min(px, 1_500_000) / (d.patch * d.patch)
	case "qwen2_5_vl":
		// The CUDA reserve is qwen25VLTowerEstimate's calibrated figure; this is the arithmetic form, for Metal: the gated MLP and a ceiling of 8192 patches.
		mlpMats = 3
		np = 8192
	default:
		return 0, 0
	}
	h, i := int64(d.hidden), int64(d.inter)
	params = int64(d.layers)*(4*h*h+int64(mlpMats)*h*i) + int64(patchIn)*h
	if mt == "gemma4" {
		params += 2 * int64(d.posTable) * h
	}
	if mt == "gemma3" {
		params += int64(np) * h // the fixed position table
	}
	// scratch: about twelve hidden-wide buffers, two MLP-wide ones, the widest projection's input copy and the pixel rows, per patch
	scratch = int64(np) * 4 * (12*h + 2*i + max(h, i) + int64(patchIn))
	return params, scratch
}

// metalTowerEstimate is what a Metal device tower holds of the Mac's memory: the projections at the form Metal uploads, the
// rest in float32, plus the scratch. The grid towers (SigLIP, Qwen2.5-VL, Qwen3.5+, GLM-OCR) upload f16 projections, or int8
// in groups of 32 for Gemma 3's int8 tower (tower_gemm_w8: a byte plus an f32 scale per 32); Gemma 4's Metal tower is float32.
// Calibrated against Gemma 3's measured footprint (metal TestS18TowerHostMemory); it runs about 10% over, the safe direction
// (the scratch term is towerParts', which over-counts Metal's).
func metalTowerEstimate(mt string, d towerDims, maxPixels int, int8 bool) int64 {
	params, scratch := towerParts(mt, d, maxPixels)
	if params == 0 {
		return 0
	}
	h := int64(d.hidden)
	// the projections: everything but the patch embed and the position tables, which stay float32
	patchIn := int64(d.inChan * d.temporal * d.patch * d.patch)
	if mt == "gemma3" || mt == "gemma4" {
		patchIn = int64(3 * d.patch * d.patch)
	}
	head := patchIn * h
	switch mt {
	case "gemma3":
		side := int64(d.imageSize) / int64(max(d.patch, 1))
		head += side * side * h
	case "gemma4":
		head += 2 * int64(d.posTable) * h
	}
	proj := params - head
	switch {
	case mt == "gemma4":
		return params*4 + scratch
	case int8:
		return proj + proj/8 + head*4 + scratch // a byte each, plus an f32 scale per 32
	}
	return proj*2 + head*4 + scratch
}

// towerReserve is what to add to Options.ExtraResidentBytes for this model's CUDA vision tower: zero unless the tower will run on CUDA (the backend is cuda, -vision-device is
// not cpu, the tower is float32 and this binary registers a CUDA tower for the family).
func towerReserve(cfg config, modelPath string) int64 {
	if cfg.towerBackend() != "cuda" && cfg.towerBackend() != "metal" {
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
	if cfg.towerBackend() == "metal" {
		// Metal's device memory is the Mac's RAM, and its guard prices this beside the decoder. Gemma 3's int8 tower is
		// Metal's own (tower_gemm_w8); every other family's int8 tower is the CPU's.
		i8 := towerInt8(mt, cfg.visionQuant, "metal")
		if !metalTowerRegistered(mt) || i8 && mt != "gemma3" {
			return 0
		}
		d, ok := readTowerDims(dir)
		if !ok {
			return 0
		}
		return metalTowerEstimate(mt, d, cfg.visionMaxPixels, i8)
	}
	if towerInt8(mt, cfg.visionQuant, "cuda") {
		// Gemma 3's SigLIP is the one family with an int8 DEVICE tower on CUDA (chosen when the card is too tight for float32,
		// or by `-vision-quant int8`); every other int8 tower is the CPU's and holds no VRAM. Pricing it at zero would let the
		// default plan take every free byte and the tower then load beside a decoder with nothing left.
		if mt == "gemma3" {
			if d, ok := readTowerDims(dir); ok {
				return towerInt8VRAMEstimate(d)
			}
		}
		return 0
	}
	registered := cudaTowerRegistered(mt)
	if !registered {
		return 0
	}
	d, ok := readTowerDims(dir)
	if !ok {
		return 0
	}
	return towerVRAMEstimate(mt, d, cfg.visionMaxPixels)
}

// towerInt8VRAMEstimate is the Gemma 3 W8A8 SigLIP tower's footprint on the device: the block weights at one byte, the patch
// embed and position table in float32, and the fixed scratch (about nine hidden-wide float32 buffers per patch). Calibrated
// against a measurement, not derived (cuda TestSiglipCUDA_int8VRAM).
func towerInt8VRAMEstimate(d towerDims) int64 {
	h, i := int64(d.hidden), int64(d.inter)
	side := int64(d.imageSize) / int64(max(d.patch, 1))
	np := side * side
	weights := int64(d.layers) * (4*h*h + 2*h*i)
	f32 := (int64(3*d.patch*d.patch)*h + np*h) * 4
	return weights + f32 + np*4*9*h
}

// cudaTowerRegistered says whether this binary registers a CUDA device tower for the model type: the registries are filled by
// a cuda build's blank import, so a plain build (and a unit test) sees none. A variable so a table test can stand in for a
// cuda build.
var cudaTowerRegistered = func(mt string) bool {
	switch mt {
	case "gemma3":
		return true // cuda/vision_register.go's factory builds the SigLIP tower (float32, or the int8 one) for either encoder
	case "gemma4":
		return slices.Contains(multimodal.Gemma4Towers(), "cuda")
	case "qwen3_5", "qwen3_5_moe", "qwen3_vl":
		return slices.Contains(multimodal.Qwen3Towers(), "cuda")
	case "glm_ocr":
		return slices.Contains(multimodal.GlmOcrTowers(), "cuda")
	case "qwen2_5_vl":
		return true // aikit's qwencuda registers through the vision package, not multimodal's registry; a cuda binary imports it
	}
	return false
}

// metalTowerRegistered says whether this binary has a Metal device tower for the model type: a metal binary registers the
// metal backend and its towers in the same package init, so a plain build (and a unit test) sees none. A variable so a table
// test can stand in for a metal build.
var metalTowerRegistered = func(mt string) bool {
	if !slices.Contains(decoder.RegisteredBackends(), "metal") {
		return false
	}
	switch mt {
	case "gemma3", "qwen2_5_vl", "qwen3_5", "qwen3_5_moe", "qwen3_vl", "glm_ocr", "gemma4":
		return true
	}
	return false
}

// qwen25VLTowerEstimate prices goinfer's float32 Qwen2.5-VL tower on CUDA (cuda/qwen25_vision.go), calibrated to a
// measurement, not derived: the weights take more than their arithmetic size (about 570 separate buffers, each rounded up to
// the allocator's 2 MiB quantum), and the first image's scratch is np * (7 hidden + 2 padded intermediate + patch dim + 2 head
// dim) * 4 bytes. The ceiling stays 8192 patches (about 1.6 MP); a larger image falls back to the CPU tower by name
// (deviceFallback). There is deliberately no extra slack for the resident build's scratch: the text build's scratch after the
// plan's probe is 0, and the allocator rounding of the TEXT model's weights is priced by the CUDA plan itself
// (cuda.packedAllocSlack).
func qwen25VLTowerEstimate(d towerDims) int64 {
	const (
		np         = 8192
		allocExtra = 337 << 20 // weights' allocator overhead
	)
	h, i := int64(d.hidden), (int64(d.inter)+63)/64*64 // the tower pads the intermediate width to a multiple of 64
	patchIn := int64(d.inChan * d.temporal * d.patch * d.patch)
	hd := h / 16 // Qwen2.5-VL's 16 heads
	params := int64(d.layers)*(4*h*h+3*h*i) + patchIn*h
	scratch := int64(np) * (7*h + 2*i + patchIn + 2*hd) * 4
	return params*4 + allocExtra + scratch
}
