package towerfit

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// TowerDims holds the vision_config dimensions used to price a device tower.
type TowerDims struct {
	Hidden, Inter, Layers, Patch, Temporal, InChan, Merge int
	PosTable                                              int
	OutputLength                                          int
	Pool                                                  int
	ImageSize                                             int
}

// ReadTowerDims parses dir/config.json's vision_config into TowerDims.
func ReadTowerDims(dir string) (TowerDims, bool) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return TowerDims{}, false
	}
	var c struct {
		Vision map[string]any `json:"vision_config"`
	}
	if json.Unmarshal(raw, &c) != nil || c.Vision == nil {
		return TowerDims{}, false
	}
	num := func(keys ...string) int {
		for _, k := range keys {
			if v, ok := c.Vision[k].(float64); ok {
				return int(v)
			}
		}
		return 0
	}
	d := TowerDims{
		Hidden:       num("hidden_size", "embed_dim"),
		Inter:        num("intermediate_size"),
		Layers:       num("num_hidden_layers", "depth"),
		Patch:        num("patch_size"),
		Temporal:     max(num("temporal_patch_size"), 1),
		InChan:       max(num("in_channels", "num_channels"), 3),
		Merge:        max(num("spatial_merge_size"), 1),
		PosTable:     num("position_embedding_size"),
		OutputLength: num("default_output_length"),
		Pool:         max(num("pooling_kernel_size"), 1),
		ImageSize:    num("image_size"),
	}
	return d, d.Hidden > 0 && d.Inter > 0 && d.Layers > 0 && d.Patch > 0
}

// TowerVRAMEstimate returns the float32 weight plus peak scratch footprint in bytes
// for a CUDA vision tower.
func TowerVRAMEstimate(mt string, d TowerDims, maxPixels int) int64 {
	if mt == "qwen2_5_vl" {
		return Qwen25VLTowerEstimate(d)
	}
	params, scratch := TowerParts(mt, d, maxPixels)
	return params*4 + scratch
}

// TowerParts returns a tower's parameter count and peak scratch bytes, or (0, 0)
// for an unrecognized family.
func TowerParts(mt string, d TowerDims, maxPixels int) (params, scratch int64) {
	patchIn := d.InChan * d.Temporal * d.Patch * d.Patch
	var mlpMats, np int
	switch mt {
	case "gemma3":
		patchIn = 3 * d.Patch * d.Patch
		mlpMats = 2
		side := d.ImageSize / max(d.Patch, 1)
		np = side * side
	case "gemma4":
		patchIn = 3 * d.Patch * d.Patch
		mlpMats = 3
		np = max(d.OutputLength, 1) * d.Pool * d.Pool
	case "qwen3_5", "qwen3_5_moe", "qwen3_vl", "qwen3_vl_moe":
		mlpMats = 2
		np = Qwen3MaxImageTokens * d.Merge * d.Merge
	case "glm_ocr":
		mlpMats = 3
		px := maxPixels
		if px <= 0 {
			px = 4816896
		}
		// Cap the standing reserve so large-ceiling models leave room for KV context;
		// oversized images fall back through DeviceFallback.
		np = min(px, 1_500_000) / (d.Patch * d.Patch)
	case "qwen2_5_vl":
		mlpMats = 3
		np = 8192
	default:
		return 0, 0
	}
	h, i := int64(d.Hidden), int64(d.Inter)
	params = int64(d.Layers)*(4*h*h+int64(mlpMats)*h*i) + int64(patchIn)*h
	if mt == "gemma4" {
		params += 2 * int64(d.PosTable) * h
	}
	if mt == "gemma3" {
		params += int64(np) * h
	}
	scratch = int64(np) * 4 * (12*h + 2*i + max(h, i) + int64(patchIn))
	return params, scratch
}

// MetalTowerEstimate prices a Metal device tower's weights (f16 or int8 projections
// for grid towers, float32 for Gemma 4) plus peak scratch in bytes.
func MetalTowerEstimate(mt string, d TowerDims, maxPixels int, int8 bool) int64 {
	params, scratch := TowerParts(mt, d, maxPixels)
	if params == 0 {
		return 0
	}
	h := int64(d.Hidden)
	patchIn := int64(d.InChan * d.Temporal * d.Patch * d.Patch)
	if mt == "gemma3" || mt == "gemma4" {
		patchIn = int64(3 * d.Patch * d.Patch)
	}
	head := patchIn * h
	switch mt {
	case "gemma3":
		side := int64(d.ImageSize) / int64(max(d.Patch, 1))
		head += side * side * h
	case "gemma4":
		head += 2 * int64(d.PosTable) * h
	}
	proj := params - head
	switch {
	case mt == "gemma4":
		return params*4 + scratch
	case int8:
		return proj + proj/8 + head*4 + scratch
	}
	return proj*2 + head*4 + scratch
}

// Reserve computes the ExtraResidentBytes reservation for cfg's device vision tower,
// or 0 when no device vision tower will hold device memory.
func Reserve(cfg Config, modelPath string) int64 {
	be := cfg.TowerBackend()
	if be != "cuda" && be != "metal" && be != "webgpu" {
		return 0
	}
	dir := cfg.VisionPath
	if dir == "" {
		dir = modelPath
	}
	if fi, err := os.Stat(dir); err != nil || !fi.IsDir() {
		return 0
	}
	mt := VisionModelType(dir)
	if cfg.RouteVisionTower(mt, false).Backend == "cpu" {
		return 0
	}
	if be == "metal" {
		i8 := TowerInt8(mt, cfg.VisionQuant, "metal")
		if !MetalTowerRegistered(mt) || i8 && mt != "gemma3" {
			return 0
		}
		d, ok := ReadTowerDims(dir)
		if !ok {
			return 0
		}
		return MetalTowerEstimate(mt, d, cfg.VisionMaxPixels, i8)
	}
	if be == "webgpu" {
		if !WebGPUTowerRegistered(mt) {
			return 0
		}
		d, ok := ReadTowerDims(dir)
		if !ok {
			return 0
		}
		if TowerInt8(mt, cfg.VisionQuant, "webgpu") {
			if mt == "gemma3" {
				return TowerInt8VRAMEstimate(d)
			}
			return 0
		}
		params, scratch := TowerParts(mt, d, cfg.VisionMaxPixels)
		return params*4 + scratch
	}
	if TowerInt8(mt, cfg.VisionQuant, "cuda") {
		if mt == "gemma3" {
			if d, ok := ReadTowerDims(dir); ok {
				return TowerInt8VRAMEstimate(d)
			}
		}
		return 0
	}
	if !CUDATowerRegistered(mt) {
		return 0
	}
	d, ok := ReadTowerDims(dir)
	if !ok {
		return 0
	}
	return TowerVRAMEstimate(mt, d, cfg.VisionMaxPixels)
}

// TowerInt8VRAMEstimate returns the Gemma 3 W8A8 SigLIP device footprint in bytes.
func TowerInt8VRAMEstimate(d TowerDims) int64 {
	h, i := int64(d.Hidden), int64(d.Inter)
	side := int64(d.ImageSize) / int64(max(d.Patch, 1))
	np := side * side
	weights := int64(d.Layers) * (4*h*h + 2*h*i)
	f32 := (int64(3*d.Patch*d.Patch)*h + np*h) * 4
	return weights + f32 + np*4*9*h
}

// Qwen25VLTowerEstimate returns the float32 Qwen2.5-VL CUDA tower footprint in bytes,
// accounting for per-buffer allocator rounding and peak scratch.
func Qwen25VLTowerEstimate(d TowerDims) int64 {
	const (
		np         = 8192
		allocExtra = 337 << 20
	)
	h, i := int64(d.Hidden), (int64(d.Inter)+63)/64*64
	patchIn := int64(d.InChan * d.Temporal * d.Patch * d.Patch)
	hd := h / 16
	params := int64(d.Layers)*(4*h*h+3*h*i) + patchIn*h
	scratch := int64(np) * (7*h + 2*i + patchIn + 2*hd) * 4
	return params*4 + allocExtra + scratch
}
