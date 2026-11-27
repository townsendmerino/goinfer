package towerfit

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"slices"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/multimodal"
)

// Qwen3MaxImageTokens is serve's per-image cap on merged image tokens for Qwen3.5+ towers.
const Qwen3MaxImageTokens = 1024

// Stderr notes printed when a WebGPU-capable vision tower stays on the CPU.
const (
	DiscreteKeptNote = "note: under --backend webgpu on a discrete GPU the vision tower runs on the CPU: its device memory cannot be priced against the card, so --vision-device gpu puts it on the GPU"
	MMProjKeptNote   = "note: a Qwen3.5+ .gguf mmproj tower runs on the CPU under --backend webgpu (it has no config.json to price); --vision-device gpu puts it on WebGPU"
)

// Config holds the serving flags that govern vision and audio tower placement
// and device-memory reservation.
type Config struct {
	Backend         string
	VisionPath      string
	VisionDevice    string
	VisionQuant     string
	VisionMaxPixels int
}

// TowerBackend returns the backend a device tower targets: the model's backend
// unless VisionDevice == "cpu" keeps the tower on the CPU.
func (c Config) TowerBackend() string {
	if c.VisionDevice == "cpu" {
		return "cpu"
	}
	return c.Backend
}

// VisionRoute is the resolved backend for a vision tower and any stderr note
// explaining why a WebGPU-capable tower was kept on the CPU.
type VisionRoute struct {
	Backend string
	Note    string
}

// RouteVisionTower decides the backend and stderr note for one vision tower.
func (c Config) RouteVisionTower(family string, mmproj bool) VisionRoute {
	be := c.TowerBackend()
	if be != "webgpu" {
		return VisionRoute{Backend: be}
	}
	quantFamily := family
	if mmproj {
		quantFamily = "qwen3_5"
	}
	if c.VisionQuant == "int8" && quantFamily != "gemma3" {
		return VisionRoute{Backend: be}
	}
	if c.VisionDevice == "gpu" {
		return VisionRoute{Backend: be}
	}
	if mmproj {
		r := VisionRoute{Backend: "cpu"}
		if WebGPUTowerRegistered(quantFamily) {
			r.Note = MMProjKeptNote
		}
		return r
	}
	if WebGPUTowerUnified() || (family == "gemma3" && c.VisionQuant != "") {
		return VisionRoute{Backend: be}
	}
	r := VisionRoute{Backend: "cpu"}
	if WebGPUTowerRegistered(family) {
		r.Note = DiscreteKeptNote
	}
	return r
}

// TowerInt8 reports whether a vision tower loads int8 weights for the given
// family, explicit quant flag, and backend.
func TowerInt8(modelType, visionQuant, backend string) bool {
	switch visionQuant {
	case "int8":
		return true
	case "f32":
		return false
	}
	switch modelType {
	case "qwen2_5_vl", "qwen3_5", "qwen3_5_moe", "qwen3_vl", "qwen3_vl_moe", "gemma4", "glm_ocr", "mistral3", "lfm2_vl":
		return false
	}
	return (backend == "webgpu" || backend == "cuda") && modelType != "gemma3"
}

// VisionModelType reads dir/config.json's model_type ("" if absent or unreadable).
func VisionModelType(dir string) string {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return ""
	}
	var c struct {
		ModelType string `json:"model_type"`
	}
	_ = json.Unmarshal(raw, &c)
	return c.ModelType
}

// CUDATowerRegistered reports whether this binary registers a CUDA device tower for mt.
var CUDATowerRegistered = func(mt string) bool {
	switch mt {
	case "gemma3":
		return true
	case "gemma4":
		return slices.Contains(multimodal.Gemma4Towers(), "cuda")
	case "qwen3_5", "qwen3_5_moe", "qwen3_vl", "qwen3_vl_moe":
		return slices.Contains(multimodal.Qwen3Towers(), "cuda")
	case "glm_ocr":
		return slices.Contains(multimodal.GlmOcrTowers(), "cuda")
	case "qwen2_5_vl":
		return true
	}
	return false
}

// WebGPUTowerRegistered reports whether this binary registers a WebGPU device tower for mt.
var WebGPUTowerRegistered = func(mt string) bool {
	switch mt {
	case "gemma3":
		return slices.Contains(multimodal.ResidentTowers(multimodal.TowerSigLIP), "webgpu")
	case "qwen2_5_vl":
		return slices.Contains(multimodal.ResidentTowers(multimodal.TowerQwen25VL), "webgpu")
	case "gemma4":
		return slices.Contains(multimodal.Gemma4Towers(), "webgpu")
	case "qwen3_5", "qwen3_5_moe", "qwen3_vl", "qwen3_vl_moe":
		return slices.Contains(multimodal.Qwen3Towers(), "webgpu")
	case "glm_ocr":
		return slices.Contains(multimodal.GlmOcrTowers(), "webgpu")
	}
	return false
}

// MetalTowerRegistered reports whether this binary registers a Metal device tower for mt.
var MetalTowerRegistered = func(mt string) bool {
	if !slices.Contains(decoder.RegisteredBackends(), "metal") {
		return false
	}
	switch mt {
	case "gemma3", "qwen2_5_vl", "qwen3_5", "qwen3_5_moe", "qwen3_vl", "qwen3_vl_moe", "glm_ocr", "gemma4":
		return true
	}
	return false
}

// WebGPUTowerUnified reports whether WebGPU device buffers share host RAM on this OS.
var WebGPUTowerUnified = func() bool { return runtime.GOOS == "darwin" }

// WebGPUTowerAvailable returns the host RAM currently available in bytes.
var WebGPUTowerAvailable = decoder.HostRAMAvailableBytes
