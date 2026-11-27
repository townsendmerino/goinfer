package serveapp

import (
	"fmt"
	"io"
	"os"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/towers"
	"github.com/townsendmerino/goinfer/internal/runtime/mem"
	"github.com/townsendmerino/goinfer/internal/runtime/prequant"
)

var (
	cudaFreeBytes                 = func() (int64, bool) { return decoder.FreeBytesFor("cuda") }
	sidecarFresh                  = prequant.SidecarPathIfFresh
	metalFreeBytes                = func() (int64, bool) { return decoder.FreeBytesFor("metal") }
	gemma3ResidentWeightsEstimate = towers.Gemma3ResidentWeightsEstimate
)

func resolveGemma3VisionQuant(cfg config, note io.Writer) config {
	if cfg.visionQuant != "" || (cfg.towerBackend() != "cuda" && cfg.towerBackend() != "metal") || len(cfg.models) != 1 {
		return cfg
	}
	if cfg.towerBackend() == "metal" {
		return resolveGemma3VisionQuantMetal(cfg, note)
	}
	path := cfg.models[0].path
	dir := cfg.visionPath
	if dir == "" {
		dir = path
	}
	if fi, err := os.Stat(dir); err != nil || !fi.IsDir() || visionModelType(dir) != "gemma3" {
		return cfg
	}
	d, ok := readTowerDims(dir)
	if !ok {
		return cfg
	}
	w, ok := towers.Gemma3ResidentWeightsEstimate(path)
	if !ok {
		return cfg
	}
	free, ok := cudaFreeBytes()
	if !ok {
		return cfg
	}
	const margin = 384 << 20
	need := w + w*55/100 + towerVRAMEstimate("gemma3", d, cfg.visionMaxPixels) + margin
	if free >= need {
		return cfg
	}
	cfg.visionQuant = "int8"
	fmt.Fprintf(note, "note: the float32 Gemma 3 vision tower is the CUDA default, but this card has %s free and the decoder, one KV slot and that tower need about %s: using the int8 device tower "+
		"(-vision-quant f32 forces float32)\n", mem.GiB(free), mem.GiB(need))
	return cfg
}

func resolveGemma3VisionQuantMetal(cfg config, note io.Writer) config {
	path := cfg.models[0].path
	dir := cfg.visionPath
	if dir == "" {
		dir = path
	}
	if fi, err := os.Stat(dir); err != nil || !fi.IsDir() || visionModelType(dir) != "gemma3" {
		return cfg
	}
	d, ok := readTowerDims(dir)
	if !ok {
		return cfg
	}
	w, ok := towers.Gemma3ResidentWeightsEstimate(path)
	if !ok {
		return cfg
	}
	free, ok := metalFreeBytes()
	if !ok {
		return cfg
	}
	w = w * 34 / 27
	dec := w
	if fi, err := os.Stat(path); err == nil && fi.IsDir() {
		o := cfg.load.Options()
		if _, fresh := sidecarFresh(path, o.Quant, "metal", decoder.EmbedInt4For(path, o)); !fresh {
			dec *= 2
		}
	}
	const margin = 384 << 20
	need := dec + w*55/100/4 + metalTowerEstimate("gemma3", d, cfg.visionMaxPixels, false) + margin
	if free >= need {
		return cfg
	}
	cfg.visionQuant = "int8"
	fmt.Fprintf(note, "note: this Mac's resident budget is %s and the decoder, one KV slot and the f16 Gemma 3 vision tower need about %s: using the int8 Metal tower "+
		"(weights in groups of 32, activations f32; -vision-quant f32 forces f16)\n", mem.GiB(free), mem.GiB(need))
	return cfg
}

func gemma3FloatDefault(cfg config, int8Tower bool) bool {
	return cfg.visionQuant == "" && (cfg.towerBackend() == "cuda" || cfg.towerBackend() == "metal") && !int8Tower
}
