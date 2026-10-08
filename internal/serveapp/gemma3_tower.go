package serveapp

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/townsendmerino/aikit/vision"
	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/prequant"
)

// attachableTower is what attachGemma3Tower needs of a vision encoder: the device attach and the release of one that did not take.
type attachableTower interface {
	EnableResident() error
	Close()
}

// attachGemma3Tower loads Gemma 3's SigLIP encoder (load(true) is the int8 one) and attaches its device-resident form.
//
// Float32 is the CUDA default since 2026-10-08 (owner): it reproduces the CPU float32 reference's reply and serves a new image in 2.4 s against the int8 tower's 4.6 s, at the price of about 1.7 GiB of VRAM
// against 0.56 (docs/tasks/task-multimodal-support-2026-10.md, "Night 2026-10-08"). A card with less room might not hold it, and a failed attach would drop the request to the CPU tower at about 21 s, which is
// worse than the int8 device tower it replaced. So when fallback is set (the float32 tower was the DEFAULT, not asked for) a float32 attach that does not take is released and the int8 tower is attached in its
// place, with a line saying why. An explicit `-vision-quant f32` sets fallback false and keeps the old behaviour: attach, or warn, or fail under -require-backend.
//
// It returns the encoder that is attached (or the CPU one), whether it is int8, and whether its device form is attached.
func attachGemma3Tower[E attachableTower](load func(int8 bool) (E, error), startInt8, fallback bool, backend string, require bool, warn io.Writer) (enc E, int8, resident bool, err error) {
	const label = "Gemma 3 SigLIP"
	enc, err = load(startInt8)
	if err != nil {
		return enc, startInt8, false, err
	}
	if !fallback {
		resident, err = attachResidentTower(label, enc, backend, require, warn)
		return enc, startInt8, resident, err
	}
	var why bytes.Buffer
	if ok, _ := attachResidentTower(label, enc, backend, false, &why); ok { // require false: a failure is a warning in why, never an error
		return enc, startInt8, true, nil
	}
	enc.Close()
	fmt.Fprintf(warn, "note: the float32 Gemma 3 vision tower did not fit on the device (%s); using the int8 device tower instead (-vision-quant f32 asks for float32 and does not fall back)\n",
		strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(why.String()), "warning:")))
	enc, err = load(true)
	if err != nil {
		return enc, true, false, err
	}
	resident, err = attachResidentTower(label, enc, backend, require, warn)
	return enc, true, resident, err
}

// cudaFreeBytes is the card's free VRAM as the driver reports it; a variable so the fit tests can stand in for a card.
var cudaFreeBytes = func() (int64, bool) { return decoder.FreeBytesFor("cuda") }

// resolveGemma3VisionQuant settles Gemma 3's UNSET -vision-quant on CUDA before anything is planned. Float32 is the default there (towerInt8), and the plan subtracts its reserve (about 2.4 GB with the margin) from the
// KV budget up front. On a card where one KV slot at the context floor no longer fits beside the decoder and that reserve, the resident build would decline and the decoder would run on the CPU, which is far worse than
// the int8 tower it replaced, and the post-attach fallback (attachGemma3Tower) cannot undo a plan already made. So the default is float32 only when the card can hold the decoder, one floor-context KV slot and the float32
// tower; otherwise it becomes int8 here, with a note, and the plan prices the small reserve. An explicit -vision-quant is never touched.
//
// The decoder and KV sizes are estimates from the checkpoint, because the model is not loaded yet: resident weights were 2.0 GB from 8.6 GB of safetensors on the 4B (0.233, taken as 0.27 for cushion; a GGUF or .giw is
// already quantized, taken at its size), and one 4096-position KV slot was 1088 MB (0.54 of the weights, taken as 0.55). The estimate leans toward int8 on a card that would just have fit float32; it never
// leans toward declining. Anything it cannot read (an hf: reference, no free-VRAM probe, several models) leaves the default alone and the post-attach fallback as the guard.
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
	w, ok := gemma3ResidentWeightsEstimate(path)
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
	fmt.Fprintf(note, "note: the float32 Gemma 3 vision tower is the CUDA default, but this card has %.1f GB free and the decoder, one KV slot and that tower need about %.1f GB: using the int8 device tower "+
		"(-vision-quant f32 forces float32)\n", float64(free)/1e9, float64(need)/1e9)
	return cfg
}

// metalFreeBytes is the Metal resident budget (the memory guard's ceiling: the lower of 70% of RAM and the live available); a variable so the tests can
// stand in for a Mac.
var metalFreeBytes = func() (int64, bool) { return decoder.FreeBytesFor("metal") }

// resolveGemma3VisionQuantMetal is resolveGemma3VisionQuant on Metal (S18 on the Mac, docs/tasks/task-multimodal-support-2026-10.md): the unset default is
// the f16 tower when the budget holds the decoder, one KV slot at Metal's 2048-position floor and that tower, and the int8 tower (tower_gemm_w8, groups of
// 32) otherwise, with a note. The decoder's bytes are an estimate from the checkpoint: about 0.34 of the safetensors on the device (Gemma 3 4B's 5.15 GB
// guard figure, less its 0.53 GB of KV, is two copies of about 2.3-2.9 GB), counted twice unless a fresh sidecar .giw will be loaded instead (a heap
// load keeps the host copy beside the device one; part 1 of S18). It leans toward int8: a wrong f16 choice can push the decoder itself to the CPU, which
// is far worse than the int8 tower.
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
	w, ok := gemma3ResidentWeightsEstimate(path)
	if !ok {
		return cfg
	}
	free, ok := metalFreeBytes()
	if !ok {
		return cfg
	}
	w = w * 34 / 27 // gemma3ResidentWeightsEstimate's 0.27 is CUDA's; Metal's int4 layout and int8-pinned head read about 0.34
	dec := w
	if fi, err := os.Stat(path); err == nil && fi.IsDir() {
		if _, fresh := prequant.SidecarPathIfFresh(path, cfg.load.Quant, "metal", cfg.load.EmbedInt4); !fresh {
			dec *= 2 // a heap load: the host copy stays beside the device one
		}
	}
	const margin = 384 << 20
	need := dec + w*55/100/4 + metalTowerEstimate("gemma3", d, cfg.visionMaxPixels, false) + margin // one f16 KV slot at 2048 (CUDA's f32 4096-slot ratio, halved twice)
	if free >= need {
		return cfg
	}
	cfg.visionQuant = "int8"
	fmt.Fprintf(note, "note: this Mac's resident budget is %.1f GB and the decoder, one KV slot and the f16 Gemma 3 vision tower need about %.1f GB: using the int8 Metal tower "+
		"(weights in groups of 32, activations f32; -vision-quant f32 forces f16)\n", float64(free)/1e9, float64(need)/1e9)
	return cfg
}

// gemma3ResidentWeightsEstimate is the device bytes the decoder's weights will take, from the checkpoint alone (see resolveGemma3VisionQuant).
func gemma3ResidentWeightsEstimate(path string) (int64, bool) {
	fi, err := os.Stat(path)
	if err != nil {
		return 0, false
	}
	if !fi.IsDir() {
		low := strings.ToLower(path)
		if strings.HasSuffix(low, ".gguf") || strings.HasSuffix(low, ".giw") {
			return fi.Size(), fi.Size() > 0
		}
		return 0, false
	}
	files, _ := filepath.Glob(filepath.Join(path, "*.safetensors"))
	var total int64
	for _, f := range files {
		if st, err := os.Stat(f); err == nil {
			total += st.Size()
		}
	}
	return total * 27 / 100, total > 0
}

// siglipLoader is how attachGemma3Tower loads Gemma 3's SigLIP encoder from dir at either precision. head loads it without its blocks (S18 on the Mac): a
// Metal tower streams them from the checkpoint as it uploads, so the host never holds the 2.17 GB of float32 blocks beside the device copy, and the CPU
// path loads them only if it ever runs (a failed attach).
func siglipLoader(dir string, head bool) func(int8 bool) (*vision.Encoder, error) {
	return func(i8 bool) (*vision.Encoder, error) {
		load := vision.LoadEncoder
		if head {
			load = vision.LoadEncoderHead
		}
		e, err := load(dir, i8)
		if err != nil {
			return nil, fmt.Errorf("load vision encoder (%s): %w", dir, err)
		}
		return e, nil
	}
}

// gemma3FloatDefault says whether the float32 tower in use is the DEFAULT, the only case in which a failed attach may fall back to int8: -vision-quant unset, on CUDA or
// (since S18) Metal, which now has an int8 device tower too, and not already int8.
func gemma3FloatDefault(cfg config, int8Tower bool) bool {
	return cfg.visionQuant == "" && (cfg.towerBackend() == "cuda" || cfg.towerBackend() == "metal") && !int8Tower
}
