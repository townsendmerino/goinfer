package multimodal

import (
	"slices"
	"sort"
	"sync"
)

// S8 (docs/tasks/task-multimodal-support-2026-10.md, "S8, plan and gates"): which GPU towers each backend's build
// registers, declared once, here. The README and docs/multimodal.md support table's tower cells are computed from this
// declaration (decoder's support-table drift test), and each backend module's test proves its live registrations equal
// its row, so a tower added or dropped without the table following fails a test rather than a reader.

// Tower families, as the support table and the declaration name them.
const (
	TowerSigLIP          = "siglip"          // Gemma 3's SigLIP (aikit vision.RegisterResident)
	TowerQwen25VL        = "qwen2.5-vl"      // Qwen2.5-VL's ViT (aikit vision.RegisterQwenResident)
	TowerQwen3           = "qwen3"           // the Qwen3.5+ and Qwen3-VL grid tower (RegisterQwen3Tower)
	TowerGlmOcr          = "glm-ocr"         // GLM-OCR's grid tower (RegisterGlmOcrTower)
	TowerGemma4          = "gemma4-vision"   // Gemma 4's vision tower (RegisterGemma4Tower)
	TowerPixtral         = "pixtral"         // Ministral 3's Pixtral tower (S10; CPU only, no backend declares it)
	TowerGemma4Audio     = "gemma4-audio"    // Gemma 4's audio tower (embeddinggemma2.RegisterAudioAccelerator)
	TowerEmbeddingGemma2 = "embeddinggemma2" // EmbeddingGemma 2's text accelerator (embeddinggemma2.RegisterAccelerator)
)

// declaredTowers is, per backend (decoder's backend names), the tower families its build registers a GPU tower for.
var declaredTowers = map[string][]string{
	"metal":  {TowerSigLIP, TowerQwen25VL, TowerQwen3, TowerGlmOcr, TowerGemma4, TowerGemma4Audio, TowerEmbeddingGemma2},
	"cuda":   {TowerSigLIP, TowerQwen25VL, TowerQwen3, TowerGlmOcr, TowerGemma4},
	"webgpu": {TowerSigLIP},
}

// DeclaredTowers returns the tower families backend's build declares, sorted.
func DeclaredTowers(backend string) []string {
	t := slices.Clone(declaredTowers[backend])
	sort.Strings(t)
	return t
}

// DeclaredTower reports whether backend's build declares a GPU tower for family.
func DeclaredTower(backend, family string) bool {
	return slices.Contains(declaredTowers[backend], family)
}

var (
	residentNamesMu sync.Mutex
	residentNames   = map[string][]string{} // tower family -> device names
)

// MarkResidentTower records that device registered a resident tower for family through one of aikit's unnamed slots
// (vision.RegisterResident for SigLIP, vision.RegisterQwenResident for Qwen2.5-VL), which cannot be asked what is
// registered. A backend's init calls it beside the aikit registration.
func MarkResidentTower(family, device string) {
	residentNamesMu.Lock()
	defer residentNamesMu.Unlock()
	if !slices.Contains(residentNames[family], device) {
		residentNames[family] = append(residentNames[family], device)
	}
}

// ResidentTowers lists the devices that marked a resident tower for family, sorted.
func ResidentTowers(family string) []string {
	residentNamesMu.Lock()
	defer residentNamesMu.Unlock()
	n := slices.Clone(residentNames[family])
	sort.Strings(n)
	return n
}

// LiveTowers is the tower families a build has registered for device, from the live registries: the named ones
// (Gemma4Towers, Qwen3Towers, GlmOcrTowers) and the resident marks (MarkResidentTower), plus extra (the embeddinggemma2
// accelerators, which this package cannot import, passed by the caller), sorted. Each backend module's S8 test compares it
// with DeclaredTowers.
func LiveTowers(device string, extra ...string) []string {
	var t []string
	add := func(family string, names []string) {
		if slices.Contains(names, device) {
			t = append(t, family)
		}
	}
	add(TowerSigLIP, ResidentTowers(TowerSigLIP))
	add(TowerQwen25VL, ResidentTowers(TowerQwen25VL))
	add(TowerQwen3, Qwen3Towers())
	add(TowerGlmOcr, GlmOcrTowers())
	add(TowerGemma4, Gemma4Towers())
	t = append(t, extra...)
	sort.Strings(t)
	return t
}

// TowerDrift compares a build's live towers with a declaration: the families declared but not registered, and those
// registered but not declared. Both empty means the support table's tower cells for that backend are true.
func TowerDrift(live, declared []string) (missing, undeclared []string) {
	for _, d := range declared {
		if !slices.Contains(live, d) {
			missing = append(missing, d)
		}
	}
	for _, l := range live {
		if !slices.Contains(declared, l) {
			undeclared = append(undeclared, l)
		}
	}
	return missing, undeclared
}
