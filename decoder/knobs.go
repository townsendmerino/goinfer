package decoder

import (
	"os"
	"strconv"
)

// Per-model operator knobs (docs/tasks/task-env-config-2026-09.md, phase 2a). These were read from the
// process environment on EVERY forward or generation, so changing the environment altered a model that was
// already loaded, and two models in one process could not differ. Each model now snapshots them ONCE, at
// Load (Options.Knobs overrides the environment per model), and every reader consults the model's snapshot.
// The environment variables keep working — they are simply read once, at Load.
//
// The values are kept RAW (as the environment would give them) and each reader below keeps the exact parsing
// its os.Getenv call had, so a default or an override means precisely what it meant before.
const (
	knobAttnGrouped      = "GOINFER_ATTN_GROUPED"
	knobAttnRowTile      = "GOINFER_ATTN_ROW_TILE"
	knobPrefillWorkers   = "GOINFER_PREFILL_ATTN_WORKERS"
	knobFusedAttention   = "GOINFER_FUSED_ATTENTION"
	knobMLANaive         = "GOINFER_MLA_NAIVE"
	knobMoEExpertMajor   = "GOINFER_MOE_EXPERT_MAJOR"
	knobBatchedPrefill   = "GOINFER_BATCHED_PREFILL"
	knobNoKVOnlyPrefill  = "GOINFER_NO_KVONLY_PREFILL"
	knobNoGreedyFastpath = "GOINFER_NO_GREEDY_FASTPATH"
	knobNoOptFwd         = "GOINFER_NO_OPTFWD"
	knobNoSampleFastpath = "GOINFER_NO_SAMPLE_FASTPATH"
	knobNoTopKFastpath   = "GOINFER_NO_TOPK_FASTPATH"
	knobOptFwdMaxTemp    = "GOINFER_OPTFWD_MAX_TEMP"
	knobCPUFastAttention = "GOINFER_CPU_FAST_ATTENTION"

	// Phase 2b: read at Load or per call on a loaded model.
	knobMoECacheExperts = "GOINFER_MOE_CACHE_EXPERTS"
	knobMoECacheSlots   = "GOINFER_MOE_CACHE_SLOTS"
	knobNoFitDefault    = "GOINFER_NO_FIT_DEFAULT"
	knobNoFitGuard      = "GOINFER_NO_FIT_GUARD"
	knobNoResidency     = "GOINFER_NO_RESIDENCY"
	knobNoResidentReuse = "GOINFER_NO_RESIDENT_REUSE"
	knobSSMResident     = "GOINFER_SSM_RESIDENT"

	// Phase 6: a rollback switch that had been filed as a diagnostic.
	knobMoEPreadCPU = "GOINFER_MOE_PREAD_CPU"
)

// cudaKnobs are phase 3's: the CUDA backend's operator knobs, snapshotted here with the rest so one mechanism
// (Load-time read, Options.Knobs override, the testhooks drift check) covers every backend. The CUDA resident
// reads them through Model.Knob; decoder itself never interprets them.
var cudaKnobs = []string{
	"GOINFER_CUDA_FAST_PREFILL", "GOINFER_CUDA_FAST_PREFILL_FLOOR", "GOINFER_CUDA_FLASH_DECODE",
	"GOINFER_CUDA_FLASH_DECODE_MIN_KEYS", "GOINFER_CUDA_FLASH_DECODE_VERIFY", "GOINFER_CUDA_NO_FUSE",
	"GOINFER_NO_LORA_CACHE", "GOINFER_PREFILL_CHUNK", "GOINFER_PREFILL_IMAGE_CHUNK", "GOINFER_SPLITKV_ATTN",
	"GOINFER_SPLITKV_MIN_KEYS",
	// Phase 6: rollback switches for default-on paths, filed as diagnostics until then.
	"GOINFER_CUDA_MOE_EXPERT_MAJOR", "GOINFER_CUDA_ATTN_FUSED_TILE", "GOINFER_MOE_DMA_OVERLAP",
	"GOINFER_MOE_PIN_REGISTER", "GOINFER_CUDA_L01_CPU_OFFLOAD",
}

var knobNames = []string{
	knobAttnGrouped, knobAttnRowTile, knobPrefillWorkers, knobFusedAttention, knobMLANaive, knobMoEExpertMajor,
	knobBatchedPrefill, knobNoKVOnlyPrefill, knobNoGreedyFastpath, knobNoOptFwd, knobNoSampleFastpath,
	knobNoTopKFastpath, knobOptFwdMaxTemp, knobCPUFastAttention,
	knobMoECacheExperts, knobMoECacheSlots, knobNoFitDefault, knobNoFitGuard, knobNoResidency,
	knobNoResidentReuse, knobSSMResident, knobMoEPreadCPU,
}

// metalKnobs are phase 4's: the Metal backend's operator knobs, same arrangement as cudaKnobs (read through
// Model.Knob by the Metal resident). GOINFER_MOE_EXPERT_MAJOR is shared with the CPU path and already listed.
var metalKnobs = []string{
	"GOINFER_METAL_ALIAS", "GOINFER_METAL_ATTN_FA", "GOINFER_METAL_BATCHED_PREFILL", "GOINFER_METAL_DECODE_LANE",
	"GOINFER_METAL_FAST_PREFILL", "GOINFER_METAL_FAST_PREFILL_FLOOR", "GOINFER_METAL_FUSED_ATTENTION",
	"GOINFER_METAL_MOE_SLOTS", "GOINFER_MOE_NOCACHE", "GOINFER_MOE_PREAD", "GOINFER_MOE_RESIDENCY",
	"GOINFER_MOE_RESIDENCY_SCOPE", "GOINFER_NO_RESIDENT_MEM_GUARD", "GOINFER_PRECISE_MATH",
}

func init() { knobNames = append(append(knobNames, cudaKnobs...), metalKnobs...) }

// Knobs is Options.Knobs: per-model knob values by environment-variable name.
type Knobs map[string]string

func (k *Knobs) values() map[string]string {
	if k == nil {
		return nil
	}
	return *k
}

// knobSet is one model's snapshot. A nil *knobSet reads the live environment — the old behaviour — for the
// structures tests build by hand without a Model (a hand-made Architecture, scratch or worker pool).
type knobSet struct {
	val    map[string]string
	set    map[string]bool
	pinned map[string]bool // set through Options.Knobs or SetKnobForTest: exempt from the drift check
}

// snapshotKnobs reads the knobs from the environment once, then applies over (Options.Knobs). A name in
// over that is not a known knob is ignored; knobs.go is the list.
func snapshotKnobs(over map[string]string) *knobSet {
	k := &knobSet{val: map[string]string{}, set: map[string]bool{}, pinned: map[string]bool{}}
	for _, n := range knobNames {
		k.val[n], k.set[n] = os.LookupEnv(n)
	}
	for n, v := range over {
		if _, known := k.set[n]; known {
			k.val[n], k.set[n], k.pinned[n] = v, true, true
		}
	}
	return k
}

func (k *knobSet) lookup(name string) (string, bool) {
	if k == nil {
		return os.LookupEnv(name)
	}
	knobDrift(k, name)
	return k.val[name], k.set[name]
}

func (k *knobSet) get(name string) string { v, _ := k.lookup(name); return v }

// pin sets one knob on this snapshot and exempts it from the drift check — the per-model replacement for a
// test's post-Load t.Setenv (SetKnobForTest / setKnob in tests). Returns a func restoring the previous state.
func (k *knobSet) pin(name, value string, set bool) (restore func()) {
	if _, known := k.set[name]; !known {
		panic("decoder: " + name + " is not a per-model knob (knobs.go)")
	}
	pv, ps, pp := k.val[name], k.set[name], k.pinned[name]
	k.val[name], k.set[name], k.pinned[name] = value, set, true
	return func() { k.val[name], k.set[name], k.pinned[name] = pv, ps, pp }
}

// Knob returns this model's value for one per-model knob and whether it is set: the snapshot taken at Load, with
// Options.Knobs applied. It is how a backend reads its own operator knobs (phase 3: CUDA), so they are read once
// per model and can differ between two models in one process. name must be on knobs.go's list — an unknown name
// panics rather than silently reading "unset", which is what a typo would otherwise do.
func (m *Model) Knob(name string) (string, bool) {
	if m.knobs != nil {
		if _, known := m.knobs.set[name]; !known {
			panic("decoder: Model.Knob(" + name + "): not a per-model knob (decoder/knobs.go)")
		}
	}
	return m.knobs.lookup(name)
}

// loadKnob is a knob read during Load BEFORE the model (and its snapshot) exists — the fit guards. Same
// precedence as the snapshot: Options.Knobs, else the environment, read once, at Load.
func loadKnob(opts Options, name string) string {
	if v, ok := opts.Knobs.values()[name]; ok {
		return v
	}
	return os.Getenv(name)
}

func (k *knobSet) attnGrouped() bool      { return k.get(knobAttnGrouped) != "0" }
func (k *knobSet) fusedAttention() bool   { return k.get(knobFusedAttention) != "0" }
func (k *knobSet) mlaNaive() bool         { return k.get(knobMLANaive) != "" }
func (k *knobSet) moeExpertMajor() bool   { return k.get(knobMoEExpertMajor) != "0" }
func (k *knobSet) cpuFastAttention() bool { return k.get(knobCPUFastAttention) != "0" }

// positiveInt is the override parse attnRowTile and prefillAttnWorkers shared: an integer >= 1, else unset.
func (k *knobSet) positiveInt(name string) (int, bool) {
	if v := k.get(name); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 {
			return n, true
		}
	}
	return 0, false
}

func (k *knobSet) optFwdTempCap() float64 {
	if v := k.get(knobOptFwdMaxTemp); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return optFwdMaxTemp
}

// bindKnobs gives a freshly constructed model its knob snapshot and shares it with the model's
// Architecture, which the deep forward helpers receive instead of the Model. Called before
// withResidency, so a backend that builds or probes during residency already sees the snapshot.
func (m *Model) bindKnobs(over map[string]string) *Model {
	m.knobs = snapshotKnobs(over)
	if m.w != nil && m.w.arch != nil {
		m.w.arch.knobs = m.knobs
	}
	return m
}
