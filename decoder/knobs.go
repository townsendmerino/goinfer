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
)

var knobNames = []string{
	knobAttnGrouped, knobAttnRowTile, knobPrefillWorkers, knobFusedAttention, knobMLANaive, knobMoEExpertMajor,
	knobBatchedPrefill, knobNoKVOnlyPrefill, knobNoGreedyFastpath, knobNoOptFwd, knobNoSampleFastpath,
	knobNoTopKFastpath, knobOptFwdMaxTemp, knobCPUFastAttention,
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
