package decoder

import (
	"slices"
	"sync"
)

// BuildQuery is a load as the fit guard sees it before the build: the options, the facts decoder resolves from the config,
// and the bytes it priced. A backend's BuildDemand report reads it and applies its own layout and constants.
type BuildQuery struct {
	Options Options
	GGUF    bool // the source is a plain .gguf file

	// FeaturesCovered is whether the backend implements every feature the architecture needs for a resident build (decoder's
	// feature table); false when there is no config or it does not resolve. Layers is the config's layer count.
	FeaturesCovered bool
	Layers          int

	// WeightBytes and ExpertBytes are the resident weights the fit guard priced and the routed experts among them; 0 where the
	// query does not price them (a .giw, or before the weights are priced).
	WeightBytes, ExpertBytes int64

	// The routed experts' geometry, 0 for a dense model: the layers that hold experts, the experts per layer, an expert's
	// intermediate width, and the hidden width.
	MoELayers, NumExperts, ExpertIntermediate, Hidden int
	// ExpertCache is whether the load asks for an expert cache (Options.MoECacheExperts, or its knob).
	ExpertCache bool
}

// BuildDemand is a backend's account, before its build, of what a load on it holds beyond the weights the fit guard prices.
type BuildDemand struct {
	Resident   bool // the build goes resident on this backend
	KVOnDevice bool // the KV and the prefill scratch are device memory, which the backend's own plan sizes: the host fit prices none
	// HostPackBytes is the host the build adds beside the weights while it packs them, where no build peak applies.
	HostPackBytes int64
	// HostBuildPeak is the host peak of building an expert cache from a .gguf; it replaces weights + KV + source when larger,
	// because the phases do not overlap.
	HostBuildPeak int64
	// PinnedExpertBytes is the page-locked host copy of the routed experts a .giw load's expert cache holds.
	PinnedExpertBytes int64
}

// BuildDemandFunc is one backend's report.
type BuildDemandFunc func(BuildQuery) BuildDemand

var (
	buildDemandMu sync.RWMutex
	buildDemands  = map[string]BuildDemandFunc{}
)

// RegisterBuildDemand installs backend's report, replacing decoder's default for it. Call it from the backend package's init.
func RegisterBuildDemand(backend string, f BuildDemandFunc) {
	buildDemandMu.Lock()
	defer buildDemandMu.Unlock()
	buildDemands[backend] = f
}

// DefaultBuildDemand is decoder's own copy of backend's report, which a backend's test holds its registered report equal to.
// A backend with no default reports nothing.
func DefaultBuildDemand(backend string, q BuildQuery) BuildDemand {
	if f, ok := defaultBuildDemands[backend]; ok {
		return f(q)
	}
	return BuildDemand{}
}

var defaultBuildDemands = map[string]BuildDemandFunc{"cuda": defaultCUDABuildDemand, "metal": defaultMetalBuildDemand}

// The defaults stand in for a backend's own report where its package is not linked, as in the decoder's tests.
func init() {
	for name, f := range defaultBuildDemands {
		RegisterBuildDemand(name, f)
	}
}

// cudaPackStageBytes is the staging memory the CUDA build streams packed weights through: up to four blocks of 8 MiB of words
// with their scale scratch (20 MiB each).
const cudaPackStageBytes = 80 << 20

// defaultCUDABuildDemand: resident when CUDA is linked, is the load's backend, does not stream, and implements the
// architecture; its KV is then device memory, and packing adds one packed layer and the staging blocks. A C' expert cache built
// from a .gguf peaks at the weights twice and the experts once; a .giw load's cache pins every routed expert at int4 with f16
// group-32 scales (9/16 byte a weight).
func defaultCUDABuildDemand(q BuildQuery) BuildDemand {
	o := q.Options
	var d BuildDemand
	d.Resident = o.Backend == "cuda" && !o.StreamWeights && slices.Contains(CompiledBackends(), "cuda") && q.FeaturesCovered
	d.KVOnDevice = d.Resident
	if d.Resident {
		d.HostPackBytes = q.WeightBytes/int64(max(q.Layers, 1)) + cudaPackStageBytes
	}
	if o.Backend == "cuda" && q.GGUF && !o.StreamWeights && q.ExpertBytes > 0 && q.ExpertCache {
		d.HostBuildPeak = 2*q.WeightBytes + q.ExpertBytes
	}
	if q.ExpertCache && d.Resident && q.NumExperts > 0 && q.ExpertIntermediate > 0 {
		d.PinnedExpertBytes = int64(q.MoELayers) * int64(q.NumExperts) * 3 * int64(q.ExpertIntermediate) * int64(q.Hidden) * 9 / 16
	}
	return d
}

// defaultMetalBuildDemand: resident when Metal is linked, is the load's backend and implements the architecture. Its KV is
// unified memory the fit guard prices as Metal allocates it.
func defaultMetalBuildDemand(q BuildQuery) BuildDemand {
	o := q.Options
	return BuildDemand{Resident: o.Backend == "metal" && slices.Contains(CompiledBackends(), "metal") && q.FeaturesCovered}
}

// buildDemandFor is the report of opts's backend for a load of cfg (nil before the config is read) with the priced bytes.
func buildDemandFor(cfg *Config, opts Options, weightBytes, expertBytes int64, gguf bool) BuildDemand {
	q := BuildQueryFor(cfg, opts, weightBytes, expertBytes, gguf)
	buildDemandMu.RLock()
	f := buildDemands[opts.Backend]
	buildDemandMu.RUnlock()
	if f == nil {
		return BuildDemand{}
	}
	return f(q)
}

// BuildQueryFor is the query the fit guard asks opts's backend for a load of cfg (nil before the config is read) with the
// priced bytes: the facts decoder resolves from the config, and the options.
func BuildQueryFor(cfg *Config, opts Options, weightBytes, expertBytes int64, gguf bool) BuildQuery {
	q := BuildQuery{Options: opts, GGUF: gguf, WeightBytes: weightBytes, ExpertBytes: expertBytes,
		ExpertCache: opts.MoECacheExperts || loadKnob(opts, knobMoECacheExperts) != ""}
	if cfg != nil {
		q.Layers = cfg.NumLayers
		c := *cfg
		if arch, _, err := resolveArchitecture(&c); err == nil {
			q.FeaturesCovered = len(missingFeatures(arch.residentFeatures(), residentBackendFeatures[opts.Backend])) == 0
			if arch.MoE != nil && arch.MoE.NumExperts > 0 && arch.MoE.IntermediateDim > 0 {
				q.MoELayers = max(c.NumLayers-max(c.FirstKDenseReplace, 0), 0)
				q.NumExperts, q.ExpertIntermediate, q.Hidden = arch.MoE.NumExperts, arch.MoE.IntermediateDim, c.HiddenDim
			}
		}
	}
	return q
}

// PlanSlackFunc is a backend's allocation rounding over the buffers a resident build of the loaded model m allocates for its
// weights: what Plan prices beside them so the dry run and the build agree. ok false is unknown, priced as 0.
type PlanSlackFunc func(m *Model) (slackBytes int64, ok bool)

var (
	planSlackMu sync.RWMutex
	planSlacks  = map[string]PlanSlackFunc{}
)

// RegisterPlanSlack installs backend's rounding report. Decoder keeps no default copy of it: the rounding follows the backend's
// own buffer layout, so its one copy is the backend's; without it the slack is 0, the true figure for every buffer under the
// allocator's quantum.
func RegisterPlanSlack(backend string, f PlanSlackFunc) {
	planSlackMu.Lock()
	defer planSlackMu.Unlock()
	planSlacks[backend] = f
}

// planSlackMemo remembers each backend's PlanSlack for one model: it is a property of the loaded weights.
type planSlackMemo struct {
	mu sync.Mutex
	of map[string]int64
}

// planSlack is backend's rounding over m's weight buffers, asked once per model and backend.
func (m *Model) planSlack(backend string) int64 {
	if m == nil || planSlackOffForTest {
		return 0
	}
	ask := func() int64 {
		planSlackMu.RLock()
		f := planSlacks[backend]
		planSlackMu.RUnlock()
		if f != nil {
			if s, ok := f(m); ok {
				return s
			}
		}
		return 0
	}
	memo := m.planSlackOf
	if memo == nil {
		return ask()
	}
	memo.mu.Lock()
	defer memo.mu.Unlock()
	if v, ok := memo.of[backend]; ok {
		return v
	}
	if memo.of == nil {
		memo.of = map[string]int64{}
	}
	v := ask()
	memo.of[backend] = v
	return v
}

// PlanSlack is backend's allocation rounding over this model's weight buffers, as Plan prices it (0 where the backend reports
// none). The CUDA build reads it instead of pricing its own copy.
func (m *Model) PlanSlack(backend string) int64 { return m.planSlack(backend) }

// planSlackOffForTest prices no allocation rounding in any plan: the before side of a test's diff.
var planSlackOffForTest bool
