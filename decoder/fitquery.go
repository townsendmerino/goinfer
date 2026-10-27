package decoder

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/townsendmerino/aikit/mmap"
	"github.com/townsendmerino/goinfer/internal/giw"
)

// FitAnswer is what Load's own memory check says of loading a path with given options, priced against the memory
// available now, without loading anything. FitQuery computes it with the functions Load calls (fitCheckFor and guardFit's
// decision for a .gguf or safetensors directory, decideGIWFit for a .giw), so the two cannot disagree.
type FitAnswer struct {
	// Guard names the check that priced it: "resident" (weights, KV and the mapped source, against 70% of available memory)
	// or "giw" (KV and scratch against available memory less 1 GB: a .giw's weights are file-backed).
	Guard string
	// Fits is whether Load proceeds. Load proceeds when the check cannot price the load; Unpriced then says why.
	Fits     bool
	Unpriced string
	// Ctx is the context priced; when CtxCapped, the smaller one Load would pin an unpinned request to.
	Ctx                                                      int
	CtxPinned                                                bool
	CtxCapped                                                bool
	MaxCtx                                                   int // the model's own window, 0 when it declares none
	NeedBytes, BudgetBytes, WeightBytes, KVBytes, AvailBytes int64
	// PinnedExpertBytes is the page-locked host copy of the routed experts that a CUDA expert cache (--moe-cache-experts) builds from a .giw, priced into
	// NeedBytes; 0 for every other load.
	PinnedExpertBytes int64
	Decline           string // Load's refusal when it does not fit, "" otherwise
	// Streamable is the refusal's DenseStreamable: modelload.Load retries such a .gguf with weight streaming.
	Streamable bool
	// DecodePath is the decode path Load would report: "cpu (<quant>)", as DecodePath begins, when the load runs on the CPU,
	// or, for a .giw on a backend whose pre-build checks can be asked (ResidencyPricer), the resident or the decline they
	// decide. "" when only the load can tell; DecodeUnknown then says why.
	DecodePath    string
	DecodeUnknown string
	// ResidentCtx is the context a resident decode path would hold, when DecodePath names one (0 otherwise).
	ResidentCtx int
}

// FitQuery answers FitAnswer for path and opts. An error is a load that would fail before its memory check, for a reason
// other than memory (an unreadable .giw, an unknown quant).
func FitQuery(path string, opts Options) (FitAnswer, error) {
	return fitQueryAt(path, opts, hostRAMAvailable)
}

// FitQueryAvailable is FitQuery against a recorded machine instead of this one: the available host memory is availableBytes
// (0 for unknown, which proceeds unpriced) rather than the probe's reading. `fit -profile` and the night queue's check use it to
// price a load before the box it will run on is in the state it will run in.
func FitQueryAvailable(path string, opts Options, availableBytes int64) (FitAnswer, error) {
	return fitQueryAt(path, opts, func() int64 { return availableBytes })
}

// fitQueryAt is FitQuery with the available-memory reading passed in.
func fitQueryAt(path string, opts Options, read func() int64) (FitAnswer, error) {
	opts = opts.withAutoBackend()
	if strings.HasSuffix(path, ".giw") {
		cfg, baked, err := readGIWConfig(path)
		if err != nil {
			return FitAnswer{}, err
		}
		d := decideGIWFitAt(cfg, opts, read)
		a := FitAnswer{Guard: "giw", Fits: d.err == nil, Ctx: d.ctx, CtxPinned: d.pinned, CtxCapped: d.pinCtx > 0,
			MaxCtx: cfg.MaxPositions, NeedBytes: d.need, BudgetBytes: d.budget, KVBytes: max(d.need-d.pinnedExperts-prefillAttnScratchBudget, 0), PinnedExpertBytes: d.pinnedExperts, AvailBytes: d.avail}
		if !d.known {
			a.Unpriced = unpricedReason(opts, cfg != nil)
			a.KVBytes = 0
		}
		if d.err != nil {
			a.Decline = d.err.Error()
		}
		a.DecodePath, a.DecodeUnknown = decodePathBeforeLoad(cfg, opts, baked)
		if a.DecodePath == "" && a.Fits {
			a.DecodePath, a.DecodeUnknown, a.ResidentCtx = planGIWResidency(path, opts)
		}
		return a, nil
	}
	quant, err := parseQuant(opts.Quant)
	if err != nil {
		return FitAnswer{}, err
	}
	if err := checkHalf16Load(quant, path, opts.Backend); err != nil {
		return FitAnswer{}, err
	}
	f := fitCheckForAt(path, opts.Quant, quant, opts, read)
	pin, decline := f.decide()
	a := FitAnswer{Guard: "resident", Fits: decline == nil, Ctx: f.effCtx, CtxPinned: f.pinned, NeedBytes: f.need(),
		BudgetBytes: f.budget(), WeightBytes: f.weightBytes, KVBytes: f.kvBytes, AvailBytes: f.availBytes}
	if f.cfg != nil {
		a.MaxCtx = f.cfg.MaxPositions
	}
	if pin > 0 {
		a.Ctx, a.CtxCapped = pin, true
		a.KVBytes = estimateKVBytes(f.cfg, pin, f.kvF16, f.kvI8)
		a.NeedBytes = max(f.cudaBuildBytes, f.weightBytes+a.KVBytes+f.srcFileBytes)
	}
	if f.noGuard || !f.known() {
		a.Unpriced = unpricedReason(opts, f.cfg != nil && f.weightBytes > 0)
	}
	if decline != nil {
		a.Decline, a.Streamable = decline.Error(), decline.DenseStreamable
	}
	a.DecodePath, a.DecodeUnknown = decodePathBeforeLoad(f.cfg, opts, opts.Quant)
	return a, nil
}

// unpricedReason says why a memory check proceeded without a verdict.
func unpricedReason(opts Options, readable bool) string {
	switch {
	case loadKnob(opts, knobNoFitGuard) != "":
		return "the fit guard is off (GOINFER_NO_FIT_GUARD)"
	case hostRAMAvailable() <= 0:
		return "this machine's available memory could not be read"
	case !readable:
		return "the checkpoint's size or config could not be read"
	}
	return "nothing is pinned and the model declares no maximum context"
}

// decodePathBeforeLoad is the decode path a load reports, where that is fixed before the weights are opened: on the CPU
// backend, a GPU backend this binary does not carry (NewBackend falls back), or Metal where the architecture fails its
// feature gate (metalWillBeResident). Anything that may go resident is decided by the backend after the weights are opened.
func decodePathBeforeLoad(cfg *Config, opts Options, quant string) (path, unknown string) {
	be := opts.Backend
	cpu := be == "" || be == "cpu" || !slices.Contains(CompiledBackends(), be) ||
		(be == "metal" && cfg != nil && !buildDemandFor(cfg, opts, 0, 0, false).Resident)
	switch {
	case !cpu:
		return "", fmt.Sprintf("%s residency is decided once the weights are opened (its memory guard and plan); a .giw is opened to ask", be)
	case quant == "":
		return "", "the quant is read from the weights"
	}
	return fmt.Sprintf("cpu (%s)", quant), ""
}

// planGIWResidency is a .giw load's decode path on a GPU backend: the bundle opened as Load opens it (mapped, its weights
// file-backed and untouched) but with no resident built, the backend's own pre-build checks asked (Model.plannedDecodePath),
// then closed.
func planGIWResidency(path string, opts Options) (decodePath, unknown string, ctx int) {
	opts.planResidency = true
	m, err := Load(path, opts)
	if err != nil {
		return "", "the .giw does not open: " + err.Error(), 0
	}
	defer m.Close()
	return m.plannedDecodePath()
}

// GIWHeaderCheck reports whether path is a .giw this build reads, from its header alone: the bundle container, the weights
// format version and a Config that resolves to an architecture. It does not read past the header, so it does not prove the
// bundle loads (a truncated payload or a missing tensor fails only at Load).
func GIWHeaderCheck(path string) error {
	_, _, err := readGIWConfig(path)
	return err
}

// readGIWConfig reads a .giw's Config and baked quant from its header without walking a weight, as Load's .giw branch does
// before guardGIWFit (resolveArchitecture included, which may backfill fields). The mapping touches only the header pages.
func readGIWConfig(path string) (*Config, string, error) {
	data, err := mapGIW(path)
	if err != nil {
		return nil, "", fmt.Errorf("decoder: mmap .giw: %w", err)
	}
	defer func() { _ = mmap.Unmap(data) }()
	blob, _, err := giw.Read(data)
	if err != nil {
		return nil, "", fmt.Errorf("decoder: parse .giw bundle: %w", err)
	}
	cfgJSON, baked, err := readGIWHead(blob)
	if err != nil {
		return nil, "", err
	}
	var cfg Config
	if err := json.Unmarshal(cfgJSON, &cfg); err != nil {
		return nil, "", &SerializeError{"config json: " + err.Error()}
	}
	if _, _, err := resolveArchitecture(&cfg); err != nil {
		return nil, "", &SerializeError{"arch: " + err.Error()}
	}
	return &cfg, baked, nil
}
