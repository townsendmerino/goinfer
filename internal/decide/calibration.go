package decide

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
)

// Calibration is autotrust's calibration.json (version 1, recorded verbatim in
// docs/measurements/decisions-d0-prior-art-2026-09-27.md): one temperature per kind, applied as z / T before the
// softmax. The loader reads version and per_kind and keeps the rest only to write it back; per_kind_family's
// semantics are unpublished and it is ignored.
type Calibration struct {
	Version       int                   `json:"version"`
	PerKind       map[string]float64    `json:"per_kind"`
	PerKindFamily map[string]any        `json:"per_kind_family"`
	Fit           map[string]FitSummary `json:"fit,omitempty"`
	Source        string                `json:"source,omitempty"`
	FitRows       int                   `json:"fit_rows,omitempty"`
	Template      string                `json:"template_version,omitempty"` // goinfer's addition: the template it was fitted under
}

// FitSummary is one kind's fit, in autotrust's shape.
type FitSummary struct {
	N        int     `json:"n"`
	T        float64 `json:"T"`
	KLBefore float64 `json:"kl_before"`
	KLAfter  float64 `json:"kl_after"`
}

// LoadCalibration reads a calibration.json. It refuses an unknown version and a non-positive temperature. The
// template a file was fitted under (goinfer's template_version; autotrust's files are bare-v1 and do not say) is
// checked by New against the decider's own, because a temperature fitted on other prompts does not describe these.
func LoadCalibration(path string) (*Calibration, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var c Calibration
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("decide: %s: %w", path, err)
	}
	if c.Version != 1 {
		return nil, fmt.Errorf("decide: %s: calibration version %d, this loader knows 1", path, c.Version)
	}
	for k, t := range c.PerKind {
		if !(t > 0) || math.IsInf(t, 0) {
			return nil, fmt.Errorf("decide: %s: temperature %v for %s", path, t, k)
		}
	}
	return &c, nil
}

// Save writes c as indented JSON.
func (c *Calibration) Save(path string) error {
	raw, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(raw, '\n'), 0o644)
}

// Temperature returns kind's temperature and whether one was fitted; 1 and false when c is nil or has none.
func (c *Calibration) Temperature(kind string) (float64, bool) {
	if c == nil {
		return 1, false
	}
	if t, ok := c.PerKind[kind]; ok {
		return t, true
	}
	return 1, false
}

// FitRow is one labelled row for the fit: the restricted log-probabilities at T = 1 (Decider.Scores) and the target
// distribution (a one-hot gold label or a soft teacher distribution), aligned.
type FitRow struct {
	Kind   string
	LogP   []float64
	Target []float64
}

// FitTemperatures fits one temperature per kind by minimizing the mean KL(target ‖ softmax(logp / T)) over that
// kind's rows. The objective is convex in β = 1/T (a log-sum-exp minus a linear term), so a golden-section search on
// β ∈ [0.02, 50] finds its minimum without a dependency. Kinds with no rows get no temperature.
func FitTemperatures(rows []FitRow, source, template string) *Calibration {
	byKind := map[string][]FitRow{}
	for _, r := range rows {
		byKind[r.Kind] = append(byKind[r.Kind], r)
	}
	c := &Calibration{Version: 1, PerKind: map[string]float64{}, PerKindFamily: map[string]any{}, Fit: map[string]FitSummary{},
		Source: source, FitRows: len(rows), Template: template}
	kinds := make([]string, 0, len(byKind))
	for k := range byKind {
		kinds = append(kinds, k)
	}
	sort.Strings(kinds)
	for _, k := range kinds {
		rs := byKind[k]
		obj := func(beta float64) float64 { return meanKL(rs, beta) }
		lo, hi := betaMin, betaMax
		const phi = 0.6180339887498949
		a, b := hi-phi*(hi-lo), lo+phi*(hi-lo)
		fa, fb := obj(a), obj(b)
		for hi-lo > 1e-7 {
			if fa < fb {
				hi, b, fb = b, a, fa
				a = hi - phi*(hi-lo)
				fa = obj(a)
			} else {
				lo, a, fa = a, b, fb
				b = lo + phi*(hi-lo)
				fb = obj(b)
			}
		}
		beta := (lo + hi) / 2
		c.PerKind[k] = 1 / beta
		c.Fit[k] = FitSummary{N: len(rs), T: 1 / beta, KLBefore: obj(1), KLAfter: obj(beta)}
	}
	return c
}

// The fit's search range for β = 1/T.
const (
	betaMin = 0.02
	betaMax = 50.0
)

// AtSearchBound reports whether a fitted temperature sits at the edge of the search (T within 0.1% of 1/betaMin or
// 1/betaMax): the objective wanted a flatter or sharper distribution than any temperature in range gives, which
// happens when the model's ranking disagrees with the targets. Such a fit is not a calibration.
func AtSearchBound(t float64) bool {
	return t >= (1/betaMin)*0.999 || t <= (1/betaMax)*1.001
}

// meanKL is the mean over rows of KL(target ‖ softmax(β·logp)), skipping options with no mass in either.
func meanKL(rows []FitRow, beta float64) float64 {
	var sum float64
	for _, r := range rows {
		maxv := math.Inf(-1)
		for _, v := range r.LogP {
			if b := beta * v; b > maxv && !math.IsInf(v, -1) {
				maxv = b
			}
		}
		var z float64
		for _, v := range r.LogP {
			if !math.IsInf(v, -1) {
				z += math.Exp(beta*v - maxv)
			}
		}
		lse := maxv + math.Log(z)
		for i, t := range r.Target {
			if t <= 0 {
				continue
			}
			lq := beta*r.LogP[i] - lse
			if math.IsInf(r.LogP[i], -1) {
				lq = math.Log(1e-300)
			}
			sum += t * (math.Log(t) - lq)
		}
	}
	return sum / float64(len(rows))
}
