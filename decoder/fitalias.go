package decoder

import "github.com/townsendmerino/goinfer/decoder/fit"

// The fit guard's and the planner's exchange types live in decoder/fit; these are the same types under their long-standing names.
type (
	Placement       = fit.Placement
	PlanRequest     = fit.PlanRequest
	Plan            = fit.Plan
	FitAnswer       = fit.FitAnswer
	FitDeclineError = fit.FitDeclineError
)

const (
	PlacementResident            = fit.PlacementResident
	PlacementExpertCached        = fit.PlacementExpertCached
	PlacementHostComputedExperts = fit.PlacementHostComputedExperts
	PlacementWeightPaged         = fit.PlacementWeightPaged
	PlacementDecline             = fit.PlacementDecline
)

// ErrWontFitResident is wrapped into every load-time fit-guard refusal; see fit.ErrWontFitResident.
var ErrWontFitResident = fit.ErrWontFitResident
