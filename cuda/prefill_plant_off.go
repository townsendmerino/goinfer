//go:build cuda && !goinfer_testhooks

package cuda

// prefillPlant is the batched prefill walk's planted defect (prefillPlantKind): none outside goinfer_testhooks builds.
const prefillPlant = prefillPlantNone

// The batched prefill steps' build-time hooks (testhooks.go): off outside goinfer_testhooks builds.
const (
	prefillStepsBuildChecksOffForTest = false
	prefillStepsEmptyLayerForTest     = -1
)
