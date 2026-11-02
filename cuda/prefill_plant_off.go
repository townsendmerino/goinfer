//go:build cuda && !goinfer_testhooks

package cuda

// prefillPlant is the batched prefill walk's planted defect (prefillPlantKind): none outside goinfer_testhooks builds.
const prefillPlant = prefillPlantNone
