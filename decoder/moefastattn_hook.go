//go:build goinfer_testhooks

package decoder

// moeFastAttnProbe lets a MEASUREMENT force f32 prefill attention on a MoE arch,
// which the shipping build refuses outright.
//
// The refusal it bypasses is reasoned but was never measured: forwardn.go argues
// that an f32 QKᵀ reassociation flips a top-k expert at a near-tie and cascades,
// so MoE "is excluded here rather than trusted to the operator". No MoE model
// appears in docs/measurements/attention-a3-kernel-ratio-2026-08-26.md, and both
// G24 divergence tests run on the DENSE bench checkpoint — including the one whose
// doc comment says it pins "MoE excluded", which asserts nothing of the sort. The
// mechanism is plausible; its magnitude is unknown, and the whole value of the
// exclusion turns on the magnitude.
//
// Never set this outside a measurement. It is not a flag and must not become one.
var moeFastAttnProbe bool
