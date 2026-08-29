//go:build !goinfer_testhooks

package decoder

// moeFastAttnProbe is a CONSTANT false in every shipping build, so the MoE
// exclusion in runLayersFromEmbedN folds away at compile time and cannot be
// reached by an operator, an env var, or a caller. The tagged twin
// (moefastattn_hook.go) turns it into a variable so a test can measure what the
// exclusion is worth — see TestA3MoEExclusionIsMeasured.
//
// A const rather than a `var x = false`: the compiler then PROVES the shipping
// guard is unchanged, instead of a reader having to trust that nothing writes it.
const moeFastAttnProbe = false
