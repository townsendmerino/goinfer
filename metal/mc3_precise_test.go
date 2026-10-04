//go:build darwin

package metal

import "testing"

// TestMC3Step_bitIdenticalPreciseMath is audit E-C02's probe (docs/audit-metal-2026-09-30.md): the batched step against
// production's single-token decode with the resident library compiled without fast math, as a native int8 model's is
// since 2026-10-04 (w8PreciseMath). The derived rows kernels are appended after the Gumbel block, whose closing pragma
// restores fp contract(fast); under precise math that may compile them differently from the kernels they derive from.
// Shallow and deep rows, as TestMC3Step_bitIdentical and _bitIdenticalDeep.
func TestMC3Step_bitIdenticalPreciseMath(t *testing.T) {
	prev := preciseMathCompile
	preciseMathCompile = true
	defer func() { preciseMathCompile = prev }()
	t.Run("shallow", func(t *testing.T) { mc3Identity(t, []int{5, 40, 300}, 1024) })
	t.Run("deep", func(t *testing.T) {
		mc3Identity(t, []int{attnFADepthFloor + 400, attnFADepthFloor - 6, 100, attnFADepthFloor + 64}, 2048)
	})
}

// TestMC3Step_w8BitIdentical (int8 slice 3, docs/tasks/task-metal-int8-2026-10.md): a native int8 resident (precise
// math, as w8PreciseMath compiles it) in the batched step, every projection as production's int8 GEMV once per row,
// against production's single-token decode: every logit equal, shallow and deep rows, the per-row and multi-row deep
// attention both.
func TestMC3Step_w8BitIdentical(t *testing.T) {
	prevQ, prevN := mc3TestQuant, nativeInt8
	mc3TestQuant, nativeInt8 = "int8int8", true
	defer func() { mc3TestQuant, nativeInt8 = prevQ, prevN }()
	t.Run("shallow", func(t *testing.T) { mc3Identity(t, []int{5, 40, 300}, 1024) })
	deep := []int{attnFADepthFloor + 400, attnFADepthFloor - 6, 100, attnFADepthFloor + 64}
	t.Run("deep", func(t *testing.T) { mc3Identity(t, deep, 2048) })
	t.Run("deep-fa-rows", func(t *testing.T) {
		prev := mc3FARowsOn
		mc3FARowsOn = true
		defer func() { mc3FARowsOn = prev }()
		mc3Identity(t, deep, 2048)
	})
}
