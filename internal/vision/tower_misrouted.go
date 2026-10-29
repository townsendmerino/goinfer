package vision

// MisroutedTowerOp is one case of MisroutedTowerOpsForTest.
type MisroutedTowerOp struct {
	Name string
	Op   TowerOp
}

// MisroutedTowerOpsForTest are ops every walker must refuse before it queues anything: the routing CheckRegs holds and
// the weight shapes CheckShape holds, against a block of hidden 8 and intermediate 32 with a fused qkv (or q, k and v
// split from it), proj, fc1 and fc2, and gate, up and down. The walkers' tests share them.
func MisroutedTowerOpsForTest() []MisroutedTowerOp {
	return []MisroutedTowerOp{
		{"attention into the residual", TowerOp{Kind: TowerAttention, Out: TowerX}},
		{"GELU from another register", TowerOp{Kind: TowerGELUTanh, In: TowerX, Out: TowerMid}},
		{"fc1 into a hidden-width register", TowerOp{Kind: TowerLinear, In: TowerXN, Out: TowerQ, W: TowerWFC1}},
		{"fc2 from a hidden-width register", TowerOp{Kind: TowerLinear, In: TowerXN, Out: TowerX, W: TowerWFC2}},
		{"fc1 added to the residual", TowerOp{Kind: TowerLinearAdd, In: TowerXN, Out: TowerX, W: TowerWFC1}},
		{"fc2 added into mid", TowerOp{Kind: TowerLinearAdd, In: TowerMid, Out: TowerMid2, W: TowerWFC2}},
		{"a linear with no input register", TowerOp{Kind: TowerLinear, Out: TowerQ, W: TowerWO}},
		{"a norm into mid", TowerOp{Kind: TowerNorm, In: TowerX, Out: TowerMid, W: TowerNorm1, Form: TowerLayerNorm}},
		{"a norm from mid", TowerOp{Kind: TowerNorm, In: TowerMid, Out: TowerXN, W: TowerNorm1, Form: TowerLayerNorm}},
		{"attention with an input register", TowerOp{Kind: TowerAttention, In: TowerXN, Out: TowerCtx}},
		{"a per-head norm on mid", TowerOp{Kind: TowerHeadNorm, Out: TowerMid, W: TowerQNorm, Form: TowerRMSNorm}},
		{"RoPE from another register", TowerOp{Kind: TowerRoPE, In: TowerXN, Out: TowerQ}},
		{"the fused qkv with an output register", TowerOp{Kind: TowerLinearQKV, In: TowerXN, Out: TowerX, W: TowerWQKV}},
		{"the fused qkv from mid", TowerOp{Kind: TowerLinearQKV, In: TowerMid, W: TowerWQKV}},
		{"the SiLU product of mid with itself", TowerOp{Kind: TowerSiLUMul, In: TowerMid, Out: TowerMid}},
		{"the SiLU product from a hidden register", TowerOp{Kind: TowerSiLUMul, In: TowerXN, Out: TowerMid}},
		{"gate from mid", TowerOp{Kind: TowerLinear, In: TowerMid, Out: TowerMid2, W: TowerWGate}},
		{"down into a hidden register from one", TowerOp{Kind: TowerLinearAdd, In: TowerXN, Out: TowerX, W: TowerWDown}},
		{"fc1 transposed: from mid into a hidden register", TowerOp{Kind: TowerLinear, In: TowerMid, Out: TowerQ, W: TowerWFC1}},
		{"fc2 transposed: from a hidden register into mid", TowerOp{Kind: TowerLinear, In: TowerXN, Out: TowerMid, W: TowerWFC2}},
		{"fc1 transposed, added to the residual from mid", TowerOp{Kind: TowerLinearAdd, In: TowerMid, Out: TowerX, W: TowerWFC1}},
		{"gate transposed: from mid into a hidden register", TowerOp{Kind: TowerLinear, In: TowerMid, Out: TowerXN, W: TowerWGate}},
		{"down transposed: from a hidden register into mid", TowerOp{Kind: TowerLinear, In: TowerXN, Out: TowerMid2, W: TowerWDown}},
		{"the fused qkv of another weight", TowerOp{Kind: TowerLinearQKV, In: TowerXN, W: TowerWO}},
	}
}
