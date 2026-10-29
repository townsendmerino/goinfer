package vision

import "testing"

// TestTowerOpCheckRegs: every op of every tower program takes the registers its kind allows, and each misrouting below
// is refused.
func TestTowerOpCheckRegs(t *testing.T) {
	progs := map[string]TowerBlock{
		"siglip":  SiglipProgram(EncoderConfig{HiddenSize: 8, NumAttentionHeads: 2, LayerNormEps: 1e-6}),
		"qwen3":   Qwen3Program(Qwen3EncoderConfig{HiddenSize: 8, NumHeads: 2}),
		"glm-ocr": GlmOcrProgram(GlmOcrEncoderConfig{HiddenSize: 8, NumHeads: 2, RMSNormEps: 1e-5}),
		"qwen2.5": Qwen25Program(QwenEncoderConfig{HiddenSize: 8, NumHeads: 2}),
	}
	for name, p := range progs {
		for i, op := range p {
			if err := op.CheckRegs(); err != nil {
				t.Errorf("%s op %d (%s): %v", name, i, op.Kind, err)
			}
		}
	}
	for r, want := range map[TowerReg]int{0: 0, TowerX: 8, TowerXN: 8, TowerCtx: 8, TowerMid: 32, TowerMid2: 32, TowerMid2 + 1: 0} {
		if got := TowerRegWidth(r, 8, 32); got != want {
			t.Errorf("TowerRegWidth(%d) = %d, want %d", r, got, want)
		}
	}
	for _, op := range []TowerOp{
		{Kind: TowerNorm, In: TowerMid, Out: TowerXN},
		{Kind: TowerNorm, In: TowerX, Out: TowerMid2},
		{Kind: TowerAttention, In: TowerXN, Out: TowerCtx},
		{Kind: TowerAttention, Out: TowerX},
		{Kind: TowerGELUTanh, In: TowerXN, Out: TowerMid},
		{Kind: TowerRoPE, Out: TowerMid},
		{Kind: TowerHeadNorm, In: TowerQ, Out: TowerQ},
		{Kind: TowerSiLUMul, In: TowerXN, Out: TowerMid},
		{Kind: TowerSiLUMul, In: TowerMid, Out: TowerMid},
		{Kind: TowerLinearQKV, In: TowerXN, Out: TowerQ},
		{Kind: TowerLinear, In: TowerXN, Out: 0},
		{Kind: TowerLinearAdd, In: 0, Out: TowerX},
		{Kind: 0},
	} {
		if err := op.CheckRegs(); err == nil {
			t.Errorf("%+v: accepted", op)
		}
	}
}
