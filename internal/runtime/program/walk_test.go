package program

import "testing"

func TestDecodeLayerWalk_CoversAllTopologies(t *testing.T) {
	rms := func(slot NormSlot) Op {
		return Op{Kind: OpNorm, In: RegH, Out: RegX, Norm: NormParams{Slot: slot, Form: NormRMS, Eps: 1e-5, AddOne: slot == SlotPostAttn || slot == SlotPostMLP}}
	}
	copyIn := func(slot NormSlot) Op {
		return Op{Kind: OpNorm, In: RegH, Out: RegX, Norm: NormParams{Slot: slot, Form: NormCopy}}
	}

	t.Run("sequential pre-norm attention + SwiGLU", func(t *testing.T) {
		ops := []Op{
			rms(SlotPreAttn),
			{Kind: OpProjQKV},
			{Kind: OpQKNorm, QKNorm: QKNormParams{Form: QKRMS, Eps: 1e-5}},
			{Kind: OpRope, Rope: RopeParams{Kernel: RopeAtKernel, InvFreq: []float64{1}}},
			{Kind: OpAttention},
			{Kind: OpProjO},
			{Kind: OpResidual},
			rms(SlotPreMLP),
			{Kind: OpFFN, FFN: FFNParams{Form: FfnGated, Act: ActSiLU}},
			{Kind: OpResidual},
		}
		lw, err := DecodeLayerWalk(ops, "test")
		if err != nil {
			t.Fatal(err)
		}
		if !lw.HasAttn || lw.Attn.InputForm != NormRMS || !lw.Attn.QKNorm || lw.Attn.OutMode != OutFusedResidual {
			t.Fatalf("unexpected Attn walk: %+v", lw.Attn)
		}
		if !lw.HasFFN || lw.FFN.InputMode != FFNInNorm || lw.FFN.InputNorm.Slot != SlotPreMLP || lw.FFN.OutMode != OutFusedResidual {
			t.Fatalf("unexpected FFN walk: %+v", lw.FFN)
		}
	})

	t.Run("post-only attention vs DeltaNet hybrid FFN input modes", func(t *testing.T) {
		attnLayer := []Op{
			copyIn(SlotPreAttn),
			{Kind: OpProjQKV},
			{Kind: OpRope, Rope: RopeParams{Kernel: RopeAtKernel, InvFreq: []float64{1}}},
			{Kind: OpAttention},
			{Kind: OpProjO},
			{Kind: OpNorm, Norm: NormParams{Slot: SlotPostAttn, Form: NormRMS, Eps: 1e-5}},
			{Kind: OpResidual},
			copyIn(SlotPreMLP),
			{Kind: OpFFN, FFN: FFNParams{Form: FfnGated, Act: ActSiLU}},
			{Kind: OpNorm, Norm: NormParams{Slot: SlotPostMLP, Form: NormRMS, Eps: 1e-5}},
			{Kind: OpResidual},
		}
		deltaLayer := []Op{
			rms(SlotPreAttn),
			{Kind: OpDeltaNet, Delta: DeltaNetOpParams{Eps: 1e-5}},
			{Kind: OpResidual},
			rms(SlotPreMLP),
			{Kind: OpFFN, FFN: FFNParams{Form: FfnGated, Act: ActSiLU}},
			{Kind: OpResidual},
		}
		lwAttn, err := DecodeLayerWalk(attnLayer, "test")
		if err != nil {
			t.Fatal(err)
		}
		lwDelta, err := DecodeLayerWalk(deltaLayer, "test")
		if err != nil {
			t.Fatal(err)
		}
		// Without per-family flags, the IR itself distinguishes post-only attention FFN (FFNInRaw + OutPostNormResidual)
		// from Olmo Hybrid's DeltaNet layer FFN (FFNInNorm + OutFusedResidual).
		if lwAttn.FFN.InputMode != FFNInRaw || lwAttn.FFN.OutMode != OutPostNormResidual {
			t.Fatalf("post-only attention FFN = %+v, want FFNInRaw + OutPostNormResidual", lwAttn.FFN)
		}
		if lwDelta.Mixer != MixerDeltaNet || lwDelta.FFN.InputMode != FFNInNorm || lwDelta.FFN.OutMode != OutFusedResidual {
			t.Fatalf("hybrid DeltaNet FFN = (mixer=%v, ffn=%+v), want DeltaNet + FFNInNorm + OutFusedResidual", lwDelta.Mixer, lwDelta.FFN)
		}
	})

	t.Run("parallel block shares input norm and emits Two residual", func(t *testing.T) {
		ops := []Op{
			rms(SlotPreAttn),
			{Kind: OpProjQKV},
			{Kind: OpRope, Rope: RopeParams{Kernel: RopeAtKernel, InvFreq: []float64{1}}},
			{Kind: OpAttention},
			{Kind: OpProjO},
			{Kind: OpFFN, Out: RegB, FFN: FFNParams{Form: FfnGated, Act: ActSiLU}},
			{Kind: OpResidual, Res: ResidualParams{Two: true}},
		}
		lw, err := DecodeLayerWalk(ops, "test")
		if err != nil {
			t.Fatal(err)
		}
		if lw.Attn.OutMode != OutParallelScratch || lw.FFN.InputMode != FFNInShared || lw.FFN.OutMode != OutTwoResidual {
			t.Fatalf("parallel walk = (attn=%+v, ffn=%+v)", lw.Attn, lw.FFN)
		}
	})

	t.Run("Gemma 4 E-model layer with v_norm, sandwich FFN, PLE and layer scale", func(t *testing.T) {
		ops := []Op{
			rms(SlotPreAttn),
			{Kind: OpGemma4Attn, G4: Gemma4AttnParams{Eps: 1e-6, AddOne: true, OwnsKV: true, InvFreq: []float64{1}}},
			{Kind: OpNorm, Norm: NormParams{Slot: SlotPostAttn, Form: NormRMS, Eps: 1e-6, AddOne: true}},
			{Kind: OpResidual},
			rms(SlotPreMLP),
			{Kind: OpFFN, FFN: FFNParams{Form: FfnGemma4Dense, Act: ActGeluTanh}},
			{Kind: OpNorm, Norm: NormParams{Slot: SlotPostMLP, Form: NormRMS, Eps: 1e-6, AddOne: true}},
			{Kind: OpResidual},
			{Kind: OpPLE, Norms: NormEpsParams{Eps: 1e-6, AddOne: true}},
			{Kind: OpLayerScale},
		}
		lw, err := DecodeLayerWalk(ops, "test")
		if err != nil {
			t.Fatal(err)
		}
		if !lw.Attn.G4VNorm || lw.Attn.OutMode != OutPostNormResidual || !lw.HasPLE || !lw.HasLayerScale {
			t.Fatalf("Gemma 4 E-model walk = %+v", lw)
		}
	})

	t.Run("FFN-only Nemotron-H MLP layer binds SlotPreAttn", func(t *testing.T) {
		ops := []Op{
			rms(SlotPreAttn),
			{Kind: OpFFN, FFN: FFNParams{Form: FfnNonGated, Act: ActReLU2}},
			{Kind: OpResidual},
		}
		lw, err := DecodeLayerWalk(ops, "test")
		if err != nil {
			t.Fatal(err)
		}
		if lw.Mixer != MixerNone || !lw.HasFFN || lw.FFN.InputNorm.Slot != SlotPreAttn || lw.FFN.Kind != SublayerNonGatedFFN {
			t.Fatalf("FFN-only walk = %+v", lw)
		}
	})
}
