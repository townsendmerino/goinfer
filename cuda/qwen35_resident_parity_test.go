//go:build cuda && goinfer_testhooks

package cuda

import (
	"os"
	"strconv"
	"testing"

	gc "github.com/eitamring/gocudrv/cuda"
	"github.com/townsendmerino/goinfer/decoder"
)

// Whole-model parity for the Gated-DeltaNet hybrid ON CUDA: the resident runner vs the CPU
// runLayersQwen35, fed the SAME fixed token sequence, logits compared as the run gets long.
//
// Long is the point. Three of every four layers carry a recurrent state that COMPOUNDS, so the
// wiring errors this family invites — a mixer misroute, a conv ring slip, the state left un-reset,
// the value-head → key-head GVA map inverted — do not show as a bad first token. They show as a
// cosine that decays. A single-token check would pass on all of them.
//
// The gated softmax layers are gated here too, and they are the half the residency PLAN did not
// anticipate: q_proj is double width ([query ‖ gate] per head) and the context is scaled by
// sigmoid(gate) before o_proj. Reading that weight as an ordinary q_proj produces plausible logits
// from the wrong tensor, which is exactly what an argmax-only check would wave through.
func TestQwen35ResidentParityCUDA(t *testing.T) {
	if os.Getenv("GOINFER_DNET_PARITY_CUDA") == "" {
		t.Skip("qwen3.5 DeltaNet resident parity (set GOINFER_DNET_PARITY_CUDA=1)")
	}
	if err := gc.Init(); err != nil {
		t.Skipf("cuInit failed (no driver): %v", err)
	}
	if _, err := gc.GetDevice(0); err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	// BOTH siblings. The MoE one is load-bearing twice over: it is the only CUDA coverage of a
	// DeltaNet mixer paired with a sparse FFN in the same layer, AND the only coverage of the
	// SIGMOID-GATED shared expert (qwen3_5_moe carries shared_expert_gate, inherited from
	// Qwen2-MoE) that FeatMoEGatedShared claims. That feature is declared on the strength of
	// this subtest, not ahead of it.
	// int4 for the MoE one, and that is a real constraint rather than a convenience: this
	// backend's stacked-expert GEMVs are W4A8 only and decline int8 experts outright. The dense
	// sibling has no experts, so it runs at the same int8int8 the WebGPU gate uses.
	for _, fx := range []struct{ name, quant string }{
		{"qwen3_5-tiny", "int8int8"},
		{"qwen3_5_moe-tiny", "int4"},
	} {
		t.Run(fx.name, func(t *testing.T) {
			qwen35ResidentParityCUDA(t, "../decoder/testdata/"+fx.name, fx.quant)
		})
	}
	if ck := os.Getenv("GOINFER_DNET_CKPT_CUDA"); ck != "" { // a bigger/real checkpoint, opt-in
		t.Run("env", func(t *testing.T) { qwen35ResidentParityCUDA(t, ck, "int8int8") })
	}
}

func qwen35ResidentParityCUDA(t *testing.T, ckpt, quant string) {
	if _, err := os.Stat(ckpt + "/model.safetensors"); err != nil {
		// stat the WEIGHTS, not the directory: the fixture's config.json is committed while
		// model.safetensors is gitignored, so a dir-existence guard would flip this skip into a
		// hard failure on a fresh clone.
		t.Skipf("no qwen3_5 fixture at %s (run scripts/pin_qwen3_5_forward.py [--moe]): %v", ckpt, err)
	}
	if v := os.Getenv("GOINFER_DNET_QUANT_CUDA"); v != "" {
		quant = v
	}

	mRes, err := decoder.Load(ckpt, decoder.Options{Backend: "cuda", Quant: quant})
	if err != nil {
		t.Fatalf("load cuda: %v", err)
	}
	defer mRes.Close()
	rf := mRes.ResidentForwardForTest()
	if rf == nil {
		// Non-vacuity: this test is worthless if it silently measures CPU-vs-CPU, and a declined
		// residency is exactly how that happens.
		t.Fatalf("qwen3.5 did not go resident (BuildResident refused) — decode path %q; decline: %s", mRes.DecodePath(), mRes.ResidentDecline())
	}
	t.Logf("resident decode path: %s", mRes.DecodePath())

	// The CPU arm runs at the SAME weight quant, so what is being measured is WIRING, not the
	// int8 gap. The resident path still quantizes activations the CPU keeps in f32, which is the
	// floor this cosine sits on — it should be flat, not decaying.
	mCPU, err := decoder.Load(ckpt, decoder.Options{Backend: "cpu", Quant: quant})
	if err != nil {
		t.Fatalf("load cpu: %v", err)
	}
	defer mCPU.Close()

	ntok := 128
	if v := os.Getenv("GOINFER_DNET_NTOK_CUDA"); v != "" {
		if n, e := strconv.Atoi(v); e == nil {
			ntok = n
		}
	}
	_, _, _, _, _, _, vocab := mCPU.Dims()
	checks := map[int]bool{1: true, 2: true, 4: true, 8: true, 16: true, 64: true, 128: true, 512: true}

	cache := mCPU.NewCache(ntok + 1)
	rf.Reset()
	worst, first, agree := 1.0, 1.0, 0
	var firstRun [][]float32 // the resident's own opening logits, for the replay check below
	for i := range ntok {
		tok := (i*131 + 7) % vocab
		lc, err := mCPU.ForwardForTest(tok, cache)
		if err != nil {
			t.Fatalf("cpu forward[%d]: %v", i, err)
		}
		lr, err := rf.Forward(mRes.EmbedResidentForTest(tok), i)
		if err != nil {
			t.Fatalf("resident forward[%d]: %v", i, err)
		}
		if len(firstRun) < 16 {
			firstRun = append(firstRun, append([]float32(nil), lr...))
		}
		if argmaxF32(lc) == argmaxF32(lr) {
			agree++
		}
		cos, maxAbs := cosF32(lc, lr)
		if i == 0 {
			first = cos
		}
		if cos < worst {
			worst = cos
		}
		if checks[i+1] || i+1 == ntok {
			t.Logf("  tok %4d: cosine=%.6f maxAbs=%.4g argmax_match=%v", i+1, cos, maxAbs, argmaxF32(lc) == argmaxF32(lr))
		}
	}
	t.Logf("  worst cosine over %d tokens: %.6f (tok1 %.6f, drift %.4f); argmax agree %d/%d",
		ntok, worst, first, first-worst, agree, ntok)

	// DRIFT is the wiring signal, and it is checked separately from the absolute floor on
	// purpose: a recurrence that is merely coarse sits flat and low, while one that is wired
	// wrong decays. Conflating the two into a single floor loses that distinction — and it is
	// the distinction that says whether to widen a tolerance or go find a bug.
	if first-worst > 0.02 {
		t.Errorf("cosine DRIFTS over the run: tok1 %.6f → worst %.6f (Δ%.4f). The recurrent state "+
			"compounds; a decaying cosine is a wiring error, not quantization noise", first, worst, first-worst)
	}
	if worst < 0.95 {
		t.Errorf("worst cosine %.6f < 0.95 — below the resident-path floor the other families hold", worst)
	}

	// SECOND GENERATION on the same model — a REPLAY, not another CPU comparison.
	//
	// The recurrent state compounds and is NOT positional, so unlike a KV cache the next sequence
	// cannot simply overwrite it; it has to be zeroed (audit C-01). The first loop cannot see
	// that: the state buffers are allocated zeroed, so a runner that never resets still passes
	// its first generation.
	//
	// Comparing generation 2 against the CPU does not see it either — measured. Deleting the
	// DeltaNet arm of residentDecoder.Reset and re-running left the cosine above 0.95, because
	// the decay gate shrinks the stale state faster than 16 tokens of comparison can notice. So
	// this replays the FIRST run's opening tokens and requires the resident to reproduce ITS OWN
	// logits. Same inputs + same (reset) state ⇒ the same output, to f32 determinism — a bound
	// nothing but leftover state can break, and one no tolerance argument can absorb.
	const replay = 16
	rf.Reset()
	worstReplay := 1.0
	for i := range replay {
		lr, err := rf.Forward(mRes.EmbedResidentForTest((i*131+7)%vocab), i)
		if err != nil {
			t.Fatalf("resident replay[%d]: %v", i, err)
		}
		if cos, _ := cosF32(firstRun[i], lr); cos < worstReplay {
			worstReplay = cos
		}
	}
	t.Logf("  replay of the first %d tokens after Reset: worst self-cosine %.9f", replay, worstReplay)
	if worstReplay < 0.9999999 {
		t.Errorf("replaying the same tokens after Reset gives DIFFERENT logits (worst self-cosine "+
			"%.9f): the recurrent state carried over from the first sequence. residentDecoder.Reset "+
			"must zero {win, dnState}, not just Mamba's {win, ssm}", worstReplay)
	}
}
